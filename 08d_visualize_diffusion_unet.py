"""
Step 8d: Visualize the UNet Diffusion Policy Rollout (06d)
===========================================================
Drop-in replacement for 08_visualize_policy_rollout.py that works with
checkpoints saved by 06d_train_diffusion_unet.py.

Usage:
    python 08d_visualize_diffusion_unet.py --checkpoint /tmp/cabinet_policy_06d/best_policy.pt --offscreen
    mjpython 08d_visualize_diffusion_unet.py --checkpoint ...
"""

import math
import os
import sys
from collections import deque

_OFFSCREEN = "--offscreen" in sys.argv

if _OFFSCREEN:
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
else:
    if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
        _env = dict(os.environ)
        _changed = False
        if _env.get("WAYLAND_DISPLAY"):
            if not _env.get("DISPLAY", "").startswith(":"):
                _env["DISPLAY"] = ":0"
                _changed = True
            if _env.get("GALLIUM_DRIVER") != "llvmpipe":
                _env["GALLIUM_DRIVER"] = "llvmpipe"
                _changed = True
            if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
                _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
                _changed = True
        if _changed:
            _env["__TELEOP_DISPLAY_OK"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, _env)
        else:
            os.environ["__TELEOP_DISPLAY_OK"] = "1"

import argparse
import time

import numpy as np
import robocasa  # noqa: F401
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy(checkpoint_path, device):
    import torch
    import torch.nn as nn
    from einops.layers.torch import Rearrange

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dim           = ckpt["state_dim"]
    action_dim          = ckpt["action_dim"]
    history_len         = ckpt["history_len"]
    chunk_size          = ckpt["chunk_size"]
    diff_steps          = ckpt["diff_steps"]
    down_dims           = tuple(ckpt["down_dims"])
    diff_step_embed_dim = ckpt["diff_step_embed_dim"]
    state_columns       = ckpt["state_columns"]

    class SinusoidalPosEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            device = x.device
            half   = self.dim // 2
            emb    = math.log(10000) / (half - 1)
            emb    = torch.exp(torch.arange(half, device=device) * -emb)
            emb    = x[:, None] * emb[None, :]
            return torch.cat([emb.sin(), emb.cos()], dim=-1)

    class Conv1dBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, n_groups=8):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.GroupNorm(n_groups, out_ch),
                nn.Mish(),
            )

        def forward(self, x):
            return self.block(x)

    class Downsample1d(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

        def forward(self, x):
            return self.conv(x)

    class Upsample1d(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

        def forward(self, x):
            return self.conv(x)

    class ConditionalResidualBlock1D(nn.Module):
        def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
            super().__init__()
            self.blocks = nn.ModuleList([
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
            ])
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange('b t -> b t 1'),
            )
            self.residual_conv = (
                nn.Conv1d(in_channels, out_channels, 1)
                if in_channels != out_channels else nn.Identity()
            )

        def forward(self, x, cond):
            out = self.blocks[0](x) + self.cond_encoder(cond)
            out = self.blocks[1](out)
            return out + self.residual_conv(x)

    class ConditionalUnet1D(nn.Module):
        def __init__(self):
            super().__init__()
            dsed = diff_step_embed_dim
            self.diffusion_step_encoder = nn.Sequential(
                SinusoidalPosEmb(dsed),
                nn.Linear(dsed, dsed * 4),
                nn.Mish(),
                nn.Linear(dsed * 4, dsed),
            )
            cond_dim = dsed + state_dim
            all_dims = [action_dim] + list(down_dims)
            in_out   = list(zip(all_dims[:-1], all_dims[1:]))

            self.down_modules = nn.ModuleList([])
            for ind, (d_in, d_out) in enumerate(in_out):
                is_last = ind >= len(in_out) - 1
                self.down_modules.append(nn.ModuleList([
                    ConditionalResidualBlock1D(d_in,  d_out, cond_dim),
                    ConditionalResidualBlock1D(d_out, d_out, cond_dim),
                    Downsample1d(d_out) if not is_last else nn.Identity(),
                ]))

            mid_dim = all_dims[-1]
            self.mid_modules = nn.ModuleList([
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim),
                ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim),
            ])

            self.up_modules = nn.ModuleList([])
            for ind, (d_in, d_out) in enumerate(reversed(in_out[1:])):
                is_last = ind >= len(in_out) - 1
                self.up_modules.append(nn.ModuleList([
                    ConditionalResidualBlock1D(d_out * 2, d_in, cond_dim),
                    ConditionalResidualBlock1D(d_in,      d_in, cond_dim),
                    Upsample1d(d_in) if not is_last else nn.Identity(),
                ]))

            start_dim = down_dims[0]
            self.final_conv = nn.Sequential(
                Conv1dBlock(start_dim, start_dim, kernel_size=3),
                nn.Conv1d(start_dim, action_dim, 1),
            )

        def forward(self, sample, timestep, global_cond):
            import einops
            x = einops.rearrange(sample, 'b h t -> b t h')
            if not torch.is_tensor(timestep):
                timestep = torch.tensor([timestep], dtype=torch.long, device=x.device)
            timestep = timestep.expand(x.shape[0])
            t_emb = self.diffusion_step_encoder(timestep)
            cond  = torch.cat([t_emb, global_cond], dim=-1)
            h = []
            for resnet, resnet2, downsample in self.down_modules:
                x = resnet(x, cond)
                x = resnet2(x, cond)
                h.append(x)
                x = downsample(x)
            for mid in self.mid_modules:
                x = mid(x, cond)
            for resnet, resnet2, upsample in self.up_modules:
                x = torch.cat((x, h.pop()), dim=1)
                x = resnet(x, cond)
                x = resnet2(x, cond)
                x = upsample(x)
            x = self.final_conv(x)
            import einops
            return einops.rearrange(x, 'b t h -> b h t')

    model = ConditionalUnet1D().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint    : {checkpoint_path}")
    print(f"  Epoch       : {ckpt['epoch']},  loss={ckpt['loss']:.6f}")
    print(f"  State dim   : {state_dim}")
    print(f"  Action dim  : {action_dim}")
    print(f"  History len : {history_len}")
    print(f"  Chunk size  : {chunk_size}")
    print(f"  Diff steps  : {diff_steps}")
    print(f"  Down dims   : {down_dims}")

    return model, state_dim, action_dim, history_len, chunk_size, diff_steps, state_columns, ckpt


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def extract_raw_state(obs, state_columns, state_dim):
    parts = []
    for col in state_columns:
        key = col.replace("observation.", "robot0_") if col.startswith("observation.") else col
        val = obs.get(col) if col in obs else obs.get(key)
        if val is not None and isinstance(val, np.ndarray):
            parts.append(val.flatten())
    if not parts:
        for k in sorted(obs.keys()):
            if isinstance(obs[k], np.ndarray) and not k.endswith("_image"):
                parts.append(obs[k].flatten())
    if not parts:
        return np.zeros(state_dim, dtype=np.float32)
    state = np.concatenate(parts).astype(np.float32)
    if len(state) < state_dim:
        state = np.pad(state, (0, state_dim - len(state)))
    elif len(state) > state_dim:
        state = state[:state_dim]
    return state


# ---------------------------------------------------------------------------
# Denoising loop
# ---------------------------------------------------------------------------

def ddpm_denoise(model, state_vec, action_dim, chunk_size, diff_steps, device):
    import torch
    betas      = torch.linspace(1e-4, 0.02, diff_steps, device=device)
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    state_t    = torch.from_numpy(state_vec).unsqueeze(0).to(device)
    x          = torch.randn(1, chunk_size, action_dim, device=device)

    with torch.no_grad():
        for i in reversed(range(diff_steps)):
            t_tensor     = torch.tensor([i], device=device, dtype=torch.long)
            pred_noise   = model(x, t_tensor, global_cond=state_t)
            alpha_bar_t  = alpha_bars[i]
            alpha_bar_t1 = alpha_bars[i - 1] if i > 0 else torch.tensor(1.0, device=device)
            x0_pred      = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
            x0_pred      = x0_pred.clamp(-1, 1)
            if i > 0:
                posterior_mean = (
                    alpha_bar_t1.sqrt() * betas[i] / (1 - alpha_bar_t) * x0_pred
                    + alphas[i].sqrt() * (1 - alpha_bar_t1) / (1 - alpha_bar_t) * x
                )
                posterior_var = betas[i] * (1 - alpha_bar_t1) / (1 - alpha_bar_t)
                x = posterior_mean + posterior_var.sqrt() * torch.randn_like(x)
            else:
                x = x0_pred

    return x.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# On-screen rollout
# ---------------------------------------------------------------------------

def run_onscreen(model, state_dim, action_dim, history_len,
                 chunk_size, diff_steps, state_columns, args):
    import torch

    device = next(model.parameters()).device

    env = robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )
    env = VisualizationWrapper(env)

    successes = 0
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        obs     = env.reset()
        ep_meta = env.get_ep_meta()
        print(f"  Task   : {ep_meta.get('lang', '')}")
        print(f"  Layout : {env.layout_id}   Style: {env.style_id}")

        zero_state   = np.zeros(state_dim, dtype=np.float32)
        history_buf  = deque([zero_state] * history_len, maxlen=history_len)
        action_queue = deque()
        success      = False

        for step in range(args.max_steps):
            raw_state = extract_raw_state(obs, state_columns, state_dim)
            history_buf.append(raw_state)
            state_vec = np.concatenate(list(history_buf))

            if not action_queue:
                chunk = ddpm_denoise(model, raw_state, action_dim, chunk_size, diff_steps, device)
                for k in range(chunk_size):
                    action_queue.append(chunk[k])

            action  = action_queue.popleft()
            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            obs, reward, done, info = env.step(action)

            if step % 20 == 0:
                is_open = any(env.fxtr.is_open(env=env, joint_names=[j])
                              for j in env.fxtr.door_joint_names)
                print(f"  step {step:4d}  reward={reward:+.3f}  "
                      f"[{'cabinet OPEN' if is_open else 'in progress'}]")

            if any(env.fxtr.is_open(env=env, joint_names=[j])
                   for j in env.fxtr.door_joint_names):
                success = True
                break

            time.sleep(1.0 / args.max_fr)

        print(f"\n  Result: {'SUCCESS' if success else 'did not open cabinet'}")
        if success:
            successes += 1

    env.close()
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ---------------------------------------------------------------------------
# Off-screen rollout
# ---------------------------------------------------------------------------

def run_offscreen(model, state_dim, action_dim, history_len,
                  chunk_size, diff_steps, state_columns, args):
    import torch
    import imageio
    from robocasa.utils.env_utils import create_env

    device    = next(model.parameters()).device
    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    cam_h, cam_w = 512, 768
    successes    = 0
    all_frames   = []

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        env = create_env(
            env_name="OpenCabinet",
            render_onscreen=False,
            seed=args.seed + ep,
            camera_widths=cam_w,
            camera_heights=cam_h,
        )
        obs     = env.reset()
        ep_meta = env.get_ep_meta()
        print(f"  Task   : {ep_meta.get('lang', '')}")
        print(f"  Layout : {env.layout_id}   Style: {env.style_id}")

        zero_state   = np.zeros(state_dim, dtype=np.float32)
        history_buf  = deque([zero_state] * history_len, maxlen=history_len)
        action_queue = deque()
        success      = False
        ep_frames    = []

        for step in range(args.max_steps):
            raw_state = extract_raw_state(obs, state_columns, state_dim)
            history_buf.append(raw_state)
            state_vec = np.concatenate(list(history_buf))

            if not action_queue:
                chunk = ddpm_denoise(model, raw_state, action_dim, chunk_size, diff_steps, device)
                for k in range(chunk_size):
                    action_queue.append(chunk[k])

            action  = action_queue.popleft()
            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            obs, reward, done, info = env.step(action)

            frame = env.sim.render(
                height=cam_h, width=cam_w, camera_name="robot0_agentview_center"
            )[::-1]
            ep_frames.append(frame)

            if step % 20 == 0:
                is_open = any(env.fxtr.is_open(env=env, joint_names=[j])
                              for j in env.fxtr.door_joint_names)
                print(f"  step {step:4d}  reward={reward:+.3f}  "
                      f"[{'cabinet OPEN' if is_open else 'in progress'}]")

            if any(env.fxtr.is_open(env=env, joint_names=[j])
                   for j in env.fxtr.door_joint_names):
                success = True
                break

        print(f"  Result: {'SUCCESS' if success else 'did not open cabinet'}  ({len(ep_frames)} frames)")
        if success:
            successes += 1

        all_frames.extend(ep_frames)
        env.close()

    print(f"\nWriting {len(all_frames)} frames to {args.video_path} ...")
    with imageio.get_writer(args.video_path, fps=args.fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    print(f"Video saved: {args.video_path}")
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize the 06d UNet diffusion policy")
    parser.add_argument("--checkpoint",   type=str,
                        default="/tmp/cabinet_policy_06d/best_policy.pt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps",    type=int, default=300)
    parser.add_argument("--offscreen",    action="store_true")
    parser.add_argument("--video_path",   type=str, default="/tmp/policy_rollout_06d.mp4")
    parser.add_argument("--fps",          type=int, default=20)
    parser.add_argument("--max_fr",       type=int, default=20)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - UNet Diffusion Policy Visualizer (06d)")
    print("=" * 60)
    print()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required.  pip install torch")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, state_dim, action_dim, history_len, chunk_size, diff_steps, state_columns, ckpt = \
        load_policy(args.checkpoint, device)

    print(f"Device    : {device}")
    print(f"Mode      : {'off-screen (video)' if args.offscreen else 'on-screen (viewer)'}")
    print(f"Episodes  : {args.num_episodes}")
    print(f"Max steps : {args.max_steps}")
    if args.offscreen:
        print(f"Output    : {args.video_path}")
    print()

    if args.offscreen:
        run_offscreen(model, state_dim, action_dim, history_len,
                      chunk_size, diff_steps, state_columns, args)
    else:
        run_onscreen(model, state_dim, action_dim, history_len,
                     chunk_size, diff_steps, state_columns, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
