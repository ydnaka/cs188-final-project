"""
Step 8c: Visualize the MLP Diffusion Policy Rollout (06c)
==========================================================
Drop-in replacement for 08_visualize_policy_rollout.py that works with
checkpoints saved by 06c_train_diffusion.py.

Usage:
    python 08c_visualize_diffusion.py --checkpoint /tmp/cabinet_policy_06c/best_policy.pt --offscreen
    mjpython 08c_visualize_diffusion.py --checkpoint ...
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

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dim   = ckpt["state_dim"]
    action_dim  = ckpt["action_dim"]
    history_len = ckpt["history_len"]
    chunk_size  = ckpt["chunk_size"]
    diff_steps  = ckpt["diff_steps"]
    hidden_dim  = ckpt.get("config", {}).get("hidden_dim", 512)
    state_columns = ckpt["state_columns"]

    class SinusoidalEmbedding(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            device = t.device
            half  = self.dim // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(half, dtype=torch.float32, device=device) / half
            )
            args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
            return torch.cat([args.sin(), args.cos()], dim=-1)

    class DiffusionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            t_emb_dim = 64
            self.t_emb = SinusoidalEmbedding(t_emb_dim)
            in_dim = state_dim + action_dim + t_emb_dim
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.block1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.act1 = nn.ReLU()
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
            )

        def forward(self, state, noisy_action, t):
            t_emb = self.t_emb(t)
            x     = torch.cat([state, noisy_action, t_emb], dim=-1)
            x     = self.input_proj(x)
            x     = self.act1(x + self.block1(x))
            return self.head(x)

    model = DiffusionMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    single_state_dim = state_dim // history_len

    print(f"Checkpoint    : {checkpoint_path}")
    print(f"  Epoch       : {ckpt['epoch']},  loss={ckpt['loss']:.6f}")
    print(f"  State dim   : {state_dim}")
    print(f"  Action dim  : {action_dim}")
    print(f"  History len : {history_len}")
    print(f"  Chunk size  : {chunk_size}")
    print(f"  Diff steps  : {diff_steps}")

    return model, single_state_dim, action_dim, history_len, chunk_size, diff_steps, state_columns, ckpt


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def extract_raw_state(obs, state_columns, single_state_dim):
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
        return np.zeros(single_state_dim, dtype=np.float32)
    state = np.concatenate(parts).astype(np.float32)
    if len(state) < single_state_dim:
        state = np.pad(state, (0, single_state_dim - len(state)))
    elif len(state) > single_state_dim:
        state = state[:single_state_dim]
    return state


# ---------------------------------------------------------------------------
# Denoising loop
# ---------------------------------------------------------------------------

def ddpm_denoise(model, state_vec, action_dim, diff_steps, device):
    import torch
    betas      = torch.linspace(1e-4, 0.02, diff_steps, device=device)
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    state_t    = torch.from_numpy(state_vec).unsqueeze(0).to(device)
    x          = torch.randn(1, action_dim, device=device)

    with torch.no_grad():
        for i in reversed(range(diff_steps)):
            t_tensor     = torch.tensor([i], device=device, dtype=torch.long)
            pred_noise   = model(state_t, x, t_tensor)
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

def run_onscreen(model, single_state_dim, action_dim, history_len,
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

    single_action_dim = action_dim // chunk_size
    successes = 0

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        obs     = env.reset()
        ep_meta = env.get_ep_meta()
        print(f"  Task   : {ep_meta.get('lang', '')}")
        print(f"  Layout : {env.layout_id}   Style: {env.style_id}")

        zero_state  = np.zeros(single_state_dim, dtype=np.float32)
        history_buf = deque([zero_state] * history_len, maxlen=history_len)
        action_queue = deque()

        success    = False
        hold_count = 0

        for step in range(args.max_steps):
            raw_state = extract_raw_state(obs, state_columns, single_state_dim)
            history_buf.append(raw_state)
            state_vec = np.concatenate(list(history_buf))

            if not action_queue:
                action_chunk = ddpm_denoise(model, state_vec, action_dim, diff_steps, device)
                for k in range(chunk_size):
                    action_queue.append(action_chunk[k * single_action_dim:(k + 1) * single_action_dim])

            action = action_queue.popleft()
            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            obs, reward, done, info = env.step(action)

            if step % 20 == 0:
                is_open = env._check_success()
                print(f"  step {step:4d}  reward={reward:+.3f}  "
                      f"[{'cabinet OPEN' if is_open else 'in progress'}]")

            if env._check_success():
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

            time.sleep(1.0 / args.max_fr)

        print(f"\n  Result: {'SUCCESS' if success else 'did not open cabinet'}")
        if success:
            successes += 1

    env.close()
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ---------------------------------------------------------------------------
# Off-screen rollout
# ---------------------------------------------------------------------------

def run_offscreen(model, single_state_dim, action_dim, history_len,
                  chunk_size, diff_steps, state_columns, args):
    import torch
    import imageio
    from robocasa.utils.env_utils import create_env

    device = next(model.parameters()).device

    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    cam_h, cam_w      = 512, 768
    single_action_dim = action_dim // chunk_size
    successes         = 0
    all_frames        = []

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

        zero_state   = np.zeros(single_state_dim, dtype=np.float32)
        history_buf  = deque([zero_state] * history_len, maxlen=history_len)
        action_queue = deque()

        success    = False
        hold_count = 0
        ep_frames  = []

        for step in range(args.max_steps):
            raw_state = extract_raw_state(obs, state_columns, single_state_dim)
            history_buf.append(raw_state)
            state_vec = np.concatenate(list(history_buf))

            if not action_queue:
                action_chunk = ddpm_denoise(model, state_vec, action_dim, diff_steps, device)
                for k in range(chunk_size):
                    action_queue.append(action_chunk[k * single_action_dim:(k + 1) * single_action_dim])

            action = action_queue.popleft()
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
                is_open = env._check_success()
                print(f"  step {step:4d}  reward={reward:+.3f}  "
                      f"[{'cabinet OPEN' if is_open else 'in progress'}]")

            if env._check_success():
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

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
    parser = argparse.ArgumentParser(description="Visualize the 06c MLP diffusion policy")
    parser.add_argument("--checkpoint",   type=str,
                        default="/tmp/cabinet_policy_06c/best_policy.pt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps",    type=int, default=300)
    parser.add_argument("--offscreen",    action="store_true")
    parser.add_argument("--video_path",   type=str, default="/tmp/policy_rollout_06c.mp4")
    parser.add_argument("--fps",          type=int, default=20)
    parser.add_argument("--max_fr",       type=int, default=20)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - MLP Diffusion Policy Visualizer (06c)")
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
    model, single_state_dim, action_dim, history_len, chunk_size, diff_steps, state_columns, ckpt = \
        load_policy(args.checkpoint, device)

    print(f"Device    : {device}")
    print(f"Mode      : {'off-screen (video)' if args.offscreen else 'on-screen (viewer)'}")
    print(f"Episodes  : {args.num_episodes}")
    print(f"Max steps : {args.max_steps}")
    if args.offscreen:
        print(f"Output    : {args.video_path}")
    print()

    if args.offscreen:
        run_offscreen(model, single_state_dim, action_dim, history_len,
                      chunk_size, diff_steps, state_columns, args)
    else:
        run_onscreen(model, single_state_dim, action_dim, history_len,
                     chunk_size, diff_steps, state_columns, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
