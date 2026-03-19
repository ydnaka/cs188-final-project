"""
Step 8a: Visualize the Temporal Context Policy Rollout (06a)
=============================================================
Drop-in replacement for 08_visualize_policy_rollout.py that works with
checkpoints saved by 06a_train_temporal.py.

Usage:
    # Off-screen video (works headless / WSL)
    python 08a_visualize_temporal.py --checkpoint /tmp/cabinet_policy_06a/best_policy.pt --offscreen

    # On-screen viewer (Mac: use mjpython)
    python 08a_visualize_temporal.py --checkpoint ...
    mjpython 08a_visualize_temporal.py --checkpoint ...

    # Multiple episodes, slower playback
    python 08a_visualize_temporal.py --checkpoint ... --num_episodes 3 --max_fr 10 --offscreen
"""

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

    input_dim   = ckpt["input_dim"]
    action_dim  = ckpt["action_dim"]
    history_len = ckpt["history_len"]
    hidden_dim  = ckpt.get("config", {}).get("hidden_dim", 512)

    class TemporalMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
            self.block1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.act1 = nn.ReLU()
            self.block2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.act1(x + self.block1(x))
            return self.block2(x)

    model = TemporalMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    single_state_dim = input_dim // history_len

    print(f"Checkpoint      : {checkpoint_path}")
    print(f"  Epoch         : {ckpt['epoch']},  loss={ckpt['loss']:.6f}")
    print(f"  History len   : {history_len}")
    print(f"  Single state  : {single_state_dim}  →  input: {input_dim}")
    print(f"  Action dim    : {action_dim}")

    return model, single_state_dim, action_dim, history_len, ckpt


def extract_raw_state(obs, single_state_dim):
    parts = []
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, np.ndarray) and not key.endswith("_image"):
            parts.append(val.flatten())
    if not parts:
        return np.zeros(single_state_dim, dtype=np.float32)
    state = np.concatenate(parts).astype(np.float32)
    if len(state) < single_state_dim:
        state = np.pad(state, (0, single_state_dim - len(state)))
    elif len(state) > single_state_dim:
        state = state[:single_state_dim]
    return state


# ---------------------------------------------------------------------------
# On-screen rollout
# ---------------------------------------------------------------------------

def run_onscreen(model, single_state_dim, action_dim, history_len, args):
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
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        print(f"  Task   : {ep_meta.get('lang', '')}")
        print(f"  Layout : {env.layout_id}   Style: {env.style_id}")

        first_state = extract_raw_state(obs, single_state_dim)
        history_buf = deque([first_state] * history_len, maxlen=history_len)

        success = False
        hold_count = 0

        for step in range(args.max_steps):
            raw_state = extract_raw_state(obs, single_state_dim)
            history_buf.append(raw_state)
            history_vec = np.concatenate(list(history_buf))

            with torch.no_grad():
                x = torch.from_numpy(history_vec).unsqueeze(0).to(device)
                action = model(x).cpu().numpy().squeeze(0)

            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            obs, reward, done, info = env.step(action)

            if step % 20 == 0:
                is_open = env._check_success()
                status = "cabinet OPEN" if is_open else "in progress"
                print(f"  step {step:4d}  reward={reward:+.3f}  [{status}]")

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
# Off-screen rollout with video
# ---------------------------------------------------------------------------

def run_offscreen(model, single_state_dim, action_dim, history_len, args):
    import torch
    import imageio
    from robocasa.utils.env_utils import create_env

    device = next(model.parameters()).device

    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    cam_h, cam_w = 512, 768
    successes = 0
    all_frames = []

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        env = create_env(
            env_name="OpenCabinet",
            render_onscreen=False,
            seed=args.seed + ep,
            camera_widths=cam_w,
            camera_heights=cam_h,
        )
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        print(f"  Task   : {ep_meta.get('lang', '')}")
        print(f"  Layout : {env.layout_id}   Style: {env.style_id}")

        first_state = extract_raw_state(obs, single_state_dim)
        history_buf = deque([first_state] * history_len, maxlen=history_len)

        success = False
        hold_count = 0
        ep_frames = []

        for step in range(args.max_steps):
            raw_state = extract_raw_state(obs, single_state_dim)
            history_buf.append(raw_state)
            history_vec = np.concatenate(list(history_buf))

            with torch.no_grad():
                x = torch.from_numpy(history_vec).unsqueeze(0).to(device)
                action = model(x).cpu().numpy().squeeze(0)

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
                status = "cabinet OPEN" if is_open else "in progress"
                print(f"  step {step:4d}  reward={reward:+.3f}  [{status}]")

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
    parser = argparse.ArgumentParser(description="Visualize the 06a temporal policy")
    parser.add_argument("--checkpoint",   type=str,
                        default="/tmp/cabinet_policy_06a/best_policy.pt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps",    type=int, default=300)
    parser.add_argument("--offscreen",    action="store_true")
    parser.add_argument("--video_path",   type=str, default="/tmp/policy_rollout_06a.mp4")
    parser.add_argument("--fps",          type=int, default=20)
    parser.add_argument("--max_fr",       type=int, default=20)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Temporal Policy Visualizer (06a)")
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
    model, single_state_dim, action_dim, history_len, ckpt = load_policy(args.checkpoint, device)

    print(f"Device    : {device}")
    print(f"Mode      : {'off-screen (video)' if args.offscreen else 'on-screen (viewer)'}")
    print(f"Episodes  : {args.num_episodes}")
    print(f"Max steps : {args.max_steps}")
    if args.offscreen:
        print(f"Output    : {args.video_path}")
    print()

    if args.offscreen:
        run_offscreen(model, single_state_dim, action_dim, history_len, args)
    else:
        print("Opening viewer window...")
        run_onscreen(model, single_state_dim, action_dim, history_len, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
