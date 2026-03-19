"""
Step 7b: Evaluate the Action Chunking Policy (06b)
===================================================
Usage:
    python 07b_evaluate_chunking.py --checkpoint /tmp/cabinet_policy_06b/best_policy.pt
    python 07b_evaluate_chunking.py --checkpoint ... --num_rollouts 50 --split target
"""

import argparse
import os
import sys
from collections import deque

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_policy(checkpoint_path, device):
    import torch
    import torch.nn as nn

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    input_dim  = ckpt["input_dim"]
    output_dim = ckpt["output_dim"]
    hidden_dim = ckpt.get("config", {}).get("hidden_dim", 512)

    class ChunkMLP(nn.Module):
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
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
                nn.Tanh(),
            )

        def forward(self, x):
            x = self.input_proj(x)
            x = self.act1(x + self.block1(x))
            return self.head(x)

    model = ChunkMLP().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded          : {checkpoint_path}")
    print(f"  Epoch         : {ckpt['epoch']},  loss={ckpt['loss']:.6f}")
    print(f"  History len   : {ckpt['history_len']}")
    print(f"  Chunk size    : {ckpt['chunk_size']}")
    print(f"  Action dim    : {ckpt['action_dim']}")

    return model, ckpt


def extract_state(obs, state_columns, single_state_dim):
    parts = []
    if state_columns and state_columns != ["synthetic"]:
        for key in state_columns:
            val = obs.get(key)
            if val is not None and isinstance(val, np.ndarray):
                parts.append(val.flatten())
    else:
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


def run_evaluation(model, ckpt, num_rollouts, max_steps, split, video_path, seed):
    import torch
    import imageio

    device        = next(model.parameters()).device
    history_len   = ckpt["history_len"]
    chunk_size    = ckpt["chunk_size"]
    action_dim    = ckpt["action_dim"]
    single_dim    = ckpt["input_dim"] // history_len
    state_columns = ckpt.get("state_columns")

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )

    video_writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {"successes": [], "episode_lengths": [], "rewards": []}

    for ep in range(num_rollouts):
        obs  = env.reset()
        lang = env.get_ep_meta().get("lang", "")

        zero        = np.zeros(single_dim, dtype=np.float32)
        history_buf = deque([zero] * history_len, maxlen=history_len)
        action_queue = deque()

        ep_reward = 0.0
        success   = False

        for step in range(max_steps):
            raw = extract_state(obs, state_columns, single_dim)
            history_buf.append(raw)

            if not action_queue:
                history_vec = np.concatenate(list(history_buf))
                with torch.no_grad():
                    x = torch.from_numpy(history_vec).unsqueeze(0).to(device)
                    chunk_flat = model(x).cpu().numpy().squeeze(0)
                for k in range(chunk_size):
                    action_queue.append(chunk_flat[k * action_dim:(k + 1) * action_dim])

            action  = action_queue.popleft()
            env_dim = env.action_dim
            if len(action) < env_dim:
                action = np.pad(action, (0, env_dim - len(action)))
            elif len(action) > env_dim:
                action = action[:env_dim]

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(frame)

            if any(
                env.fxtr.is_open(env=env, joint_names=[j])
                for j in env.fxtr.door_joint_names
                ):
                success = True
                break

        results["successes"].append(success)
        results["episode_lengths"].append(step + 1)
        results["rewards"].append(ep_reward)

        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep+1:3d}/{num_rollouts}: {status:7s} "
              f"(steps={step+1:4d}, reward={ep_reward:.1f}) "
              f'layout={env.layout_id}, style={env.style_id}, task="{lang}"')

    if video_writer:
        video_writer.close()
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   type=str,
                        default="/tmp/cabinet_policy_06b/best_policy.pt")
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--max_steps",   type=int, default=500)
    parser.add_argument("--split",       type=str, default="pretrain",
                        choices=["pretrain", "target"])
    parser.add_argument("--video_path",  type=str, default=None)
    parser.add_argument("--seed",        type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: pip install torch"); sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Action Chunking Evaluation (06b)")
    print("=" * 60)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_policy(args.checkpoint, device)

    print_section(f"Evaluating on '{args.split}' ({args.num_rollouts} episodes)")

    results = run_evaluation(
        model=model, ckpt=ckpt,
        num_rollouts=args.num_rollouts, max_steps=args.max_steps,
        split=args.split, video_path=args.video_path, seed=args.seed,
    )

    print_section("Results")
    n_success    = sum(results["successes"])
    success_rate = n_success / args.num_rollouts * 100
    print(f"  Split         : {args.split}")
    print(f"  Successes     : {n_success}/{args.num_rollouts}")
    print(f"  Success rate  : {success_rate:.1f}%")
    print(f"  Avg ep length : {np.mean(results['episode_lengths']):.1f} steps")
    print(f"  Avg reward    : {np.mean(results['rewards']):.3f}")
    if args.video_path:
        print(f"  Video         : {args.video_path}")


if __name__ == "__main__":
    main()
