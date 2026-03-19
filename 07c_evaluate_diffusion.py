"""
Step 7c: Evaluate the Diffusion Policy (06c)
=============================================
Drop-in replacement for 07_evaluate_policy.py that works with checkpoints
saved by 06c_train_diffusion.py.

Key differences from 07b:
  - Reconstructs DiffusionMLP with SinusoidalEmbedding
  - Inference uses a DDPM denoising loop instead of a chunk execution queue
  - Reads diff_steps, chunk_size, history_len from checkpoint

Usage:
    python 07c_evaluate_diffusion.py --checkpoint /tmp/cabinet_policy_06c/best_policy.pt
    python 07c_evaluate_diffusion.py --checkpoint ... --num_rollouts 50
    python 07c_evaluate_diffusion.py --checkpoint ... --split target
    python 07c_evaluate_diffusion.py --checkpoint ... --video_path /tmp/eval_06c.mp4
"""

import argparse
import math
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

    # --- Rebuild exact same classes as 06c ---
    class SinusoidalEmbedding(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            device = t.device
            half   = self.dim // 2
            freqs  = torch.exp(
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

    print(f"Loaded policy   : {checkpoint_path}")
    print(f"  Epoch         : {ckpt['epoch']},  loss={ckpt['loss']:.6f}")
    print(f"  State dim     : {state_dim}")
    print(f"  Action dim    : {action_dim}")
    print(f"  History len   : {history_len}")
    print(f"  Chunk size    : {chunk_size}")
    print(f"  Diff steps    : {diff_steps}")

    return model, state_dim, action_dim, history_len, chunk_size, diff_steps, state_columns


# ---------------------------------------------------------------------------
# State extraction — uses locked column order from checkpoint
# ---------------------------------------------------------------------------

def extract_raw_state(obs, state_columns, state_dim):
    """
    Build state vector using the column order locked at training time.
    Falls back to sorted keys if a column isn't found in obs.
    """
    parts = []
    for col in state_columns:
        # Map parquet column names to obs dict keys
        # e.g. "observation.state" -> try stripping "observation." prefix
        key = col.replace("observation.", "robot0_") if col.startswith("observation.") else col
        # Try the direct key first, then the stripped version
        val = obs.get(col) if col in obs else obs.get(key)
        if val is not None and isinstance(val, np.ndarray):
            parts.append(val.flatten())

    # Fallback: sort all non-image keys
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

def ddpm_denoise(model, state_vec, action_dim, diff_steps, device):
    """
    Run the DDPM reverse process to sample one action from noise.

    Returns a numpy array of shape (action_dim,).
    """
    import torch

    # Linear beta schedule — same as what 06c used during training
    betas      = torch.linspace(1e-4, 0.02, diff_steps, device=device)
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    state_t = torch.from_numpy(state_vec).unsqueeze(0).to(device)

    # Start from pure noise
    x = torch.randn(1, action_dim, device=device)

    with torch.no_grad():
        for i in reversed(range(diff_steps)):
            t_tensor   = torch.tensor([i], device=device, dtype=torch.long)
            pred_noise = model(state_t, x, t_tensor)

            alpha_bar_t  = alpha_bars[i]
            alpha_bar_t1 = alpha_bars[i - 1] if i > 0 else torch.tensor(1.0, device=device)

            # DDPM posterior mean
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

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
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(model, state_dim, action_dim, history_len, chunk_size,
                   diff_steps, state_columns, num_rollouts, max_steps,
                   split, video_path, seed):
    import torch
    import imageio

    device = next(model.parameters()).device

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
        obs     = env.reset()
        ep_meta = env.get_ep_meta()
        lang    = ep_meta.get("lang", "")

        # Zero-pad history buffer at episode start
        zero_state  = np.zeros(state_dim, dtype=np.float32)
        history_buf = deque([zero_state] * history_len, maxlen=history_len)

        # Chunk execution queue
        action_queue = deque()

        ep_reward = 0.0
        success   = False

        for step in range(max_steps):
            raw_state = extract_raw_state(obs, state_columns, state_dim)
            history_buf.append(raw_state)
            state_vec = np.concatenate(list(history_buf))

            # Re-query when the queue is empty
            if not action_queue:
                action_chunk = ddpm_denoise(
                    model, raw_state, action_dim,
                    diff_steps, device
                )
                # Split flat chunk into individual actions
                single_action_dim = action_dim // chunk_size
                for k in range(chunk_size):
                    action_queue.append(action_chunk[k * single_action_dim:(k + 1) * single_action_dim])

            action = action_queue.popleft()

            env_action_dim = env.action_dim
            if len(action) < env_action_dim:
                action = np.pad(action, (0, env_action_dim - len(action)))
            elif len(action) > env_action_dim:
                action = action[:env_action_dim]

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
        print(
            f"  Episode {ep+1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={step+1:4d}, reward={ep_reward:.1f}) "
            f'layout={env.layout_id}, style={env.style_id}, task="{lang}"'
        )

    if video_writer:
        video_writer.close()
    env.close()
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the 06c diffusion policy")
    parser.add_argument("--checkpoint",   type=str,
                        default="/tmp/cabinet_policy_06c/best_policy.pt")
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--max_steps",    type=int, default=500)
    parser.add_argument("--split",        type=str, default="pretrain",
                        choices=["pretrain", "target"])
    parser.add_argument("--video_path",   type=str, default=None)
    parser.add_argument("--seed",         type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch required.  pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Diffusion Policy Evaluation (06c)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model, state_dim, action_dim, history_len, chunk_size, diff_steps, state_columns = \
        load_policy(args.checkpoint, device)

    print_section(f"Evaluating on '{args.split}' split  ({args.num_rollouts} episodes)")

    results = run_evaluation(
        model=model,
        state_dim=state_dim,
        action_dim=action_dim,
        history_len=history_len,
        chunk_size=chunk_size,
        diff_steps=diff_steps,
        state_columns=state_columns,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
    )

    print_section("Results")
    num_success  = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100
    print(f"  Split         : {args.split}")
    print(f"  Episodes      : {args.num_rollouts}")
    print(f"  Successes     : {num_success}/{args.num_rollouts}")
    print(f"  Success rate  : {success_rate:.1f}%")
    print(f"  Avg ep length : {np.mean(results['episode_lengths']):.1f} steps")
    print(f"  Avg reward    : {np.mean(results['rewards']):.3f}")
    if args.video_path:
        print(f"\n  Video saved to: {args.video_path}")


if __name__ == "__main__":
    main()
