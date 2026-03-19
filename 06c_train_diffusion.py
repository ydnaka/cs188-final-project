"""
Step 6c: Minimal Diffusion Policy
===================================
Improvement #3. Same dataset/history/chunking setup as 06b, but replaces
MSE regression with a diffusion-based action generator.

WHAT CHANGES vs 06b:
  1. The model now takes (state, noisy_action, noise_timestep) as input
     and predicts the NOISE added to the action — not the action itself.
  2. Training loss: mse(predicted_noise, actual_noise) — one line different.
  3. Inference: run a 10-step denoising loop instead of a single forward pass.
  4. A simple linear noise schedule (beta) controls how much noise is added
     at each diffusion timestep.

WHAT STAYS THE SAME as 06b:
  - Dataset class (augmented data, locked column order, zero-padding)
  - Action chunking (predict K actions at once)
  - Temporal history (stack last k states)
  - AdamW + cosine LR + gradient clipping
  - Checkpoint format (same keys, plus diffusion-specific ones)

Usage:
    python 06c_train_diffusion.py
    python 06c_train_diffusion.py --chunk_size 8 --history_len 4 --diff_steps 100
"""

import argparse
import os
import sys
from collections import deque

import numpy as np


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def get_dataset_path():
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path
    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


# ---------------------------------------------------------------------------
# Dataset  (identical to 06b)
# ---------------------------------------------------------------------------

class ActionChunkDataset:
    def __init__(self, dataset_path, history_len=4, chunk_size=8,
                 max_episodes=None, use_aug=True):
        import pyarrow.parquet as pq

        self.history_len   = history_len
        self.chunk_size    = chunk_size
        self.X             = []
        self.y             = []
        self.state_columns = None

        aug_dir = os.path.join(dataset_path, "augmented")
        raw_dir = os.path.join(dataset_path, "data", "chunk-000")
        if not os.path.exists(raw_dir):
            raw_dir = os.path.join(dataset_path, "lerobot", "data", "chunk-000")

        if use_aug and os.path.exists(aug_dir) and os.listdir(aug_dir):
            data_dir = aug_dir
            print(f"  [Dataset] Using augmented data: {aug_dir}")
        else:
            data_dir = raw_dir
            print(f"  [Dataset] Using raw data: {raw_dir}")

        parquet_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
        if max_episodes:
            parquet_files = parquet_files[:max_episodes]

        BASE_STATE_COLS = ["observation.state"]
        AUG_STATE_COLS  = [
            "observation.handle_pos", "observation.handle_to_eef_pos",
            "observation.door_openness", "observation.handle_xaxis",
            "observation.hinge_direction",
        ]
        ACTION_COLS = ["action"]

        episodes_loaded = 0
        for pf in parquet_files:
            table = pq.read_table(os.path.join(data_dir, pf))
            df    = table.to_pandas()

            if self.state_columns is None:
                s_cols = [c for c in BASE_STATE_COLS if c in df.columns]
                if not s_cols:
                    s_cols = [c for c in df.columns
                              if any(k in c for k in ("gripper","base","eef","observation.state"))]
                aug_cols = [c for c in AUG_STATE_COLS if c in df.columns]
                self.state_columns = s_cols + aug_cols

            a_cols = [c for c in ACTION_COLS if c in df.columns]
            if not a_cols:
                a_cols = [c for c in df.columns if "action" in c and "observation" not in c]
            if not self.state_columns or not a_cols:
                continue

            ep_states, ep_actions = [], []
            for _, row in df.iterrows():
                s = []
                for c in self.state_columns:
                    v = row[c]
                    s.extend(v.flatten().tolist() if isinstance(v, np.ndarray) else [float(v)])
                a = []
                for c in a_cols:
                    v = row[c]
                    a.extend(v.flatten().tolist() if isinstance(v, np.ndarray) else [float(v)])
                if s and a:
                    ep_states.append(np.array(s, dtype=np.float32))
                    ep_actions.append(np.array(a, dtype=np.float32))

            if not ep_states:
                continue

            state_dim  = len(ep_states[0])
            zero_state = np.zeros(state_dim, dtype=np.float32)
            T          = len(ep_states)
            buf = deque([zero_state] * history_len, maxlen=history_len)

            for t in range(T):
                buf.append(ep_states[t])
                history_vec  = np.concatenate(list(buf))
                chunk        = [ep_actions[min(t + k, T - 1)] for k in range(chunk_size)]
                action_chunk = np.concatenate(chunk)
                self.X.append(history_vec)
                self.y.append(action_chunk)

            episodes_loaded += 1

        if not self.X:
            print("WARNING: No data loaded — generating synthetic data.")
            self._generate_synthetic(history_len, chunk_size)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        print(f"  Episodes  : {episodes_loaded}")
        print(f"  Samples   : {len(self.X)}")
        print(f"  Input dim : {self.X.shape[-1]}")
        print(f"  Output dim: {self.y.shape[-1]}")

    def _generate_synthetic(self, history_len, chunk_size):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            self.X.append(rng.standard_normal(history_len * 16).astype(np.float32))
            self.y.append(rng.standard_normal(chunk_size * 12).astype(np.float32) * 0.1)
        self.state_columns = ["synthetic"]


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

class LinearNoiseSchedule:
    """
    Pre-computes the alpha/beta tables for the forward (noising) process.

    At diffusion timestep t, a clean action x0 is corrupted to:
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    where noise ~ N(0, I).

    The network is trained to predict 'noise' given (state, x_t, t).
    """

    def __init__(self, num_steps=100, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        betas          = np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)
        alphas         = 1.0 - betas
        alpha_bar      = np.cumprod(alphas)

        # Store as tensors — moved to device during training
        import torch
        self.betas      = torch.from_numpy(betas)
        self.alphas     = torch.from_numpy(alphas)
        self.alpha_bar  = torch.from_numpy(alpha_bar)

    def to(self, device):
        self.betas     = self.betas.to(device)
        self.alphas    = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def add_noise(self, x0, t, noise):
        """
        Forward process: corrupt clean action x0 at timestep t.
        x0, noise: (B, action_dim)
        t: (B,) long tensor of diffusion timesteps
        """
        ab  = self.alpha_bar[t].view(-1, 1)          # (B, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def denoise_step(self, model, state, x_t, t_scalar):
        """
        One DDPM denoising step: x_t → x_{t-1}.
        Called inside the inference loop.
        """
        import torch
        device = x_t.device
        t_tensor = torch.full((x_t.shape[0],), t_scalar, dtype=torch.long, device=device)

        with torch.no_grad():
            pred_noise = model(state, x_t, t_tensor)

        alpha     = self.alphas[t_scalar]
        alpha_bar = self.alpha_bar[t_scalar]
        beta      = self.betas[t_scalar]

        # Predicted clean action
        x0_pred = (x_t - (1 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # Mean of the posterior q(x_{t-1} | x_t, x0)
        mean = (alpha.sqrt() * (1 - alpha_bar / alpha) * x_t
                + (alpha_bar / alpha).sqrt() * beta * x0_pred) / (1 - alpha_bar)

        if t_scalar == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + beta.sqrt() * noise


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(state_dim, action_dim, hidden_dim=512, diff_steps=100):
    """
    Noise prediction network: f(state, noisy_action, t) → predicted_noise.

    The diffusion timestep t is embedded via a sinusoidal encoding (same idea
    as positional encodings in Transformers) and concatenated with the state
    and noisy action before being fed into the MLP.
    """
    import torch
    import torch.nn as nn
    import math

    class SinusoidalEmbedding(nn.Module):
        """Encodes a scalar timestep t into a fixed-size vector."""
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            # t: (B,) long tensor
            device = t.device
            half   = self.dim // 2
            freqs  = torch.exp(
                -math.log(10000) * torch.arange(half, dtype=torch.float32, device=device) / half
            )
            args   = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
            return torch.cat([args.sin(), args.cos()], dim=-1)     # (B, dim)

    class DiffusionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            t_emb_dim = 64

            self.t_emb = SinusoidalEmbedding(t_emb_dim)

            # Input: concat(state, noisy_action, t_embedding)
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
            # NOTE: no Tanh here — we're predicting unbounded noise, not actions

        def forward(self, state, noisy_action, t):
            t_emb = self.t_emb(t)
            x     = torch.cat([state, noisy_action, t_emb], dim=-1)
            x     = self.input_proj(x)
            x     = self.act1(x + self.block1(x))
            return self.head(x)

    return DiffusionMLP()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print_section("Minimal Diffusion Policy")

    dataset_path = get_dataset_path()
    print(f"Dataset root: {dataset_path}\n")

    ds = ActionChunkDataset(
        dataset_path,
        history_len=config["history_len"],
        chunk_size=config["chunk_size"],
        max_episodes=config.get("max_episodes", 50),
        use_aug=config.get("use_aug", True),
    )

    loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.X), torch.from_numpy(ds.y)),
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    state_dim  = ds.X.shape[-1]
    action_dim = ds.y.shape[-1]   # chunk_size * single_action_dim
    diff_steps = config["diff_steps"]

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(state_dim, action_dim,
                            config.get("hidden_dim", 512), diff_steps).to(device)
    schedule  = LinearNoiseSchedule(num_steps=diff_steps).to(device)

    print(f"\n  Device      : {device}")
    print(f"  Parameters  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Diff steps  : {diff_steps}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"])

    print_section("Training")
    print(f"  Epochs      : {config['epochs']}")
    print(f"  Batch size  : {config['batch_size']}")
    print(f"  LR          : {config['learning_rate']}")
    print(f"  Chunk size  : {config['chunk_size']}")
    print(f"  History len : {config['history_len']}")
    print()

    ckpt_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_06c")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss    = float("inf")
    loss_history = []

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss, n = 0.0, 0

        for X_b, y_b in loader:
            X_b = X_b.to(device)   # (B, state_dim)
            y_b = y_b.to(device)   # (B, action_dim)  — clean action chunk

            # Sample a random diffusion timestep for each item in the batch
            t = torch.randint(0, diff_steps, (X_b.shape[0],), device=device)

            # Sample noise and corrupt the clean action
            noise  = torch.randn_like(y_b)
            x_t    = schedule.add_noise(y_b, t, noise)

            # CHANGE: predict the noise, not the action
            pred_noise = model(X_b, x_t, t)
            loss       = nn.functional.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        scheduler.step()
        avg = epoch_loss / max(n, 1)
        loss_history.append(avg)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{config['epochs']}  loss={avg:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if avg < best_loss:
            best_loss = avg
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "loss":             best_loss,
                "state_dim":        state_dim,
                "action_dim":       action_dim,
                "history_len":      config["history_len"],
                "chunk_size":       config["chunk_size"],
                "diff_steps":       diff_steps,
                "state_columns":    ds.state_columns,
                "config":           config,
            }, os.path.join(ckpt_dir, "best_policy.pt"))

    torch.save({
        "epoch":            config["epochs"],
        "model_state_dict": model.state_dict(),
        "loss":             avg,
        "state_dim":        state_dim,
        "action_dim":       action_dim,
        "history_len":      config["history_len"],
        "chunk_size":       config["chunk_size"],
        "diff_steps":       diff_steps,
        "state_columns":    ds.state_columns,
        "config":           config,
    }, os.path.join(ckpt_dir, "final_policy.pt"))

    np.savetxt(os.path.join(ckpt_dir, "loss_history.txt"),
               loss_history, header="epoch_avg_noise_mse_loss")

    print(f"\n  Training complete!")
    print(f"  Best loss   : {best_loss:.6f}")
    print(f"  Checkpoints : {ckpt_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="06c: Minimal diffusion policy")
    parser.add_argument("--diff_steps",     type=int,   default=100)
    parser.add_argument("--chunk_size",     type=int,   default=8)
    parser.add_argument("--history_len",    type=int,   default=4)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--hidden_dim",     type=int,   default=512)
    parser.add_argument("--max_episodes",   type=int,   default=50)
    parser.add_argument("--checkpoint_dir", type=str,   default="/tmp/cabinet_policy_06c")
    parser.add_argument("--no_aug",         action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  06c: Minimal Diffusion Policy")
    print("=" * 60)

    train({
        "diff_steps":     args.diff_steps,
        "chunk_size":     args.chunk_size,
        "history_len":    args.history_len,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "learning_rate":  args.lr,
        "hidden_dim":     args.hidden_dim,
        "max_episodes":   args.max_episodes,
        "checkpoint_dir": args.checkpoint_dir,
        "use_aug":        not args.no_aug,
    })


if __name__ == "__main__":
    main()
