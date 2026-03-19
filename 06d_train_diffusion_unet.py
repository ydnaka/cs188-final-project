"""
Step 6d: Diffusion Policy with Conditional U-Net (UNet1D)
==========================================================
Improvement #3. Replaces the MLP noise predictor from the previous 06c
with the ConditionalUnet1D from Chi et al. (RSS 2023).

Key changes vs the MLP diffusion version:
  - Noise predictor is now a 1D convolutional U-Net (~15M params)
  - Actions are treated as sequences: (B, chunk_size, action_dim)
  - State is passed as global_cond: (B, state_dim)
  - All other logic (dataset, zero-padding, column locking) identical to 06b

The helper classes (SinusoidalPosEmb, Conv1dBlock, Downsample1d, Upsample1d,
ConditionalResidualBlock1D, ConditionalUnet1D) are inlined so this script
has no dependency on the diffusion_policy package.

Requires: einops  (pip install einops — likely already installed via robosuite)

Usage:
    python 06d_train_diffusion_unet.py
    python 06d_train_diffusion_unet.py --epochs 300 --max_episodes 200
"""

import argparse
import math
import os
import sys
from collections import deque
from typing import Union

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
# Inlined U-Net components (from real-stanford/diffusion_policy)
# ---------------------------------------------------------------------------

def build_unet(action_dim, chunk_size, state_dim,
               down_dims=(128, 256, 512), diff_step_embed_dim=128):
    """
    Instantiate ConditionalUnet1D with all helper classes inlined.
    Returns the model. ~15M params with default down_dims.
    """
    import torch
    import torch.nn as nn
    import einops
    from einops.layers.torch import Rearrange

    # -- Positional embedding for diffusion timestep --
    class SinusoidalPosEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            device = x.device
            half = self.dim // 2
            emb = math.log(10000) / (half - 1)
            emb = torch.exp(torch.arange(half, device=device) * -emb)
            emb = x[:, None] * emb[None, :]
            return torch.cat([emb.sin(), emb.cos()], dim=-1)

    # -- 1D conv block with GroupNorm + Mish --
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

    # -- Conditional residual block with FiLM conditioning --
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

    # -- Full Conditional U-Net --
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
            cond_dim = dsed + state_dim  # timestep emb + flattened state

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
            """
            sample      : (B, chunk_size, action_dim)  — noisy actions
            timestep    : (B,) long
            global_cond : (B, state_dim)               — current state
            returns     : (B, chunk_size, action_dim)  — predicted noise
            """
            import einops
            # rearrange to (B, action_dim, chunk_size) for conv layers
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
            return einops.rearrange(x, 'b t h -> b h t')

    model = ConditionalUnet1D()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  U-Net parameters: {n_params:,}")
    return model


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
            print(f"  [Dataset] Using augmented data")
        else:
            data_dir = raw_dir
            print(f"  [Dataset] Using raw data")

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
                self.state_columns = s_cols + [c for c in AUG_STATE_COLS if c in df.columns]

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

            zero_state = np.zeros(len(ep_states[0]), dtype=np.float32)
            buf = deque([zero_state] * history_len, maxlen=history_len)
            T   = len(ep_states)

            for t in range(T):
                buf.append(ep_states[t])
                history_vec  = np.concatenate(list(buf))
                chunk        = [ep_actions[min(t + k, T - 1)] for k in range(chunk_size)]
                action_chunk = np.stack(chunk)  # (chunk_size, action_dim)
                self.X.append(history_vec)
                self.y.append(action_chunk)

            episodes_loaded += 1

        if not self.X:
            print("WARNING: No data — generating synthetic data.")
            self._generate_synthetic(history_len, chunk_size)

        self.X = np.array(self.X, dtype=np.float32)           # (N, history_len*state_dim)
        self.y = np.array(self.y, dtype=np.float32)           # (N, chunk_size, action_dim)

        print(f"  Episodes  : {episodes_loaded}")
        print(f"  Samples   : {len(self.X)}")
        print(f"  State dim : {self.X.shape[-1]}")
        print(f"  Action shape: {self.y.shape[1:]}")

    def _generate_synthetic(self, history_len, chunk_size):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            self.X.append(rng.standard_normal(history_len * 16).astype(np.float32))
            self.y.append(rng.standard_normal((chunk_size, 12)).astype(np.float32) * 0.1)
        self.state_columns = ["synthetic"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print_section("Diffusion Policy — ConditionalUnet1D")
    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}\n")

    ds = ActionChunkDataset(
        dataset_path,
        history_len   = config["history_len"],
        chunk_size    = config["chunk_size"],
        max_episodes  = config.get("max_episodes"),
        use_aug       = config.get("use_aug", True),
    )

    state_dim  = ds.X.shape[-1]
    action_dim = ds.y.shape[-1]   # single-step action dim
    chunk_size = config["chunk_size"]
    diff_steps = config["diff_steps"]

    loader = DataLoader(
        TensorDataset(torch.from_numpy(ds.X), torch.from_numpy(ds.y)),
        batch_size = config["batch_size"],
        shuffle    = True,
        drop_last  = True,
        num_workers= 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_unet(
        action_dim          = action_dim,
        chunk_size          = chunk_size,
        state_dim           = state_dim,
        down_dims           = tuple(config.get("down_dims", [128, 256, 512])),
        diff_step_embed_dim = config.get("diff_step_embed_dim", 128),
    ).to(device)

    print(f"\n  Device     : {device}")

    # Linear beta schedule
    betas      = torch.linspace(1e-4, 0.02, diff_steps, device=device)
    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    print_section("Training")
    print(f"  Epochs      : {config['epochs']}")
    print(f"  Batch size  : {config['batch_size']}")
    print(f"  Diff steps  : {diff_steps}")
    print(f"  Chunk size  : {chunk_size}")
    print()

    ckpt_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_06d")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss    = float("inf")
    loss_history = []

    def save(path, epoch, loss):
        torch.save({
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "loss":             loss,
            "state_dim":        state_dim,
            "action_dim":       action_dim,
            "history_len":      config["history_len"],
            "chunk_size":       chunk_size,
            "diff_steps":       diff_steps,
            "down_dims":        config.get("down_dims", [128, 256, 512]),
            "diff_step_embed_dim": config.get("diff_step_embed_dim", 128),
            "state_columns":    ds.state_columns,
            "config":           config,
        }, path)

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss, n = 0.0, 0

        for X_b, y_b in loader:
            X_b = X_b.to(device)                    # (B, state_dim)
            y_b = y_b.to(device)                    # (B, chunk_size, action_dim)

            # Sample random diffusion timesteps
            t = torch.randint(0, diff_steps, (X_b.shape[0],), device=device)

            # Add noise to action chunks
            noise        = torch.randn_like(y_b)
            ab           = alpha_bars[t].view(-1, 1, 1)
            noisy_action = ab.sqrt() * y_b + (1 - ab).sqrt() * noise

            # Predict the noise
            pred_noise = model(noisy_action, t, global_cond=X_b)
            loss = nn.functional.mse_loss(pred_noise, noise)

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
            save(os.path.join(ckpt_dir, "best_policy.pt"), epoch, best_loss)

    save(os.path.join(ckpt_dir, "final_policy.pt"), config["epochs"], avg)
    np.savetxt(os.path.join(ckpt_dir, "loss_history.txt"), loss_history, header="epoch_avg_mse_loss")

    print(f"\n  Training complete!")
    print(f"  Best loss   : {best_loss:.6f}")
    print(f"  Checkpoints : {ckpt_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="06d: Diffusion policy with ConditionalUnet1D")
    parser.add_argument("--chunk_size",     type=int,   default=8)
    parser.add_argument("--history_len",    type=int,   default=4)
    parser.add_argument("--diff_steps",     type=int,   default=100)
    parser.add_argument("--epochs",         type=int,   default=300)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--max_episodes",   type=int,   default=None,
                        help="None = use all available episodes")
    parser.add_argument("--checkpoint_dir", type=str,   default="/tmp/cabinet_policy_06d")
    parser.add_argument("--no_aug",         action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  06d: Diffusion Policy — ConditionalUnet1D")
    print("=" * 60)

    train({
        "chunk_size":     args.chunk_size,
        "history_len":    args.history_len,
        "diff_steps":     args.diff_steps,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "learning_rate":  args.lr,
        "max_episodes":   args.max_episodes,
        "checkpoint_dir": args.checkpoint_dir,
        "use_aug":        not args.no_aug,
        "down_dims":      [128, 256, 512],
        "diff_step_embed_dim": 128,
    })


if __name__ == "__main__":
    main()
