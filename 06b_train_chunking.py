"""
Step 6b: Behavior Cloning with Action Chunking
================================================
Improvement #2. Builds on 06a (temporal context) and adds:

  CHANGE 1 — Action chunking: predict the next K actions at once.
    Input:  [s_{t-k+1}, ..., s_t]          shape: (history_len * state_dim,)
    Output: [a_t, a_{t+1}, ..., a_{t+K-1}] shape: (chunk_size * action_dim,)
    At inference we execute the chunk open-loop, then re-query.
    This produces smoother, more temporally coherent motion.

  CHANGE 2 — Zero-pad history at episode start instead of repeating s_0.
    The network never saw repeated-state windows during training, so
    repeating s_0 pushed it into an unfamiliar input region. Zeros are
    a neutral "no history yet" signal.

  CHANGE 3 — Column order is locked at training time and saved in the
    checkpoint. Eval scripts load this order and match it exactly,
    preventing the scrambled-state bug from 06a.

Usage:
    python 06b_train_action_chunking.py
    python 06b_train_action_chunking.py --chunk_size 16 --history_len 4 --epochs 100
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
# Dataset
# ---------------------------------------------------------------------------

class ActionChunkDataset:
    """
    Builds (history_vec, action_chunk) pairs from the dataset.

    history_vec : concat of last history_len states, zero-padded at episode start
    action_chunk: concat of next chunk_size actions (last action repeated at ep end)

    Column order is stored in self.state_columns so it can be saved in the
    checkpoint and reproduced exactly at eval time.
    """

    def __init__(self, dataset_path, history_len=4, chunk_size=8,
                 max_episodes=None, use_aug=True):
        import pyarrow.parquet as pq

        self.history_len   = history_len
        self.chunk_size    = chunk_size
        self.X             = []
        self.y             = []
        self.state_columns = None  # locked on first episode

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

            # Lock column order on first episode
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

            ep_states  = []
            ep_actions = []
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

            # CHANGE 2: zero-pad history buffer
            buf = deque([zero_state] * history_len, maxlen=history_len)

            for t in range(T):
                buf.append(ep_states[t])
                history_vec = np.concatenate(list(buf))

                # Action chunk: repeat last action to pad end of episode
                chunk = [ep_actions[min(t + k, T - 1)] for k in range(chunk_size)]
                action_chunk = np.concatenate(chunk)

                self.X.append(history_vec)
                self.y.append(action_chunk)

            episodes_loaded += 1

        if not self.X:
            print("WARNING: No data loaded — generating synthetic data.")
            self._generate_synthetic(history_len, chunk_size)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        print(f"  Episodes      : {episodes_loaded}")
        print(f"  Samples       : {len(self.X)}")
        print(f"  Input dim     : {self.X.shape[-1]}")
        print(f"  Output dim    : {self.y.shape[-1]}")

    def _generate_synthetic(self, history_len, chunk_size):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            self.X.append(rng.standard_normal(history_len * 16).astype(np.float32))
            self.y.append(rng.standard_normal(chunk_size * 12).astype(np.float32) * 0.1)
        self.state_columns = ["synthetic"]


# ---------------------------------------------------------------------------
# Model  (same backbone as 06a, wider output head)
# ---------------------------------------------------------------------------

def build_model(input_dim, output_dim, hidden_dim=512):
    import torch.nn as nn

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

    return ChunkMLP()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print_section("Action Chunking Behavior Cloning")

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

    input_dim  = ds.X.shape[-1]
    output_dim = ds.y.shape[-1]
    action_dim = output_dim // config["chunk_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(input_dim, output_dim, config.get("hidden_dim", 512)).to(device)

    print(f"\n  Device      : {device}")
    print(f"  Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    print_section("Training")
    print(f"  Epochs      : {config['epochs']}")
    print(f"  Batch size  : {config['batch_size']}")
    print(f"  LR          : {config['learning_rate']}")
    print(f"  Chunk size  : {config['chunk_size']}")
    print(f"  History len : {config['history_len']}")
    print()

    ckpt_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_06b")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_loss    = float("inf")
    loss_history = []

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss, n = 0.0, 0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            loss = nn.functional.mse_loss(model(X_b), y_b)
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
                "input_dim":        input_dim,
                "action_dim":       action_dim,
                "output_dim":       output_dim,
                "history_len":      config["history_len"],
                "chunk_size":       config["chunk_size"],
                "state_columns":    ds.state_columns,  # CHANGE 3
                "config":           config,
            }, os.path.join(ckpt_dir, "best_policy.pt"))

    torch.save({
        "epoch":            config["epochs"],
        "model_state_dict": model.state_dict(),
        "loss":             avg,
        "input_dim":        input_dim,
        "action_dim":       action_dim,
        "output_dim":       output_dim,
        "history_len":      config["history_len"],
        "chunk_size":       config["chunk_size"],
        "state_columns":    ds.state_columns,
        "config":           config,
    }, os.path.join(ckpt_dir, "final_policy.pt"))

    np.savetxt(os.path.join(ckpt_dir, "loss_history.txt"), loss_history, header="epoch_avg_mse_loss")

    print(f"\n  Training complete!")
    print(f"  Best loss   : {best_loss:.6f}")
    print(f"  Checkpoints : {ckpt_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="06b: Action chunking BC policy")
    parser.add_argument("--chunk_size",     type=int,   default=8)
    parser.add_argument("--history_len",    type=int,   default=4)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--hidden_dim",     type=int,   default=512)
    parser.add_argument("--max_episodes",   type=int,   default=50)
    parser.add_argument("--checkpoint_dir", type=str,   default="/tmp/cabinet_policy_06b")
    parser.add_argument("--no_aug",         action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  06b: Action Chunking BC Policy")
    print("=" * 60)

    train({
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
