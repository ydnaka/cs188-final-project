"""
Step 6a: Behavior Cloning with Temporal Context (State History)
================================================================
Improvement #1 over the baseline MLP in 06_train_policy.py.

KEY CHANGE: Instead of feeding a single state s_t, we stack the last k states:
    [s_{t-k+1}, ..., s_{t-1}, s_t]  →  MLP  →  action_t

This gives the network a short-term memory of where the robot has been,
which helps it infer velocity, momentum, and whether its approach is working.
The architecture stays a simple MLP — the only change is the wider input.

Requires augmented data from 05b_augment_handle_data.py (auto-detected).
If augmented data is not found, falls back to the raw parquet data.

Usage:
    python 06a_train_temporal.py
    python 06a_train_temporal.py --history_len 8 --epochs 100
    python 06a_train_temporal.py --no_aug   # force raw data even if aug exists
"""

import argparse
import os
import sys
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class TemporalCabinetDataset:
    """
    Loads state-action pairs and builds (history, action) training samples.

    For each timestep t inside an episode we build:
        X[t] = concat(s_{t-k+1}, ..., s_t)   shape: (k * state_dim,)
        y[t] = a_t                             shape: (action_dim,)

    Timesteps near the start of an episode are padded by repeating s_0.

    Augmented columns loaded when available:
        observation.handle_pos          (3)
        observation.handle_to_eef_pos   (3)
        observation.door_openness       (1)
        observation.handle_xaxis        (3)
        observation.hinge_direction     (1)
    """

    def __init__(self, dataset_path, history_len=4, max_episodes=None, use_aug=True):
        import pyarrow.parquet as pq

        self.history_len = history_len
        self.X = []  # (history, ) flattened
        self.y = []  # action

        # ------------------------------------------------------------------
        # Locate data directory (raw or augmented)
        # ------------------------------------------------------------------
        aug_dir = os.path.join(dataset_path, "augmented")
        raw_dir = os.path.join(dataset_path, "data", "chunk-000")
        if not os.path.exists(raw_dir):
            raw_dir = os.path.join(dataset_path, "lerobot", "data", "chunk-000")

        if use_aug and os.path.exists(aug_dir) and os.listdir(aug_dir):
            data_dir = aug_dir
            print(f"  [Dataset] Using AUGMENTED data: {aug_dir}")
        else:
            data_dir = raw_dir
            if use_aug:
                print("  [Dataset] Augmented data not found — using raw parquet data.")
                print("            Run 05b_augment_handle_data.py to add handle features.")
            else:
                print(f"  [Dataset] Using raw data: {raw_dir}")

        parquet_files = sorted(
            f for f in os.listdir(data_dir) if f.endswith(".parquet")
        )
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        if max_episodes:
            parquet_files = parquet_files[:max_episodes]

        # ------------------------------------------------------------------
        # Column names we want (in order)
        # ------------------------------------------------------------------
        BASE_STATE_COLS = [
            "observation.state",          # flattened low-dim state vector
        ]
        AUG_STATE_COLS = [
            "observation.handle_pos",
            "observation.handle_to_eef_pos",
            "observation.door_openness",
            "observation.handle_xaxis",
            "observation.hinge_direction",
        ]
        ACTION_COLS = ["action"]

        episodes_loaded = 0
        for pf in parquet_files:
            table = pq.read_table(os.path.join(data_dir, pf))
            df = table.to_pandas()

            # Find which state/action columns are actually present
            state_cols = [c for c in BASE_STATE_COLS if c in df.columns]
            # Fallback: any column whose name contains gripper/base/eef
            if not state_cols:
                state_cols = [
                    c for c in df.columns
                    if any(k in c for k in ("gripper", "base", "eef", "observation.state"))
                ]
            aug_cols = [c for c in AUG_STATE_COLS if c in df.columns]
            action_cols = [c for c in ACTION_COLS if c in df.columns]
            if not action_cols:
                action_cols = [c for c in df.columns if "action" in c and "observation" not in c]

            if not state_cols or not action_cols:
                continue

            # Build per-row state and action arrays
            ep_states = []
            ep_actions = []
            for _, row in df.iterrows():
                s_parts = []
                for c in state_cols + aug_cols:
                    val = row[c]
                    if isinstance(val, np.ndarray):
                        s_parts.extend(val.flatten().tolist())
                    else:
                        s_parts.append(float(val))

                a_parts = []
                for c in action_cols:
                    val = row[c]
                    if isinstance(val, np.ndarray):
                        a_parts.extend(val.flatten().tolist())
                    else:
                        a_parts.append(float(val))

                if s_parts and a_parts:
                    ep_states.append(np.array(s_parts, dtype=np.float32))
                    ep_actions.append(np.array(a_parts, dtype=np.float32))

            if not ep_states:
                continue

            # ------------------------------------------------------------------
            # Build history windows for this episode
            # ------------------------------------------------------------------
            # Pad the start by repeating the first state
            buf = deque([ep_states[0]] * history_len, maxlen=history_len)
            for t, (state, action) in enumerate(zip(ep_states, ep_actions)):
                buf.append(state)
                history_vec = np.concatenate(list(buf))  # shape: (k * state_dim,)
                self.X.append(history_vec)
                self.y.append(action)

            episodes_loaded += 1

        if not self.X:
            print("WARNING: No state-action pairs loaded. Generating synthetic data.")
            self._generate_synthetic(history_len)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        print(f"  Episodes loaded : {episodes_loaded}")
        print(f"  Samples         : {len(self.X)}")
        print(f"  History window  : {history_len} steps")
        print(f"  Input dim       : {self.X.shape[-1]}  ({history_len} × {self.X.shape[-1]//history_len})")
        print(f"  Action dim      : {self.y.shape[-1]}")

    def _generate_synthetic(self, history_len):
        rng = np.random.default_rng(42)
        state_dim = 16
        action_dim = 12
        for _ in range(1000):
            self.X.append(rng.standard_normal(history_len * state_dim).astype(np.float32))
            self.y.append(rng.standard_normal(action_dim).astype(np.float32) * 0.1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import torch
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(input_dim, action_dim, hidden_dim=512):
    """
    MLP identical in spirit to the baseline, but with:
      - Wider input layer (accepts k * state_dim)
      - LayerNorm after the first projection for training stability
      - Residual connection in the middle block
    """
    import torch.nn as nn

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
            x = self.act1(x + self.block1(x))  # residual
            return self.block2(x)

    return TemporalMLP()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    print_section("Temporal Context Behavior Cloning")

    dataset_path = get_dataset_path()
    print(f"Dataset root: {dataset_path}")

    dataset = TemporalCabinetDataset(
        dataset_path,
        history_len=config["history_len"],
        max_episodes=config.get("max_episodes", 50),
        use_aug=config.get("use_aug", True),
    )

    X_tensor = torch.from_numpy(dataset.X)
    y_tensor = torch.from_numpy(dataset.y)
    torch_dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(
        torch_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    input_dim  = dataset.X.shape[-1]
    action_dim = dataset.y.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(input_dim, action_dim, hidden_dim=config.get("hidden_dim", 512)).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Device       : {device}")
    print(f"  Parameters   : {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-4,
    )
    # Cosine LR schedule — smoothly decays to 0 by the last epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    print_section("Training")
    print(f"  Epochs       : {config['epochs']}")
    print(f"  Batch size   : {config['batch_size']}")
    print(f"  LR           : {config['learning_rate']}")
    print(f"  History len  : {config['history_len']}")
    print()

    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_06a")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    loss_history = []

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(X_batch)
            loss = nn.functional.mse_loss(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:4d}/{config['epochs']}  "
                  f"loss={avg_loss:.6f}  lr={lr_now:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "input_dim": input_dim,
                    "action_dim": action_dim,
                    "history_len": config["history_len"],
                    "config": config,
                },
                os.path.join(checkpoint_dir, "best_policy.pt"),
            )

    # Final checkpoint
    torch.save(
        {
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
            "loss": avg_loss,
            "input_dim": input_dim,
            "action_dim": action_dim,
            "history_len": config["history_len"],
            "config": config,
        },
        os.path.join(checkpoint_dir, "final_policy.pt"),
    )

    # Save loss curve as plain text (easy to import into your report plots)
    loss_path = os.path.join(checkpoint_dir, "loss_history.txt")
    np.savetxt(loss_path, loss_history, header="epoch_avg_mse_loss")

    print(f"\n  Training complete!")
    print(f"  Best MSE loss  : {best_loss:.6f}")
    print(f"  Checkpoints in : {checkpoint_dir}")
    print(f"  Loss log       : {loss_path}")

    return checkpoint_dir


# ---------------------------------------------------------------------------
# Inference helper — how to use the trained model at rollout time
# ---------------------------------------------------------------------------

class TemporalPolicyRunner:
    """
    Wraps the trained model for use inside an eval loop.

    Usage:
        runner = TemporalPolicyRunner(checkpoint_path)
        runner.reset()
        for t in range(horizon):
            obs = env.step(action)[0]
            state = extract_state(obs)        # numpy array, same features as training
            action = runner.act(state)        # numpy array
    """

    def __init__(self, checkpoint_path, device=None):
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.history_len = ckpt["history_len"]
        self.input_dim   = ckpt["input_dim"]
        self.action_dim  = ckpt["action_dim"]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = build_model(self.input_dim, self.action_dim)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._buf = None

    def reset(self):
        """Call at the start of each episode to clear the history buffer."""
        self._buf = None

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Given current state (numpy array), return action (numpy array).
        Maintains internal history buffer automatically.
        """
        import torch

        if self._buf is None:
            # Pad start of episode with copies of the first state
            self._buf = deque([state] * self.history_len, maxlen=self.history_len)
        self._buf.append(state)

        history_vec = np.concatenate(list(self._buf))
        x = torch.from_numpy(history_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.model(x).squeeze(0).cpu().numpy()
        return action


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="06a: Behavior cloning with temporal state history"
    )
    parser.add_argument("--history_len",    type=int,   default=4,      help="Number of past states to stack (k)")
    parser.add_argument("--epochs",         type=int,   default=100,    help="Training epochs")
    parser.add_argument("--batch_size",     type=int,   default=64,     help="Batch size")
    parser.add_argument("--lr",             type=float, default=3e-4,   help="Learning rate")
    parser.add_argument("--hidden_dim",     type=int,   default=512,    help="MLP hidden layer width")
    parser.add_argument("--max_episodes",   type=int,   default=50,     help="Max episodes to load from dataset")
    parser.add_argument("--checkpoint_dir", type=str,   default="/tmp/cabinet_policy_06a")
    parser.add_argument("--no_aug",         action="store_true",        help="Ignore augmented data, use raw parquet")
    args = parser.parse_args()

    print("=" * 60)
    print("  06a: Temporal Context BC Policy")
    print("=" * 60)

    config = {
        "history_len":    args.history_len,
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "learning_rate":  args.lr,
        "hidden_dim":     args.hidden_dim,
        "max_episodes":   args.max_episodes,
        "checkpoint_dir": args.checkpoint_dir,
        "use_aug":        not args.no_aug,
    }

    train(config)

    print_section("What this adds vs baseline (06)")
    print(
        "  - State input is now a window of the last k states concatenated.\n"
        "  - The network can infer velocity and direction of motion from\n"
        "    the difference between consecutive states.\n"
        "  - LayerNorm + residual connection improve training stability.\n"
        "  - AdamW + cosine LR schedule for better convergence.\n"
        "\n"
        "  At eval time, use TemporalPolicyRunner (defined in this file)\n"
        "  to manage the history buffer automatically.\n"
        "\n"
        "  Next improvement: python 06b_train_action_chunking.py"
    )


if __name__ == "__main__":
    main()
