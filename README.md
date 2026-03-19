# CS 188 Final Project — Cabinet Door Opening Robot
## by Andy Kasbarian

This project extends the [OpenCabinet starter project](https://github.com/HoldenGs/cs188-cabinet-door-project) by Holden Grissett (UCLA CS 188 TA). It trains a PandaOmron robot to open cabinet doors in the RoboCasa simulation environment using an incremental series of behavior cloning policy improvements.

**Website:** https://ydnaka.github.io/cs188-final-website/  
**Report:** https://ydnaka.github.io/cs188-final-website/report.pdf

---

## Prerequisites

Complete the setup from the original project first:

```bash
git clone https://github.com/HoldenGs/cs188-cabinet-door-project
cd cs188-cabinet-door-project
./install.sh
source .venv/bin/activate
cd cabinet_door_project
python 00_verify_installation.py
python 04_download_dataset.py
```

Once `00_verify_installation.py` passes and the dataset is downloaded, clone this repo and copy the scripts into the same `cabinet_door_project/` directory.

### Additional dependency (required for 06d only)

```bash
pip install einops
```

---

## Project Structure

```
cabinet_door_project/
  05b_augment_handle_data.py   # Augment dataset with handle features (from Holden Grissett)
  06_train_policy.py           # Baseline MLP (from original project, unmodified)
  06a_train_temporal.py        # MLP + temporal context
  06b_train_action_chunking.py # MLP + temporal context + action chunking
  06c_train_diffusion.py       # MLP diffusion policy
  06d_train_diffusion_unet.py  # 1D U-Net diffusion policy
  07_evaluate_policy.py        # Evaluate baseline (modified: single-door success, default path)
  07a_evaluate_temporal.py     # Evaluate 06a
  07b_evaluate_chunking.py     # Evaluate 06b
  07c_evaluate_diffusion.py    # Evaluate 06c
  07d_evaluate_diffusion_unet.py # Evaluate 06d
  08_visualize_policy_rollout.py  # Visualize baseline (from original project)
  08a_visualize_temporal.py    # Visualize 06a
  08b_visualize_chunking.py    # Visualize 06b
  08c_visualize_diffusion.py   # Visualize 06c
  08d_visualize_diffusion_unet.py # Visualize 06d
  Notes.md                     # Evaluation results for all runs
```

---

## Reproducing Results

### Step 1 — Augment the dataset

Run this once before training any policy except the baseline:

```bash
python 05b_augment_handle_data.py
```

This adds 11 dimensions to the state vector: handle world position, handle-to-eef vector, door openness, handle axis, and hinge direction.

---

### Baseline MLP

```bash
python 06_train_policy.py
python 07_evaluate_policy.py
python 08_visualize_policy_rollout.py --offscreen --num_episodes 3
```

Checkpoints saved to: `/tmp/cabinet_policy_checkpoints/`

---

### 06a — MLP with Temporal Context

```bash
python 06a_train_temporal.py
python 07a_evaluate_temporal.py
python 08a_visualize_temporal.py --offscreen --num_episodes 3
```

Checkpoints saved to: `/tmp/cabinet_policy_06a/`

Key change: stacks last k=4 states as input. Includes zero-padding fix and locked column order.

---

### 06b — MLP with Action Chunking

```bash
python 06b_train_action_chunking.py
python 07b_evaluate_chunking.py
python 08b_visualize_chunking.py --offscreen --num_episodes 3
```

Checkpoints saved to: `/tmp/cabinet_policy_06b/`

Key change: predicts next K=8 actions at once and executes them open-loop.

---

### 06c — MLP Diffusion Policy

```bash
python 06c_train_diffusion.py
python 07c_evaluate_diffusion.py
python 08c_visualize_diffusion.py --offscreen --num_episodes 3
```

Checkpoints saved to: `/tmp/cabinet_policy_06c/`

Key change: replaces MSE regression with DDPM noise prediction. 100 denoising steps at inference.

---

### 06d — 1D U-Net Diffusion Policy

```bash
python 06d_train_diffusion_unet.py --epochs 300
python 07d_evaluate_diffusion_unet.py
python 08d_visualize_diffusion_unet.py --offscreen --num_episodes 3
```

Checkpoints saved to: `/tmp/cabinet_policy_06d/`

Key change: replaces MLP noise predictor with ConditionalUnet1D (~10.5M params) with FiLM conditioning. Requires `einops`. Training takes approximately 3 hours on an RTX 3060 Mobile.

---

## Evaluation Notes

- All policies evaluated over 20 episodes on the `pretrain` kitchen split
- Maximum 500 steps per episode
- Success criterion: any single cabinet door reaches ≥90% of its full open range
- All policies achieved 0% success rate
- 06b and 06d show qualitatively purposeful motion toward the handle; other policies exhibit random flailing

See `Notes.md` for full evaluation output.

---

## Training Arguments

All training scripts accept common arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 (300 for 06d) | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--max_episodes` | 50 (None for 06d) | Max episodes to load |
| `--checkpoint_dir` | `/tmp/cabinet_policy_06*/` | Where to save checkpoints |
| `--no_aug` | False | Use raw data instead of augmented |

06b and 06d also accept `--chunk_size` (default 8) and `--history_len` (default 4).  
06c and 06d also accept `--diff_steps` (default 100).

---

## AI Disclosure

Claude (Anthropic) assisted in writing and debugging all scripts in this repository (06a through 08d). Script `05b_augment_handle_data.py` was provided by Holden Grissett and was not written by the author or AI. All code was reviewed, debugged, and understood by the author.
