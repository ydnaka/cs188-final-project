For all below evaluations the success criteria was changed to only consider the opening of one of the doors as a success.

# 6: Original dataset, original MLP policy
Training 50 Epochs

```
Loading dataset...
Loaded 17317 state-action pairs
State dim:  16
Action dim: 12

Device: cuda

============================================================
  Training
============================================================
Epochs:     50
Batch size: 32
LR:         0.0001
  Epoch    1/50  Loss: 0.123711
  Epoch   10/50  Loss: 0.063570
  Epoch   20/50  Loss: 0.043296
  Epoch   30/50  Loss: 0.037430
  Epoch   40/50  Loss: 0.033461
  Epoch   50/50  Loss: 0.030034

Training complete!
Best loss:        0.030034
Best checkpoint:  /tmp/cabinet_policy_checkpoints/best_policy.pt
Final checkpoint: /tmp/cabinet_policy_checkpoints/final_policy.pt
```
```
============================================================
  OpenCabinet - Policy Evaluation
============================================================
Device: cuda
Loaded policy from: /tmp/cabinet_policy_checkpoints/best_policy.pt
  Trained for 49 epochs, loss=0.030034
  State dim: 16, Action dim: 12

============================================================
  Evaluating on pretrain split (20 episodes)
============================================================
[robosuite INFO] Loading controller configuration from: /home/lu/CS188/cs188-cabinet-door-project/robosuite/robosuite/controllers/config/robots/default_pandaomron.json (composite_controller_factory.py:121)
  Episode   1/20: FAIL    (steps= 500, reward=0.0) layout=53, style=37, task="Open the cabinet door."
  Episode   2/20: FAIL    (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
  Episode   3/20: FAIL    (steps= 500, reward=0.0) layout=26, style=13, task="Open the cabinet doors."
  Episode   4/20: FAIL    (steps= 500, reward=0.0) layout=55, style=46, task="Open the cabinet door."
  Episode   5/20: FAIL    (steps= 500, reward=0.0) layout=59, style=38, task="Open the cabinet door."
  Episode   6/20: FAIL    (steps= 500, reward=0.0) layout=47, style=43, task="Open the cabinet doors."
  Episode   7/20: FAIL    (steps= 500, reward=0.0) layout=59, style=26, task="Open the cabinet door."
  Episode   8/20: FAIL    (steps= 500, reward=0.0) layout=50, style=47, task="Open the cabinet doors."
  Episode   9/20: FAIL    (steps= 500, reward=0.0) layout=27, style=21, task="Open the cabinet door."
  Episode  10/20: FAIL    (steps= 500, reward=0.0) layout=29, style=17, task="Open the cabinet doors."
  Episode  11/20: FAIL    (steps= 500, reward=0.0) layout=42, style=31, task="Open the cabinet doors."
  Episode  12/20: FAIL    (steps= 500, reward=0.0) layout=54, style=37, task="Open the cabinet door."
  Episode  13/20: FAIL    (steps= 500, reward=0.0) layout=43, style=29, task="Open the cabinet doors."
  Episode  14/20: FAIL    (steps= 500, reward=0.0) layout=43, style=21, task="Open the cabinet doors."
  Episode  15/20: FAIL    (steps= 500, reward=0.0) layout=53, style=12, task="Open the cabinet doors."
  Episode  16/20: FAIL    (steps= 500, reward=0.0) layout=22, style=49, task="Open the cabinet doors."
  Episode  17/20: FAIL    (steps= 500, reward=0.0) layout=55, style=48, task="Open the cabinet door."
  Episode  18/20: FAIL    (steps= 500, reward=0.0) layout=53, style=40, task="Open the cabinet doors."
  Episode  19/20: FAIL    (steps= 500, reward=0.0) layout=32, style=14, task="Open the cabinet door."
  Episode  20/20: FAIL    (steps= 500, reward=0.0) layout=27, style=27, task="Open the cabinet doors."

============================================================
  Evaluation Results
============================================================
  Split:          pretrain
  Episodes:       20
  Successes:      0/20
  Success rate:   0.0%
  Avg ep length:  500.0 steps
  Avg reward:     0.000
```


# 6a: Augmented dataset, temporal history MLP
Training 100 Epochs

```
  Episodes loaded : 50
  Samples         : 17317
  History window  : 4 steps
  Input dim       : 108  (4 × 27)
  Action dim      : 12

  Device       : cuda
  Parameters   : 716,556

============================================================
  Training
============================================================
  Epochs       : 100
  Batch size   : 64
  LR           : 0.0003
  History len  : 4

  Epoch    1/100  loss=0.098842  lr=3.00e-04
  Epoch   10/100  loss=0.022442  lr=2.93e-04
  Epoch   20/100  loss=0.015504  lr=2.71e-04
  Epoch   30/100  loss=0.012805  lr=2.38e-04
  Epoch   40/100  loss=0.010602  lr=1.96e-04
  Epoch   50/100  loss=0.008589  lr=1.50e-04
  Epoch   60/100  loss=0.006700  lr=1.04e-04
  Epoch   70/100  loss=0.005510  lr=6.18e-05
  Epoch   80/100  loss=0.004016  lr=2.86e-05
  Epoch   90/100  loss=0.003154  lr=7.34e-06
  Epoch  100/100  loss=0.002832  lr=0.00e+00

  Training complete!
  Best MSE loss  : 0.002832
  Checkpoints in : /tmp/cabinet_policy_06a
  Loss log       : /tmp/cabinet_policy_06a/loss_history.txt
```
```
============================================================
  OpenCabinet - Temporal Policy Evaluation (06a)
============================================================
Device: cuda

Loaded policy   : /tmp/cabinet_policy_06a/best_policy.pt
  Epoch         : 99,  loss=0.002832
  History len   : 4
  Single state  : 27  →  input: 108
  Action dim    : 12

============================================================
  Evaluating on 'pretrain' split  (20 episodes)
============================================================
[robosuite INFO] Loading controller configuration from: /home/lu/CS188/cs188-cabinet-door-project/robosuite/robosuite/controllers/config/robots/default_pandaomron.json (composite_controller_factory.py:121)
  Episode   1/20: FAIL    (steps= 500, reward=0.0) layout=53, style=37, task="Open the cabinet door."
  Episode   2/20: FAIL    (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
  Episode   3/20: FAIL    (steps= 500, reward=0.0) layout=26, style=13, task="Open the cabinet doors."
  Episode   4/20: FAIL    (steps= 500, reward=0.0) layout=55, style=46, task="Open the cabinet door."
  Episode   5/20: FAIL    (steps= 500, reward=0.0) layout=59, style=38, task="Open the cabinet door."
  Episode   6/20: FAIL    (steps= 500, reward=0.0) layout=47, style=43, task="Open the cabinet doors."
  Episode   7/20: FAIL    (steps= 500, reward=0.0) layout=59, style=26, task="Open the cabinet door."
  Episode   8/20: FAIL    (steps= 500, reward=0.0) layout=50, style=47, task="Open the cabinet doors."
  Episode   9/20: FAIL    (steps= 500, reward=0.0) layout=27, style=21, task="Open the cabinet door."
  Episode  10/20: FAIL    (steps= 500, reward=0.0) layout=29, style=17, task="Open the cabinet doors."
  Episode  11/20: FAIL    (steps= 500, reward=0.0) layout=42, style=31, task="Open the cabinet doors."
  Episode  12/20: FAIL    (steps= 500, reward=0.0) layout=54, style=37, task="Open the cabinet door."
  Episode  13/20: FAIL    (steps= 500, reward=0.0) layout=43, style=29, task="Open the cabinet doors."
  Episode  14/20: FAIL    (steps= 500, reward=0.0) layout=43, style=21, task="Open the cabinet doors."
  Episode  15/20: FAIL    (steps= 500, reward=0.0) layout=53, style=12, task="Open the cabinet doors."
  Episode  16/20: FAIL    (steps= 500, reward=0.0) layout=22, style=49, task="Open the cabinet doors."
  Episode  17/20: FAIL    (steps= 500, reward=0.0) layout=55, style=48, task="Open the cabinet door."
  Episode  18/20: FAIL    (steps= 500, reward=0.0) layout=53, style=40, task="Open the cabinet doors."
  Episode  19/20: FAIL    (steps= 500, reward=0.0) layout=32, style=14, task="Open the cabinet door."
  Episode  20/20: FAIL    (steps= 500, reward=0.0) layout=27, style=27, task="Open the cabinet doors."

============================================================
  Results
============================================================
  Split         : pretrain
  Episodes      : 20
  Successes     : 0/20
  Success rate  : 0.0%
  Avg ep length : 500.0 steps
  Avg reward    : 0.000
```
This version also has odd backwards motion not present in the regular dataset. The likely cause is that when feeding it the first 4 states, the states before the first state are duplicates of it, rather than zero-padding. The next iterations will fix this.

# 6b. Augmented dataset, temporal & chunking MLP
Training 100 Epochs

```
  Episodes      : 50
  Samples       : 17317
  Input dim     : 108
  Output dim    : 96

  Device      : cuda
  Parameters  : 738,144

============================================================
  Training
============================================================
  Epochs      : 100
  Batch size  : 64
  LR          : 0.0003
  Chunk size  : 8
  History len : 4

  Epoch    1/100  loss=0.100103  lr=3.00e-04
  Epoch   10/100  loss=0.019851  lr=2.93e-04
  Epoch   20/100  loss=0.014152  lr=2.71e-04
  Epoch   30/100  loss=0.010896  lr=2.38e-04
  Epoch   40/100  loss=0.008926  lr=1.96e-04
  Epoch   50/100  loss=0.006992  lr=1.50e-04
  Epoch   60/100  loss=0.005589  lr=1.04e-04
  Epoch   70/100  loss=0.004336  lr=6.18e-05
  Epoch   80/100  loss=0.003513  lr=2.86e-05
  Epoch   90/100  loss=0.002977  lr=7.34e-06
  Epoch  100/100  loss=0.002809  lr=0.00e+00

  Training complete!
  Best loss   : 0.002809
  Checkpoints : /tmp/cabinet_policy_06b
```
```
============================================================
  OpenCabinet - Action Chunking Evaluation (06b)
============================================================
Loaded          : /tmp/cabinet_policy_06b/best_policy.pt
  Epoch         : 99,  loss=0.002809
  History len   : 4
  Chunk size    : 8
  Action dim    : 12

============================================================
  Evaluating on 'pretrain' (20 episodes)
============================================================
[robosuite INFO] Loading controller configuration from: /home/lu/CS188/cs188-cabinet-door-project/robosuite/robosuite/controllers/config/robots/default_pandaomron.json (composite_controller_factory.py:121)
  Episode   1/20: FAIL    (steps= 500, reward=0.0) layout=53, style=37, task="Open the cabinet door."
  Episode   2/20: FAIL    (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
  Episode   3/20: FAIL    (steps= 500, reward=0.0) layout=26, style=13, task="Open the cabinet doors."
  Episode   4/20: FAIL    (steps= 500, reward=0.0) layout=55, style=46, task="Open the cabinet door."
  Episode   5/20: FAIL    (steps= 500, reward=0.0) layout=59, style=38, task="Open the cabinet door."
  Episode   6/20: FAIL    (steps= 500, reward=0.0) layout=47, style=43, task="Open the cabinet doors."
  Episode   7/20: FAIL    (steps= 500, reward=0.0) layout=59, style=26, task="Open the cabinet door."
  Episode   8/20: FAIL    (steps= 500, reward=0.0) layout=50, style=47, task="Open the cabinet doors."
  Episode   9/20: FAIL    (steps= 500, reward=0.0) layout=27, style=21, task="Open the cabinet door."
  Episode  10/20: FAIL    (steps= 500, reward=0.0) layout=29, style=17, task="Open the cabinet doors."
  Episode  11/20: FAIL    (steps= 500, reward=0.0) layout=42, style=31, task="Open the cabinet doors."
  Episode  12/20: FAIL    (steps= 500, reward=0.0) layout=54, style=37, task="Open the cabinet door."
  Episode  13/20: FAIL    (steps= 500, reward=0.0) layout=43, style=29, task="Open the cabinet doors."
  Episode  14/20: FAIL    (steps= 500, reward=0.0) layout=43, style=21, task="Open the cabinet doors."
  Episode  15/20: FAIL    (steps= 500, reward=0.0) layout=53, style=12, task="Open the cabinet doors."
  Episode  16/20: FAIL    (steps= 500, reward=0.0) layout=22, style=49, task="Open the cabinet doors."
  Episode  17/20: FAIL    (steps= 500, reward=0.0) layout=55, style=48, task="Open the cabinet door."
  Episode  18/20: FAIL    (steps= 500, reward=0.0) layout=53, style=40, task="Open the cabinet doors."
  Episode  19/20: FAIL    (steps= 500, reward=0.0) layout=32, style=14, task="Open the cabinet door."
  Episode  20/20: FAIL    (steps= 500, reward=0.0) layout=27, style=27, task="Open the cabinet doors."

============================================================
  Results
============================================================
  Split         : pretrain
  Successes     : 0/20
  Success rate  : 0.0%
  Avg ep length : 500.0 steps
  Avg reward    : 0.000
```

# 6c. Minimal diffusion policy
Training 300 Epochs & max number of episodes
```
  Episodes  : 107
  Samples   : 37492
  Input dim : 108
  Output dim: 96

  Device      : cuda
  Parameters  : 820,064
  Diff steps  : 100

============================================================
  Training
============================================================
  Epochs      : 300
  Batch size  : 64
  LR          : 0.0003
  Chunk size  : 8
  History len : 4

  Epoch    1/300  loss=0.458597  lr=3.00e-04
  Epoch   10/300  loss=0.107892  lr=2.99e-04
  Epoch   20/300  loss=0.095834  lr=2.97e-04
  Epoch   30/300  loss=0.089445  lr=2.93e-04
  Epoch   40/300  loss=0.085802  lr=2.87e-04
  Epoch   50/300  loss=0.084268  lr=2.80e-04
  Epoch   60/300  loss=0.080257  lr=2.71e-04
  Epoch   70/300  loss=0.078816  lr=2.61e-04
  Epoch   80/300  loss=0.076595  lr=2.50e-04
  Epoch   90/300  loss=0.075244  lr=2.38e-04
  Epoch  100/300  loss=0.073558  lr=2.25e-04
  Epoch  110/300  loss=0.072644  lr=2.11e-04
  Epoch  120/300  loss=0.071023  lr=1.96e-04
  Epoch  130/300  loss=0.069664  lr=1.81e-04
  Epoch  140/300  loss=0.068882  lr=1.66e-04
  Epoch  150/300  loss=0.066911  lr=1.50e-04
  Epoch  160/300  loss=0.065769  lr=1.34e-04
  Epoch  170/300  loss=0.064773  lr=1.19e-04
  Epoch  180/300  loss=0.063811  lr=1.04e-04
  Epoch  190/300  loss=0.062686  lr=8.90e-05
  Epoch  200/300  loss=0.060591  lr=7.50e-05
  Epoch  210/300  loss=0.060311  lr=6.18e-05
  Epoch  220/300  loss=0.060329  lr=4.96e-05
  Epoch  230/300  loss=0.058921  lr=3.85e-05
  Epoch  240/300  loss=0.058795  lr=2.86e-05
  Epoch  250/300  loss=0.057616  lr=2.01e-05
  Epoch  260/300  loss=0.057803  lr=1.30e-05
  Epoch  270/300  loss=0.057664  lr=7.34e-06
  Epoch  280/300  loss=0.056386  lr=3.28e-06
  Epoch  290/300  loss=0.057469  lr=8.22e-07
  Epoch  300/300  loss=0.056799  lr=0.00e+00

  Training complete!
  Best loss   : 0.056180
  Checkpoints : /tmp/cabinet_policy_06c
```
```
============================================================
  OpenCabinet - Diffusion Policy Evaluation (06c)
============================================================
Device: cuda

Loaded policy   : /tmp/cabinet_policy_06c/best_policy.pt
  Epoch         : 274,  loss=0.056180
  State dim     : 108
  Action dim    : 96
  History len   : 4
  Chunk size    : 8
  Diff steps    : 100

============================================================
  Evaluating on 'pretrain' split  (20 episodes)
============================================================
[robosuite INFO] Loading controller configuration from: /home/lu/CS188/cs188-cabinet-door-project/robosuite/robosuite/controllers/config/robots/default_pandaomron.json (composite_controller_factory.py:121)
  Episode   1/20: FAIL    (steps= 500, reward=0.0) layout=53, style=37, task="Open the cabinet door."
  Episode   2/20: FAIL    (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
  Episode   3/20: FAIL    (steps= 500, reward=0.0) layout=26, style=13, task="Open the cabinet doors."
  Episode   4/20: FAIL    (steps= 500, reward=0.0) layout=55, style=46, task="Open the cabinet door."
  Episode   5/20: FAIL    (steps= 500, reward=0.0) layout=59, style=38, task="Open the cabinet door."
  Episode   6/20: FAIL    (steps= 500, reward=0.0) layout=47, style=43, task="Open the cabinet doors."
  Episode   7/20: FAIL    (steps= 500, reward=0.0) layout=59, style=26, task="Open the cabinet door."
  Episode   8/20: FAIL    (steps= 500, reward=0.0) layout=50, style=47, task="Open the cabinet doors."
  Episode   9/20: FAIL    (steps= 500, reward=0.0) layout=27, style=21, task="Open the cabinet door."
  Episode  10/20: FAIL    (steps= 500, reward=0.0) layout=29, style=17, task="Open the cabinet doors."
  Episode  11/20: FAIL    (steps= 500, reward=0.0) layout=42, style=31, task="Open the cabinet doors."
  Episode  12/20: FAIL    (steps= 500, reward=0.0) layout=54, style=37, task="Open the cabinet door."
  Episode  13/20: FAIL    (steps= 500, reward=0.0) layout=43, style=29, task="Open the cabinet doors."
  Episode  14/20: FAIL    (steps= 500, reward=0.0) layout=43, style=21, task="Open the cabinet doors."
  Episode  15/20: FAIL    (steps= 500, reward=0.0) layout=53, style=12, task="Open the cabinet doors."
  Episode  16/20: FAIL    (steps= 500, reward=0.0) layout=22, style=49, task="Open the cabinet doors."
  Episode  17/20: FAIL    (steps= 500, reward=0.0) layout=55, style=48, task="Open the cabinet door."
  Episode  18/20: FAIL    (steps= 500, reward=0.0) layout=53, style=40, task="Open the cabinet doors."
  Episode  19/20: FAIL    (steps= 500, reward=0.0) layout=32, style=14, task="Open the cabinet door."
  Episode  20/20: FAIL    (steps= 500, reward=0.0) layout=27, style=27, task="Open the cabinet doors."

============================================================
  Results
============================================================
  Split         : pretrain
  Episodes      : 20
  Successes     : 0/20
  Success rate  : 0.0%
  Avg ep length : 500.0 steps
  Avg reward    : 0.000
```

# 6d. Minimal diffusion with unet
Training 300 epochs and using all episodes (note that this is a previous version that mistakenly overwrote policy 6c, the current version won't do that anymore)
```
  Episodes  : 107
  Samples   : 37492
  State dim : 108
  Action shape: (8, 12)
  U-Net parameters: 10,581,004

  Device     : cuda

============================================================
  Training
============================================================
  Epochs      : 300
  Batch size  : 64
  Diff steps  : 100
  Chunk size  : 8

  Epoch    1/300  loss=0.342053  lr=1.00e-04
  Epoch   10/300  loss=0.066786  lr=9.97e-05
  Epoch   20/300  loss=0.057990  lr=9.89e-05
  Epoch   30/300  loss=0.053237  lr=9.76e-05
  Epoch   40/300  loss=0.048949  lr=9.57e-05
  Epoch   50/300  loss=0.045336  lr=9.33e-05
  Epoch   60/300  loss=0.042097  lr=9.05e-05
  Epoch   70/300  loss=0.039054  lr=8.72e-05
  Epoch   80/300  loss=0.036527  lr=8.35e-05
  Epoch   90/300  loss=0.034121  lr=7.94e-05
  Epoch  100/300  loss=0.032369  lr=7.50e-05
  Epoch  110/300  loss=0.030715  lr=7.03e-05
  Epoch  120/300  loss=0.028635  lr=6.55e-05
  Epoch  130/300  loss=0.027102  lr=6.04e-05
  Epoch  140/300  loss=0.026441  lr=5.52e-05
  Epoch  150/300  loss=0.025009  lr=5.00e-05
  Epoch  160/300  loss=0.024081  lr=4.48e-05
  Epoch  170/300  loss=0.022959  lr=3.96e-05
  Epoch  180/300  loss=0.021364  lr=3.45e-05
  Epoch  190/300  loss=0.020888  lr=2.97e-05
  Epoch  200/300  loss=0.019866  lr=2.50e-05
  Epoch  210/300  loss=0.019047  lr=2.06e-05
  Epoch  220/300  loss=0.018846  lr=1.65e-05
  Epoch  230/300  loss=0.017955  lr=1.28e-05
  Epoch  240/300  loss=0.016831  lr=9.55e-06
  Epoch  250/300  loss=0.016796  lr=6.70e-06
  Epoch  260/300  loss=0.015949  lr=4.32e-06
  Epoch  270/300  loss=0.015636  lr=2.45e-06
  Epoch  280/300  loss=0.015466  lr=1.09e-06
  Epoch  290/300  loss=0.015195  lr=2.74e-07
  Epoch  300/300  loss=0.014952  lr=0.00e+00

  Training complete!
  Best loss   : 0.014860
  Checkpoints : /tmp/cabinet_policy_06c
```
```
============================================================
  OpenCabinet - UNet Diffusion Policy Evaluation (06d)
============================================================
Device: cuda

Loaded policy       : /tmp/cabinet_policy_06d/best_policy.pt
  Epoch             : 296,  loss=0.014860
  State dim         : 108
  Action dim        : 12
  History len       : 4
  Chunk size        : 8
  Diff steps        : 100
  Down dims         : (128, 256, 512)

============================================================
  Evaluating on 'pretrain' split  (20 episodes)
============================================================
[robosuite INFO] Loading controller configuration from: /home/lu/CS188/cs188-cabinet-door-project/robosuite/robosuite/controllers/config/robots/default_pandaomron.json (composite_controller_factory.py:121)
  Episode   1/20: FAIL    (steps= 500, reward=0.0) layout=53, style=37, task="Open the cabinet door."
  Episode   2/20: FAIL    (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
  Episode   3/20: FAIL    (steps= 500, reward=0.0) layout=26, style=13, task="Open the cabinet doors."
  Episode   4/20: FAIL    (steps= 500, reward=0.0) layout=55, style=46, task="Open the cabinet door."
  Episode   5/20: FAIL    (steps= 500, reward=0.0) layout=59, style=38, task="Open the cabinet door."
  Episode   6/20: FAIL    (steps= 500, reward=0.0) layout=47, style=43, task="Open the cabinet doors."
  Episode   7/20: FAIL    (steps= 500, reward=0.0) layout=59, style=26, task="Open the cabinet door."
  Episode   8/20: FAIL    (steps= 500, reward=0.0) layout=50, style=47, task="Open the cabinet doors."
  Episode   9/20: FAIL    (steps= 500, reward=0.0) layout=27, style=21, task="Open the cabinet door."
  Episode  10/20: FAIL    (steps= 500, reward=0.0) layout=29, style=17, task="Open the cabinet doors."
  Episode  11/20: FAIL    (steps= 500, reward=0.0) layout=42, style=31, task="Open the cabinet doors."
  Episode  12/20: FAIL    (steps= 500, reward=0.0) layout=54, style=37, task="Open the cabinet door."
  Episode  13/20: FAIL    (steps= 500, reward=0.0) layout=43, style=29, task="Open the cabinet doors."
  Episode  14/20: FAIL    (steps= 500, reward=0.0) layout=43, style=21, task="Open the cabinet doors."
  Episode  15/20: FAIL    (steps= 500, reward=0.0) layout=53, style=12, task="Open the cabinet doors."
  Episode  16/20: FAIL    (steps= 500, reward=0.0) layout=22, style=49, task="Open the cabinet doors."
  Episode  17/20: FAIL    (steps= 500, reward=0.0) layout=55, style=48, task="Open the cabinet door."
  Episode  18/20: FAIL    (steps= 500, reward=0.0) layout=53, style=40, task="Open the cabinet doors."
  Episode  19/20: FAIL    (steps= 500, reward=0.0) layout=32, style=14, task="Open the cabinet door."
  Episode  20/20: FAIL    (steps= 500, reward=0.0) layout=27, style=27, task="Open the cabinet doors."

============================================================
  Results
============================================================
  Split         : pretrain
  Episodes      : 20
  Successes     : 0/20
  Success rate  : 0.0%
  Avg ep length : 500.0 steps
  Avg reward    : 0.000
```
