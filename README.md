# CS 188 Final Project
## by Andy Kasbarian

This project is based on this: https://github.com/HoldenGs/cs188-cabinet-door-project

If you set up that first so that you get to the point where you can run `python 00_verify_installation.py`, then you can use code from this project

All files from the original project except `07_evaluate_policy.py` have been modified. The mentioned file was modified to change the success condition from opening all cabinet doors to opening only one cabinet door, and to add the default path for the trained best policy.

## AI disclosure
Claude helped me create all of the original files. Script `05b_augment_handle_data.py` is from Holden Grissett. All code was reviewed, debugged, and understood by me.

## Baseline MLP:

1. `python 06_*` to train
2. `python 07_*` to evaluate
3. `python 08_*` to visualize

## MLP with temporal context:

1. `python 05b_*` to augment dataset
2. `python 06a_*` to train
3. `python 07a_*` to evaluate
4. `python 08a_*` to visualize

The next ones assume you've already run `python 05b_*`

## MLP with temporal + action chunking

1. `python 06b_*` to train
2. `python 07b_*` to evaluate
3. `python 08b_*` to visualize

## MLP with diffusion

1. `python 06c_*` to train
2. `python 07c_*` to evaluate
3. `python 08c_*` to visualize

## U-Net with Diffusion

1. `python 06d_*` to train
2. `python 07d_*` to evaluate
3. `python 08d_*` to visualize

`Notes.md` contains the results for all of the runs.
