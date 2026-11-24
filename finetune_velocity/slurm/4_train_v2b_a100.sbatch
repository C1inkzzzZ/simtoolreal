#!/bin/bash
# v2b (full A100): continue from the v1 final checkpoint with the v2 settings in velocity.yaml
# (no lead-in, dense velocity tracking, Leo's end values epsKp 1 cm / epsVFrac 0.15), 24576 envs (6 x 4096).
# v1 used about 18 GB with 12288 envs, so 24576 should fit in 40 GB; check logs/gpu_<job>.csv.
# Smoke-test first on a MIG slice:
#   sbatch --export=ALL,NAME=13_vel_smoke_v2b,CKPT=runs/20_vel_ft_v1/nn/last_20_vel_ft_v1_ep_6000_rew__8364.4_.pth \
#       finetune_velocity/slurm/2_smoke_mig.sbatch
#SBATCH --job-name=vel_ft_v2b
#SBATCH --account=cmsc848x-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source finetune_velocity/slurm/_common.sh
python finetune_velocity/launch_from_pretrained.py --task velocity --num-envs 24576 --name 30_vel_ft_v2b \
    --checkpoint runs/20_vel_ft_v1/nn/last_20_vel_ft_v1_ep_6000_rew__8364.4_.pth \
    --max-epochs 3500 --save-every 250
