#!/bin/bash
# Trajectory goals (full A100): README section 14. 24576 envs, starting from CKPT (v2 final or the released model;
# missing obs terms are zero-widened automatically). Pass the start checkpoint and a run name:
#   sbatch --export=ALL,NAME=40_traj_v1,CKPT=runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth \
#       finetune_velocity/slurm/6_train_traj_a100.sbatch
#   (released model: CKPT=pretrained_policy/model.pth)
# Smoke-test first on a MIG slice:
#   sbatch --export=ALL,TASK=trajectory,NAME=16_traj_smoke,CKPT=<same checkpoint> finetune_velocity/slurm/2_smoke_mig.sbatch
#SBATCH --job-name=traj_ft
#SBATCH --account=cmsc848x-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source finetune_velocity/slurm/_common.sh
python finetune_velocity/launch_from_pretrained.py --task trajectory --num-envs 24576 --name ${NAME:-40_traj_v1} \
    --checkpoint ${CKPT:-pretrained_policy/model.pth} --max-epochs 3500 --save-every 250
