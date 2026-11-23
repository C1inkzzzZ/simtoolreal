#!/bin/bash
# Step 3 (full A100): the first velocity fine-tuning run. 12288 envs (6 SAPG blocks x 2048), lr 5e-5,
# 20 critic-only epochs, then actor + critic. Checkpoints every 200 epochs in runs/20_vel_ft_v1/nn/.
# To continue after the time limit: --load-mode resume --checkpoint runs/20_vel_ft_v1/nn/<last>.pth
#SBATCH --job-name=vel_ft_v1
#SBATCH --account=cmsc848x-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source finetune_velocity/slurm/_common.sh
python finetune_velocity/launch_from_pretrained.py --task velocity --num-envs 12288 --name 20_vel_ft_v1 \
    --max-epochs 6000 --save-every 200
