#!/bin/bash
# Step 1 (MIG slice, starts right away): the widened checkpoint must behave exactly like the released one.
# Velocity goals off, arm speed x1, actor frozen for the whole run (only the critic trains).
# Expect: success_ratio / successes in the log about the same as runs/00_ft_sanity_tol1 at the same num_envs.
#SBATCH --job-name=vel_zero_init
#SBATCH --account=cmsc848x-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out
source finetune_velocity/slurm/_common.sh
python finetune_velocity/launch_from_pretrained.py --task velocity --num-envs 768 --name 10_vel_zero_init \
    --max-epochs 60 --critic-warmup 60 \
    task.env.velocity.velocityGoalProb=0.0 "task.env.velocity.armSpeedMult=[1.0,1.0]"
