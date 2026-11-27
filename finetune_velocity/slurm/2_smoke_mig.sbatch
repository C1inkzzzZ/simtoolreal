#!/bin/bash
# Step 2 (MIG slice): the full velocity setup runs end to end. 768 envs, 150 epochs (20 critic-only).
# Re-run under a new name: sbatch --export=ALL,NAME=12_vel_smoke2 finetune_velocity/slurm/2_smoke_mig.sbatch
# From another checkpoint: add CKPT=runs/<run>/nn/<file>.pth to --export; extra overrides: EXTRA="key=value ..."
# Check in the log: [velocity] lines appear, infeasible < ~0.3, attempts > 0, no NaN, no crash; fps.
#SBATCH --job-name=vel_smoke
#SBATCH --account=cmsc848x-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
source finetune_velocity/slurm/_common.sh
python finetune_velocity/launch_from_pretrained.py --task velocity --num-envs 768 --name ${NAME:-11_vel_smoke} --checkpoint ${CKPT:-pretrained_policy/model.pth} \
    --max-epochs 150 task.env.velocity.printEvery=80 ${EXTRA}
