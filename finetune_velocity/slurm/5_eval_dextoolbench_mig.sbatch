#!/bin/bash
# DexToolBench (24 object/task pairs x 10 episodes) for one policy on a MIG slice. Results: evals/ft_<NAME>/summary.txt
#   sbatch --export=ALL,NAME=pretrained,CFG=pretrained_policy/config.yaml,CKPT=pretrained_policy/model.pth \
#       finetune_velocity/slurm/5_eval_dextoolbench_mig.sbatch
#   sbatch --export=ALL,NAME=v2,CFG=runs/31_vel_ft_v2/config.yaml,CKPT=runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth,LEVEL=0.10 \
#       finetune_velocity/slurm/5_eval_dextoolbench_mig.sbatch
# Re-submitting with the same NAME skips the pairs that already have an eval.json.
#SBATCH --job-name=dtb_eval
#SBATCH --account=cmsc848x-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_1g.5gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
source finetune_velocity/slurm/_common.sh
LEVEL_ARG=""
if [ -n "${LEVEL}" ]; then LEVEL_ARG="--fixed-level ${LEVEL}"; fi
echo "eval ${NAME}: config ${CFG}, checkpoint ${CKPT}, level ${LEVEL:-none}"
python finetune_velocity/run_dextoolbench_eval.py --name "${NAME}" --config "${CFG}" --checkpoint "${CKPT}" ${LEVEL_ARG}
