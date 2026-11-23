# Sourced by the sbatch scripts in this folder: environment for Isaac Gym on Zaratan compute nodes.
module load gcc/11.3.0  # gymtorch's JIT build needs GCC >= 9
cd ~/scratch.cmsc848x/simtoolreal
source .venv/bin/activate
export PYTHONPATH=$PWD
mkdir -p logs
nvidia-smi
# GPU memory / utilization every 30 s, for sizing num_envs later
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
    > logs/gpu_${SLURM_JOB_ID}.csv &
