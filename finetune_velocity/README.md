# Velocity-goal fine-tuning of the released SimToolReal policy (experiment v1)

Short English version of `README_zh.md`, which has the full rationale for each change.

Question: does conditioning on a goal **velocity** (in addition to pose and orientation) help force-sensitive tool tasks?
This branch fine-tunes the released Isaac Gym checkpoint (DexToolBench baseline 93.3%) on pose + velocity goals,
following Leo's MuJoCo pose-velocity setup wherever SimToolReal's structure allows.

## What changes

* **New task `SimToolRealVelocity`** (`isaacgymenvs/tasks/simtoolreal/velocity_env.py`) subclasses `SimToolReal`.
  The base env only gains two empty hooks for extra observation terms.
* **Goals**: 50% official static goals (unchanged hold rule: 1.5 cm keypoint distance for 10 steps, 100 per near step),
  50% velocity goals. The first goal of an episode (the pick-up) is always static.
* **Velocity goal**:
  * The strike pose is an official delta goal.
  * The direction is drawn within a cone around the head's face axis (30° → 180° by curriculum).
    25% of goals are steep downward strikes; for these the tool is turned so that its head faces down.
  * A **lead-in** waypoint sits 12 cm behind the strike pose, with the strike orientation. It counts as reached at a
    keypoint distance ≤ 3 cm, with no hold and no bonus.
  * The lead-in, strike and follow-through (+8 cm) points must lie in the target volume, with the head above the
    table. Otherwise the goal is resampled, and falls back to a static goal.
* **Attempt**: the first strike step with fixed-size keypoint distance ≤ eps_kp (6 → 3 cm, compared directly, not
  ×1.5).
* **Reward**: on the attempt, the env pays `1000·exp(−|v_head − v_g|/(σ|v_g|))` once, then moves to the next goal.
  The error is linear, as in Leo's code; the midterm report's squared form is a typo.
  σ goes from 1.0 to 0.25 over the curriculum.
  The velocity success used by the curriculum and logs is |dv| < eps_v·|v_g|, with eps_v going from 0.6 to 0.25.
* **v_head** is measured at the **head center**: r_head = handle_len/2 + head_len_x/2 along +x, read back from the
  generated URDFs. It is the finite difference of the head position over one control step, which sidesteps the
  root-velocity-at-COM question.
* **Keypoints stay official**, centered on the handle, because they are policy inputs.
* **Arm speed**: the released arm moves at most about 0.15 rad/s (0.1 × 1.5 / 60 rad per step). A performance-gated
  multiplier raises it from ×1.5 to ×6. It does not use the official tyler curriculum, which is gated on episode
  success ratio and wall-clock minutes, and is tied to obs-dropout switches.
* **Curriculum**: one level drives σ, eps_v, eps_kp, the cone, the speed range (ending at 0.3–0.6 m/s) and the arm
  multiplier.
  * The level is decided in blocks of 2048 finished velocity goals: up at ≥ 0.5 success, down at < 0.2 (Leo's
    thresholds).
  * It only goes up while static success is ≥ 0.5.
* **Observations**:
  * Policy and critic get `goal_velocity` = [v_g, v_head, v_head − v_g] / 0.5 m/s (9 dims). The scaling is manual
    because the released normalizer's count is huge, so its statistics for new dims barely move.
  * The critic also gets a 2-dim lead-in / strike phase one-hot.
  * DR and delays are unchanged.
* **Checkpoint widening** (`finetune_velocity/expand_checkpoint.py`):
  * Zero columns are inserted at index 140 (critic: 162), before the 32 SAPG extra-param columns.
  * The normalizer gets mean 0 and var 1 for the new dims.
  * Outputs stay identical whatever the new dims contain (tested).
  * The critic is loaded strictly.
* **Training**:
  * 12288 envs (6 SAPG blocks × 2048), actor lr 5e-5, successTolerance fixed at 0.01.
  * For the first 20 epochs the actor lr is 0, so only the critic learns. The adaptive lr is reset after each warmup
    epoch.

## Run (Zaratan)

```bash
python finetune_velocity/tests/test_cpu.py                      # CPU tests
sbatch finetune_velocity/slurm/1_zero_init_check_mig.sbatch     # must match the static sanity run
sbatch finetune_velocity/slurm/2_smoke_mig.sbatch               # end-to-end check on a MIG slice
sbatch finetune_velocity/slurm/3_train_a100.sbatch              # first real run
```

For a DexToolBench eval of a fine-tuned policy, pass
`--extra-overrides task.env.velocity.curriculum.fixedLevel=<final vel/level>` so the arm speed matches training.
