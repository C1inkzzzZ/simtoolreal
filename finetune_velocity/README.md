# Fine-tuning the released SimToolReal checkpoint with velocity goals: experiment v1

This document explains every change on the `velocity-finetune` branch, why it was made, and how to run it on Zaratan.
It is the full English version of `README_zh.md`; the two have the same content.

Research question: **does giving the policy a goal velocity in addition to the goal pose (pose + orientation + velocity
goal) help force-sensitive tool tasks?**
This step covers training only: fine-tuning the released Isaac Gym policy so that it passes through a given pose at a
given velocity.

---

## 1. Approach

1. Instead of training from scratch, we continue from the released checkpoint (DexToolBench baseline: 93.3% overall)
   and change only what velocity goals require. The policy's existing grasping, transport and pose-tracking skills are
   reused as they are.
2. Half of the goals stay official static goals, with the official rules unchanged; the other half become velocity
   goals. The static goals prevent forgetting: the fine-tuned policy should still solve DexToolBench.
3. The velocity design follows Leo's MuJoCo setup wherever possible, since it is already validated there
   (pose-velocity: plate 76% / nail 96% / sweep 58%). It deviates only where SimToolReal's structure requires it; each
   such place is explained below.
4. The new observations enter the network through zero-weight widening: at the start of fine-tuning, the policy and the
   critic output exactly what the released ones output, and they learn to use the velocity inputs from there.

---

## 2. Files

| File | Content |
|---|---|
| `isaacgymenvs/tasks/simtoolreal/velocity_env.py` | New task `SimToolRealVelocity` (subclass of the official `SimToolReal`): goals, reward, observations, arm speed, curriculum, logging |
| `isaacgymenvs/tasks/simtoolreal/velocity_goals.py` | Pure torch functions, no Isaac Gym dependency, testable on CPU: cone sampling, velocity-goal sampling, velocity matching, curriculum |
| `isaacgymenvs/tasks/simtoolreal/env.py` | Only two empty hooks added (`_extra_obs_type_sizes`, `_add_extra_obs`); official behaviour unchanged |
| `isaacgymenvs/tasks/__init__.py` | Registers `SimToolRealVelocity` |
| `finetune_velocity/velocity.yaml` | All velocity-goal parameters (each can be overridden on the command line) |
| `finetune_velocity/expand_checkpoint.py` | Zero-weight widening of the checkpoint, with a check that old weights are unchanged and the new columns are zero |
| `finetune_velocity/launch_from_pretrained.py` | Launcher: `--task static` (the earlier sanity runs) or `--task velocity`; widens the checkpoint automatically, critic warmup, strict critic loading |
| `finetune_velocity/slurm/*.sbatch` | Zero-init check (MIG), smoke test (MIG), v1 main run (full A100), v2 main run (full A100, section 11) |
| `finetune_velocity/tests/test_cpu.py` | CPU unit tests, including a numerical check that widening leaves the outputs unchanged |
| `dextoolbench/eval_isaacgym.py` | Camera sensors off (compute nodes have no graphics device); new `--extra-overrides` argument |

---

## 3. Changes and rationale

### 3.1 Goal types: 50% static + 50% velocity

Whenever a new goal is drawn (except the first goal of an episode, which includes picking the tool up), it becomes a
velocity goal with probability `velocityGoalProb = 0.5`; otherwise it is an official static goal.

- **Static goals (v_g = 0)** follow the official rule exactly:
  - The goal is reached when the keypoint distance ≤ `successTolerance × keypointScale` = 0.01 × 1.5 = 1.5 cm,
    accumulated over 10 steps.
  - Each step near the goal pays 1000/10 = 100.
  - Why: this is what the released checkpoint was trained on. Keeping it unchanged preserves what the policy already
    knows, and DexToolBench results stay directly comparable to the baseline.
- **Velocity goals** have two stages, a **lead-in** and then a **strike** (see 3.2).
  - Why 50%: velocity goals need enough samples to be learned, while static goals need a large enough share to
    prevent forgetting. We start at 1:1 and adjust if the static success rate drops noticeably.
- **The first goal stays static** (`velocityOnFirstGoal: False`). It includes picking the tool up from the table,
  which is hard enough on its own without a velocity requirement on top.

### 3.2 Velocity-goal geometry (strike + lead-in)

1. **Strike pose**: drawn with the official delta sampler, like a normal goal: ≤ 10 cm translation and ≤ 90°
   rotation from the previous goal, clamped to the official target volume.
   - Why: same distribution as the official goals, so the pose changes are familiar to the policy.
2. **Direction d**: uniform within a cone around the head's face axis. The cone angle widens from 30° to 180° over
   the curriculum (Leo's `vel_cone_deg`).
   - 30° means striking with the head's face; at 180° any direction is allowed, which covers sweeping and scraping.
3. **25% steep downward strikes** (d_z ≤ −0.85, Leo's `down_frac` / `down_max_z`): a downward direction is drawn
   first, then the tool is turned by the smallest rotation that brings its head axis within the cone of d.
   - Why: in random orientations the head axis almost never points down, so without forcing them, downward strikes
     (like hammering a nail) would be very rare in training. Leo's setup does the same.
4. **Lead-in waypoint**: 12 cm behind the strike pose (along −d), with the strike orientation.
   - It counts as reached at a keypoint distance ≤ 3 cm, with no hold and no bonus (the keypoint progress reward still
     applies). The goal then switches to the strike.
   - Why a lead-in: the policy needs a run-up to be at the goal velocity when it reaches the strike pose. Without one,
     the shortest path often comes in from the side, with the wrong velocity direction.
   - Why no hold: the policy should be able to pass through the lead-in while already moving.
5. **Feasibility check**:
   - The strike tool origin must lie inside the official target volume.
   - The lead-in and follow-through points (follow-through: 8 cm past the strike) may leave the volume by at most
     `volumeMargin = 10 cm`. This margin was added after the smoke test; see section 8.
   - The head center of all three points must be above `minHeadZ = 0.58 m` (the table top is at about 0.53 m).
   - Each goal tries 8 candidates and takes the first feasible one. If none is feasible, it falls back to a static goal.
   - Why: there is room for a run-up before the strike and for follow-through after it, and the head is never asked to
     go through the table. `vel/infeasible_frac` logs the fallback rate.
6. **Speed**: drawn from U(speedLo, speedHi), raised by the curriculum from 0.15–0.35 m/s to Leo's 0.3–0.6 m/s. The
   start range was originally 0.05–0.15 m/s and was raised after the smoke test; see section 8.

**Head face axis per tool type.** The tool frame has its origin at the handle center, with +x pointing from the handle
to the head. Defaults are in `velocity_env.DEFAULT_HEAD_AXIS` and can be changed through `headAxis` in the yaml.

| Type | Axis | Two faces | Why |
|---|---|---|---|
| hammer | +y | yes | The hammer head is long along y, with a striking face at each end |
| screwdriver / marker | +x | no | The tip points along the handle |
| spatula | +x | no | The blade pushes out along +x |
| eraser / brush | +z | yes | Normal of the large flat faces (erasers and brushes work with a flat face) |

These axes are my judgment from the geometry, not an official definition. Since the cone eventually widens to 180°,
they only shape the direction distribution early in the curriculum.

### 3.3 Attempt rule: keypoint distance ≤ eps_kp (6 cm → 3 cm)

During the strike stage, the first step with keypoint distance ≤ `epsKp` is the attempt. The velocity reward is paid
once on that step, and then the next goal is drawn.

- The threshold is compared directly with the official fixed-size keypoint distance (`keypoints_max_dist_fixed_size`),
  **without multiplying by keypointScale 1.5**.
  - Why: the official rule uses `success_tolerance × 1.5` because its tolerance is defined before scaling. Leo's
    `eps_kp` is already a direct distance (6 → 3 cm); multiplying by 1.5 again would give 9 → 4.5 cm, which would not
    match his setting.
- Why start at 6 cm rather than the official curriculum's starting 11 cm: Leo found that with a large threshold, the
  attempt fires too early, while the head is still lining up near the lead-in. The oracle succeeded only 42% of the
  time at 0.11, against 94% at 0.06.
- Static goals keep the official 1.5 cm / 10-step hold rule.

### 3.4 Velocity reward: `1000 × exp(−|v_head − v_g| / (σ |v_g|))`, paid once at the attempt

- The formula matches Leo's code (`rewards.velocity_match`): the error is **linear**. **The squared form in the midterm
  report is a typo and should be corrected.**
- σ (`sigmaFrac`) tightens from 1.0 to 0.25 (Leo's final value) over the curriculum.
  - Why: at the start the policy barely controls its speed on purpose. With a small σ the reward would be near 0 and
    give no gradient signal. With σ = 1, a velocity error as large as the goal speed still earns 37%.
- Dividing by |v_g| makes the error relative, so fast and slow goals are about equally hard.
- **Velocity success** (used by the curriculum and the logs, not by the reward): an attempt with
  |dv| < epsVFrac × |v_g|, where epsVFrac tightens from 0.6 to 0.25 (Leo's `eps_v_frac`).

### 3.5 Velocity measured at the head center; keypoints stay at the official handle center

- **Keypoints unchanged**: the official 4 keypoints are centered on the handle, and they are themselves policy inputs
  (`keypoints_rel_goal`, `keypoints_rel_palm`). Moving them to the head, as in Leo's setup, would change the meaning of
  existing inputs and amount to discarding the pretrained weights.
- **Velocity at the head center**: what a strike does depends on the head's velocity, not the handle center's. The two
  differ a lot when the tool rotates (v_head = v_obj + ω × r_head).
  - **Head offset**: r_head = (handle_len/2 + head_len_x/2, 0, 0), read back from the generated URDFs (the official
    generator does not return head sizes). For the eraser, which has no head, the handle center is used.
  - **Measurement**: the finite difference of the head position over one control step (1/60 s), rather than the root
    state velocity.
  - Why: PhysX's root linear velocity may be the velocity of the center of mass rather than of the frame origin (the
    tool's center of mass is shifted toward the head). The finite difference does not depend on that convention.
  - Right after a reset there is no previous position, so that one step uses the analytic estimate
    v_obj + ω × R r_head.
- **Side effect**: the attempt is decided by the handle-centered keypoints, while the velocity is measured at the head.
  At a keypoint distance ≤ 3 cm the orientation error is about ≤ 16°, so the head position error is within a few
  centimeters. `vel/head_err_at_attempt` logs the actual head position error at each attempt. If it turns out large,
  a head-error condition can be added to the attempt rule later.

### 3.6 Arm speed: why it is raised, and why not with the official tyler curriculum

- The official joint-target change per step is `armMovingAverage × dofSpeedScale × dt × a` = 0.1 × 1.5 / 60 =
  0.0025 rad. At full action that is only about **0.15 rad/s**, far from what 0.3–0.6 m/s strikes need. (Despite its
  name, `hand_dof_speed_scale` in the code applies to the 7 arm joints.)
- What we do: the curriculum multiplies this speed by `armSpeedMult`, from 1.5× up to 6× (about 0.9 rad/s). Static goals
  use the same arm speed, since there is only one policy.
- Why not the official tyler curriculum (`dofSpeedScaleFinal`, etc.):
  1. It advances on "average success ratio over the episode's 50 goals > 0.6", which has nothing to do with velocity
     performance.
  2. It limits advancement by wall-clock minutes, so MIG and A100 runs would behave differently.
  3. It also drives several "gradually turn off observations" switches, which would be confounded with our change.
  We use the velocity-gated curriculum below instead; the official tyler curriculum is left as it is.
- The starting arm speed was to be checked in the smoke test via `vel/speed_ratio` (|v_head|/|v_g|): if it were far
  below 1 at level 0, the arm would be too slow and the start of `armSpeedMult` should be raised. (It turned out to be
  the opposite; see section 8.)

### 3.7 Curriculum: one level drives every velocity parameter

A level ∈ [0, 1] linearly interpolates σ, epsVFrac, epsKp, the cone angle, the speed range and the arm-speed multiplier
together (as in Leo's `CurriculumCfg`).

- **Decisions in blocks**: every `minSamples = 2048` finished velocity goals, the velocity success rate of that block
  decides:
  - ≥ 0.5: level up by 0.05;
  - < 0.2: level down by 0.05;
  - otherwise unchanged.
  The thresholds are Leo's. The counters are then cleared for the next block.
- **Static guard**: the level may only go up while static goals succeed at least 50% of the time, so velocity is never
  bought with the existing skills.
- At least 160 control steps (about 10 PPO epochs) separate two level changes, so the policy has time to adapt.
- The level is stored in the checkpoint's env state and restored with `--load-mode resume`. For evaluation it is fixed
  with `curriculum.fixedLevel`.

### 3.8 New observations: v_g, v_head, v_head − v_g (9 dims, divided by 0.5 m/s)

- Appended after the original 140 dims, for both the policy and the critic (the critic gets them after its 162-dim
  state).
- **v_head − v_g** could be computed from the other two, but giving the error directly helps the network learn to
  correct it faster.
- **During the lead-in, the strike's v_g is already shown**, so the policy can start accelerating; that is the purpose
  of the lead-in.
- **Manual scaling by 0.5 m/s**: the official input normalizer (RunningMeanStd) has accumulated a huge number of
  samples, so the mean and variance of the new dims barely move and effectively stay at their initial values (0, 1).
  The values are therefore scaled by hand to around 1 (0.6 m/s → 1.2).
- **No extra noise** ("clean" values), but the whole observation vector still goes through the official random delay
  queue (the observation-delay DR is kept).
- **One deviation from the plan we agreed on**: the critic also gets a 2-dim `velocity_phase` (one-hot lead-in /
  strike); the policy does not.
  - Why: the value of the lead-in and the strike differs by a 0–1000 bonus, so a critic that cannot tell the stages
    apart gives noisier value estimates.
  - The policy does not need it: in both stages the action is "move toward the goal at v_g".
  - This only affects the critic's input, and it is zero-initialized as well.

### 3.9 Zero-weight widening of the checkpoint

- **Network inputs**: the actor's LSTM reads [obs 140 | SAPG params 32], and the critic's MLP reads
  [state 162 | SAPG params 32]. rl_games appends a block id to the env observation, and the network replaces it with
  that block's learned 32-dim vector.
- **Where the new dims go**: column 140 of the first weight matrix (162 for the critic), **before** the 32 SAPG-param
  columns. The script inserts zero columns there and gives the normalizer mean 0 and variance 1 for the new dims.
- **Result**: whatever the new inputs contain, the first layer's pre-activations are unchanged, so the outputs are
  exactly those of the released model. This is checked numerically with mock LSTM and MLP networks in
  `tests/test_cpu.py`, and in the real environment by `1_zero_init_check`.
- **Strict critic loading**: rl_games' weights mode silently skips loading the critic when its shapes do not match,
  which would hide a widening mistake. The launcher loads it strictly instead: any mismatch raises an error.
- **Weights only**: the optimizer state and the rest are dropped, so the widened file can only be loaded with
  `--load-mode weights`.

### 3.10 Critic warmup: the first 20 epochs train only the critic

- The new task has a different return distribution (an extra one-time velocity bonus), so the old critic's values are
  biased. Updating the policy right away could let badly biased advantages pull the pretrained policy off course
  first.
- **Implementation**: for the first 20 epochs, the actor's learning rate is set to 0. The critic has its own optimizer
  and trains normally.
- Note: the adaptive learning-rate schedule keeps raising the lr while the KL is 0. The lr is therefore reset to the
  base value after every warmup epoch, so that warmup does not end with a huge learning rate.

### 3.11 What stays official

- **Domain randomization**: action and observation delays and object-state noise all stay official. The official DR
  includes random velocity impulses on the object, which occasionally disturb an attempt; this is kept on purpose.
- **`successTolerance` is fixed at 0.01**: the official config starts its tolerance curriculum at 0.075. The 0.01 the
  checkpoint reached is stored in the env state, which weights mode does not restore (confirmed in the sanity runs).
- **Per-goal timer**: the timer (600 steps) also resets when the lead-in is reached, so the strike stage gets its full
  time.
- **`successes` count**: static successes + strike attempts, i.e. goals reached. This keeps the official meaning of
  "at most 50 goals per episode"; velocity quality is tracked separately.

### 3.12 Training hyperparameters

| Item | Value | Why |
|---|---|---|
| Envs | 12288 (6 SAPG blocks × 2048) | The checkpoint fixes 6 blocks. 6144 envs used about 12 GB in our test; 12288 leaves headroom on a 40 GB A100. The official setup appears to use 24576 envs (6 × 4096), which would not fit in 40 GB. |
| Actor learning rate | 5e-5 (official 1e-4) | Smaller steps when fine-tuning, to reduce forgetting |
| Critic warmup | 20 epochs | See 3.10 |
| Load mode | weights | Fresh optimizer, epoch starts at 0 |

---

## 4. Comparison with Leo's pose-velocity implementation

| Item | Leo (MuJoCo) | This implementation (Isaac Gym fine-tuning) | Reason |
|---|---|---|---|
| Keypoint center | Tool head | Handle center (official) | Keypoints are inputs of the pretrained policy |
| Velocity measured at | Tool head | Tool head (finite difference) | Same |
| Reward | 1000·exp(−\|dv\|/(σ\|v_g\|)), σ = 0.25 | Same formula, σ from 1.0 → 0.25 | Starts from a pretrained policy, so the signal starts loose |
| Attempt threshold | eps_kp 6 → 3 cm | Same | — |
| After the attempt | Episode ends | Next goal (episode continues) | Official episodes have many goals |
| Static goals | A separate baseline | 50% mixed into the same training run | Prevent forgetting |
| Lead-in | Reachability check point | Explicit intermediate goal (12 cm, no hold) | The official policy only knows "reach a pose", so it needs a point to reach first |
| Training | BC (DAgger) + PPO | PPO only, from the official weights | We already have a policy that can manipulate |
| Actions | Head twist / joints | Official joint targets + arm-speed multiplier | Action space unchanged |

---

## 5. Running on Zaratan

```bash
cd ~/scratch.cmsc848x/simtoolreal
source .venv/bin/activate
git fetch origin && git checkout velocity-finetune

# 1. CPU tests (the login node is enough)
python finetune_velocity/tests/test_cpu.py

# 2. Zero-init check (MIG slice, about 10 minutes)
sbatch finetune_velocity/slurm/1_zero_init_check_mig.sbatch
# 3. Smoke test (MIG slice, about 7 minutes); a re-run under a new name:
#    sbatch --export=ALL,NAME=12_vel_smoke2 finetune_velocity/slurm/2_smoke_mig.sbatch
sbatch finetune_velocity/slurm/2_smoke_mig.sbatch
# 4. Only when 2 and 3 look right: the main run (full A100, queues)
sbatch finetune_velocity/slurm/3_train_a100.sbatch
```

The first velocity run creates `pretrained_policy/model_velocity_obs149.pth` (the widened weights); later runs reuse it.

**What to check**

- **Zero-init check**: `successes` and `true_objective_mean` should be close to those of `00_ft_sanity_tol1` at the same
  point (both use 768 envs). If they are clearly worse, the widening or the observation wiring is wrong; do not go on.
- **Smoke test**:
  - a `[velocity] ...` line every 80 steps;
  - `infeasible` well below 0.3;
  - `attempts` > 0;
  - no NaN, no crash;
  - note the fps.
- **Main run**: watch whether `vel/level` rises, and watch:
  - `vel/success_rate`;
  - `static/success_rate`, which should not drop noticeably;
  - `vel/speed_ratio`, which should approach 1;
  - `vel/head_err_at_attempt`.

**DexToolBench evaluation of a fine-tuned policy** (static tasks; fixed goal trajectories turn velocity goals off
automatically):

```bash
python dextoolbench/eval_isaacgym.py ... --config-path runs/20_vel_ft_v1/config.yaml \
    --checkpoint-path runs/20_vel_ft_v1/nn/<ckpt>.pth \
    --extra-overrides task.env.velocity.curriculum.fixedLevel=<vel/level at the end of training>
```

`fixedLevel` sets the arm-speed multiplier. It must match the level at the end of training, otherwise the policy runs
at an arm speed it was not trained with.

To run all 24 object/task pairs in one job (10 episodes each, the same protocol as the official
`run_all_evals_isaacgym.py`) and write a summary to `evals/ft_<name>/summary.txt`:

```bash
sbatch --export=ALL,NAME=v2,CFG=runs/31_vel_ft_v2/config.yaml,CKPT=runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth,LEVEL=0.10 \
    finetune_velocity/slurm/5_eval_dextoolbench_mig.sbatch
python finetune_velocity/run_dextoolbench_eval.py --summarize evals/ft_v2 evals/ft_pretrained   # compare
```

---

## 6. Logged metrics (TensorBoard: `runs/<name>/summaries`)

| Metric | Meaning |
|---|---|
| `vel/level` and `vel/sigma_frac`, `eps_kp`, `cone_deg`, `speed_hi`, `arm_speed_mult` | Current curriculum level and the parameters it sets |
| `vel/success_rate` | Velocity successes / finished velocity goals (drives the curriculum) |
| `vel/attempt_rate` | Fraction of velocity goals with an attempt (for the rest, the episode ended during the lead-in or the strike) |
| `vel/q_v`, `vel/dv_rel`, `vel/speed_ratio`, `vel/cos_dir` | At the attempt: velocity match, relative error, speed ratio, direction cosine |
| `vel/head_err_at_attempt`, `vel/kp_dist_at_attempt` | At the attempt: head position error, keypoint distance |
| `vel/infeasible_frac` | Fraction of velocity goals that fell back to static goals because they were infeasible |
| `vel/down_frac_of_attempts`, `vel/down_success_given_attempt` | Share of downward strikes among attempts, and their success rate |
| `vel/fd_vs_analytic_rel` | At the attempt: \|finite-difference − analytic head velocity\| / \|v_g\| (cross-check, see section 8) |
| `static/success_rate` | Static-goal success rate (the main forgetting metric) |
| `kp_rew/*`, `bonus/*` | Keypoint reward and bonus split by goal type |
| `successes`, `success_ratio` | Official metrics (goals reached per episode) |

---

## 7. Known risks and open questions

1. **Is the arm fast enough at level 0?** Check `vel/speed_ratio` in the smoke test. (Answered: it is; see section 8.)
2. **Feasibility of downward strikes**: with the head pointing down, the feasibility check fails more often, so the
   realized share of downward strikes may be below 25% (`vel/down_frac_of_attempts`).
3. **Head axis choices**, especially for brush and eraser, are my judgment and can be changed in the yaml.
4. **Observation delay**: the official env delays the whole observation vector randomly (by up to a few steps). This
   makes v_head lag and may limit velocity accuracy. If needed, the delay could be turned off for the velocity terms
   only, as an ablation.
5. **The attempt and the head position do not match exactly** (see 3.5); watch `vel/head_err_at_attempt`.
6. **Velocity-task evaluation** (plate / nail / sweep) has not been ported to Isaac Gym yet; that is the next step.
   Until then, the `vel/*` training metrics are the main evidence.

---

## 8. Smoke test (v1) results and the adjustments that followed

**Zero-init check** (`10_vel_zero_init`: 768 envs, 60 epochs, actor frozen, velocity goals off, arm speed ×1):

| | Widened model | Released model (`00_ft_sanity_tol1`, same point) |
|---|---|---|
| successes | 0.875 | 0.914 |
| true_objective_mean | 8.10 | 7.48 |
| Static success rate | 0.887 | — |

The two agree within random variation, so the zero-weight widening is correct. Also, since the critic is loaded
strictly, the run finishing at all shows that the shapes match exactly.

**Smoke test** (`11_vel_smoke`: 768 envs, 150 epochs, the first 20 critic only): the whole pipeline ran end to end, with
no crash and no NaN.

| Metric | Observed | Reading |
|---|---|---|
| Static success rate | 0.86–0.95 | Close to the zero-init check; velocity goals did not break existing skills |
| attempt_rate | 0.84–1.0 | Almost every velocity goal reaches the strike pose |
| head_err_at_attempt | 4.5–5 cm | Head position error at the attempt is within the 6 cm threshold |
| Share of downward strikes | 0.28 | Close to the intended 25% |
| **infeasible_frac** | **0.42–0.69** | More than half of the velocity goals fell back to static goals; only about 17% of steps were spent on velocity goals |
| **speed_ratio** | **3.4–5.3** | At the attempt the head moves 4–5× the goal speed, about 0.4 m/s |
| q_v | 0.05–0.15 | Very weak reward signal (about 50–150 per attempt) |
| level | Stayed at 0 | Expected within 150 epochs |

**Adjustment 1: the lead-in and follow-through may leave the target volume by 10 cm** (`volumeMargin`)

- Why it was needed: the official delta sampler clamps goals onto the faces of the target volume, so many strike points
  sit right on its boundary. The lead-in (12 cm back) or the follow-through (8 cm forward) then leaves the volume. In
  the CPU test, strike points were drawn uniformly inside the volume and 63% were feasible; in actual training only
  30%–60% were.
- Why it is sound: the target volume is only where official goals are sampled, not a limit of the arm's reach, and
  10 cm beyond it is still within reach in front of the robot. The strike point itself is still kept inside the volume,
  and the check that the head stays above the table is unchanged.

**Adjustment 2: start speed range raised from 0.05–0.15 m/s to 0.15–0.35 m/s** (still ending at 0.3–0.6)

- Why it was needed: I had worried that the arm would be too slow, but the opposite was true. The released policy
  already moves the head at about 0.4 m/s at the attempt (`head_speed_on_vel_goals` averaged 0.29 m/s), 4–5× the
  original goal speeds. With σ = 1 that gives q_v = exp(−4) ≈ 0.02–0.1, which is almost no gradient.
- Effect: starting near the policy's natural speed puts |dv|/|v_g| at about 0.5–1.5 at the start of the curriculum,
  so q_v is 0.2–0.6, a much stronger signal.
- The arm-speed multiplier stays as it was (1.5 → 6). If `speed_ratio` falls well below 1 once the level rises in the
  main run, the multiplier will be raised.

**New metric `vel/fd_vs_analytic_rel`**: at the attempt, the difference between the finite-difference velocity and the
analytic estimate (v_obj + ω × r), divided by |v_g|. A large value means PhysX's root velocity is not the origin's
velocity (but the center of mass's), which is the reason for the finite difference in 3.5. It is a cross-check only
and plays no part in training.

**Speed**: the smoke test ran at about 9.3k fps total on a MIG slice.

**Second smoke test** (`12_vel_smoke2`: after the adjustments, otherwise the same settings):

| Metric | First | Second |
|---|---|---|
| infeasible_frac | 0.42–0.69 | **0.00–0.05** |
| speed_ratio | 3.4–5.3 | **1.4–2.0** |
| q_v | 0.05–0.15 | **0.34–0.46** |
| Velocity success rate (eps_v = 0.6) | 0–0.05 | **0.14–0.30** |
| Static success rate | 0.86–0.95 | 0.82–0.99 (block count 2315/2580 = 0.90) |
| fd_vs_analytic_rel | — | 0.34–0.60 |
| fps total (MIG) | 9.3k | 9.3k (zero-init check: 9.4k, so the new code does not slow training down) |

- Both adjustments worked as intended: almost all velocity goals are feasible, and the velocity reward now carries a
  clear signal.
- Curriculum block: velocity success 511/2060 = 0.25, below the 0.5 threshold, so the level stayed at 0. That is
  expected within 150 epochs; the policy has to learn it in the main run.
- `fd_vs_analytic_rel` is about 0.45: the two velocity estimates differ by about 0.45 × |v_g| (about 0.1 m/s). This is
  consistent with the root linear velocity being the center-of-mass velocity rather than the origin's: when the tool
  rotates, ω × r_com is about that size. Measuring at the head by finite difference is therefore necessary.

---

## 9. Main run v1 results (`20_vel_ft_v1`)

Setup: full A100, 12288 envs, 6000 epochs (about 1.18 billion frames), 4 h 28 min (about 73k fps on average, about
18 GB of 40 GB GPU memory). It ended normally (`MAX EPOCHS NUM!`). Final checkpoint:
`runs/20_vel_ft_v1/nn/last_20_vel_ft_v1_ep_6000_rew__8364.4_.pth`.

**Velocity success rate per curriculum block** (each block about 2300 finished velocity goals; 1995 blocks in total):

| Block | 1 | 101 | 301 | 501 | 801 | 1001 | 1301 | 1601 | 1901 | Last (1995) |
|---|---|---|---|---|---|---|---|---|---|---|
| Velocity success | 0.18 | 0.28 | 0.32 | 0.35 | 0.40 | 0.39 | 0.40 | 0.42 | 0.44 | **0.45** |
| Static success | 0.85 | 0.91 | 0.89 | 0.88 | 0.88 | 0.88 | 0.90 | 0.88 | 0.89 | **0.89** |

**Attempt metrics at the end of training** (last few print windows):

| Metric | Value | Reading |
|---|---|---|
| level | 0.00 (never rose) | Going up needs block success ≥ 0.5; it ended at 0.45 |
| attempt_rate | 0.90–0.92 | Most velocity goals reach the strike |
| q_v | 0.53–0.54 | 0.34–0.46 in the smoke test |
| \|dv\|/\|v_g\| | 0.69–0.73 | Relative velocity error |
| speed_ratio | 1.17–1.22 | Speed about 20% too high |
| cos_dir | 0.83–0.84 | About 33° off in direction; the main source of error |
| head_err_at_attempt | 5.0 cm | Unchanged throughout |
| infeasible_frac | 0.02–0.03 | |
| fd_vs_analytic_rel | 0.36–0.39 | |

**Conclusions**

1. **No forgetting**: static-goal success stayed at 0.88–0.90 throughout, the same as in the zero-init check (0.887).
   Keeping 50% static goals and a small learning rate works.
2. **Velocity control is being learned, but slowly**: block success rose from 0.18 to 0.45, but only by about 0.01 per
   100 blocks in the second half. It did not cross 0.5 in this run, so the curriculum stayed at level 0 (σ = 1,
   epsKp = 6 cm, cone 30°, speed 0.15–0.35 m/s, arm ×1.5).
3. **The error is mainly direction, not magnitude**: the speed is about 20% too high, but the direction is about 33° off,
   so most of |dv|/|v_g| ≈ 0.7 is perpendicular to d. My reading:
   - the lead-in only requires 3 cm, and the attempt fires 6 cm before the strike;
   - the released policy has learned to head straight for the goal center, so lateral offset turns into direction
     error;
   - the velocity reward is paid only once, at the attempt, which is too sparse to teach the policy to line up
     laterally in advance.
4. **Difference from Leo**: Leo's reward also settles velocity only at the attempt, but he first initialized with BC
   (DAgger) from an oracle; here there is only PPO.

**Candidate changes for v2** (not decided yet):
- a per-step velocity-tracking reward in the strike stage;
- lead-in tightened to 2 cm;
- two new diagnostics at the attempt: lateral offset and distance along d;
- 24576 envs;
- continue from v1's final checkpoint.

---

## 10. Changes in v2b (smoke-tested only, no main run)

> This section is the v2b design. The smoke test showed that removing the lead-in badly breaks the velocity direction, so the main run uses v2 instead (lead-in restored, the other changes kept); see section 11.

v2b continues from v1's final checkpoint. All changes below live in `velocity.yaml`. Setting them back to the v1 values
restores v1's behaviour: `useLeadIn: True`, `denseVelWeight: 0`, epsKp end 0.03, epsVFrac and σ end 0.25. Section 3
describes v1; where it differs from this section, this section applies.

### 10.1 Leo's feedback adopted: curriculum end values

| Parameter | v1 end | v2 end | Source |
|---|---|---|---|
| epsKp (attempt threshold) | 3 cm | **1 cm** | The value Leo actually used (his code defaults were not updated) |
| epsVFrac (velocity success threshold) | 0.25 | **0.15** | Same |
| σ (reward width) | 0.25 | **0.15** | Matches the epsVFrac end value: Leo's code comment says σ equals eps_v at level 1. Right at the success threshold the reward is exp(−1) ≈ 0.37 × 1000 |

- These only affect the later curriculum (at level 0, σ is still 1.0). v1 never left level 0, so they are unrelated to v1's results.
- Risk: at 0.6 m/s the head moves 1 cm per control step, about the size of the 1 cm threshold, so it may step over the
  attempt region. Once the level rises, watch whether `attempt_rate` drops.

### 10.2 No lead-in waypoint (`useLeadIn: False`, Leo's suggestion)

- **Leo's reasoning**: the lead-in was a patch for training from scratch. Fine-tuning a policy that can already reach
  may not need it.
- **Behaviour now**: a velocity goal starts directly as the strike pose, and the policy chooses its own run-up.
- **The feasibility check stays**: 12 cm behind and 8 cm past the strike must still be feasible, and the head must stay
  above the table. The run-up room is still there; the policy is just no longer required to reach that point first.
- **How this relates to v1's results**:
  - v1's main error was direction (about 33°). One possible cause I suggested was lateral offset from the lead-in's
    3 cm tolerance; the new diagnostics (10.4) test this directly.
  - On the other hand, without a lead-in the policy may have to loop around from in front of the goal, which could
    make the direction error larger. Answering this is one aim of v2b.
- The critic's 2-dim phase input is kept (it always reads strike), so the network is unchanged and v1's weights load
  directly.

### 10.3 New: per-step velocity tracking in the strike stage (`denseVelWeight: 3`, cap `denseVelBudgetFrac: 0.1`)

- **What**: each strike step before the attempt pays 3 × q_v(t), with the same formula and the same σ as the attempt
  reward. The total per goal is capped at 0.1 × 1000 = 100, i.e. 10% of the attempt bonus.
- **Why**: in v1 the velocity reward came only once, at the attempt. Block success then rose by only about 0.01 per
  100 blocks in the second half, and the direction error stayed large. A dense term gives feedback on direction and
  speed along the whole approach.
- **Why the cap**: without it, the policy could avoid the attempt and keep moving along d to farm per-step reward
  (each goal's timer allows 600 steps, and 600 × 3 = 1800 is more than the attempt bonus). With a cap of 100, the
  attempt bonus remains the main reward.
- This term is not part of Leo's design; it is new in v2b.
- v2b changes 10.2 and 10.3 at the same time. If the two effects need to be separated, a "lead-in removed only" run
  (v2a: `denseVelWeight=0`) can be added.

### 10.4 New diagnostics

| Metric | Meaning | Purpose |
|---|---|---|
| `vel/lateral_at_attempt` | Head offset from the goal head center perpendicular to d, at the attempt | Test whether lateral offset causes the direction error |
| `vel/along_at_attempt` | Offset along d at the attempt (negative: the goal not reached yet) | How early the attempt fires |
| `vel/v_par_rel` | Velocity component along d / \|v_g\| | How well the speed matches |
| `vel/v_perp_rel` | Velocity component perpendicular to d / \|v_g\| | Size of the direction error |
| `bonus/dense_per_strike_step` | Average dense reward per strike step | Actual size of the dense reward |

### 10.5 Training setup

| Item | v1 | v2b | Why |
|---|---|---|---|
| Initial weights | Released checkpoint (zero-widened) | v1 final checkpoint `last_20_vel_ft_v1_ep_6000_rew__8364.4_.pth` | v1 learned some velocity control and kept its static skills |
| Envs | 12288 (6 × 2048) | **24576 (6 × 4096, as in the official setup)** | v1 used only about 18 GB of GPU memory; each SAPG block is back to the official size |
| Epochs | 6000 | 3500 | Each epoch now has twice the data: 3500 epochs ≈ 1.4 billion frames, about 4 h at an estimated 100k fps |
| Everything else | lr 5e-5, critic warmup 20 epochs, successTolerance 0.01 | Same | |

**Running it**
```bash
git pull
python finetune_velocity/tests/test_cpu.py
# Smoke test on a MIG slice first (about 7 minutes):
sbatch --export=ALL,NAME=13_vel_smoke_v2b,CKPT=runs/20_vel_ft_v1/nn/last_20_vel_ft_v1_ep_6000_rew__8364.4_.pth \
    finetune_velocity/slurm/2_smoke_mig.sbatch
# Then the main run:
# (v2b never got a main run; the main run is 4_train_v2_a100.sbatch, section 11)
```

---

## 11. v2b smoke test results, and the final v2 setup (`31_vel_ft_v2`)

**v2b smoke test** (`13_vel_smoke_v2b`: MIG, 768 envs, 150 epochs, from the v1 final checkpoint, no lead-in, dense
reward on). The same v1 policy with and without the lead-in:

| Metric | End of v1 (with lead-in) | v2b smoke (no lead-in) |
|---|---|---|
| Velocity success | 0.45 | **0.03–0.08** |
| cos_dir | 0.83 (about 33° off) | **0.04–0.21 (about 80°–88° off)** |
| q_v | 0.53 | 0.25–0.29 |
| v_par_rel / v_perp_rel | — | 0.08–0.25 / **0.92–1.12** |
| lateral / along (at the attempt) | — | 3.3–3.8 cm / −0.2 to −0.4 cm |
| speed_ratio | 1.2 | 1.1–1.4 |
| Static success | 0.89 | 0.83–0.91 |
| dense/step | — | 0.69–0.78 |

**Reading**

- **What happens**: the tool slides into the goal region from the side.
  - The velocity is almost entirely perpendicular to d (`v_perp ≈ 1`, `v_par ≈ 0.1`).
  - The attempt fires just as the tool is level with the goal along d (`along ≈ 0`), with about 3.7 cm of lateral
    offset.
- **Why**: without a lead-in, the policy goes straight from the previous goal to the strike point, and that path's
  direction has nothing to do with d. The v1 policy's 0.45 relied on the lead-in bringing it behind the strike along d.
- **Learning a run-up alone is hard**: going behind first and then accelerating means moving away from the goal first,
  which is a hard exploration problem. There was no sign of it within 150 epochs.
- **Conclusion**: for **fine-tuning**, the lead-in is not just a patch for training from scratch; it is what sets the
  approach direction. Leo's judgment came from training from scratch, and this comparison can go back to him.

**Final v2 setup** (main run `31_vel_ft_v2`)

| Item | v1 | v2 |
|---|---|---|
| Lead-in | Yes (12 cm, reached within 3 cm) | **Yes** (same as v1) |
| Dense velocity reward | None | **3 × q_v per step, at most 100 per goal** (10.3) |
| epsKp end | 3 cm | **1 cm** (Leo) |
| epsVFrac end | 0.25 | **0.15** (Leo) |
| σ end | 0.25 | **0.15** (same as epsVFrac) |
| New diagnostics | None | **lateral / along / v_par / v_perp / dense** (10.4) |
| Initial weights | Released checkpoint | **v1 final checkpoint** |
| Envs | 12288 | **24576** |
| Epochs | 6000 | 3500 |

At the start of the curriculum (level 0), the only behavioural difference between v2 and v1 is the dense reward, so
v2's results can be attributed to it directly. The three end values only matter once the level rises.

**v2 smoke test** (`14_vel_smoke_v2`: MIG, 768 envs, 150 epochs, from the v1 final checkpoint), last 5 print windows:

| Metric | Value |
|---|---|
| Velocity success | 0.34–0.61 (each window covers only 16 steps, so it is noisy) |
| q_v | 0.51–0.59 |
| cos_dir | 0.81–0.90 |
| v_par_rel / v_perp_rel | 0.94–1.24 / 0.41–0.55 |
| lateral / along (at the attempt) | 1.4–2.0 cm / −4.0 to −4.7 cm |
| Static success | 0.84–0.93 |
| dense/step | 1.20–1.36 |

- **Back to v1's level**: with the lead-in restored, the metrics return to where v1 ended (cos about 0.83, success
  about 0.45), and the dense reward is being paid normally.
- **Where the direction error comes from** (first evidence from the new diagnostics):
  - The attempt fires about 4.4 cm before the goal along d, with about 1.7 cm of lateral offset.
  - If the policy is converging on the goal center at that point, its direction is off by atan(1.7/4.4) ≈ 21°. That
    explains most of v_perp/v_par ≈ 0.5 (about 27°).
  - The rest may come from velocity due to tool rotation (`fd_vs_analytic_rel` about 0.4).
- **Prediction**: as the curriculum tightens epsKp from 6 cm to 1 cm, the attempt moves closer to the goal, so the
  direction error from lateral convergence should shrink, while lateral alignment becomes more demanding.

**A direction to try later** (to meet Leo's goal of dropping the lead-in):
- No explicit intermediate goal: draw d first, then place the strike point 10–15 cm ahead of the previous goal along d.
  The tool then already starts behind the strike, and a straight approach has the right direction.
- The cost: the tool is still rotating during the approach, which adds rotational velocity at the head.

**Running it**
```bash
git pull
# Smoke test (MIG, about 7 minutes)
sbatch --export=ALL,NAME=14_vel_smoke_v2,CKPT=runs/20_vel_ft_v1/nn/last_20_vel_ft_v1_ep_6000_rew__8364.4_.pth \
    finetune_velocity/slurm/2_smoke_mig.sbatch
# If it looks right:
sbatch finetune_velocity/slurm/4_train_v2_a100.sbatch
```

---

## 12. Main run v2 results (`31_vel_ft_v2`)

Setup: full A100, 24576 envs, from the v1 final checkpoint, 3500 epochs (about 1.4 billion frames), 4 h 21 min, ended
normally.
- Average 97.6k fps.
- Peak GPU memory 26.4 GB of 40 GB; the linear extrapolation before training estimated about 30 GB.
- Final checkpoint: `runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth`.

**Curriculum**: 1736 blocks in total (each about 3000–3500 finished velocity goals), two level-ups, no level-down:

| When | Block success | Event |
|---|---|---|
| Block 1 | 1571/3048 = 0.515 | Up to level 0.05 (skill inherited from v1, plus the dense reward) |
| Blocks 51–1201 | 0.41–0.49, mostly 0.45–0.47 | Stays at 0.05 |
| Around block 1220 | 1763/3511 = 0.502 | Up to level 0.10 |
| Blocks 1251–1736 | 0.44–0.48 | Stays at 0.10 until the end |

**Metrics at the end of training.** At level 0.10 the parameters are σ ≈ 0.92, epsVFrac ≈ 0.555, epsKp ≈ 5.5 cm,
cone 45°, speed 0.165–0.375 m/s and arm ×1.95.

| Metric | End of v1 (level 0) | End of v2 (level 0.10) |
|---|---|---|
| Velocity success | 0.45 | 0.45–0.48 |
| q_v | 0.53 | 0.53–0.54 |
| \|dv\|/\|v_g\| | 0.70 | 0.64–0.66 |
| speed_ratio | 1.2 | 1.11–1.15 |
| cos_dir | 0.83 | 0.84 |
| v_par_rel / v_perp_rel | — | 0.95–0.99 / 0.43–0.44 |
| lateral / along (at the attempt) | — | 1.7 cm / −3.9 to −4.0 cm |
| head_err_at_attempt | 5.0 cm | 4.5 cm |
| Static success | 0.89 | **0.91** |
| dense/step | — | 1.28–1.29 |

**Conclusions**

1. **No forgetting; static skill slightly better**: static success 0.90–0.92, above v1's 0.89.
2. **Speed magnitude is essentially learned**: the component along d is v_par ≈ 0.97 |v_g|, and the speed ratio fell
   from 1.2 to about 1.13.
3. **The direction error barely improved and is the main bottleneck**:
   - cos went from 0.83 to only 0.84, and the perpendicular component v_perp is about 0.44 |v_g| (about 24°).
   - At level 0.10 the success threshold is |dv| < 0.555 |v_g|, and the perpendicular component alone uses up most of
     that margin. Success is therefore stuck at about 0.46.
   - The two level-ups came mostly from block-to-block noise: both blocks were only at 0.50–0.52.
4. **The source of the direction error is now essentially pinned down**:
   - At the attempt the head is about 1.7 cm off the strike line laterally and still about 3.9 cm short along d.
   - If the policy is converging on the goal point at that moment, its direction is off by
     atan(1.7 / 3.9) ≈ 23.6°. With speed_ratio 1.13 this gives v_perp ≈ 1.13 × sin(23.6°) ≈ 0.45 |v_g|, almost
     exactly the observed 0.43–0.44.
   - In other words, **the policy does not move along the strike line; in the last few centimeters it corrects its
     lateral offset toward the goal point, and that correction is the direction error.**
5. **The dense reward helped only a little**: it made the speed magnitude more accurate, but the per-step q_v is
   driven mostly by the size of the error and gives little guidance toward approaching along the line.
6. **More training alone will not fix this**: at this pace, completing all 20 curriculum levels is not realistic. The
   design needs to target lateral alignment instead (section 13, to be decided).

---

## 13. Diagnosis before v3: where does the lateral offset come from?

**Question**: in v2 the head is about 1.7 cm off the strike line at the attempt, which explains almost all of the
direction error (section 12). There are two candidate changes for v3:
- **A**: tighten the lead-in tolerance from 3 cm to 1.5 cm;
- **B**: add shaping for moving along the line during the strike stage.

A alone is enough only if **the lateral offset at the attempt is mostly inherited from the moment the lead-in is
reached**.
- Why this is plausible: the lead-in only requires 3 cm and no hold. The policy arrives from the previous goal,
  carrying sideways velocity unrelated to d, and then has only 12 cm of run-up.
- But the offset could also arise during the run-up, e.g. from tool rotation, or from overshooting after passing
  through the lead-in. Then A would not help much, and B is needed.

**New diagnostics**, measured at the step the lead-in is reached, relative to the strike line (through the head center
at the strike pose, along d):

| Metric | Meaning |
|---|---|
| `vel/lateral_at_leadin` | Head distance from the strike line |
| `vel/along_at_leadin` | Offset along d (ideally about −12 cm) |
| `vel/v_par_rel_at_leadin` / `vel/v_perp_rel_at_leadin` | Velocity along / perpendicular to d, divided by \|v_g\| |

**How to decide**: run a MIG smoke test from the v2 final policy, with unchanged training settings and the curriculum
starting at level 0.10.
- **Lateral offset at the attempt ≲ at the lead-in**: the offset is inherited, so v3 uses A.
- **Clearly larger at the attempt**: the offset arises during the run-up, so v3 uses B (or A+B).
- Also look at the perpendicular velocity at the lead-in: if it is large, the policy is cutting through the lead-in at
  an angle.

```bash
sbatch --export=ALL,NAME=15_vel_smoke_diag,CKPT=runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth,EXTRA=task.env.velocity.curriculum.initLevel=0.1 \
    finetune_velocity/slurm/2_smoke_mig.sbatch
```

**Diagnosis results** (`15_vel_smoke_diag`: MIG, 768 envs, 150 epochs, from the v2 final policy, level 0.15, last 5
print windows):

| | When the lead-in is reached | At the attempt |
|---|---|---|
| Lateral distance | 1.5–1.7 cm | 1.7–2.1 cm |
| Position along d | −10.9 to −11.1 cm | −3.1 to −3.7 cm |
| Velocity along d, v_par / \|v_g\| | **−0.26 to −0.37** | 0.90–1.09 |
| Perpendicular velocity, v_perp / \|v_g\| | **0.55–0.72** | 0.40–0.55 |

The other metrics are close to the end of v2: velocity success 0.24–0.51, cos 0.72–0.87, static success 0.87–0.95.

**Reading**

1. **Lateral position is not the problem**: the lateral offset is already only about 1.6 cm when the lead-in is
   reached, and about 1.9 cm at the attempt, almost unchanged. Tightening the lead-in tolerance from 3 cm to 1.5 cm
   (the original option A) would therefore do almost nothing for the lateral position.
2. **The problem is the velocity when the lead-in is reached**:
   - The velocity along d is **negative** (about −0.3 |v_g|): the tool arrives at the lead-in moving **backwards**,
     coming back from the strike side.
   - On top of that there is about 0.6 |v_g| of sideways velocity.
   - Why: the previous goal is near the strike point (delta sampling keeps it ≤ 10 cm away), while the lead-in is 12 cm
     behind the strike, so the policy has to move along −d to reach it. With no hold, the goal switches to the strike
     the moment the tool enters the 3 cm region.
3. **What this does to the run-up**: within the 12 cm run-up, the policy first has to cancel the backward and sideways
   velocity, then accelerate along d. The size of the lateral offset barely changes, but the sideways velocity persists
   the whole time: the head keeps crossing from one side of the line to the other, and at the attempt there is still
   about 0.46 |v_g| of perpendicular velocity.
4. **Conclusion**: the root cause of the direction error is the wrong motion state at the start of the run-up, not an
   inaccurate start position.
   - The original option A does not address it.
   - B (shaping for moving along the line) would only address it indirectly.

---

## 14. Next design: smooth trajectory goals with velocities (in progress, branch `trajectory-goals`)

**Why the design changes**: the diagnosis in section 13 showed that the direction error comes from a **dynamically
inconsistent goal sequence**: the policy entered the run-up moving backwards and sideways. Another fix would be to
make the lead-in a static goal (stop, then strike). But that only trains stop-then-strike single impacts and does
nothing for long-horizon tasks such as wiping, sweeping and chained motions, which runs against the project's aim of
general velocity-conditioned tool use. (Leo has been asked for his view.)

### 14.1 Decisions made

| Decision | What | Why |
|---|---|---|
| Goal form | A sequence of waypoints, each with a pose and a head velocity. Zero-velocity waypoints are official static goals (1.5 cm, hold 10 steps) | Static goals, single strikes, sweeps and wipes are all special cases of one framework. The previous waypoint is always behind the current one, with its velocity pointing at it, so there is no more backing into the run-up |
| Goal sequence | Pick up the tool, then a series of blocks. **40%** are static blocks (an official delta goal). A motion block is: set pose (static) → moving waypoints → end stop (static) | Keeps the official "reach and hold" structure, to prevent forgetting |
| Advancing | **Event-driven**: a waypoint counts as passed when it is reached (keypoint distance ≤ eps_kp) **or crossed** (its normal plane). Straying more than about 15 cm abandons the block and jumps to its end stop | Consistent with SimToolReal. With the "crossed" condition, a near miss does not force the policy to turn back |
| Lookahead | The **next** waypoint is added to the observation (relative keypoints + velocity, 15 dims, zero-initialized) | Waypoints switch every few steps and observations are delayed by up to 3 steps, so the policy needs to see the next waypoint in advance |
| Reward | Static waypoints: the official reward. Moving waypoints: B × q_pos × q_v when passed, about 1000 in total per block. **The dense velocity reward from v2 is dropped** | Per-waypoint rewards are already dense; the total does not depend on block length |
| Training | **Fine-tune** SimToolReal, not from scratch | Grasping, holding and robustness are already learned; compute and time are limited; the comparison is cleaner |
| Initial model | **Pending** the DexToolBench evaluation of v2 | v2 if it shows no forgetting, otherwise the released model |

### 14.2 Trajectory sampler (`isaacgymenvs/tasks/simtoolreal/trajectory_goals.py`, pure torch)

- **The path is defined at the head center**, built from 1..n segments joined with continuous tangents. Two kinds of
  segment:
  - **bend segments**: cubic Bezier curves, with the orientation slerped toward a random rotation;
  - **arc segments**: rigid rotation about a wrist-like pivot. The head sweeps a circular arc, so the motion is
    rotation-dominated (hammer swings, spatula flips).
- **Speed profile**: a smooth quintic ramp up from rest, a plateau, and a smooth ramp back down to rest.
  - The ramp fractions are random; short ramp-downs look like strikes.
  - The peak speed is capped by the acceleration limit (3 m/s²).
  - Each waypoint's goal velocity is the path tangent times the speed there, so the sequence is dynamically
    consistent by construction.
- Waypoints are about 3 cm apart (one every ~4 control steps at 0.5 m/s). The feasibility check is the same as in
  v1/v2, applied along the whole path.
- Curriculum (level 0 → 1):

  | Parameter | Level 0 | Level 1 |
  |---|---|---|
  | Peak speed limit | 0.35 m/s | 0.6 m/s |
  | Bend | 0° | 90° |
  | Rotation per segment | 0° | 150° |
  | Arc share (max swept angle) | 0 (0°) | 40% (180°) |
  | Segments per block | 1 | 1–6 |
  | Start-direction cone | 30° | 180° |

  25% of blocks are downward strikes.
- CPU tests: `finetune_velocity/tests/test_trajectories.py`. They check:
  - speed limits, rest at the end, the acceleration limit;
  - waypoint spacing, and velocity along the tangent;
  - pose consistency and feasibility;
  - that level 0 is straight and does not rotate.

### 14.3 Coverage check against DexToolBench (`finetune_velocity/tools/compare_dextoolbench.py`)

DexToolBench trajectories have poses but no timestamps, so only the geometry can be compared. **Principle:
DexToolBench is a floor for coverage, not the training target.** The sampler's ranges should contain it with room to
spare, so that the policy does not work only on these 24 tasks.

| Metric | DexToolBench (24 tasks) | Sampler, level 1 |
|---|---|---|
| Orientation change per cm of head travel, median / 90th pct | 3.4 / 15.9 °/cm | 3.8 / 7.0 °/cm |
| Turning angle per 3 cm, median / 90th pct | 17° / 129° | 12° / 34° |
| Head path / handle path, median / 90th pct | 1.03 / 1.52 | 1.08 / 1.76 |
| Length of one continuous motion, median / max | 0.79 / 1.41 m | 0.29 / 1.49 m |

Figures: `finetune_velocity/figures/geometry_vs_dextoolbench.png`, and sample blocks in
`finetune_velocity/figures/trajectory_chunks.png` (`finetune_velocity/tools/plot_trajectories.py`).

**Known gaps and limitations**

1. **Turns sharper than 90° (reversals)**: physically these require a stop, so they are designed as
   "stop → next block" rather than within one block. This is also why a single block is shorter than a whole
   DexToolBench task.
2. **Rotation in place** (screwdriver spin; it accounts for most of the 15.9 °/cm at the 90th percentile): the head's
   linear velocity is about zero, which the "head linear velocity" interface cannot express. This needs an
   angular-velocity goal later.
3. **Speed magnitude** has no real data to calibrate against; it uses Leo's 0.3–0.6 m/s. At level 1 the acceleration
   limit and path length bring the mean peak to about 0.37 m/s, with the 90th percentile at about 0.5 m/s.
4. The tool never touches anything during training; whether velocity conditioning helps apply force can only be shown
   by evaluating contact tasks.
5. Calibrating the geometry on DexToolBench and also evaluating static tasks on it is a mild leak. Conclusions about
   generalization should rest on tasks that were not used for calibration (e.g. Leo's plate / nail / sweep).

### 14.4 Next

1. Integrate into the env:
   - a waypoint queue;
   - event-driven advancing, including the "crossed" check and the abandon rule;
   - the reward, the lookahead observation and phase information for the critic;
   - diagnostics and the curriculum.
2. Make the widening and launch scripts support any old observation size (the start may be v2 at 149 dims or the
   released model at 140).
3. Decide the initial model, then MIG smoke test → A100 main run.
4. Later: a separate test trajectory set (four typical motion types, plus stress tests beyond the training ranges).

---

## 15. Environment implementation of trajectory goals (`SimToolRealTrajectory`)

| File | Content |
|---|---|
| `isaacgymenvs/tasks/simtoolreal/trajectory_env.py` | New task. It subclasses `SimToolRealVelocity` (keeping head velocity, velocity obs, the arm-speed multiplier, the curriculum class and the obs hooks) and replaces the goal logic |
| `isaacgymenvs/tasks/simtoolreal/trajectory_goals.py` | The sampler (section 14), plus `waypoint_pass`, a pure function that decides whether a waypoint is passed (CPU-tested) |
| `finetune_velocity/trajectory.yaml` | All parameters and curriculum ranges |
| `finetune_velocity/launch_from_pretrained.py` | New `--task trajectory`. Checkpoint widening is now generic: it adds whatever inputs the checkpoint lacks for the task (released 140/162 → 164/189, or v2's 149/173 → 164/189) |
| `finetune_velocity/slurm/6_train_traj_a100.sbatch` | Main run. The smoke test is `2_smoke_mig.sbatch` with `TASK=trajectory` |

**Goal phases** (`phase`): PLAIN (static block), SET (set pose, static), MOVE (moving waypoint), END (end stop,
static).
- PLAIN, SET and END use the official hold rule. Only these count toward `successes`, so the official "at most 50
  goals per episode" keeps its meaning.
- A MOVE waypoint that is passed (reached or crossed) pays B × q_pos × q_v, with B = 1000 / the block's number of
  moving waypoints.
- Straying more than 15 cm abandons the block and jumps to END.
- The per-goal timer (600 steps) restarts whenever the goal advances.

**New observations**
- `goal_lookahead` (15 dims, policy and critic): the next waypoint's keypoints relative to the tool's current
  keypoints (12), plus the next waypoint's velocity divided by 0.5 m/s (3). In SET, the next waypoint is the first
  moving one; on a static goal it is the current goal itself.
- `traj_progress` (1 dim, critic only): the fraction of the block's moving waypoints still ahead.
- `velocity_phase` keeps v2's 2 dims, now meaning (0, on a moving waypoint).

**Curriculum**: decided on moving-waypoint success: position within eps_kp and velocity error
< eps_v × max(|v_g|, 0.1), at least 4096 samples per block. The static guard is unchanged.

**Logs**: the `[trajectory] step ...` line, and `traj/*` in TensorBoard:
- waypoint success rate, share reached, abandon rate;
- q_pos, q_v, v_par, v_perp, lateral offset;
- head speed vs goal speed;
- blocks completed and failed, infeasible share, share of steps spent moving.

**Running it**
```bash
# smoke test (MIG): starting from v2 or from the released model both work
sbatch --export=ALL,TASK=trajectory,NAME=16_traj_smoke,CKPT=runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth \
    finetune_velocity/slurm/2_smoke_mig.sbatch
# main run (A100); CKPT depends on the initial-model decision
sbatch --export=ALL,NAME=40_traj_v1,CKPT=<checkpoint> finetune_velocity/slurm/6_train_traj_a100.sbatch
```
