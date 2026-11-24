# 在官方 SimToolReal checkpoint 上加入速度目标做微调：第一次实验（v1）

本文档说明 `velocity-finetune` 分支里每一处改动是什么、为什么这样做，以及怎么在 Zaratan 上运行。
英文完整版见 `README.md`（给 Leo），两份内容一致。

研究问题：**在 pose 之外再给策略一个目标速度（pose + orientation + velocity goal），能否帮助需要力的工具任务？**
这一步只做训练端：在 Isaac Gym 里把官方模型微调成能"按指定速度经过指定位姿"的策略。

---

## 1. 总体思路

1. 不从头训练，而是从官方发布的 checkpoint（DexToolBench 基线总成功率 93.3%）继续训练，只改速度目标必需的部分。
   这样策略原有的抓取、搬运和姿态跟踪能力可以直接复用。
2. 一半目标保持官方的静态目标（官方规则完全不变），另一半换成速度目标。静态目标的作用是防止遗忘：
   微调后的策略仍应能完成 DexToolBench。
3. 速度相关的设计尽量照搬 Leo 在 MuJoCo 里已经验证过的设置（pose-velocity：plate 76% / nail 96% / sweep 58%）。
   只有在 SimToolReal 的结构要求不同的地方才改动，这些地方都在下文写明了原因。
4. 新增的观测通过"零权重扩展"接入网络：微调开始时，策略和 critic 的输出与官方模型逐位相同，然后才慢慢学会使用速度信息。

---

## 2. 文件清单

| 文件 | 内容 |
|---|---|
| `isaacgymenvs/tasks/simtoolreal/velocity_env.py` | 新任务 `SimToolRealVelocity`（继承官方 `SimToolReal`），包含目标、奖励、观测、臂速、课程和日志 |
| `isaacgymenvs/tasks/simtoolreal/velocity_goals.py` | 纯 torch 函数，不依赖 Isaac Gym，可以在 CPU 上测试：锥内采样、速度目标采样、速度匹配、课程 |
| `isaacgymenvs/tasks/simtoolreal/env.py` | 只加了 2 个空钩子（`_extra_obs_type_sizes`、`_add_extra_obs`），官方行为不变 |
| `isaacgymenvs/tasks/__init__.py` | 注册 `SimToolRealVelocity` |
| `finetune_velocity/velocity.yaml` | 速度目标的全部参数（命令行可覆盖） |
| `finetune_velocity/expand_checkpoint.py` | 零权重扩展 checkpoint，并校验原有权重未变、新增列全为 0 |
| `finetune_velocity/launch_from_pretrained.py` | 启动脚本：`--task static`（原 sanity）或 `--task velocity`；自动扩展权重、critic 预热、严格加载 critic |
| `finetune_velocity/slurm/*.sbatch` | 零初始化检查（MIG）、冒烟测试（MIG）、v1 正式训练（A100）、v2b 正式训练（A100，第 10 节） |
| `finetune_velocity/tests/test_cpu.py` | CPU 单元测试，包括"扩展后输出完全不变"的数值检查 |
| `dextoolbench/eval_isaacgym.py` | 关闭相机传感器（计算节点没有图形设备）；新增 `--extra-overrides` 参数 |

---

## 3. 改动与理由

### 3.1 目标类型：50% 静态 + 50% 速度

每次换目标（每个 episode 的第一个目标，即抓起工具那一步，除外）时，以 `velocityGoalProb = 0.5` 的概率生成速度目标，否则生成官方静态目标。

- **静态目标（v_g = 0）**：完全沿用官方规则。keypoint 距离 ≤ `successTolerance × keypointScale` = 0.01 × 1.5 = 1.5 cm，累计 10 步算成功，每个"在目标附近"的步给 1000/10 = 100 奖励。
  - 理由：这就是官方 checkpoint 的训练目标。保持不变，就不会破坏它已学会的能力，DexToolBench 的结果也能和基线直接比较。
- **速度目标**：分两段，先 **lead-in**，再 **strike**（见 3.2）。
  - 理由（为什么是 50%）：速度目标需要足够多的样本才能学会；静态目标的比例又要足够高才能防止遗忘。先从 1:1 开始，如果静态成功率明显下降再调整。
- **第一个目标保持静态**（`velocityOnFirstGoal: False`）：第一个目标包含从桌上抓起工具。这一步本身就难，不宜再叠加速度要求。

### 3.2 速度目标的几何（strike + lead-in）

1. **strike 位姿**：先用官方的 delta 采样得到一个普通目标（相对上一个目标平移 ≤ 10 cm、旋转 ≤ 90°，并夹在官方 target volume 内）。
   - 理由：和官方目标的分布一致，策略熟悉这种位姿变化。
2. **方向 d**：在工具头 face 轴的锥内均匀采样，锥角由课程控制，从 30° 放宽到 180°（Leo 的 `vel_cone_deg`）。
   - 30° 相当于"用工具头的打击面去打"。180° 时方向任意，相当于扫、刮。
3. **25% 的陡峭下击**（d_z ≤ −0.85，Leo 的 `down_frac` / `down_max_z`）：先采样向下的方向，再把工具转一个最小角度，使头部轴落在 d 的锥内。
   - 理由：在随机姿态下，头部轴几乎不会朝下；不强制加入的话，钉钉子这类下击动作在训练中极少出现。Leo 的实验里也是这样处理的。
4. **lead-in 点**：位于 strike 位姿后方 12 cm（沿 −d），姿态与 strike 相同。keypoint 距离 ≤ 3 cm 即算到达，不需要停留，也没有 bonus（keypoint 进步奖励照常给）。到达后目标切换为 strike。
   - 理由：策略需要一段助跑才能在到达 strike 位姿时达到目标速度。没有 lead-in 时，最近的路线常常是从侧面绕过去，速度方向不对。之所以不要求停留，是为了让策略可以带着速度直接冲过 lead-in。
5. **可行性检查**：strike 点的工具原点必须在官方 target volume 内；lead-in 和 follow-through（strike 前方 8 cm）两个点允许超出 target volume 最多 `volumeMargin = 10 cm`。三个点的工具头中心都必须高于 `minHeadZ = 0.58 m`（桌面约 0.53 m）。10 cm 的余量是冒烟测试后加的，见第 8 节。每个目标试 8 个候选，取第一个可行的；8 个都不可行时退回静态目标。
   - 理由：保证击打前有助跑空间、击打后有穿出空间，工具头也不会被要求穿过桌面。日志里的 `vel/infeasible_frac` 记录退回比例。
6. **速度大小**：服从 U(speedLo, speedHi)，由课程从 0.15–0.35 m/s 逐步增加到 Leo 的 0.3–0.6 m/s。起点原本是 0.05–0.15 m/s，冒烟测试后提高了，见第 8 节。

**各工具类型的头部"打击面"轴**（工具坐标系：原点在手柄中心，+x 从手柄指向头部），见 `velocity_env.DEFAULT_HEAD_AXIS`，可在 yaml 的 `headAxis` 中修改：

| 类型 | 轴 | 双面 | 理由 |
|---|---|---|---|
| hammer | +y | 是 | 锤头在 y 方向较长，两端各有一个打击面 |
| screwdriver / marker | +x | 否 | 尖端沿手柄方向 |
| spatula | +x | 否 | 铲面沿 +x 推出 |
| eraser / brush | +z | 是 | 大平面的法向（橡皮擦和刷子用的是平面） |

这一项是我根据几何形状做的判断，不是官方定义。由于锥角最终会放宽到 180°，它只影响课程早期的方向分布。

### 3.3 尝试（attempt）判定：keypoint 距离 ≤ eps_kp（6 cm → 3 cm）

在 strike 阶段，第一个满足 keypoint 距离 ≤ `epsKp` 的步就是"尝试"。这一步只结算一次速度奖励，然后换下一个目标。

- 阈值直接与官方 fixed-size keypoint 距离（`keypoints_max_dist_fixed_size`）比较，**不再乘 keypointScale 1.5**。
  - 理由：官方用 `success_tolerance × 1.5` 是因为它的 tolerance 定义在缩放之前。Leo 的 `eps_kp` 本身就是直接比较的距离（6→3 cm），如果再乘 1.5 会变成 9→4.5 cm，与他的设置不一致。
- 为什么起点是 6 cm 而不是官方课程起点的 11 cm：Leo 发现阈值太大时，attempt 会在工具头还在 lead-in 附近对齐时就提前触发（oracle 在 0.11 时只有 42% 成功，0.06 时 94%）。
- 静态目标仍用 1.5 cm、10 步停留的官方规则，不受影响。

### 3.4 速度奖励：`1000 × exp(−|v_head − v_g| / (σ |v_g|))`，只在 attempt 时给一次

- 公式与 Leo 代码（`rewards.velocity_match`）一致，误差是**线性**的。**中期报告里写成平方是笔误，需要更正。**
- σ（`sigmaFrac`）由课程从 1.0 收紧到 0.25（Leo 的最终值）。
  - 理由：开始时策略几乎不会主动控制速度。σ 小的话，奖励接近 0，没有梯度信号。σ = 1 时，速度误差等于目标速度也还能拿到 37%。
- 除以 |v_g| 是相对误差，这样快目标和慢目标难度相近。
- **速度成功**（用于课程和日志，不影响奖励）：attempt 且 |dv| < epsVFrac × |v_g|，epsVFrac 从 0.6 收紧到 0.25（Leo 的 `eps_v_frac`）。

### 3.5 速度在工具头中心测量；keypoint 保持官方的手柄中心

- **keypoint 不改**：官方的 4 个 keypoint 以手柄中心为中心，并且它们本身就是策略的输入（`keypoints_rel_goal`、`keypoints_rel_palm`）。改成 Leo 的以头部为中心，会改变已有输入的含义，等于抛弃预训练权重。
- **速度在头部中心测**：击打效果取决于头部的速度，而不是手柄中心的速度。工具旋转时两者差别很大（v_head = v_obj + ω × r_head）。
  - 头部偏移 r_head = (handle_len/2 + head_len_x/2, 0, 0)，在生成工具时从 URDF 中读回（官方生成器不返回头部尺寸）。没有头部的 eraser 取手柄中心。
  - **测量方式**：用头部位置在一个控制步（1/60 s）内的有限差分计算速度，而不是根状态速度。理由是 PhysX 的根线速度可能是质心速度而不是原点速度（工具质心偏向头部），有限差分则不依赖这个约定。刚 reset 的那一步没有上一时刻位置，改用解析估计 v_obj + ω × R r_head。
- **副作用**：attempt 由手柄中心的 keypoint 判定，速度却在头部测量。keypoint 距离 ≤ 3 cm 时，姿态误差约 ≤ 16°，头部位置误差约在几厘米内。日志 `vel/head_err_at_attempt` 记录每次 attempt 时头部的实际位置误差；如果偏大，后续可以给 attempt 加一个头部误差条件。

### 3.6 臂速：为什么要提速，为什么不用官方的 tyler curriculum

- 官方每步关节目标的变化为 `armMovingAverage × dofSpeedScale × dt × a` = 0.1 × 1.5 / 60 = 0.0025 rad，即动作满幅时只有约 **0.15 rad/s**，远达不到 0.3–0.6 m/s 的击打速度。（代码里 `hand_dof_speed_scale` 这个名字虽然带 hand，实际作用于手臂的 7 个关节。）
- 做法：课程把这个速度乘以 `armSpeedMult`，从 1.5 倍增加到 6 倍（约 0.9 rad/s）。静态目标也使用同样的臂速，因为策略只有一个。
- 为什么不直接用官方的 tyler curriculum（`dofSpeedScaleFinal` 等）：
  1. 它按"每 episode 50 个目标的平均成功比例 > 0.6"升级，与速度表现无关；
  2. 它按墙钟时间（分钟）限制升级频率，MIG 和 A100 的速度不同，结果就会不同；
  3. 它同时控制若干"逐渐关闭观测"的开关，混在一起不好分析。
  所以改用下面这个只看速度表现的课程（官方 tyler 课程保持原样）。
- 课程开始时的臂速需要在冒烟测试中确认：看 `vel/speed_ratio`（|v_head|/|v_g|）。如果 level 0 时它远小于 1，说明手臂太慢，应提高 `armSpeedMult` 的起点。

### 3.7 课程：一个 level 同时控制所有速度参数

level ∈ [0, 1] 同时线性插值 σ、epsVFrac、epsKp、锥角、速度范围、臂速倍数（与 Leo 的 `CurriculumCfg` 相同）。

- 按"块"决策：每累计 `minSamples = 2048` 个结束的速度目标，看这一块的速度成功率。≥ 0.5 升一级（+0.05），< 0.2 降一级（阈值来自 Leo），其余不变；然后清空计数，开始下一块。
- **静态保护**：只有静态目标的成功率 ≥ 0.5 时才允许升级，避免为了速度牺牲原有能力。
- 两次变化之间至少间隔 160 个控制步（约 10 个 PPO epoch），给策略适应时间。
- level 保存在 checkpoint 的 env state 中，`--load-mode resume` 时会恢复；评估时用 `curriculum.fixedLevel` 固定。

### 3.8 新观测：v_g、v_head、v_head − v_g（9 维，除以 0.5 m/s）

- 追加在原 140 维之后，策略和 critic 都能看到（critic 的 162 维状态之后同样追加）。
- **v_head − v_g** 本可以由前两项算出，但直接给出误差能让网络更快学会修正。
- **lead-in 阶段也给出 strike 的 v_g**：让策略提前加速，这正是 lead-in 的目的。
- **手动缩放 0.5 m/s**：官方的输入归一化器（RunningMeanStd）已累计了海量样本（count 非常大），新维度的均值和方差几乎不会再被更新，会一直停留在初始值（0, 1）。因此需要手动把数值缩放到 1 左右（0.6 m/s → 1.2）。
- **不另加噪声**（"干净值"），但整个观测向量仍然经过官方的随机延迟队列（观测延迟这项 DR 保留）。
- **偏离原计划的一处**：critic 额外得到 2 维 `velocity_phase`（lead-in / strike 的 one-hot），策略没有。
  - 理由：lead-in 和 strike 的价值相差一个 0–1000 的 bonus，critic 如果分不清当前处于哪个阶段，价值估计会更噪。策略不需要这个信息：两个阶段的动作都是"朝目标按 v_g 运动"。这只影响 critic 的输入，同样是零初始化。

### 3.9 零权重扩展 checkpoint

- 网络输入的结构：actor 的 LSTM 读取 [obs 140 | SAPG 参数 32]，critic 的 MLP 读取 [state 162 | SAPG 参数 32]。rl_games 会在 env 观测后附加 block id，网络再把它替换为该 block 学到的 32 维向量。
- 因此新维度在第一层权重中的位置是第 140（critic 为 162）列，在 32 列 SAPG 参数**之前**。脚本在这里插入全零列，并把归一化器的新维度设为均值 0、方差 1。
- 结果：无论新输入是什么，第一层的预激活都不变，所以输出与官方模型完全相同（`tests/test_cpu.py` 用模拟的 LSTM 和 MLP 做了数值验证，`1_zero_init_check` 在真实环境中验证）。
- rl_games 的 weights 模式在 critic 形状不匹配时会静默跳过加载，这会掩盖扩展错误。启动脚本改为严格加载：不匹配就直接报错。
- 只保留权重（丢弃优化器状态等），只能用 `--load-mode weights` 加载。

### 3.10 critic 预热：前 20 个 epoch 只训练 critic

- 新任务的回报分布不同（多了一次性的速度 bonus），旧 critic 的价值估计会偏。如果一开始就更新策略，偏差大的 advantage 可能先把预训练策略带偏。
- 实现：前 20 个 epoch 把 actor 的学习率设为 0（critic 有自己的优化器，正常训练）。
  - 注意：adaptive 学习率调度在 KL = 0 时会不断调高学习率，所以每个预热 epoch 之后都把它重置为基础学习率，避免预热结束时出现一个超大的学习率。

### 3.11 其余保持官方设置的部分

- 域随机化、动作和观测延迟、物体状态噪声全部保持官方设置。注意官方 DR 包括对物体的随机速度冲量，偶尔会干扰某次 attempt，这是有意保留的。
- `successTolerance` 固定为 0.01。原因：官方配置从 0.075 开始课程，checkpoint 训练到的 0.01 保存在 env state 里，weights 模式不会恢复。sanity 实验已经验证过这一点。
- 每个目标的计时器（600 步）在到达 lead-in 时也会重置，所以 strike 阶段有完整的时间。
- `successes` 计数 = 静态成功数 + strike attempt 数（即"到达过的目标数"），沿用官方"每 episode 最多 50 个目标"的语义。速度质量另行统计。

### 3.12 训练超参数

| 项 | 值 | 理由 |
|---|---|---|
| 环境数 | 12288（6 个 SAPG block × 2048） | block 数由 checkpoint 固定为 6；6144 个环境实测约 12 GB 显存，12288 个在 40 GB 的 A100 上留有余量 |
| actor 学习率 | 5e-5（官方 1e-4） | 微调时步子小一点，减少遗忘 |
| critic 预热 | 20 epoch | 见 3.10 |
| 加载模式 | weights | 新优化器，epoch 从 0 开始 |

---

## 4. 与 Leo 的 pose-velocity 实现的对照

| 项 | Leo（MuJoCo） | 本实现（Isaac Gym 微调） | 原因 |
|---|---|---|---|
| keypoint 中心 | 工具头 | 手柄中心（官方） | keypoint 是预训练策略的输入 |
| 速度测量点 | 工具头 | 工具头（有限差分） | 一致 |
| 奖励 | 1000·exp(−\|dv\|/(σ\|v_g\|))，σ = 0.25 | 同一公式，σ 由 1.0 → 0.25 | 从预训练开始，先给宽松信号 |
| attempt 阈值 | eps_kp 6 → 3 cm | 同 | — |
| attempt 之后 | episode 结束 | 换下一个目标（episode 继续） | 官方是多目标 episode |
| 静态目标 | 另一个基线 | 50% 混合在同一个训练里 | 防止遗忘 |
| lead-in | 可达性检查点 | 显式的中间目标（12 cm，不停留） | 官方策略只会"到达位姿"，需要一个先到的点 |
| 训练方式 | BC（DAgger）+ PPO | 只用 PPO 从官方权重开始 | 已经有一个会操作的策略 |
| 动作 | head twist / 关节 | 官方关节目标 + 臂速倍数 | 不改动作空间 |

---

## 5. 在 Zaratan 上运行

```bash
cd ~/scratch.cmsc848x/simtoolreal
# 0. 还原之前手动改过的文件，把未跟踪的旧启动脚本移开，然后切换到新分支
git checkout -- dextoolbench/eval_isaacgym.py
mv finetune_velocity/launch_from_pretrained.py ~/launch_from_pretrained.old.py
git fetch origin && git checkout velocity-finetune

# 1. CPU 测试（登录节点即可）
source .venv/bin/activate
python finetune_velocity/tests/test_cpu.py

# 2. 零初始化检查（MIG，约 30 分钟）
sbatch finetune_velocity/slurm/1_zero_init_check_mig.sbatch
# 3. 冒烟测试（MIG）
sbatch finetune_velocity/slurm/2_smoke_mig.sbatch
# 4. 1、2 都正常后再提交正式训练（整块 A100，需要排队）
sbatch finetune_velocity/slurm/3_train_a100.sbatch
```

第一次运行时会自动生成 `pretrained_policy/model_velocity_obs149.pth`（扩展后的权重），以后直接复用。

**判断标准**

- 零初始化检查：日志中 `success_ratio`、`successes` 应与 `00_ft_sanity_tol1` 相近（同为 768 个环境）。如果明显更差，说明扩展或观测接入有问题，不要继续。
- 冒烟测试：每 80 步出现一行 `[velocity] ...`；`infeasible` 大约 < 0.3；`attempts` > 0；没有 NaN，没有崩溃；记下 fps。
- 正式训练：重点看 `vel/level` 是否上升、`vel/success_rate`、`static/success_rate`（不应明显下降）、`vel/speed_ratio`（应接近 1）、`vel/head_err_at_attempt`。

**用 DexToolBench 评估微调后的策略**（静态任务；固定的目标轨迹会自动关闭速度目标）：

```bash
python dextoolbench/eval_isaacgym.py ... --config-path runs/20_vel_ft_v1/config.yaml \
    --checkpoint-path runs/20_vel_ft_v1/nn/<ckpt>.pth \
    --extra-overrides task.env.velocity.curriculum.fixedLevel=<训练结束时的 vel/level>
```

`fixedLevel` 决定臂速倍数，必须与训练结束时的 level 一致，否则策略会在另一个臂速下运行。

---

## 6. 日志指标（TensorBoard：`runs/<name>/summaries`）

| 指标 | 含义 |
|---|---|
| `vel/level` 及 `vel/sigma_frac`、`eps_kp`、`cone_deg`、`speed_hi`、`arm_speed_mult` | 当前课程等级及对应参数 |
| `vel/success_rate` | 速度成功数 / 结束的速度目标数（课程依据） |
| `vel/attempt_rate` | 有 attempt 的速度目标比例（另一部分在 lead-in 或 strike 途中 episode 就结束了） |
| `vel/q_v`、`vel/dv_rel`、`vel/speed_ratio`、`vel/cos_dir` | attempt 时的速度匹配度、相对误差、速度大小比、方向余弦 |
| `vel/head_err_at_attempt`、`vel/kp_dist_at_attempt` | attempt 时工具头位置误差、keypoint 距离 |
| `vel/infeasible_frac` | 因不可行而退回静态目标的比例 |
| `vel/down_frac_of_attempts`、`vel/down_success_given_attempt` | 下击在 attempt 中的占比及其成功率 |
| `static/success_rate` | 静态目标成功率（防遗忘的主要指标） |
| `kp_rew/*`、`bonus/*` | 按目标类型拆分的 keypoint 奖励和 bonus |
| `successes`、`success_ratio` | 官方指标（每个 episode 到达的目标数） |

---

## 7. 已知风险和待验证项

1. **level 0 时手臂是否足够快**：看冒烟测试中的 `vel/speed_ratio`。
2. **下击的可行比例**：工具头朝下时，可行性检查更容易失败，实际下击占比可能低于 25%（`vel/down_frac_of_attempts`）。
3. **头部轴的选择**（尤其是 brush、eraser）只是我的判断，可以在 yaml 里修改。
4. **观测延迟**：官方对整个观测向量做随机延迟（最多几步）。这会让 v_head 滞后，可能限制速度精度。如有需要，可以对速度项单独关闭延迟，作为消融实验。
5. **attempt 与头部位置不完全一致**（见 3.5）：通过 `vel/head_err_at_attempt` 观察。
6. **速度任务评估**（plate / nail / sweep）尚未移植到 Isaac Gym，是下一步的工作。在那之前，训练日志中的 `vel/*` 是主要证据。

---

## 8. 冒烟测试（v1）的结果和随后的调整

**零初始化检查**（`10_vel_zero_init`，768 个环境，60 epoch，actor 冻结，关闭速度目标，臂速 ×1）：

| | 扩展后的模型 | 官方模型（`00_ft_sanity_tol1`，同一时间点） |
|---|---|---|
| successes | 0.875 | 0.914 |
| true_objective_mean | 8.10 | 7.48 |
| 静态目标成功率 | 0.887 | — |

两者在随机波动范围内一致，说明零权重扩展是正确的。另外，critic 是严格加载的，能跑完本身就说明形状完全匹配。

**冒烟测试**（`11_vel_smoke`，768 个环境，150 epoch，前 20 个只训练 critic）：完整流程跑通，没有崩溃，也没有 NaN。

| 指标 | 观察到的值 | 解读 |
|---|---|---|
| 静态目标成功率 | 0.86–0.95 | 与零初始化时相近，引入速度目标没有破坏原有能力 |
| attempt_rate | 0.84–1.0 | 速度目标几乎都能到达 strike 位姿 |
| head_err_at_attempt | 4.5–5 cm | attempt 时工具头位置误差在 6 cm 阈值内 |
| 下击占比 | 0.28 | 接近设计的 25% |
| **infeasible_frac** | **0.42–0.69** | 一半以上的速度目标退回成了静态目标，花在速度目标上的步数只有约 17% |
| **speed_ratio** | **3.4–5.3** | attempt 时工具头的速度是目标速度的 4–5 倍，即约 0.4 m/s |
| q_v | 0.05–0.15 | 奖励信号很弱（每次 attempt 约 50–150 分） |
| level | 一直是 0 | 150 个 epoch 内预期如此 |

**调整 1：lead-in 和 follow-through 允许超出 target volume 10 cm**（`volumeMargin`）

- 原因：官方的 delta 采样会把目标夹到 target volume 的表面上，所以很多 strike 点正好贴着边界，lead-in（后退 12 cm）或 follow-through（前进 8 cm）一伸出去就出界。CPU 测试里 strike 点是在整个区域内均匀采样的，可行率有 63%；而实际训练中只有 30%–60%。
- 为什么这样改是合理的：target volume 只是官方采样目标位置的范围，并不是手臂能到达的边界，向外 10 cm 依然在机器人前方的可达区域内。strike 点本身仍然限制在区域内，工具头高于桌面的检查也保持不变。

**调整 2：初始速度范围从 0.05–0.15 m/s 提高到 0.15–0.35 m/s**（最终仍是 0.3–0.6）

- 原因：我原本担心手臂太慢，事实正好相反。官方策略在 attempt 时工具头已经有约 0.4 m/s（`head_speed_on_vel_goals` 平均 0.29 m/s），比原来设的目标速度快 4–5 倍。按 σ = 1 计算，q_v = exp(−4) ≈ 0.02–0.1，奖励几乎没有梯度。把起点提高到接近策略自然的运动速度，课程开始时 |dv|/|v_g| 大约在 0.5–1.5，q_v 就有 0.2–0.6，信号强得多。
- 臂速倍数暂时不变（1.5 → 6）。如果正式训练中 level 升高后 `speed_ratio` 明显小于 1，再提高倍数。

**新增日志 `vel/fd_vs_analytic_rel`**：attempt 时，有限差分速度和解析估计（v_obj + ω × r）之差除以 |v_g|。如果这个值很大，说明 PhysX 根状态的速度不是原点速度（而是质心速度），这正是 3.5 节选择有限差分的原因。这一项只用来交叉检查，不参与训练。

**速度**：冒烟测试在 MIG 上的 fps total 约为 9.3k。

**第二次冒烟测试**（`12_vel_smoke2`，改动后，其余设置相同）：

| 指标 | 第一次 | 第二次 |
|---|---|---|
| infeasible_frac | 0.42–0.69 | **0.00–0.05** |
| speed_ratio | 3.4–5.3 | **1.4–2.0** |
| q_v | 0.05–0.15 | **0.34–0.46** |
| 速度成功率（eps_v = 0.6） | 0–0.05 | **0.14–0.30** |
| 静态目标成功率 | 0.86–0.95 | 0.82–0.99（块统计 2315/2580 = 0.90） |
| fd_vs_analytic_rel | — | 0.34–0.60 |
| fps total（MIG） | 9.3k | 9.3k（零初始化检查为 9.4k，新增代码对速度没有影响） |

- 两处调整都达到了预期：几乎所有速度目标都可行，速度奖励也有了明显的信号。
- 课程块统计：速度成功率为 511/2060 = 0.25，低于升级阈值 0.5，所以 level 保持 0。150 个 epoch 内这是预期的，需要正式训练中策略逐渐学会。
- `fd_vs_analytic_rel` 约为 0.45，即两种速度估计相差约 0.45 × |v_g|（约 0.1 m/s）。这与根状态线速度是质心速度、而不是原点速度的解释一致：工具转动时，ω × r_com 正好是这个量级。这说明在头部用有限差分是必要的。

---

## 9. 正式训练 v1 的结果（`20_vel_ft_v1`）

设置：整块 A100，12288 个环境，6000 个 epoch（约 11.8 亿帧），用时 4 小时 28 分（平均约 73k fps，显存约 18 GB / 40 GB），正常结束（`MAX EPOCHS NUM!`）。最终 checkpoint：`runs/20_vel_ft_v1/nn/last_20_vel_ft_v1_ep_6000_rew__8364.4_.pth`。

**课程块速度成功率的变化**（每块约 2300 个结束的速度目标；整轮共 1995 个块）：

| 块 | 1 | 101 | 301 | 501 | 801 | 1001 | 1301 | 1601 | 1901 | 最后（1995） |
|---|---|---|---|---|---|---|---|---|---|---|
| 速度成功率 | 0.18 | 0.28 | 0.32 | 0.35 | 0.40 | 0.39 | 0.40 | 0.42 | 0.44 | **0.45** |
| 静态成功率 | 0.85 | 0.91 | 0.89 | 0.88 | 0.88 | 0.88 | 0.90 | 0.88 | 0.89 | **0.89** |

**训练结束时的 attempt 指标**（最后几个打印窗口）：

| 指标 | 值 | 解读 |
|---|---|---|
| level | 0.00（整轮未升级） | 升级要求块成功率 ≥ 0.5，最终是 0.45 |
| attempt_rate | 0.90–0.92 | 绝大多数速度目标都能到达 strike |
| q_v | 0.53–0.54 | smoke 时为 0.34–0.46 |
| \|dv\|/\|v_g\| | 0.69–0.73 | 相对速度误差 |
| speed_ratio | 1.17–1.22 | 速度大小偏快约 20% |
| cos_dir | 0.83–0.84 | 方向偏约 33°，是误差的主要来源 |
| head_err_at_attempt | 5.0 cm | 一直没变 |
| infeasible_frac | 0.02–0.03 | |
| fd_vs_analytic_rel | 0.36–0.39 | |

**结论**

1. **没有遗忘**：静态目标成功率全程稳定在 0.88–0.90，与零初始化检查（0.887）相同。50% 静态目标加小学习率的做法是有效的。
2. **速度控制在学，但很慢**：块成功率从 0.18 涨到 0.45，后半程每 100 个块只涨约 0.01，没有在这一轮跨过 0.5，所以课程停在 level 0（σ = 1、epsKp = 6 cm、锥角 30°、速度 0.15–0.35 m/s、臂速 ×1.5）。
3. **误差主要来自方向，而不是大小**：速度大小偏快约 20%，方向偏约 33°。按 |dv|/|v_g| ≈ 0.7 拆分，大部分来自垂直于 d 的分量。我认为原因是：lead-in 只要求 3 cm，attempt 在离 strike 6 cm 时就触发；而官方策略学的是"直奔目标中心"，侧向偏差就变成了方向误差。速度奖励只在 attempt 那一步给一次，太稀疏，很难教会策略提前在侧向对准。
4. **与 Leo 的差别**：Leo 的奖励同样只在 attempt 时结算速度，但他先用 oracle 做了 BC（DAgger）初始化；这里只有 PPO。

**v2 的候选改动**（待定）：strike 阶段加每步的速度跟踪奖励；lead-in 收紧到 2 cm；新增 attempt 时的侧向偏差和沿 d 方向距离两个诊断量；环境数加到 24576；从 v1 的最终 checkpoint 继续训练。

---

## 10. v2b 的改动（`30_vel_ft_v2b`）

v2b 从 v1 的最终 checkpoint 继续训练。下面的改动都写在 `velocity.yaml` 里，设成 v1 的值就能还原 v1 的行为（`useLeadIn: True`、`denseVelWeight: 0`、epsKp 终点 0.03、epsVFrac 终点 0.25）。第 3 节描述的是 v1，与本节不同之处以本节为准。

### 10.1 采纳 Leo 的反馈：课程终点

| 参数 | v1 终点 | v2 终点 | 来源 |
|---|---|---|---|
| epsKp（attempt 阈值） | 3 cm | **1 cm** | Leo 实验实际使用的值（他代码里的默认值忘了改） |
| epsVFrac（速度成功线） | 0.25 | **0.15** | 同上 |
| σ（奖励宽度） | 0.25 | 0.25（未改） | 已向 Leo 确认：σ 是否也跟着改到 0.15。他在代码注释中写的是 σ 等于 level 1 的 eps_v |

- 这三项只影响课程后期，v1 一直停在 level 0，所以它们与 v1 的结果无关。
- 风险：0.6 m/s 时每个控制步移动 1 cm，与 1 cm 的阈值相当，可能一步跨过判定范围。level 升高后要看 `attempt_rate` 是否下降。

### 10.2 去掉 lead-in 中间目标（`useLeadIn: False`，Leo 的建议）

- **Leo 的理由**：lead-in 是从头训练时的补丁；我们从一个已经会 reach 的策略微调，不一定需要。
- **现在的行为**：速度目标一开始就是 strike 位姿；助跑路线由策略自己决定。
- **保留可行性检查**：strike 后方 12 cm、前方 8 cm 的空间仍要可行，工具头不能低于桌面。场景中依然有助跑空间，只是不再强制先到达那个点。
- **与 v1 结论的关系**：v1 的主要误差是方向（约 33°）。我当时猜测原因之一是 lead-in 只要求 3 cm 带来的侧向偏差；去掉 lead-in 后，新的诊断量（10.4）可以直接检验这一点。另一方面，没有 lead-in，策略可能需要从目标前方绕一圈，方向误差也可能变大。这是 v2b 要回答的问题之一。
- critic 的阶段标记那 2 维仍保留（一直显示 strike），网络结构不变，可以直接加载 v1 的权重。

### 10.3 新增：strike 阶段每步的速度跟踪奖励（`denseVelWeight: 3`，上限 `denseVelBudgetFrac: 0.1`）

- **做法**：strike 阶段 attempt 之前的每一步给 3 × q_v(t)，q_v 与 attempt 奖励用同一个公式和同一个 σ。每个目标累计最多 0.1 × 1000 = 100，即 attempt bonus 的 10%。
- **为什么要加**：v1 中速度奖励只在 attempt 那一步给一次，后半程速度成功率每 100 个块只涨约 0.01，方向误差一直很大。稠密奖励让策略在接近目标的整个过程中，都能得到"方向和大小对不对"的反馈。
- **为什么要设上限**：如果不设上限，策略可能故意不触发 attempt，持续沿 d 移动来刷每步奖励（每个目标的计时器有 600 步，600 × 3 = 1800，超过 attempt bonus）。设了 100 的上限后，奖励的大头仍是 attempt bonus。
- 这一项不是 Leo 设计里的内容，是 v2b 新增的。v2b 是同时改了 10.2 和 10.3；如果结果需要拆开看两者各自的作用，再补跑一组"只去 lead-in"（v2a：`denseVelWeight=0`）。

### 10.4 新增诊断量

| 指标 | 含义 | 用途 |
|---|---|---|
| `vel/lateral_at_attempt` | attempt 时工具头偏离目标头部中心、垂直于 d 的距离 | 检验"侧向偏差导致方向误差" |
| `vel/along_at_attempt` | attempt 时沿 d 方向的偏移（负数表示还没到达目标） | 看 attempt 触发得有多早 |
| `vel/v_par_rel` | 速度沿 d 的分量 / \|v_g\| | 速度大小匹配得如何 |
| `vel/v_perp_rel` | 速度垂直于 d 的分量 / \|v_g\| | 方向误差的大小 |
| `bonus/dense_per_strike_step` | strike 阶段平均每步的稠密奖励 | 稠密奖励的实际量级 |

### 10.5 训练设置

| 项 | v1 | v2b | 理由 |
|---|---|---|---|
| 初始权重 | 官方 checkpoint（零扩展） | v1 最终 checkpoint `last_20_vel_ft_v1_ep_6000_rew__8364.4_.pth` | v1 已学到一部分速度控制，静态能力也没有退化 |
| 环境数 | 12288（6 × 2048） | **24576（6 × 4096，与官方相同）** | v1 只用了约 18 GB 显存；每个 SAPG block 回到官方大小 |
| epoch 数 | 6000 | 3500 | 每个 epoch 的数据量翻倍，3500 个 epoch 约 14 亿帧，按约 10 万 fps 估算约 4 小时 |
| 其余 | 学习率 5e-5，critic 预热 20 epoch，successTolerance 0.01 | 相同 | |

**运行**
```bash
git pull
python finetune_velocity/tests/test_cpu.py
# 先在 MIG 上冒烟（约 7 分钟）：
sbatch --export=ALL,NAME=13_vel_smoke_v2b,CKPT=runs/20_vel_ft_v1/nn/last_20_vel_ft_v1_ep_6000_rew__8364.4_.pth \
    finetune_velocity/slurm/2_smoke_mig.sbatch
# 正常后提交正式训练：
sbatch finetune_velocity/slurm/4_train_v2b_a100.sbatch
```
