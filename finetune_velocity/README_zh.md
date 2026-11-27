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
| `finetune_velocity/slurm/*.sbatch` | 零初始化检查（MIG）、冒烟测试（MIG）、v1 正式训练（A100）、v2 正式训练（A100，第 11 节） |
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

一次跑完全部 24 个物体/任务组合（每个 10 个 episode，和官方 `run_all_evals_isaacgym.py` 的协议相同），并生成汇总 `evals/ft_<名字>/summary.txt`：

```bash
sbatch --export=ALL,NAME=v2,CFG=runs/31_vel_ft_v2/config.yaml,CKPT=runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth,LEVEL=0.10 \
    finetune_velocity/slurm/5_eval_dextoolbench_mig.sbatch
python finetune_velocity/run_dextoolbench_eval.py --summarize evals/ft_v2 evals/ft_pretrained   # 对比
```

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

## 10. v2b 的改动（只做了冒烟测试，没有正式训练）

> 本节是 v2b 的设计。冒烟测试表明去掉 lead-in 会严重破坏速度方向，所以正式训练改用 v2（恢复 lead-in，其余改动保留），见第 11 节。

v2b 从 v1 的最终 checkpoint 继续训练。下面的改动都写在 `velocity.yaml` 里，设成 v1 的值就能还原 v1 的行为（`useLeadIn: True`、`denseVelWeight: 0`、epsKp 终点 0.03、epsVFrac 和 σ 终点 0.25）。第 3 节描述的是 v1，与本节不同之处以本节为准。

### 10.1 采纳 Leo 的反馈：课程终点

| 参数 | v1 终点 | v2 终点 | 来源 |
|---|---|---|---|
| epsKp（attempt 阈值） | 3 cm | **1 cm** | Leo 实验实际使用的值（他代码里的默认值忘了改） |
| epsVFrac（速度成功线） | 0.25 | **0.15** | 同上 |
| σ（奖励宽度） | 0.25 | **0.15** | 与 epsVFrac 终点一致：Leo 代码注释中 σ 等于 level 1 的 eps_v。刚好达到成功线时奖励为 exp(−1) ≈ 0.37 × 1000 |

- 这三项只影响课程后期（level 0 时 σ 仍是 1.0），v1 一直停在 level 0，所以它们与 v1 的结果无关。
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
# （v2b 没有提交正式训练；正式训练见第 11 节的 4_train_v2_a100.sbatch）
```

---

## 11. v2b 冒烟测试结果，以及 v2 的最终设置（`31_vel_ft_v2`）

**v2b 冒烟测试**（`13_vel_smoke_v2b`：MIG，768 个环境，150 epoch，从 v1 最终 checkpoint 开始，无 lead-in，有稠密奖励）。同一个 v1 策略，有 lead-in 和没有 lead-in 的对比：

| 指标 | v1 结束时（有 lead-in） | v2b 冒烟（无 lead-in） |
|---|---|---|
| 速度成功率 | 0.45 | **0.03–0.08** |
| cos_dir | 0.83（偏约 33°） | **0.04–0.21（偏约 80°–88°）** |
| q_v | 0.53 | 0.25–0.29 |
| v_par_rel / v_perp_rel | — | 0.08–0.25 / **0.92–1.12** |
| lateral / along（attempt 时） | — | 3.3–3.8 cm / −0.2 到 −0.4 cm |
| speed_ratio | 1.2 | 1.1–1.4 |
| 静态成功率 | 0.89 | 0.83–0.91 |
| dense/step | — | 0.69–0.78 |

**解读**

- 速度几乎全部垂直于 d（`v_perp ≈ 1`、`v_par ≈ 0.1`），attempt 恰好在工具与目标在 d 方向上平齐时触发（`along ≈ 0`），侧向偏差约 3.7 cm。也就是说，工具是从侧面横着进入目标区域的。
- 原因：没有 lead-in 时，策略从上一个目标直接走向 strike 点，这条路线的方向与 d 无关。v1 策略的 0.45 正是靠 lead-in 把它带到 d 的后方。
- 要自己学会"先绕到后方再加速"，策略需要先远离目标，这是很难探索的行为。150 个 epoch 里看不到这种迹象。
- 结论：对**微调**来说，lead-in 不只是从头训练时的补丁，而是决定接近方向的关键。Leo 的判断基于从头训练的经验；这组数据可以反馈给他。

**v2 的最终设置**（正式训练 `31_vel_ft_v2`）

| 项 | v1 | v2 |
|---|---|---|
| lead-in | 有（12 cm，3 cm 内算到达） | **有**（与 v1 相同） |
| 稠密速度奖励 | 无 | **3 × q_v / 步，每个目标上限 100**（10.3） |
| epsKp 终点 | 3 cm | **1 cm**（Leo） |
| epsVFrac 终点 | 0.25 | **0.15**（Leo） |
| σ 终点 | 0.25 | **0.15**（与 epsVFrac 一致） |
| 新诊断量 | 无 | **lateral / along / v_par / v_perp / dense**（10.4） |
| 初始权重 | 官方 checkpoint | **v1 最终 checkpoint** |
| 环境数 | 12288 | **24576** |
| epoch | 6000 | 3500 |

v2 与 v1 在课程开始阶段（level 0）的唯一行为差别是稠密奖励，所以 v2 的结果可以直接归因于稠密奖励。三个终点值只在 level 升高后才起作用。

**v2 冒烟测试**（`14_vel_smoke_v2`：MIG，768 个环境，150 epoch，从 v1 最终 checkpoint 开始）最后 5 个打印窗口：

| 指标 | 值 |
|---|---|
| 速度成功率 | 0.34–0.61（每个窗口只有 16 步，波动大） |
| q_v | 0.51–0.59 |
| cos_dir | 0.81–0.90 |
| v_par_rel / v_perp_rel | 0.94–1.24 / 0.41–0.55 |
| lateral / along（attempt 时） | 1.4–2.0 cm / −4.0 到 −4.7 cm |
| 静态成功率 | 0.84–0.93 |
| dense/step | 1.20–1.36 |

- 恢复 lead-in 后，各项指标回到 v1 结束时的水平（cos 约 0.83，成功率约 0.45），稠密奖励也在正常发放。
- 新诊断量第一次给出了方向误差的来源：attempt 在离目标沿 d 还有约 4.4 cm 时触发，此时侧向偏差约 1.7 cm。如果策略这时正朝目标中心收拢，方向就会偏 atan(1.7/4.4) ≈ 21°，可以解释 v_perp/v_par ≈ 0.5（约 27°）的大部分。剩余部分可能来自工具转动带来的速度（`fd_vs_analytic_rel` 约 0.4）。
- 由此预测：随着课程把 epsKp 从 6 cm 收紧到 1 cm，attempt 会更靠近目标，侧向收拢造成的方向误差应该减小，但对侧向对准的要求会更高。

**以后可以试的方向**（回应 Leo 想去掉 lead-in 的意图）：不设显式的中间目标，而是先采样方向 d，再把 strike 点放在上一个目标沿 d 前方 10–15 cm 处。这样工具本来就在 strike 点的后方，直线接近的方向自然正确。代价是接近途中工具还在转动，会给工具头带来额外的转动速度。

**运行**
```bash
git pull
# 冒烟（MIG，约 7 分钟）
sbatch --export=ALL,NAME=14_vel_smoke_v2,CKPT=runs/20_vel_ft_v1/nn/last_20_vel_ft_v1_ep_6000_rew__8364.4_.pth \
    finetune_velocity/slurm/2_smoke_mig.sbatch
# 正常后：
sbatch finetune_velocity/slurm/4_train_v2_a100.sbatch
```

---

## 12. 正式训练 v2 的结果（`31_vel_ft_v2`）

设置：整块 A100，24576 个环境，从 v1 最终 checkpoint 开始，3500 个 epoch（约 14 亿帧），用时 4 小时 21 分，正常结束。平均速度 97.6k fps；显存峰值 26.4 GB / 40 GB（训练前按线性外推的估计约 30 GB）。最终 checkpoint：`runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth`。

**课程**：共 1736 个块（每块约 3000–3500 个结束的速度目标），升级两次，没有降级：

| 时间点 | 块成功率 | 事件 |
|---|---|---|
| 第 1 块 | 1571/3048 = 0.515 | 升到 level 0.05（从 v1 继承的能力，加上稠密奖励） |
| 第 51–1201 块 | 0.41–0.49，大多为 0.45–0.47 | 停在 0.05 |
| 约第 1220 块 | 1763/3511 = 0.502 | 升到 level 0.10 |
| 第 1251–1736 块 | 0.44–0.48 | 停在 0.10，直到结束 |

**训练结束时的指标**（level 0.10，对应 σ ≈ 0.92、epsVFrac ≈ 0.555、epsKp ≈ 5.5 cm、锥角 45°、速度 0.165–0.375 m/s、臂速 ×1.95）：

| 指标 | v1 结束（level 0） | v2 结束（level 0.10） |
|---|---|---|
| 速度成功率 | 0.45 | 0.45–0.48 |
| q_v | 0.53 | 0.53–0.54 |
| \|dv\|/\|v_g\| | 0.70 | 0.64–0.66 |
| speed_ratio | 1.2 | 1.11–1.15 |
| cos_dir | 0.83 | 0.84 |
| v_par_rel / v_perp_rel | — | 0.95–0.99 / 0.43–0.44 |
| lateral / along（attempt 时） | — | 1.7 cm / −3.9 到 −4.0 cm |
| head_err_at_attempt | 5.0 cm | 4.5 cm |
| 静态成功率 | 0.89 | **0.91** |
| dense/step | — | 1.28–1.29 |

**结论**

1. **没有遗忘，静态能力略有提升**：静态成功率 0.90–0.92，高于 v1 的 0.89。
2. **速度大小基本学会了**：沿 d 的速度分量 v_par ≈ 0.97 |v_g|，速度大小比从 1.2 降到约 1.13。
3. **方向误差几乎没有改善，是主要瓶颈**：cos 0.83 → 0.84，垂直分量 v_perp ≈ 0.44 |v_g|（约 24°）。在 level 0.10 时成功线是 |dv| < 0.555 |v_g|，光是垂直分量就用掉了大部分余量，所以成功率卡在约 0.46，课程升级主要靠块与块之间的随机波动（两次升级的块成功率都只有 0.50–0.52）。
4. **方向误差的来源已经基本确定**：attempt 时工具头侧向偏离 strike 直线约 1.7 cm，沿 d 方向还差约 3.9 cm。如果策略此时朝目标中心收拢，方向会偏 atan(1.7 / 3.9) ≈ 23.6°；按 speed_ratio 1.13 计算，v_perp ≈ 1.13 × sin(23.6°) ≈ 0.45 |v_g|，与观测到的 0.43–0.44 几乎完全吻合。也就是说，**策略没有沿着 strike 直线运动，而是在最后几厘米朝目标点修正侧向偏差**，这个修正本身就是方向误差。
5. **稠密奖励的作用有限**：它让速度大小更准，但每步的 q_v 主要由误差大小决定，对"沿直线接近"的引导很弱。
6. 按目前的速度，课程走完 20 级不现实；继续堆训练时长意义不大，需要针对侧向对准改设计（第 13 节，待定）。

---

## 13. v3 之前的诊断：侧向偏差是从哪里来的

**问题**：v2 中 attempt 时工具头偏离 strike 直线约 1.7 cm，这几乎完全解释了方向误差（第 12 节）。v3 的候选改动有两个：
- **A**：把 lead-in 的到达容差从 3 cm 收紧到 1.5 cm；
- **B**：在 strike 阶段加沿直线运动的 shaping。

只用 A 的前提是：**attempt 时的侧向偏差主要是从到达 lead-in 那一刻继承来的**。
- 支持这个前提的理由：lead-in 只要求 3 cm、不要求停留，策略又是从上一个目标赶过来、带着与 d 无关的侧向速度进入 lead-in 的，然后只有 12 cm 的助跑距离。
- 但偏差也可能是助跑途中才产生的，比如工具转动、冲过 lead-in 后过冲。这种情况下 A 帮助不大，需要 B。

**新增诊断量**（在到达 lead-in 的那一步，相对于 strike 直线测量；这条直线过 strike 时工具头的中心，方向为 d）：

| 指标 | 含义 |
|---|---|
| `vel/lateral_at_leadin` | 工具头离 strike 直线的距离 |
| `vel/along_at_leadin` | 沿 d 方向的偏移（理想值约 −12 cm） |
| `vel/v_par_rel_at_leadin` / `vel/v_perp_rel_at_leadin` | 速度沿 d / 垂直于 d 的分量，除以 \|v_g\| |

**判断方式**：用 v2 的最终策略在 MIG 上跑一次 smoke，训练设置不变，课程从 level 0.10 开始。
- **attempt 时的侧向偏差 ≲ 到达 lead-in 时的**：偏差是继承来的，v3 用 A。
- **attempt 时的侧向偏差明显更大**：偏差在助跑途中产生，v3 用 B（或 A+B）。
- 同时参考到达 lead-in 时的垂直速度：如果它很大，说明策略是斜着冲过 lead-in 的。

```bash
sbatch --export=ALL,NAME=15_vel_smoke_diag,CKPT=runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth,EXTRA=task.env.velocity.curriculum.initLevel=0.1 \
    finetune_velocity/slurm/2_smoke_mig.sbatch
```

**诊断结果**（`15_vel_smoke_diag`：MIG，768 个环境，150 epoch，从 v2 最终策略开始，level 0.15，最后 5 个打印窗口）：

| | 到达 lead-in 时 | attempt 时 |
|---|---|---|
| 侧向距离 lateral | 1.5–1.7 cm | 1.7–2.1 cm |
| 沿 d 的位置 along | −10.9 到 −11.1 cm | −3.1 到 −3.7 cm |
| 沿 d 的速度 v_par / \|v_g\| | **−0.26 到 −0.37** | 0.90–1.09 |
| 垂直速度 v_perp / \|v_g\| | **0.55–0.72** | 0.40–0.55 |

其余指标与 v2 结束时相近：速度成功率 0.24–0.51，cos 0.72–0.87，静态成功率 0.87–0.95。

**解读**

1. **侧向位置不是问题**：到达 lead-in 时侧向偏差已经只有约 1.6 cm，到 attempt 时约 1.9 cm，几乎不变。所以把 lead-in 容差从 3 cm 收紧到 1.5 cm（原方案 A），对侧向位置几乎没有作用。
2. **问题在于到达 lead-in 时的速度**：沿 d 的速度是**负的**（约 −0.3 |v_g|），也就是工具是**倒着**、从 strike 那一侧退到 lead-in 的；同时还有约 0.6 |v_g| 的侧向速度。原因是上一个目标在 strike 点附近（delta 采样距上一个目标 ≤ 10 cm），而 lead-in 在 strike 后方 12 cm，所以策略要往 −d 方向走才能到达 lead-in；又因为不要求停留，一进入 3 cm 范围就立刻切换到 strike。
3. 所以策略在 12 cm 的助跑里，要先抵消往回走和横向的速度，再沿 d 加速。侧向距离的大小基本没变，但横向速度在这段时间里一直存在，工具头在直线两侧来回穿过，到 attempt 时仍有约 0.46 |v_g| 的垂直速度。
4. **结论**：方向误差的根源是"助跑开始时的运动状态不对"，而不是起点位置不准。原方案 A 不对症；B（沿直线的 shaping）也只能间接处理这个问题。

---

## 14. 下一版设计：带速度的平滑轨迹目标（进行中，分支 `trajectory-goals`）

**为什么要改设计**：第 13 节的诊断说明，方向误差的根源是**目标序列在动力学上不连贯**：策略是倒着、斜着进入助跑的。另一种改法是"lead-in 设成静态目标、停稳后再击打"，但那只能训练出"停 → 击打"的单次击打动作，对擦、扫、连续多段运动这类长程任务没有帮助，偏离了项目"通用的速度条件化工具操作"的目标。（已向 Leo 征求意见。）

### 14.1 已确定的设计

| 决定 | 内容 | 理由 |
|---|---|---|
| 目标形式 | 一串路点，每个都带位姿和工具头速度；速度为 0 的路点就是官方静态目标（1.5 cm、停 10 步） | 静态目标、单次击打、扫、擦都是同一框架的特例；前一个路点总在当前路点后方，速度方向指向它，不会再"倒着助跑" |
| 目标序列 | 抓起工具 → 若干"块"；静态块（官方 delta 目标）占 **40%**，运动块 = 设定位姿（静态）→ 带速度的路点 → 终点停点（静态） | 保留官方"到达并停稳"的结构，防止遗忘 |
| 推进方式 | **按事件推进**：到达（keypoint 距离 ≤ eps_kp）**或越过**路点的法平面，都算经过；偏离超过约 15 cm 就放弃这一块，直接跳到终点停点 | 与 SimToolReal 一致；有"越过"这个条件，擦边错过时不必掉头回来补 |
| 前瞻 | 观测中加入**下一个**路点（相对 keypoint 位置 + 速度，15 维，零初始化） | 路点每隔几步切换，观测还有最多 3 步的延迟，需要提前知道下一个路点 |
| 奖励 | 静态路点用官方奖励；运动路点在经过时给 B × q_pos × q_v，一块内所有运动路点合计约 1000；**去掉 v2 的稠密速度奖励** | 每个运动路点本身已经足够"稠密"；奖励总额与块的长短无关 |
| 训练方式 | 在 SimToolReal 上**微调**，不从零开始 | 抓取、握持、抗干扰都已学会；算力和时间有限；对照实验更干净 |
| 起点模型 | **待定**：等 v2 的 DexToolBench 评估结果 | 若 v2 没有遗忘就从 v2 开始，否则从官方模型开始 |

### 14.2 轨迹采样器（`isaacgymenvs/tasks/simtoolreal/trajectory_goals.py`，纯 torch）

- **路径定义在工具头中心上**，由 1 到 n 段拼接而成，段与段之间切线连续。两种段：
  - **弯曲段**：三次 Bézier 曲线，姿态平滑插值到一个随机转动；
  - **划弧段**：工具绕手腕附近的转动中心做刚体转动，工具头沿圆弧运动，以转动为主（锤子挥动、翻铲子）。
- **速度曲线**：从静止平滑加速（五次 smoothstep）、经过平台段、再平滑减速到静止。加减速段的比例随机（减速段短的更像击打），峰值速度受加速度上限（3 m/s²）约束。每个路点的目标速度 = 路径切线 × 该点的速度大小，所以整条序列在动力学上是连贯的。
- 路点间距约 3 cm（0.5 m/s 时约每 4 个控制步一个路点）；可行性检查与 v1/v2 相同，扩展到整条路径。
- 课程（level 0 → 1）：峰值速度上限 0.35 → 0.6 m/s；弯曲 0° → 90°；每段转动 0° → 150°；划弧比例 0 → 40%（最大扫过角 0° → 180°）；每块段数 1 → 1–6；起始方向锥角 30° → 180°；另有 25% 的下击。
- CPU 测试：`finetune_velocity/tests/test_trajectories.py`，检查速度上限、终点静止、加速度上限、路点间距、速度沿切线、位姿一致、可行性、level 0 为直线且不转动。

### 14.3 用 DexToolBench 检查覆盖范围（`finetune_velocity/tools/compare_dextoolbench.py`）

DexToolBench 的轨迹只有位姿、没有时间戳，所以只能比较几何形状。**原则：DexToolBench 是覆盖范围的下限，不是训练目标**，采样范围应包住它并留有余量，避免只对这 24 个任务有效。

| 指标 | DexToolBench（24 个任务） | 采样器 level 1 |
|---|---|---|
| 每厘米工具头位移对应的姿态变化，中位数 / 第 90 百分位 | 3.4 / 15.9 °/cm | 3.8 / 7.0 °/cm |
| 每 3 cm 的转弯角，中位数 / 第 90 百分位 | 17° / 129° | 12° / 34° |
| 工具头路程 / 手柄路程，中位数 / 第 90 百分位 | 1.03 / 1.52 | 1.08 / 1.76 |
| 一次连续运动的长度，中位数 / 最大 | 0.79 / 1.41 m | 0.29 / 1.49 m |

图：`finetune_velocity/figures/geometry_vs_dextoolbench.png`，以及采样示例 `finetune_velocity/figures/trajectory_chunks.png`（`finetune_velocity/tools/plot_trajectories.py`）。

**已知差距和局限**

1. **> 90° 的急转弯（往返）**：在物理上需要先停下来，所以设计成"停 → 下一块"，而不是放在一块之内。也正因为如此，单块的长度比 DexToolBench 一整条任务短。
2. **原地旋转**（螺丝刀 spin，转动速率第 90 百分位 15.9 °/cm 主要来自它）：工具头线速度接近 0，"工具头线速度"这个接口表达不了，需要以后加角速度目标。
3. **速度大小**没有真实数据可以校准，沿用 Leo 的 0.3–0.6 m/s。level 1 时，受加速度上限和路径长度的约束，峰值速度平均约 0.37 m/s，第 90 百分位约 0.5 m/s。
4. 训练中工具不接触任何物体；"速度条件能帮助用力"只能通过接触任务的评估来验证。
5. 用 DexToolBench 校准几何分布、又用它评估静态任务，存在轻微的数据泄漏；泛化能力的结论应以没有参与校准的任务（如 Leo 的 plate / nail / sweep）为准。

### 14.4 接下来

1. 接入环境：路点队列、按事件推进（含"越过"判定和放弃机制）、奖励、前瞻观测、critic 的阶段信息、诊断量、课程。
2. 让扩展脚本和启动脚本支持任意的旧观测维度（起点可能是 v2 的 149 维，也可能是官方的 140 维）。
3. 定下起点模型，然后 MIG smoke → A100 正式训练。
4. 稍后再做：独立的测试轨迹集（四类典型运动，加上超出训练范围的压力测试）。

---

## 15. 轨迹目标的环境实现（`SimToolRealTrajectory`）

| 文件 | 内容 |
|---|---|
| `isaacgymenvs/tasks/simtoolreal/trajectory_env.py` | 新任务，继承 `SimToolRealVelocity`（沿用工具头速度、速度观测、臂速倍数、课程类和观测钩子），替换目标逻辑 |
| `isaacgymenvs/tasks/simtoolreal/trajectory_goals.py` | 采样器（第 14 节），以及判定"经过"路点的纯函数 `waypoint_pass`（有 CPU 测试） |
| `finetune_velocity/trajectory.yaml` | 全部参数和课程范围 |
| `finetune_velocity/launch_from_pretrained.py` | 新增 `--task trajectory`；权重扩展改为通用：按任务需要的输入维度，自动补齐 checkpoint 缺少的部分（官方 140/162 → 164/189，或 v2 的 149/173 → 164/189） |
| `finetune_velocity/slurm/6_train_traj_a100.sbatch` | 正式训练；smoke 用 `2_smoke_mig.sbatch` 加 `TASK=trajectory` |

**目标的阶段**（`phase`）：PLAIN（静态块）、SET（设定位姿，静态）、MOVE（运动路点）、END（终点停点，静态）。
- PLAIN、SET、END 都用官方的停留规则，只有这三种会计入 `successes`，这样官方"每个 episode 最多 50 个目标"的含义保持不变。
- MOVE 路点"经过"（到达或越过）时，奖励为 B × q_pos × q_v，其中 B = 1000 / 该块的运动路点数；偏离超过 15 cm 就放弃这一块，直接跳到 END。
- 每次目标前进，每个目标的计时器（600 步）都重新开始。

**新增观测**
- `goal_lookahead`（15 维，策略和 critic 都有）：下一个路点的 keypoint 相对工具当前 keypoint 的位置（12 维），加上下一个路点的速度除以 0.5 m/s（3 维）。在 SET 阶段，下一个路点是第一个运动路点；在静态目标上，就是当前目标本身。
- `traj_progress`（1 维，只给 critic）：当前块剩余运动路点的比例。
- `velocity_phase` 沿用 v2 的 2 维，现在含义是（0，是否在运动路点上）。

**课程**：判定依据是运动路点的成功率，即位置在 eps_kp 内且速度误差 < eps_v × max(|v_g|, 0.1)，每块至少 4096 个样本；静态保护不变。

**日志**：`[trajectory] step ...` 这一行，以及 TensorBoard 中的 `traj/*`，包括：
- 路点成功率、到达比例、放弃率；
- q_pos、q_v、v_par、v_perp、侧向偏差；
- 工具头速度与目标速度；
- 完成、失败的块数，不可行比例，运动步数占比。

**运行**
```bash
# smoke（MIG）：从 v2 或官方模型开始都可以
sbatch --export=ALL,TASK=trajectory,NAME=16_traj_smoke,CKPT=runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth \
    finetune_velocity/slurm/2_smoke_mig.sbatch
# 正式训练（A100），CKPT 取决于起点模型的决定
sbatch --export=ALL,NAME=40_traj_v1,CKPT=<checkpoint> finetune_velocity/slurm/6_train_traj_a100.sbatch
```

## 16. 轨迹 smoke 1（`16_traj_smoke`，作业 24298137）和两处修改

这次 smoke 在 MIG 上从 v2 的最终 checkpoint 开始，正常跑完、没有报错，说明整套流程（权重扩展、新观测、阶段、路点判定、课程、日志）在 Isaac Gym 里都能跑通。

**正常的部分**
- 静态成功率 0.91–0.95，停留目标没有退化。
- 放弃率 0–0.02，完成的块远多于失败的块。
- 运动路点上工具头的平均速度（0.20–0.23 m/s）和目标平均速度（0.22–0.23 m/s）一致。

**问题 1：路点提前前进。**
- 98–100% 的"经过"都是由"到达"条件触发的，而且触发时工具头离路点还差约 4.4 cm（`along −4.4 cm`，keypoint 距离 5.1 cm）。
- 原因："到达"用的是课程容差 eps_kp，level 0 时是 6 cm，而路点间距只有 3 cm。容差比间距还大，工具落后一两个路点也能"到达"，目标就继续往前走，奖励实际上在奖励落后。
- 改法：把"前进规则"和"成功规则"分开。
  - 目标前进的条件：工具头越过路点的法平面，或者 keypoint 距离 ≤ `reachDist` = 1.5 cm。1.5 cm 是间距的一半，所以工具落后整整一个路点时，目标不可能前进。
  - 课程容差 eps_kp（6 cm → 1 cm）只用来判断这次经过算不算成功（位置在 eps_kp 内，且速度误差 < eps_v × max(|v_g|, 0.1)），不再决定目标什么时候前进。
  - 经过时的奖励不变（B × q_pos × q_v）。因为现在经过发生在离路点更近的地方，q_pos 是在更有意义的位置上算的。

**问题 2：在运动路点上的时间太少（5–8% 的步数）。**
- 大部分时间花在静态停留上：每个运动块需要两次"移过去并停住"（之前的设定位姿和之后的终点），再加上本来就有 40% 的静态块，速度跟踪的学习信号被稀释了。
- 改法：**串联块**。一个块的终点停稳之后，以 `chainProb` = 0.5 的概率直接从这里开始下一个运动块：目标直接跳到新块的第一个运动路点，不再先去新的设定位姿。
  - 终点本来就是静止状态，可以作为下一块的合法起点，每个串联块省掉一次移动和一次停留。
  - 这样还会出现"停 → 运动 → 停 → 运动"的连续动作，和长程工具使用（连续几下击打、来回扫）更像。
  - 代价：串联块不能是"先转向朝下"的下击，因为那需要先转动工具姿态，也就需要一次停留。所以串联块的 `down_frac` = 0，下击仍然来自从新设定位姿开始的块。
  - 另外 50% 的终点，以及所有 PLAIN 目标，都和以前一样走官方路径：生成新的 delta 目标，它可能成为新块的设定位姿。
  - 新统计量 `traj/chained_frac`（`[trajectory]` 行里的 "chained"）是已开始的块中串联块的比例，预计约为 0.5 × 走到终点的块所占比例。

**smoke 2 的预期**
- 由"到达"触发的经过比例大幅下降（大部分变成越过），经过时的 `along` 和 keypoint 距离明显变小。
- 运动路点上的步数占比上升。
- `chained` 约为 0.5。
- 静态成功率仍保持在 0.9 左右。

```bash
sbatch --export=ALL,TASK=trajectory,NAME=17_traj_smoke2,CKPT=runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth \
    finetune_velocity/slurm/2_smoke_mig.sbatch
```

**smoke 2 结果（`17_traj_smoke2`，作业 24298309，150 epoch，level 0）**：两处修改都按预期生效。

| | smoke 1 | smoke 2 |
|---|---|---|
| 由"到达"触发的经过比例 | 0.98–1.00 | 0.44–0.57（其余是越过） |
| 运动路点步数占比 | 0.05–0.08 | 0.16–0.19 |
| 已开始块中串联块的比例 | – | 0.24 → 0.47（随着块走到终点而上升） |
| 静态成功率 | 0.91–0.95 | 0.90–0.95 |
| 放弃率 | 0–0.02 | 0–0.01 |
| 工具头 / 目标速度（m/s） | 0.20–0.23 / 0.22–0.23 | 0.16–0.18 / 0.20–0.22 |

- 路点成功率约 0.40（q_pos 0.67，q_v 0.49），课程停在 level 0（1713/4181 < 0.5）。速度跟踪是短板：|dv|/|v_g| ≈ 0.84，v_par ≈ 0.87，v_perp ≈ 0.55，而侧向偏差只有约 1.1 cm。也就是说工具头基本在路径上，但速度方向在晃。现在目标只有在工具真正到达后才会前进，所以工具头速度比目标慢约 20%。这正是正式训练要学的东西，150 epoch 还看不出学习趋势。
- 即使在 level 0，也有约 23–32% 的采样块不可行（会退回普通静态目标，所以实际运动块比例约为 0.6 × 0.72 ≈ 0.43）。这是代价，不是 bug；需要的话可以调大 `tries`。
