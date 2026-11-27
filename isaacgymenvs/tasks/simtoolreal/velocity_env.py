"""SimToolReal with pose + orientation + velocity goals, for fine-tuning the released checkpoint.

Only behaviour that the velocity goals need is changed; everything else (assets, domain randomization, delays, the
official reward terms, the static "hold" success rule) is inherited unchanged from SimToolReal. The design and the
reason for every change are written up in finetune_velocity/README_zh.md (Chinese) and README.md (English).

Goal kinds (self.goal_kind):
  STATIC  official goal (v_g = 0): success after successSteps steps within successTolerance * keypointScale,
          bonus reachGoalBonus / successSteps per near step. Exactly the official rule.
  LEADIN  (only with useLeadIn, v1) first half of a velocity goal: the strike orientation, leadIn m behind the strike
          pose along -d. Reached at keypoint distance <= leadInTolerance; no hold, no bonus (keypoint reward applies).
  STRIKE  the strike pose, observed goal velocity v_g. The first step with keypoint distance <= eps_kp is the attempt;
          it pays reachGoalBonus * exp(-|v_head - v_g| / (sigma * |v_g|)) once and ends the goal. With
          denseVelWeight > 0 (v2) the strike steps before it also pay denseVelWeight * q_v, capped per goal.

v_head is the velocity of the head center (finite difference of its position over one control step), not of the
handle-center frame the keypoints are built on.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple

# isort: off
from isaacgym import gymapi  # noqa: F401  (must precede torch)
import torch
from torch import Tensor
# isort: on

from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from isaacgymenvs.tasks.simtoolreal.velocity_goals import (
    VelocityCurriculum,
    VelocityGoalParams,
    lerp,
    quat_rotate_xyzw,
    sample_velocity_goals,
    velocity_match,
)

STATIC, LEADIN, STRIKE = 0, 1, 2

# Head "face" axis in the tool frame (+x = handle -> head) per primitive type, and which types can strike with either
# face. Hammer heads are long along y, with a face at each end. Screwdriver / marker tips and the spatula blade point
# along +x. Eraser (no head) and brush: the large faces normal to z.
DEFAULT_HEAD_AXIS = {
    "hammer": [0.0, 1.0, 0.0],
    "screwdriver": [1.0, 0.0, 0.0],
    "marker": [1.0, 0.0, 0.0],
    "spatula": [1.0, 0.0, 0.0],
    "eraser": [0.0, 0.0, 1.0],
    "brush": [0.0, 0.0, 1.0],
    "unknown": [1.0, 0.0, 0.0],
}
DEFAULT_SYMMETRIC = ["hammer", "eraser", "brush"]

# Diagnostic accumulators (one GPU vector, read back every statsEvery steps).
ACC_NAMES = [
    "vel_goals_sampled",  # velocity goals requested (velocityGoalProb draws)
    "vel_goals_infeasible",  # requested but no feasible candidate -> stayed static
    "leadin_reached",
    "attempts",
    "vel_success",  # attempt with |dv| < eps_v_frac |v_g|
    "vel_fail_no_attempt",  # episode ended during the lead-in or the strike
    "sum_qv",
    "sum_dv_rel",  # |dv| / |v_g|
    "sum_speed_ratio",  # |v_head| / |v_g|
    "sum_cos",  # cos(v_head, v_g)
    "sum_head_err",  # m, head center to goal head center at the attempt
    "sum_attempt_kp",  # keypoint distance at the attempt
    "down_attempts",
    "down_success",
    "static_success",
    "static_fail",
    "steps_static",
    "steps_leadin",
    "steps_strike",
    "kp_rew_static",
    "kp_rew_leadin",
    "kp_rew_strike",
    "bonus_static",
    "bonus_strike",
    "sum_head_speed_vel",  # |v_head| on lead-in + strike steps
    "sum_fd_vs_analytic",  # |v_fd - v_analytic| / |v_g| at the attempt (cross-check of the head velocity estimate)
    "sum_lateral",  # m, head offset from the goal head center perpendicular to d, at the attempt
    "sum_along",  # m, head offset along d at the attempt (negative: still before the goal)
    "sum_v_par_rel",  # (v_head . d) / |v_g| at the attempt
    "sum_v_perp_rel",  # |v_head - (v_head . d) d| / |v_g| at the attempt (the direction part of the error)
    "dense_rew",  # dense velocity reward paid (sum over steps)
    # At the moment the lead-in is reached, relative to the strike line (through the strike head center along d):
    "sum_lead_lateral",  # m, head distance from the line
    "sum_lead_along",  # m, head offset along d (about -leadIn)
    "sum_lead_v_par_rel",  # (v_head . d) / |v_g|
    "sum_lead_v_perp_rel",  # |v_head - (v_head . d) d| / |v_g|
]
ACC = {k: i for i, k in enumerate(ACC_NAMES)}


class SimToolRealVelocity(SimToolReal):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        vcfg = dict(cfg["env"].get("velocity", {}) or {})
        self.vcfg = vcfg
        self.vel_goal_prob = float(vcfg.get("velocityGoalProb", 0.5))
        # Fixed goal trajectories (DexToolBench evaluation) stay static goals; the obs terms and arm speed stay.
        self.vel_enabled = self.vel_goal_prob > 0 and not cfg["env"].get("useFixedGoalStates", False)
        self.vel_on_first_goal = bool(vcfg.get("velocityOnFirstGoal", False))
        self.vel_obs_scale = float(vcfg.get("obsVelScale", 0.5))
        self.vel_leadin_tol = float(vcfg.get("leadInTolerance", 0.03))
        # v2: the lead-in waypoint is optional (Leo: it was a patch for training from scratch). When off, a velocity goal
        # starts directly as the strike; the feasibility check still keeps room for a run-up behind it.
        self.vel_use_lead_in = bool(vcfg.get("useLeadIn", True))
        # v2: dense velocity tracking during the strike stage: denseVelWeight * q_v per step, at most
        # denseVelBudgetFrac * reachGoalBonus per goal (the cap removes any incentive to avoid the attempt).
        self.vel_dense_weight = float(vcfg.get("denseVelWeight", 0.0))
        self.vel_dense_budget_frac = float(vcfg.get("denseVelBudgetFrac", 0.1))
        self.vel_stats_every = int(vcfg.get("statsEvery", 16))
        self.vel_print_every = int(vcfg.get("printEvery", 160))
        self.vel_ranges = {
            "sigma_frac": vcfg.get("sigmaFrac", [1.0, 0.15]),
            "eps_v_frac": vcfg.get("epsVFrac", [0.6, 0.25]),
            "eps_kp": vcfg.get("epsKp", [0.06, 0.03]),
            "cone_deg": vcfg.get("coneDeg", [30.0, 180.0]),
            "speed_lo": vcfg.get("speedLo", [0.15, 0.3]),
            "speed_hi": vcfg.get("speedHi", [0.35, 0.6]),
            "arm_speed_mult": vcfg.get("armSpeedMult", [1.5, 6.0]),
        }
        self.vel_geom = dict(
            lead_in=float(vcfg.get("leadIn", 0.12)),
            follow_through=float(vcfg.get("followThrough", 0.08)),
            min_head_z=float(vcfg.get("minHeadZ", 0.58)),
            down_frac=float(vcfg.get("downFrac", 0.25)),
            down_max_z=float(vcfg.get("downMaxZ", -0.85)),
            tries=int(vcfg.get("tries", 8)),
            volume_margin=float(vcfg.get("volumeMargin", 0.1)),
        )
        cur = dict(vcfg.get("curriculum", {}) or {})
        fixed_level = cur.get("fixedLevel", None)
        self.vel_curriculum = VelocityCurriculum(
            level=float(fixed_level if fixed_level is not None else cur.get("initLevel", 0.0)),
            step=float(cur.get("step", 0.05)),
            up_threshold=float(cur.get("upThreshold", 0.5)),
            down_threshold=float(cur.get("downThreshold", 0.2)),
            min_samples=int(cur.get("minSamples", 2048)),
            static_guard=float(cur.get("staticGuard", 0.5)),
            min_steps_between=int(cur.get("minStepsBetween", 160)),
            fixed=fixed_level is not None,
        )
        self.head_axis_by_type = {**DEFAULT_HEAD_AXIS, **dict(vcfg.get("headAxis", {}) or {})}
        self.head_symmetric_types = set(vcfg.get("symmetricHeadTypes", DEFAULT_SYMMETRIC))
        self._vel_step_count = 0
        self._vel_stats = {}
        self._vel_apply_level()

        # Filled by _main_object_assets_and_scales (called inside super().__init__ -> create_sim -> _create_envs).
        self._asset_head_offsets = None
        self._asset_types = None

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self._vel_alloc()

    # ------------------------------------------------------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------------------------------------------------------
    def _vel_apply_level(self):
        lv = self.vel_curriculum.level
        self.vel_cur = {k: lerp(v, lv) for k, v in self.vel_ranges.items()}
        # Applies to static goals too, and also when velocity goals are off (evaluation of a fine-tuned policy must
        # use the arm speed it was trained at: set curriculum.fixedLevel).
        self._vel_arm_mult = self.vel_cur["arm_speed_mult"]

    def _main_object_assets_and_scales(self, object_asset_root, tmp_assets_dir):
        files, scales, need_vhacds = super()._main_object_assets_and_scales(object_asset_root, tmp_assets_dir)
        # The generator does not return head sizes, so read the head center back from each URDF while the temporary
        # files still exist: the second <visual> of the single link is the head, at origin x = handle_len/2 + head_x/2.
        offsets, types = [], []
        for f in files:
            off = 0.0
            try:
                link = ET.parse(str(f)).getroot().find("link")
                visuals = link.findall("visual") if link is not None else []
                if len(visuals) >= 2:
                    off = float(visuals[1].find("origin").get("xyz").split()[0])
            except Exception as exc:  # known (mesh) objects: keep the handle center
                print(f"[velocity] could not read head offset from {f}: {exc}")
            parts = Path(str(f)).name.split("_")
            t = parts[1] if len(parts) > 1 and parts[1] in self.head_axis_by_type else "unknown"
            offsets.append(off)
            types.append(t)
        self._asset_head_offsets = offsets
        self._asset_types = types
        return files, scales, need_vhacds

    def _vel_alloc(self):
        n, dev = self.num_envs, self.device
        self.goal_kind = torch.zeros(n, dtype=torch.long, device=dev)
        self.vel_goal = torch.zeros(n, 3, device=dev)
        self.vel_goal_is_down = torch.zeros(n, dtype=torch.bool, device=dev)
        self.pending_strike_pose = torch.zeros(n, 7, device=dev)
        self.dense_paid = torch.zeros(n, device=dev)
        self.head_pos = torch.zeros(n, 3, device=dev)
        self.head_vel = torch.zeros(n, 3, device=dev)
        self.head_vel_analytic = torch.zeros(n, 3, device=dev)
        self._head_prev_pos = torch.zeros(n, 3, device=dev)
        self._head_prev_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._acc = torch.zeros(len(ACC_NAMES), device=dev)
        self._block = torch.zeros(4, device=dev)  # vel_success, vel_total, static_success, static_total

        n_assets = len(self._asset_head_offsets) if self._asset_head_offsets else 1
        offs = self._asset_head_offsets or [0.0]
        types = self._asset_types or ["unknown"]
        asset_idx = torch.arange(n) % n_assets  # same assignment as _create_envs
        off_t = torch.tensor(offs, dtype=torch.float)[asset_idx]
        self.head_offset = torch.zeros(n, 3)
        self.head_offset[:, 0] = off_t
        self.head_offset = self.head_offset.to(dev)
        axes = torch.tensor([self.head_axis_by_type[t] for t in types], dtype=torch.float)
        axes = axes / axes.norm(dim=-1, keepdim=True)
        self.head_axis = axes[asset_idx].to(dev)
        sym = torch.tensor([t in self.head_symmetric_types for t in types], dtype=torch.bool)
        self.head_axis_symmetric = sym[asset_idx].to(dev)

        tv_min = self.target_volume_origin + self.target_volume_extent[:, 0]
        tv_max = self.target_volume_origin + self.target_volume_extent[:, 1]
        self._vel_volume = (tv_min, tv_max)
        print(
            f"[velocity] enabled={self.vel_enabled} goal_prob={self.vel_goal_prob} level={self.vel_curriculum.level:.2f} "
            f"head offset mean {off_t.mean():.3f} m (min {off_t.min():.3f}, max {off_t.max():.3f}); "
            f"types {sorted(set(types))}; target volume {tv_min.tolist()} .. {tv_max.tolist()}"
        )

    def _vel_ready(self) -> bool:
        return hasattr(self, "goal_kind")

    # ------------------------------------------------------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------------------------------------------------------
    def _extra_obs_type_sizes(self) -> dict:
        # goal_velocity: v_g, v_head, v_head - v_g (each 3, divided by obsVelScale m/s). Policy and critic.
        # velocity_phase: one-hot (lead-in, strike). Critic only (stateList), see README.
        return {"goal_velocity": 9, "velocity_phase": 2}

    def _add_extra_obs(self, obs_dict: dict) -> None:
        if not self._vel_ready():
            obs_dict["goal_velocity"] = torch.zeros(self.num_envs, 9, device=self.device)
            obs_dict["velocity_phase"] = torch.zeros(self.num_envs, 2, device=self.device)
            return
        s = 1.0 / self.vel_obs_scale
        obs_dict["goal_velocity"] = torch.cat(
            (self.vel_goal * s, self.head_vel * s, (self.head_vel - self.vel_goal) * s), dim=-1
        )
        obs_dict["velocity_phase"] = torch.stack(
            ((self.goal_kind == LEADIN).float(), (self.goal_kind == STRIKE).float()), dim=-1
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Head kinematics
    # ------------------------------------------------------------------------------------------------------------------
    def populate_sim_buffers(self):
        out = super().populate_sim_buffers()
        if not self._vel_ready():
            return out
        r_w = quat_rotate_xyzw(self.object_rot, self.head_offset)
        self.head_pos = self.object_pos + r_w
        # Analytic estimate, used only right after a reset (no previous position) and logged as a cross-check: the
        # root linear velocity may refer to the center of mass rather than the frame origin, so it is not the primary.
        self.head_vel_analytic = self.object_linvel + torch.cross(self.object_angvel, r_w, dim=-1)
        fd = (self.head_pos - self._head_prev_pos) / self.control_dt
        self.head_vel = torch.where(self._head_prev_valid.unsqueeze(-1), fd, self.head_vel_analytic)
        self._head_prev_pos = self.head_pos.clone()
        self._head_prev_valid[:] = True
        return out

    # ------------------------------------------------------------------------------------------------------------------
    # Arm speed
    # ------------------------------------------------------------------------------------------------------------------
    @property
    def hand_dof_speed_scale(self) -> float:
        # Despite its name this is the arm joint-target speed (pre_physics_step: target += scale * dt * a[:, :7]).
        # The arm moves alpha * scale * dt per step, about 0.15 rad/s at the released settings, too slow for
        # 0.3-0.6 m/s strikes; the velocity curriculum multiplies it (armSpeedMult).
        return SimToolReal.hand_dof_speed_scale.fget(self) * getattr(self, "_vel_arm_mult", 1.0)

    # ------------------------------------------------------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------------------------------------------------------
    def _write_goal_actor(self, ids: Tensor):
        gi = self.goal_object_indices[ids]
        self.root_state_tensor[gi, 0:7] = self.goal_states[ids, 0:7]
        self.root_state_tensor[gi, 7:13] = 0.0
        self.deferred_set_actor_root_state_tensor_indexed([gi])

    def _set_static(self, ids: Tensor):
        self.goal_kind[ids] = STATIC
        self.vel_goal[ids] = 0.0
        self.vel_goal_is_down[ids] = False
        self.dense_paid[ids] = 0.0

    def _maybe_make_velocity_goals(self, ids: Tensor):
        draw = torch.rand(len(ids), device=self.device) < self.vel_goal_prob
        sel = ids[draw]
        if len(sel) == 0:
            return
        params = VelocityGoalParams(
            cone_deg=self.vel_cur["cone_deg"],
            speed_lo=self.vel_cur["speed_lo"],
            speed_hi=self.vel_cur["speed_hi"],
            **self.vel_geom,
        )
        out = sample_velocity_goals(
            strike_pos=self.goal_states[sel, 0:3],
            strike_quat=self.goal_states[sel, 3:7],
            head_offset=self.head_offset[sel],
            head_axis=self.head_axis[sel],
            head_axis_symmetric=self.head_axis_symmetric[sel],
            volume_min=self._vel_volume[0],
            volume_max=self._vel_volume[1],
            params=params,
        )
        feas = out["feasible"]
        self._acc[ACC["vel_goals_sampled"]] += len(sel)
        self._acc[ACC["vel_goals_infeasible"]] += (~feas).sum()
        ok = sel[feas]
        if len(ok) == 0:
            return
        quat = out["quat"][feas]
        self.vel_goal[ok] = out["v_goal"][feas]
        self.vel_goal_is_down[ok] = out["is_down"][feas]
        self.dense_paid[ok] = 0.0
        if self.vel_use_lead_in:
            self.pending_strike_pose[ok, 0:3] = self.goal_states[ok, 0:3]
            self.pending_strike_pose[ok, 3:7] = quat
            self.goal_kind[ok] = LEADIN
            self.goal_states[ok, 0:3] = out["leadin_pos"][feas]
        else:
            # The sampled pose is the strike; down strikes may have turned the orientation.
            self.goal_kind[ok] = STRIKE
        self.goal_states[ok, 3:7] = quat
        self._write_goal_actor(ok)

    def _start_strike(self, ids: Tensor):
        self.goal_states[ids, 0:7] = self.pending_strike_pose[ids]
        self.goal_kind[ids] = STRIKE
        self.dense_paid[ids] = 0.0
        self._write_goal_actor(ids)
        # Same per-goal bookkeeping as SimToolReal.reset_target_pose (the keypoint reward restarts from here).
        self.reset_goal_buf[ids] = 0
        self.near_goal_steps[ids] = 0
        self.closest_keypoint_max_dist[ids] = -1
        self.closest_keypoint_max_dist_fixed_size[ids] = -1

    def reset_target_pose(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True, is_first_goal=True) -> None:
        if (
            not self._vel_ready()
            or not self.vel_enabled
            or not tensor_reset
            or reset_buf_idxs is not None
            or len(env_ids) == 0
        ):
            super().reset_target_pose(env_ids, reset_buf_idxs, tensor_reset=tensor_reset, is_first_goal=is_first_goal)
            if self._vel_ready() and tensor_reset and len(env_ids) > 0:
                self._set_static(env_ids.long())
            return

        env_ids = env_ids.long()
        if is_first_goal:
            to_strike_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        else:
            to_strike_mask = self.goal_kind[env_ids] == LEADIN
        fresh = env_ids[~to_strike_mask]
        to_strike = env_ids[to_strike_mask]

        if len(fresh) > 0:
            # Official sampling (delta goal from the previous goal, or the first goal of an episode).
            super().reset_target_pose(fresh, None, tensor_reset=True, is_first_goal=is_first_goal)
            self._set_static(fresh)
            if not is_first_goal or self.vel_on_first_goal:
                self._maybe_make_velocity_goals(fresh)
        if len(to_strike) > 0:
            self._start_strike(to_strike)

    def reset_idx(self, env_ids: Tensor, reset_buf_idxs=None, episode_reset=True, tensor_reset=True) -> None:
        if self._vel_ready() and len(env_ids) > 0 and tensor_reset:
            ids = env_ids.long()
            kind = self.goal_kind[ids]
            done_this_step = self.reset_goal_buf[ids] > 0
            # progress 0 only happens at the very first reset of the run (all envs start with reset_buf = 1).
            alive = self.progress_buf[ids] > 0
            # Velocity goal cut short: still in the lead-in, or in the strike without an attempt.
            vel_fail = alive & ((kind == LEADIN) | ((kind == STRIKE) & ~done_this_step))
            # Static goal cut short (not the extra goal sampled after the last allowed success).
            static_fail = (
                alive
                & (kind == STATIC)
                & ~done_this_step
                & (self.successes[ids] < self.max_consecutive_successes)
            )
            nv = vel_fail.sum()
            ns = static_fail.sum()
            self._acc[ACC["vel_fail_no_attempt"]] += nv
            self._acc[ACC["static_fail"]] += ns
            self._block[1] += nv
            self._block[3] += ns
            self._head_prev_valid[ids] = False
        super().reset_idx(env_ids, reset_buf_idxs, episode_reset=episode_reset, tensor_reset=tensor_reset)

    # ------------------------------------------------------------------------------------------------------------------
    # Reward (SimToolReal.compute_kuka_reward with the velocity-goal changes marked VELOCITY)
    # ------------------------------------------------------------------------------------------------------------------
    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        # With only static goals this is exactly SimToolReal.compute_kuka_reward.
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        keypoint_rew, keypoint_rew_fixed_size = self._keypoint_reward(lifted_object)
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            keypoint_rew = keypoint_rew_fixed_size

        keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale
        kp_dist = (
            self.keypoints_max_dist_fixed_size if self.cfg["env"]["fixedSizeKeypointReward"] else self.keypoints_max_dist
        )

        # VELOCITY: goal kinds
        is_static = self.goal_kind == STATIC
        is_lead = self.goal_kind == LEADIN
        is_strike = self.goal_kind == STRIKE

        # Static goals: the official hold rule, unchanged.
        near_goal: Tensor = (kp_dist <= keypoint_success_tolerance) & is_static
        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            self.near_goal_steps = (self.near_goal_steps + near_goal) * near_goal
        else:
            self.near_goal_steps += near_goal
        static_success = (self.near_goal_steps >= self.success_steps) & is_static

        # VELOCITY: lead-in reached / strike attempt
        lead_reached = is_lead & (kp_dist <= self.vel_leadin_tol)
        attempt = is_strike & (kp_dist <= self.vel_cur["eps_kp"])
        q_v, dv, speed = velocity_match(self.head_vel, self.vel_goal, self.vel_cur["sigma_frac"])
        vel_success = attempt & (dv < self.vel_cur["eps_v_frac"] * speed)

        is_success = static_success | attempt  # goals completed (counted in self.successes)
        goal_advance = is_success | lead_reached  # goals that move on this step
        self.successes += is_success
        self.reset_goal_buf[:] = goal_advance

        object_lin_vel_penalty = -torch.sum(torch.square(self.object_linvel), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_angvel), dim=-1)

        self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        self.rewards_episode["raw_hand_delta_penalty"] += hand_delta_penalty
        self.rewards_episode["raw_lifting_rew"] += lifting_rew
        self.rewards_episode["raw_keypoint_rew"] += keypoint_rew
        self.rewards_episode["raw_object_lin_vel_penalty"] += object_lin_vel_penalty
        self.rewards_episode["raw_object_ang_vel_penalty"] += object_ang_vel_penalty

        fingertip_delta_rew *= self.distance_delta_rew_scale
        hand_delta_penalty *= self.distance_delta_rew_scale * 0  # currently disabled (as in SimToolReal)
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale
        object_lin_vel_penalty *= self.object_lin_vel_penalty_scale
        object_ang_vel_penalty *= self.object_ang_vel_penalty_scale

        kuka_actions_penalty, hand_actions_penalty = self._action_penalties()

        bonus_static = near_goal * (self.reach_goal_bonus / self.success_steps)
        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            bonus_static = static_success * self.reach_goal_bonus
        # VELOCITY: paid once, on the attempt
        bonus_strike = attempt * (self.reach_goal_bonus * q_v)
        # VELOCITY v2: dense tracking on the strike steps before the attempt, capped per goal
        dense_rew = torch.zeros_like(q_v)
        if self.vel_dense_weight > 0:
            budget = self.vel_dense_budget_frac * self.reach_goal_bonus
            want = self.vel_dense_weight * q_v * (is_strike & ~attempt)
            dense_rew = torch.minimum(want, (budget - self.dense_paid).clamp_min(0.0))
            self.dense_paid += dense_rew
        bonus_rew = bonus_static + bonus_strike + dense_rew

        reward = (
            fingertip_delta_rew
            + hand_delta_penalty
            + lifting_rew
            + lift_bonus_rew
            + keypoint_rew
            + kuka_actions_penalty
            + hand_actions_penalty
            + bonus_rew
            + object_lin_vel_penalty
            + object_ang_vel_penalty
        )
        self.rew_buf[:] = reward

        # VELOCITY: the per-goal clock also restarts when the lead-in is reached
        resets = self._compute_resets(goal_advance)
        self.reset_buf[:] = resets
        if self.cfg["env"]["forceNoReset"]:
            self.reset_buf[:] = False

        self.extras["successes"] = self.prev_episode_successes
        self.extras["success_ratio"] = self.prev_episode_successes.mean().item() / self.max_consecutive_successes
        self.extras["closest_keypoint_max_dist"] = self.prev_episode_closest_keypoint_max_dist
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (hand_delta_penalty, "hand_delta_penalty"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (kuka_actions_penalty, "kuka_actions_penalty"),
            (hand_actions_penalty, "hand_actions_penalty"),
            (bonus_rew, "bonus_rew"),
            (object_lin_vel_penalty, "object_lin_vel_penalty"),
            (object_ang_vel_penalty, "object_ang_vel_penalty"),
            (reward, "total_reward"),
        ]
        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        self._vel_accumulate(
            is_static, is_lead, is_strike, keypoint_rew, bonus_static, bonus_strike, static_success, lead_reached,
            attempt, vel_success, q_v, dv, speed, kp_dist, dense_rew,
        )
        self._vel_update_curriculum_and_stats()
        return self.rew_buf, is_success

    # ------------------------------------------------------------------------------------------------------------------
    # Diagnostics and curriculum
    # ------------------------------------------------------------------------------------------------------------------
    def _vel_accumulate(self, is_static, is_lead, is_strike, keypoint_rew, bonus_static, bonus_strike, static_success,
                        lead_reached, attempt, vel_success, q_v, dv, speed, kp_dist, dense_rew):
        a = self._acc
        af = attempt.float()
        spd = speed.clamp_min(1e-6)
        head_speed = self.head_vel.norm(dim=-1)
        goal_head = self.goal_pos + quat_rotate_xyzw(self.goal_rot, self.head_offset)
        head_err = (self.head_pos - goal_head).norm(dim=-1)
        cos = (self.head_vel * self.vel_goal).sum(-1) / (head_speed.clamp_min(1e-6) * spd)
        down = attempt & self.vel_goal_is_down
        d_hat = self.vel_goal / spd.unsqueeze(-1)
        err = self.head_pos - goal_head
        along = (err * d_hat).sum(-1)
        lateral = (err - along.unsqueeze(-1) * d_hat).norm(dim=-1)
        v_par = (self.head_vel * d_hat).sum(-1)
        v_perp = (self.head_vel - v_par.unsqueeze(-1) * d_hat).norm(dim=-1)
        # Same quantities at the lead-in, measured against the strike line (the strike pose is pending there).
        lf = lead_reached.float()
        strike_head = self.pending_strike_pose[:, 0:3] + quat_rotate_xyzw(
            self.pending_strike_pose[:, 3:7], self.head_offset
        )
        err_l = self.head_pos - strike_head
        along_l = (err_l * d_hat).sum(-1)
        lateral_l = (err_l - along_l.unsqueeze(-1) * d_hat).norm(dim=-1)
        vel_steps = is_lead | is_strike
        vals = torch.stack(
            [
                torch.zeros((), device=self.device),  # vel_goals_sampled (counted at sampling)
                torch.zeros((), device=self.device),  # vel_goals_infeasible
                lead_reached.float().sum(),
                af.sum(),
                vel_success.float().sum(),
                torch.zeros((), device=self.device),  # vel_fail_no_attempt (counted at reset)
                (q_v * af).sum(),
                (dv / spd * af).sum(),
                (head_speed / spd * af).sum(),
                (cos * af).sum(),
                (head_err * af).sum(),
                (kp_dist * af).sum(),
                down.float().sum(),
                (down & vel_success).float().sum(),
                static_success.float().sum(),
                torch.zeros((), device=self.device),  # static_fail (counted at reset)
                is_static.float().sum(),
                is_lead.float().sum(),
                is_strike.float().sum(),
                (keypoint_rew * is_static).sum(),
                (keypoint_rew * is_lead).sum(),
                (keypoint_rew * is_strike).sum(),
                bonus_static.sum(),
                bonus_strike.sum(),
                (head_speed * vel_steps).sum(),
                ((self.head_vel - self.head_vel_analytic).norm(dim=-1) / spd * af).sum(),
                (lateral * af).sum(),
                (along * af).sum(),
                (v_par / spd * af).sum(),
                (v_perp / spd * af).sum(),
                dense_rew.sum(),
                (lateral_l * lf).sum(),
                (along_l * lf).sum(),
                (v_par / spd * lf).sum(),
                (v_perp / spd * lf).sum(),
            ]
        )
        a += vals
        self._block += torch.stack(
            [vel_success.float().sum(), af.sum(), static_success.float().sum(), static_success.float().sum()]
        )
        # vel_success / vel_total / static_success / static_total; the failures are added in reset_idx.

    def _vel_update_curriculum_and_stats(self):
        self._vel_step_count += 1
        if self._vel_step_count % self.vel_stats_every == 0:
            a = self._acc.cpu().tolist()
            g = {k: a[i] for k, i in ACC.items()}
            n_att = max(g["attempts"], 1.0)
            n_vel_end = g["attempts"] + g["vel_fail_no_attempt"]
            n_static_end = g["static_success"] + g["static_fail"]
            st = {
                "vel/level": self.vel_curriculum.level,
                "vel/sigma_frac": self.vel_cur["sigma_frac"],
                "vel/eps_v_frac": self.vel_cur["eps_v_frac"],
                "vel/eps_kp": self.vel_cur["eps_kp"],
                "vel/cone_deg": self.vel_cur["cone_deg"],
                "vel/speed_hi": self.vel_cur["speed_hi"],
                "vel/arm_speed_mult": self._vel_arm_mult,
                "vel/infeasible_frac": g["vel_goals_infeasible"] / max(g["vel_goals_sampled"], 1.0),
                "vel/attempt_rate": g["attempts"] / max(n_vel_end, 1.0),
                "vel/success_rate": g["vel_success"] / max(n_vel_end, 1.0),
                "vel/success_given_attempt": g["vel_success"] / n_att,
                "vel/q_v": g["sum_qv"] / n_att,
                "vel/dv_rel": g["sum_dv_rel"] / n_att,
                "vel/speed_ratio": g["sum_speed_ratio"] / n_att,
                "vel/cos_dir": g["sum_cos"] / n_att,
                "vel/head_err_at_attempt": g["sum_head_err"] / n_att,
                "vel/kp_dist_at_attempt": g["sum_attempt_kp"] / n_att,
                "vel/down_success_given_attempt": g["down_success"] / max(g["down_attempts"], 1.0),
                "vel/down_frac_of_attempts": g["down_attempts"] / n_att,
                "vel/leadin_reached": g["leadin_reached"],
                "vel/attempts": g["attempts"],
                "vel/fd_vs_analytic_rel": g["sum_fd_vs_analytic"] / n_att,
                "vel/lateral_at_attempt": g["sum_lateral"] / n_att,
                "vel/along_at_attempt": g["sum_along"] / n_att,
                "vel/v_par_rel": g["sum_v_par_rel"] / n_att,
                "vel/v_perp_rel": g["sum_v_perp_rel"] / n_att,
                "bonus/dense_per_strike_step": g["dense_rew"] / max(g["steps_strike"], 1.0),
                "vel/lateral_at_leadin": g["sum_lead_lateral"] / max(g["leadin_reached"], 1.0),
                "vel/along_at_leadin": g["sum_lead_along"] / max(g["leadin_reached"], 1.0),
                "vel/v_par_rel_at_leadin": g["sum_lead_v_par_rel"] / max(g["leadin_reached"], 1.0),
                "vel/v_perp_rel_at_leadin": g["sum_lead_v_perp_rel"] / max(g["leadin_reached"], 1.0),
                "vel/head_speed_on_vel_goals": g["sum_head_speed_vel"] / max(g["steps_leadin"] + g["steps_strike"], 1.0),
                "static/success_rate": g["static_success"] / max(n_static_end, 1.0),
                "static/successes": g["static_success"],
                "kp_rew/static_per_step": g["kp_rew_static"] / max(g["steps_static"], 1.0),
                "kp_rew/leadin_per_step": g["kp_rew_leadin"] / max(g["steps_leadin"], 1.0),
                "kp_rew/strike_per_step": g["kp_rew_strike"] / max(g["steps_strike"], 1.0),
                "bonus/static_per_step": g["bonus_static"] / max(g["steps_static"], 1.0),
                "bonus/strike_per_attempt": g["bonus_strike"] / n_att,
                "vel/frac_steps_on_vel_goals": (g["steps_leadin"] + g["steps_strike"])
                / max(g["steps_static"] + g["steps_leadin"] + g["steps_strike"], 1.0),
            }
            self._vel_stats = st
            self._acc.zero_()

            b = self._block.cpu().tolist()
            decision = self.vel_curriculum.decide(b[0], b[1], b[2], b[3], self._vel_step_count)
            if decision is not None:
                print(
                    f"[velocity] curriculum block: vel success {b[0]:.0f}/{b[1]:.0f}, static {b[2]:.0f}/{b[3]:.0f} "
                    f"-> {decision}, level {self.vel_curriculum.level:.2f}"
                )
                self._block.zero_()
                if decision in ("up", "down"):
                    self._vel_apply_level()

            if self._vel_step_count % self.vel_print_every == 0:
                print(
                    f"[velocity] step {self._vel_step_count} level {st['vel/level']:.2f} arm x{st['vel/arm_speed_mult']:.2f} | "
                    f"vel succ {st['vel/success_rate']:.2f} attempt {st['vel/attempt_rate']:.2f} q_v {st['vel/q_v']:.2f} "
                    f"|dv|/|vg| {st['vel/dv_rel']:.2f} speed ratio {st['vel/speed_ratio']:.2f} cos {st['vel/cos_dir']:.2f} "
                    f"head err {st['vel/head_err_at_attempt'] * 100:.1f} cm | static succ {st['static/success_rate']:.2f} | "
                    f"infeasible {st['vel/infeasible_frac']:.2f} | fd-analytic {st['vel/fd_vs_analytic_rel']:.2f} | "
                    f"v_par {st['vel/v_par_rel']:.2f} v_perp {st['vel/v_perp_rel']:.2f} "
                    f"lateral {st['vel/lateral_at_attempt'] * 100:.1f} cm along {st['vel/along_at_attempt'] * 100:.1f} cm | "
                    f"dense/step {st['bonus/dense_per_strike_step']:.2f} | at lead-in: "
                    f"lateral {st['vel/lateral_at_leadin'] * 100:.1f} cm along {st['vel/along_at_leadin'] * 100:.1f} cm "
                    f"v_par {st['vel/v_par_rel_at_leadin']:.2f} v_perp {st['vel/v_perp_rel_at_leadin']:.2f}"
                )
        self.extras.update(self._vel_stats)

    # ------------------------------------------------------------------------------------------------------------------
    # Checkpointed env state (resume keeps the curriculum level)
    # ------------------------------------------------------------------------------------------------------------------
    def get_env_state(self):
        state = super().get_env_state()
        state["vel_level"] = float(self.vel_curriculum.level)
        return state

    def set_env_state(self, env_state):
        level = None
        if env_state is not None:
            level = env_state.pop("vel_level", None)
        super().set_env_state(env_state)
        if level is not None and not self.vel_curriculum.fixed:
            self.vel_curriculum.level = float(level)
            self._vel_apply_level()
            print(f"[velocity] restored curriculum level {level:.2f}")
