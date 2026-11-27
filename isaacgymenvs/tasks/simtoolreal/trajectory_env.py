"""SimToolReal with smooth trajectory goals (pose + head velocity per waypoint), for fine-tuning.

Builds on SimToolRealVelocity (head kinematics, velocity observations, arm-speed multiplier, curriculum class, the
two observation hooks) and replaces its goal logic. Design and rationale: finetune_velocity/README section 14.

Goal sequence of an episode: the official first goal (pick-up), then blocks:
  static block (prob 1 - motionBlockProb): one official delta goal.
  motion block: SET (the set pose, an official delta goal, possibly turned for a downward start) -> MOVE waypoints
                along a smooth head path, each with a head velocity -> END (the end stop).
SET, END and static blocks are hold goals with the official rule (successTolerance * keypointScale for successSteps
steps, reachGoalBonus / successSteps per near step). A MOVE waypoint is passed when it is reached (keypoint distance
<= eps_kp) or crossed (the head is past its normal plane); passing pays B * q_pos * q_v once, with B = passBudget *
reachGoalBonus / (number of moving waypoints of the block). If the head strays more than abandonDist from the current
waypoint, the rest of the block is abandoned and the goal jumps to its end stop. Only hold goals count as successes
(the official "at most maxConsecutiveSuccesses goals per episode" keeps its meaning).
"""

from typing import Tuple

# isort: off
from isaacgym import gymapi  # noqa: F401  (must precede torch)
import torch
from torch import Tensor
# isort: on

from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
from isaacgymenvs.tasks.simtoolreal.trajectory_goals import PassParams, TrajParams, sample_motion_chunks, waypoint_pass
from isaacgymenvs.tasks.simtoolreal.velocity_env import STATIC, STRIKE, SimToolRealVelocity
from isaacgymenvs.tasks.simtoolreal.velocity_goals import VelocityCurriculum, lerp, quat_rotate_xyzw

PLAIN, SET, MOVE, END = 0, 1, 2, 3

T_ACC_NAMES = [
    "blocks_sampled", "blocks_infeasible", "blocks_started", "blocks_completed", "blocks_failed", "abandons",
    "passes", "reached", "crossed", "pass_success", "sum_qpos", "sum_qv", "sum_dv_rel", "sum_v_par_rel",
    "sum_v_perp_rel", "sum_lateral", "sum_along", "sum_kp_at_pass", "sum_head_speed_move", "sum_goal_speed_move",
    "static_success", "static_fail", "steps_hold", "steps_move", "kp_rew_hold", "kp_rew_move", "bonus_hold",
    "bonus_move", "sum_fd_vs_analytic",
]
T_ACC = {k: i for i, k in enumerate(T_ACC_NAMES)}


class SimToolRealTrajectory(SimToolRealVelocity):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        tcfg = dict(cfg["env"].get("trajectory", {}) or {})
        self.tcfg = tcfg
        self.traj_motion_prob = float(tcfg.get("motionBlockProb", 0.6))
        self.traj_pass_budget = float(tcfg.get("passBudget", 1.0))
        self.traj_ranges = {
            "v_max": tcfg.get("vMax", [0.35, 0.6]),
            "bend_deg": tcfg.get("bendDeg", [0.0, 90.0]),
            "rot_max_deg": tcfg.get("rotMaxDeg", [0.0, 150.0]),
            "n_seg_max": tcfg.get("nSegMax", [1, 6]),
            "cone_deg": tcfg.get("coneDeg", [30.0, 180.0]),
            "p_arc": tcfg.get("pArc", [0.0, 0.4]),
            "arc_max_deg": tcfg.get("arcMaxDeg", [0.0, 180.0]),
            "eps_kp": tcfg.get("epsKp", [0.06, 0.01]),
            "sigma_pos": tcfg.get("sigmaPos", [0.05, 0.01]),
            "sigma_v": tcfg.get("sigmaV", [1.0, 0.15]),
            "eps_v": tcfg.get("epsV", [0.6, 0.15]),
            "arm_speed_mult": tcfg.get("armSpeedMult", [1.5, 6.0]),
        }
        self.traj_fixed = dict(
            a_max=float(tcfg.get("aMax", 3.0)),
            spacing=float(tcfg.get("spacing", 0.03)),
            k_max=int(tcfg.get("kMax", 48)),
            down_frac=float(tcfg.get("downFrac", 0.25)),
            volume_margin=float(tcfg.get("volumeMargin", 0.10)),
            min_head_z=float(tcfg.get("minHeadZ", 0.58)),
            tries=int(tcfg.get("tries", 8)),
        )
        self.traj_v_floor = float(tcfg.get("vFloor", 0.1))
        self.traj_abandon = float(tcfg.get("abandonDist", 0.15))
        cur = dict(tcfg.get("curriculum", {}) or {})
        fixed_level = cur.get("fixedLevel", None)
        self._traj_curriculum = VelocityCurriculum(
            level=float(fixed_level if fixed_level is not None else cur.get("initLevel", 0.0)),
            step=float(cur.get("step", 0.05)),
            up_threshold=float(cur.get("upThreshold", 0.5)),
            down_threshold=float(cur.get("downThreshold", 0.2)),
            min_samples=int(cur.get("minSamples", 4096)),
            static_guard=float(cur.get("staticGuard", 0.5)),
            min_steps_between=int(cur.get("minStepsBetween", 160)),
            fixed=fixed_level is not None,
        )
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        # One curriculum for everything (the velocity env's own block/curriculum is not used here).
        self.vel_curriculum = self._traj_curriculum
        self._vel_apply_level()

    # ------------------------------------------------------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------------------------------------------------------
    def _vel_apply_level(self):
        if not hasattr(self, "traj_ranges"):  # called by SimToolRealVelocity.__init__ before our attributes exist
            return super()._vel_apply_level()
        lv = self._traj_curriculum.level
        c = {k: lerp(v, lv) for k, v in self.traj_ranges.items()}
        self.traj_cur = c
        self.traj_params = TrajParams(
            v_max=c["v_max"], bend_deg=c["bend_deg"], rot_max_deg=c["rot_max_deg"],
            n_seg_max=max(1, int(round(c["n_seg_max"]))), cone_deg=c["cone_deg"], p_arc=c["p_arc"],
            arc_max_deg=c["arc_max_deg"], **self.traj_fixed,
        )
        self.pass_params = PassParams(eps_kp=c["eps_kp"], sigma_pos=c["sigma_pos"], sigma_v=c["sigma_v"],
                                      eps_v=c["eps_v"], v_floor=self.traj_v_floor, abandon_dist=self.traj_abandon)
        self._vel_arm_mult = c["arm_speed_mult"]
        # the velocity env's own dict (used in its print / stats code paths) stays defined
        self.vel_cur = {k: lerp(v, lv) for k, v in self.vel_ranges.items()}

    def _vel_alloc(self):
        super()._vel_alloc()
        n, dev, K = self.num_envs, self.device, self.traj_fixed["k_max"]
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.traj_pos = torch.zeros(n, K, 3, device=dev)
        self.traj_quat = torch.zeros(n, K, 4, device=dev)
        self.traj_quat[..., 3] = 1.0
        self.traj_vel = torch.zeros(n, K, 3, device=dev)
        self.traj_tan = torch.zeros(n, K, 3, device=dev)
        self.traj_tan[..., 0] = 1.0
        self.traj_head = torch.zeros(n, K, 3, device=dev)
        self.traj_n = torch.ones(n, dtype=torch.long, device=dev)
        self.traj_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.abandon_flag = torch.zeros(n, dtype=torch.bool, device=dev)
        self._tacc = torch.zeros(len(T_ACC_NAMES), device=dev)
        self._tblock = torch.zeros(4, device=dev)  # pass_success, pass_total (+abandons), static_success, static_total
        print(f"[trajectory] motion block prob {self.traj_motion_prob}, level {self._traj_curriculum.level:.2f}, "
              f"k_max {K}, enabled {self.vel_enabled}")

    def _traj_ready(self) -> bool:
        return hasattr(self, "phase")

    # ------------------------------------------------------------------------------------------------------------------
    # Observations: + goal_lookahead (next waypoint: keypoints rel. to the object's, and its velocity), traj_progress
    # ------------------------------------------------------------------------------------------------------------------
    def _extra_obs_type_sizes(self) -> dict:
        sizes = super()._extra_obs_type_sizes()
        sizes.update({"goal_lookahead": 3 * self.num_keypoints + 3, "traj_progress": 1})
        return sizes

    def _next_waypoint_index(self) -> Tensor:
        nxt = torch.where(self.phase == SET, torch.zeros_like(self.traj_idx), self.traj_idx + 1)
        return torch.minimum(nxt, self.traj_n - 1)

    def _add_extra_obs(self, obs_dict: dict) -> None:
        super()._add_extra_obs(obs_dict)
        n = self.num_envs
        if not self._traj_ready():
            obs_dict["goal_lookahead"] = torch.zeros(n, 3 * self.num_keypoints + 3, device=self.device)
            obs_dict["traj_progress"] = torch.zeros(n, 1, device=self.device)
            return
        rows = torch.arange(n, device=self.device)
        in_block = (self.phase == SET) | (self.phase == MOVE)
        nidx = self._next_waypoint_index()
        n_pos = torch.where(in_block.unsqueeze(-1), self.traj_pos[rows, nidx], self.goal_states[:, 0:3])
        n_quat = torch.where(in_block.unsqueeze(-1), self.traj_quat[rows, nidx], self.goal_states[:, 3:7])
        n_vel = torch.where(in_block.unsqueeze(-1), self.traj_vel[rows, nidx], torch.zeros_like(self.vel_goal))
        rel = []
        for i in range(self.num_keypoints):
            kp = n_pos + quat_rotate_xyzw(n_quat, self.object_keypoint_offsets[:, i] * self.object_scale_noise_multiplier)
            rel.append(self.obj_keypoint_pos[:, i] - kp)
        obs_dict["goal_lookahead"] = torch.cat(rel + [n_vel / self.vel_obs_scale], dim=-1)
        n_moving = (self.traj_n - 1).clamp(min=1).float()
        remaining = torch.where(
            self.phase == SET, torch.ones(n, device=self.device),
            torch.where(self.phase == MOVE, (self.traj_n - 1 - self.traj_idx).float() / n_moving,
                        torch.zeros(n, device=self.device)),
        )
        obs_dict["traj_progress"] = remaining.unsqueeze(-1)

    # ------------------------------------------------------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------------------------------------------------------
    def _set_goal(self, ids: Tensor, pos: Tensor, quat: Tensor, vel: Tensor, phase: int):
        self.goal_states[ids, 0:3] = pos
        self.goal_states[ids, 3:7] = quat
        self.vel_goal[ids] = vel
        self.phase[ids] = phase
        self.goal_kind[ids] = STRIKE if phase == MOVE else STATIC  # velocity_phase obs: (0, moving)
        self._write_goal_actor(ids)
        # same per-goal bookkeeping as SimToolReal.reset_target_pose (the keypoint reward restarts from here)
        self.reset_goal_buf[ids] = 0
        self.near_goal_steps[ids] = 0
        self.closest_keypoint_max_dist[ids] = -1
        self.closest_keypoint_max_dist_fixed_size[ids] = -1

    def _go_to_waypoint(self, ids: Tensor, idx: Tensor):
        """Current goal = waypoint idx of the block (the end stop is a hold goal)."""
        self.traj_idx[ids] = idx
        is_end = idx >= self.traj_n[ids] - 1
        for mask, phase in ((~is_end, MOVE), (is_end, END)):
            sel = ids[mask]
            if len(sel) == 0:
                continue
            i = idx[mask]
            vel = self.traj_vel[sel, i] if phase == MOVE else torch.zeros(len(sel), 3, device=self.device)
            self._set_goal(sel, self.traj_pos[sel, i], self.traj_quat[sel, i], vel, phase)

    def _maybe_start_blocks(self, ids: Tensor):
        """ids just got an official delta goal: turn some into motion blocks (that goal becomes the set pose)."""
        sel = ids[torch.rand(len(ids), device=self.device) < self.traj_motion_prob]
        if len(sel) == 0:
            return
        out = sample_motion_chunks(
            self.goal_states[sel, 0:3], self.goal_states[sel, 3:7], self.head_offset[sel], self.head_axis[sel],
            self.head_axis_symmetric[sel], self._vel_volume[0], self._vel_volume[1], self.traj_params,
        )
        feas = out["feasible"]
        self._tacc[T_ACC["blocks_sampled"]] += len(sel)
        self._tacc[T_ACC["blocks_infeasible"]] += (~feas).sum()
        ok = sel[feas]
        if len(ok) == 0:
            return
        for name, buf in (("wp_pos", self.traj_pos), ("wp_quat", self.traj_quat), ("wp_vel", self.traj_vel),
                          ("wp_tan", self.traj_tan), ("wp_head", self.traj_head)):
            buf[ok] = out[name][feas]
        self.traj_n[ok] = out["n_wp"][feas]
        self.traj_idx[ok] = 0
        self.phase[ok] = SET
        # the set pose may have been turned so that the head faces a downward start direction
        self.goal_states[ok, 3:7] = out["set_quat"][feas]
        self._write_goal_actor(ok)

    def reset_target_pose(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True, is_first_goal=True) -> None:
        plain = (
            not self._traj_ready() or not self.vel_enabled or not tensor_reset or reset_buf_idxs is not None
            or len(env_ids) == 0
        )
        if plain:
            SimToolReal.reset_target_pose(self, env_ids, reset_buf_idxs, tensor_reset=tensor_reset,
                                          is_first_goal=is_first_goal)
            if self._traj_ready() and tensor_reset and len(env_ids) > 0:
                ids = env_ids.long()
                self._set_static(ids)
                self.phase[ids] = PLAIN
            return

        ids = env_ids.long()
        if is_first_goal:
            fresh, move, set_done = ids, ids[:0], ids[:0]
        else:
            ph = self.phase[ids]
            move, set_done = ids[ph == MOVE], ids[ph == SET]
            fresh = ids[(ph == PLAIN) | (ph == END)]
        if len(move) > 0:
            nxt = torch.where(self.abandon_flag[move], self.traj_n[move] - 1, self.traj_idx[move] + 1)
            self.abandon_flag[move] = False
            self._go_to_waypoint(move, nxt)
        if len(set_done) > 0:
            self._tacc[T_ACC["blocks_started"]] += len(set_done)
            self._go_to_waypoint(set_done, torch.zeros_like(set_done))
        if len(fresh) > 0:
            SimToolReal.reset_target_pose(self, fresh, None, tensor_reset=True, is_first_goal=is_first_goal)
            self._set_static(fresh)
            self.phase[fresh] = PLAIN
            if not is_first_goal:
                self._maybe_start_blocks(fresh)

    def reset_idx(self, env_ids: Tensor, reset_buf_idxs=None, episode_reset=True, tensor_reset=True) -> None:
        if self._traj_ready() and len(env_ids) > 0 and tensor_reset:
            ids = env_ids.long()
            ph = self.phase[ids]
            done = self.reset_goal_buf[ids] > 0
            alive = self.progress_buf[ids] > 0
            hold = ph != MOVE
            static_fail = alive & hold & ~done & (self.successes[ids] < self.max_consecutive_successes)
            block_fail = alive & ((ph == SET) | (ph == MOVE) | ((ph == END) & ~done))
            move_fail = alive & (ph == MOVE) & ~done  # the waypoint in progress counts as missed
            self._tacc[T_ACC["static_fail"]] += static_fail.sum()
            self._tacc[T_ACC["blocks_failed"]] += block_fail.sum()
            self._tblock[1] += move_fail.sum()
            self._tblock[3] += static_fail.sum()
            self._head_prev_valid[ids] = False
            self.abandon_flag[ids] = False
        # bypass SimToolRealVelocity.reset_idx (its statistics are for v1/v2 goals)
        SimToolReal.reset_idx(self, env_ids, reset_buf_idxs, episode_reset=episode_reset, tensor_reset=tensor_reset)

    # ------------------------------------------------------------------------------------------------------------------
    # Reward (SimToolReal.compute_kuka_reward with the trajectory changes marked TRAJ)
    # ------------------------------------------------------------------------------------------------------------------
    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        keypoint_rew, keypoint_rew_fixed_size = self._keypoint_reward(lifted_object)
        if self.cfg["env"]["fixedSizeKeypointReward"]:
            keypoint_rew = keypoint_rew_fixed_size
        keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale
        kp_dist = (
            self.keypoints_max_dist_fixed_size if self.cfg["env"]["fixedSizeKeypointReward"] else self.keypoints_max_dist
        )

        # TRAJ: hold goals (static blocks, set poses, end stops) use the official rule, unchanged
        is_move = self.phase == MOVE
        is_hold = ~is_move
        near_goal: Tensor = (kp_dist <= keypoint_success_tolerance) & is_hold
        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            self.near_goal_steps = (self.near_goal_steps + near_goal) * near_goal
        else:
            self.near_goal_steps += near_goal
        static_success = (self.near_goal_steps >= self.success_steps) & is_hold

        # TRAJ: moving waypoints are passed (reached or crossed) or abandoned
        rows = torch.arange(self.num_envs, device=self.device)
        pp = waypoint_pass(kp_dist, self.head_pos, self.head_vel, self.traj_head[rows, self.traj_idx],
                           self.traj_tan[rows, self.traj_idx], self.vel_goal, self.pass_params)
        passed = pp["passed"] & is_move
        abandon = pp["abandon"] & is_move
        self.abandon_flag |= abandon
        n_moving = (self.traj_n - 1).clamp(min=1).float()
        bonus_move = passed * (self.traj_pass_budget * self.reach_goal_bonus / n_moving) * pp["q_pos"] * pp["q_v"]

        goal_advance = static_success | passed | abandon
        self.successes += static_success  # only hold goals count toward maxConsecutiveSuccesses
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

        bonus_hold = near_goal * (self.reach_goal_bonus / self.success_steps)
        if self.cfg["env"]["forceConsecutiveNearGoalSteps"]:
            bonus_hold = static_success * self.reach_goal_bonus
        bonus_rew = bonus_hold + bonus_move

        reward = (
            fingertip_delta_rew + hand_delta_penalty + lifting_rew + lift_bonus_rew + keypoint_rew
            + kuka_actions_penalty + hand_actions_penalty + bonus_rew + object_lin_vel_penalty + object_ang_vel_penalty
        )
        self.rew_buf[:] = reward

        # TRAJ: the per-goal clock restarts whenever the goal moves on
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
            (fingertip_delta_rew, "fingertip_delta_rew"), (hand_delta_penalty, "hand_delta_penalty"),
            (lifting_rew, "lifting_rew"), (lift_bonus_rew, "lift_bonus_rew"), (keypoint_rew, "keypoint_rew"),
            (kuka_actions_penalty, "kuka_actions_penalty"), (hand_actions_penalty, "hand_actions_penalty"),
            (bonus_rew, "bonus_rew"), (object_lin_vel_penalty, "object_lin_vel_penalty"),
            (object_ang_vel_penalty, "object_ang_vel_penalty"), (reward, "total_reward"),
        ]
        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        self._traj_accumulate(is_hold, is_move, keypoint_rew, bonus_hold, bonus_move, static_success, passed, abandon,
                              pp, kp_dist)
        self._traj_update_curriculum_and_stats()
        return self.rew_buf, static_success

    # ------------------------------------------------------------------------------------------------------------------
    # Diagnostics and curriculum
    # ------------------------------------------------------------------------------------------------------------------
    def _traj_accumulate(self, is_hold, is_move, keypoint_rew, bonus_hold, bonus_move, static_success, passed,
                         abandon, pp, kp_dist):
        pf = passed.float()
        zero = torch.zeros((), device=self.device)
        succ = passed & pp["success"]
        end_done = static_success & (self.phase == END)
        head_speed = self.head_vel.norm(dim=-1)
        goal_speed = self.vel_goal.norm(dim=-1)
        fd_rel = (self.head_vel - self.head_vel_analytic).norm(dim=-1) / goal_speed.clamp(min=self.traj_v_floor)
        vals = torch.stack([
            zero, zero, zero, end_done.float().sum(), zero, abandon.float().sum(),
            pf.sum(), (passed & pp["reached"]).float().sum(), (passed & pp["crossed"] & ~pp["reached"]).float().sum(),
            succ.float().sum(), (pp["q_pos"] * pf).sum(), (pp["q_v"] * pf).sum(), (pp["dv_rel"] * pf).sum(),
            (pp["v_par_rel"] * pf).sum(), (pp["v_perp_rel"] * pf).sum(), (pp["lateral"] * pf).sum(),
            (pp["along"] * pf).sum(), (kp_dist * pf).sum(), (head_speed * is_move).sum(), (goal_speed * is_move).sum(),
            static_success.float().sum(), zero, is_hold.float().sum(), is_move.float().sum(),
            (keypoint_rew * is_hold).sum(), (keypoint_rew * is_move).sum(), bonus_hold.sum(), bonus_move.sum(),
            (fd_rel * pf).sum(),
        ])
        self._tacc += vals
        # curriculum blocks: moving-waypoint success over passed + abandoned (+ missed at reset), static guard
        self._tblock += torch.stack([succ.float().sum(), pf.sum() + abandon.float().sum(),
                                     static_success.float().sum(), static_success.float().sum()])

    def _traj_update_curriculum_and_stats(self):
        self._vel_step_count += 1
        if self._vel_step_count % self.vel_stats_every == 0:
            a = self._tacc.cpu().tolist()
            g = {k: a[i] for k, i in T_ACC.items()}
            n_pass = max(g["passes"], 1.0)
            n_static_end = g["static_success"] + g["static_fail"]
            c = self.traj_cur
            st = {
                "traj/level": self._traj_curriculum.level,
                "traj/v_max": c["v_max"], "traj/eps_kp": c["eps_kp"], "traj/sigma_v": c["sigma_v"],
                "traj/eps_v": c["eps_v"], "traj/arm_speed_mult": self._vel_arm_mult,
                "traj/n_seg_max": self.traj_params.n_seg_max, "traj/p_arc": c["p_arc"],
                "traj/infeasible_frac": g["blocks_infeasible"] / max(g["blocks_sampled"], 1.0),
                "traj/blocks_started": g["blocks_started"], "traj/blocks_completed": g["blocks_completed"],
                "traj/blocks_failed": g["blocks_failed"],
                "traj/abandon_rate": g["abandons"] / max(g["passes"] + g["abandons"], 1.0),
                "traj/pass_success_rate": g["pass_success"] / max(g["passes"] + g["abandons"], 1.0),
                "traj/reached_frac_of_passes": g["reached"] / n_pass,
                "traj/q_pos": g["sum_qpos"] / n_pass, "traj/q_v": g["sum_qv"] / n_pass,
                "traj/dv_rel": g["sum_dv_rel"] / n_pass, "traj/v_par_rel": g["sum_v_par_rel"] / n_pass,
                "traj/v_perp_rel": g["sum_v_perp_rel"] / n_pass, "traj/lateral_at_pass": g["sum_lateral"] / n_pass,
                "traj/along_at_pass": g["sum_along"] / n_pass, "traj/kp_dist_at_pass": g["sum_kp_at_pass"] / n_pass,
                "traj/head_speed_moving": g["sum_head_speed_move"] / max(g["steps_move"], 1.0),
                "traj/goal_speed_moving": g["sum_goal_speed_move"] / max(g["steps_move"], 1.0),
                "traj/frac_steps_moving": g["steps_move"] / max(g["steps_move"] + g["steps_hold"], 1.0),
                "traj/fd_vs_analytic_rel": g["sum_fd_vs_analytic"] / n_pass,
                "static/success_rate": g["static_success"] / max(n_static_end, 1.0),
                "kp_rew/hold_per_step": g["kp_rew_hold"] / max(g["steps_hold"], 1.0),
                "kp_rew/move_per_step": g["kp_rew_move"] / max(g["steps_move"], 1.0),
                "bonus/hold_per_step": g["bonus_hold"] / max(g["steps_hold"], 1.0),
                "bonus/move_per_pass": g["bonus_move"] / n_pass,
            }
            self._vel_stats = st
            self._tacc.zero_()

            b = self._tblock.cpu().tolist()
            decision = self._traj_curriculum.decide(b[0], b[1], b[2], b[3], self._vel_step_count)
            if decision is not None:
                print(f"[trajectory] curriculum block: waypoint success {b[0]:.0f}/{b[1]:.0f}, "
                      f"static {b[2]:.0f}/{b[3]:.0f} -> {decision}, level {self._traj_curriculum.level:.2f}")
                self._tblock.zero_()
                if decision in ("up", "down"):
                    self._vel_apply_level()

            if self._vel_step_count % self.vel_print_every == 0:
                print(
                    f"[trajectory] step {self._vel_step_count} level {st['traj/level']:.2f} "
                    f"arm x{st['traj/arm_speed_mult']:.2f} | wp succ {st['traj/pass_success_rate']:.2f} "
                    f"reached {st['traj/reached_frac_of_passes']:.2f} abandon {st['traj/abandon_rate']:.2f} | "
                    f"q_pos {st['traj/q_pos']:.2f} q_v {st['traj/q_v']:.2f} |dv|/|vg| {st['traj/dv_rel']:.2f} "
                    f"v_par {st['traj/v_par_rel']:.2f} v_perp {st['traj/v_perp_rel']:.2f} "
                    f"lateral {st['traj/lateral_at_pass'] * 100:.1f} cm | speed head/goal "
                    f"{st['traj/head_speed_moving']:.2f}/{st['traj/goal_speed_moving']:.2f} | static succ "
                    f"{st['static/success_rate']:.2f} | blocks done {st['traj/blocks_completed']:.0f} failed "
                    f"{st['traj/blocks_failed']:.0f} infeasible {st['traj/infeasible_frac']:.2f} | "
                    f"moving steps {st['traj/frac_steps_moving']:.2f}"
                )
        self.extras.update(self._vel_stats)
