"""Pure torch helpers for velocity-conditioned goals (no Isaac Gym import, so they run and are tested on CPU).

Conventions follow the SimToolReal env:
  * quaternions are xyzw (Isaac Gym order);
  * the tool frame has its origin at the handle center, +x along the handle toward the head;
  * a goal pose is the pose of that frame (the official keypoints are built around it).

See finetune_velocity/README_zh.md for why each choice was made.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import torch
from torch import Tensor


# ----------------------------------------------------------------------------------------------------------------------
# Quaternion helpers (xyzw). Small re-implementations so this file does not need isaacgym.torch_utils.
# ----------------------------------------------------------------------------------------------------------------------
def quat_rotate_xyzw(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4, xyzw)."""
    q_vec = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def quat_mul_xyzw(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product a * b (both xyzw): rotate by b first, then by a."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def quat_from_axis_angle_xyzw(axis: Tensor, angle: Tensor) -> Tensor:
    """axis (..., 3), need not be unit; angle (...) in rad."""
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    half = 0.5 * angle
    return torch.cat((axis * torch.sin(half).unsqueeze(-1), torch.cos(half).unsqueeze(-1)), dim=-1)


def any_perpendicular(v: Tensor) -> Tensor:
    """A unit vector perpendicular to each unit vector v (..., 3)."""
    ref = torch.zeros_like(v)
    use_x = v[..., 0].abs() < 0.9
    ref[..., 0] = use_x.float()
    ref[..., 1] = (~use_x).float()
    p = torch.cross(v, ref, dim=-1)
    return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def cone_sample(center: Tensor, max_deg: float, generator: Optional[torch.Generator] = None) -> Tensor:
    """Unit vectors uniform (by area) within max_deg of the unit vectors `center` (..., 3).

    Same distribution as Leo's goals.cone_sample: cos(angle) uniform in [cos(max), 1], random azimuth.
    max_deg >= 180 gives a uniform direction on the sphere.
    """
    shape = center.shape[:-1]
    dev = center.device
    cos_max = math.cos(math.radians(min(max_deg, 180.0)))
    u = torch.rand(shape, device=dev, generator=generator)
    cos_a = cos_max + (1.0 - cos_max) * u
    sin_a = torch.sqrt((1.0 - cos_a * cos_a).clamp_min(0.0))
    phi = (2.0 * torch.rand(shape, device=dev, generator=generator) - 1.0) * math.pi
    p1 = any_perpendicular(center)
    p2 = torch.cross(center, p1, dim=-1)
    perp = torch.cos(phi).unsqueeze(-1) * p1 + torch.sin(phi).unsqueeze(-1) * p2
    out = cos_a.unsqueeze(-1) * center + sin_a.unsqueeze(-1) * perp
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-9)


# ----------------------------------------------------------------------------------------------------------------------
# Reward pieces
# ----------------------------------------------------------------------------------------------------------------------
def velocity_match(v_head: Tensor, v_goal: Tensor, sigma_frac: float) -> Tuple[Tensor, Tensor, Tensor]:
    """q_v = exp(-|v_head - v_g| / (sigma_frac * |v_g|)), as in Leo's rewards.velocity_match.

    Returns (q_v, |dv|, |v_g|). |v_g| is clamped away from 0 only inside the exponent; callers mask static goals.
    The linear (not squared) error matches Leo's code; the midterm report's squared formula is a typo.
    """
    dv = (v_head - v_goal).norm(dim=-1)
    speed = v_goal.norm(dim=-1)
    q_v = torch.exp(-dv / (sigma_frac * speed.clamp_min(1e-6)))
    return q_v, dv, speed


# ----------------------------------------------------------------------------------------------------------------------
# Curriculum
# ----------------------------------------------------------------------------------------------------------------------
def lerp(pair, level: float) -> float:
    """Linear interpolation of a (start, end) pair at level in [0, 1]."""
    a, b = float(pair[0]), float(pair[1])
    return a + (b - a) * float(level)


@dataclass
class VelocityCurriculum:
    """One difficulty level in [0, 1] drives every velocity-goal knob (as Leo's CurriculumCfg does).

    The level moves in blocks: once at least `min_samples` velocity goals have finished since the last decision,
    the velocity success rate of that block decides: >= up_threshold (and static goals still fine) -> level up,
    < down_threshold -> level down, otherwise stay. The block counters are then cleared.
    """

    level: float = 0.0
    step: float = 0.05
    up_threshold: float = 0.5
    down_threshold: float = 0.2
    min_samples: int = 2048
    static_guard: float = 0.5  # level only goes up while static goals succeed at least this often (0 disables)
    min_steps_between: int = 0  # control steps between two level changes
    fixed: bool = False  # True: never move (evaluation, ablations)

    def __post_init__(self):
        self.level = float(min(max(self.level, 0.0), 1.0))
        self.last_change_step = -(10**12)

    def decide(self, vel_success: float, vel_total: float, static_success: float, static_total: float,
               step_count: int) -> Optional[str]:
        """Return 'up', 'down', 'stay' after a decision, or None while the block is still filling."""
        if self.fixed or vel_total < self.min_samples:
            return None
        if step_count - self.last_change_step < self.min_steps_between:
            return None
        rate = vel_success / max(vel_total, 1.0)
        static_rate = static_success / static_total if static_total > 0 else 1.0
        old = self.level
        if rate >= self.up_threshold and static_rate >= self.static_guard:
            self.level = min(1.0, self.level + self.step)
        elif rate < self.down_threshold:
            self.level = max(0.0, self.level - self.step)
        if self.level != old:
            self.last_change_step = step_count
            return "up" if self.level > old else "down"
        return "stay"


# ----------------------------------------------------------------------------------------------------------------------
# Goal sampling
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class VelocityGoalParams:
    cone_deg: float  # max angle between the strike direction and the head axis
    speed_lo: float  # m/s
    speed_hi: float  # m/s
    down_frac: float = 0.25  # fraction of steep downward strikes (Leo: GoalCfg.down_frac)
    down_max_z: float = -0.85  # a downward strike has direction z <= this (Leo: GoalCfg.down_max_z)
    lead_in: float = 0.12  # m behind the strike pose, along -d (Leo: GoalCfg.lead_in)
    follow_through: float = 0.08  # m past the strike pose, along +d, must stay feasible (Leo: GoalCfg.follow_through)
    min_head_z: float = 0.58  # head center of all three points must be above this (table top is about 0.53 m)
    tries: int = 8


def sample_velocity_goals(
    strike_pos: Tensor,
    strike_quat: Tensor,
    head_offset: Tensor,
    head_axis: Tensor,
    head_axis_symmetric: Tensor,
    volume_min: Tensor,
    volume_max: Tensor,
    params: VelocityGoalParams,
    generator: Optional[torch.Generator] = None,
):
    """Turn already-sampled strike poses (the official delta goals) into velocity goals.

    strike_pos (N, 3), strike_quat (N, 4 xyzw): official delta goals (tool-frame origin = handle center).
    head_offset (N, 3): head center in the tool frame. head_axis (N, 3): unit "face" axis in the tool frame.
    head_axis_symmetric (N,) bool: the tool strikes equally well with -head_axis (hammer heads have two faces).

    For each env, `tries` candidates are drawn and the first feasible one is used:
      * a fraction down_frac of candidates is a steep downward strike: d is drawn in the cap z <= down_max_z, then
        the tool is turned (smallest rotation) until its head axis is within cone_deg of d;
      * the others keep the sampled orientation and draw d within cone_deg of the head axis;
      * feasible: the lead-in (p - lead_in d), strike (p) and follow-through (p + follow_through d) tool origins lie in
        the target volume, and the head center of each is above min_head_z.

    Returns dict with
      feasible (N,) bool, quat (N, 4) final strike orientation, v_goal (N, 3), leadin_pos (N, 3), direction (N, 3),
      is_down (N,) bool.
    """
    n = strike_pos.shape[0]
    k = params.tries
    dev = strike_pos.device

    def rep(x):
        return x.unsqueeze(1).expand(n, k, *x.shape[1:]).reshape(n * k, *x.shape[1:])

    p = rep(strike_pos)
    q = rep(strike_quat)
    r_head = rep(head_offset)
    axis_local = rep(head_axis)
    sym = rep(head_axis_symmetric)
    m = n * k

    is_down = torch.rand(m, device=dev, generator=generator) < params.down_frac

    # Head axis in the world at the sampled orientation. Symmetric heads: pick a face at random (normal goals) or the
    # face closer to the downward direction (down goals, so the turn below is as small as possible).
    axis_w = quat_rotate_xyzw(q, axis_local)
    flip_rand = (torch.rand(m, device=dev, generator=generator) < 0.5) & sym

    # Downward directions: uniform in the cap of half angle acos(-down_max_z) around -z.
    down_center = torch.zeros(m, 3, device=dev)
    down_center[:, 2] = -1.0
    cap_deg = math.degrees(math.acos(min(1.0, max(-1.0, -params.down_max_z))))
    d_down = cone_sample(down_center, cap_deg, generator)
    flip_down = sym & ((axis_w * d_down).sum(-1) < 0.0)
    flip = torch.where(is_down, flip_down, flip_rand)
    axis_w = torch.where(flip.unsqueeze(-1), -axis_w, axis_w)

    # Normal goals: d within the cone around the head axis.
    d_norm = cone_sample(axis_w, params.cone_deg, generator)

    # Down goals: turn the tool so that the head axis is within cone_deg of d (rotate about axis_w x d).
    cos_ad = (axis_w * d_down).sum(-1).clamp(-1.0, 1.0)
    ang = torch.acos(cos_ad)
    excess = (ang - math.radians(params.cone_deg)).clamp_min(0.0)
    rot_axis = torch.cross(axis_w, d_down, dim=-1)
    degenerate = rot_axis.norm(dim=-1) < 1e-6
    rot_axis = torch.where(degenerate.unsqueeze(-1), any_perpendicular(axis_w), rot_axis)
    q_turn = quat_from_axis_angle_xyzw(rot_axis, excess)
    q_down = quat_mul_xyzw(q_turn, q)
    q_down = q_down / q_down.norm(dim=-1, keepdim=True)

    d = torch.where(is_down.unsqueeze(-1), d_down, d_norm)
    q_final = torch.where(is_down.unsqueeze(-1), q_down, q)

    speed = params.speed_lo + (params.speed_hi - params.speed_lo) * torch.rand(m, device=dev, generator=generator)
    v = d * speed.unsqueeze(-1)

    lead = p - params.lead_in * d
    follow = p + params.follow_through * d
    head_w = quat_rotate_xyzw(q_final, r_head)

    vmin = volume_min.view(1, 3)
    vmax = volume_max.view(1, 3)
    ok = torch.ones(m, dtype=torch.bool, device=dev)
    for pt in (lead, p, follow):
        ok &= ((pt >= vmin) & (pt <= vmax)).all(-1)
        ok &= (pt[:, 2] + head_w[:, 2]) >= params.min_head_z

    ok = ok.view(n, k)
    first = torch.argmax(ok.float(), dim=1)  # first feasible try (argmax returns the first max; 0 if none)
    feasible = ok.any(dim=1)
    pick = torch.arange(n, device=dev) * k + first

    return dict(
        feasible=feasible,
        quat=q_final[pick],
        v_goal=v[pick],
        leadin_pos=lead[pick],
        direction=d[pick],
        is_down=is_down[pick],
    )
