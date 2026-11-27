"""Smooth random trajectory goals for velocity-conditioned fine-tuning (pure torch, no Isaac Gym; CPU-testable).

A *motion chunk* replaces v1/v2's "lead-in + strike":

    set pose (static goal, official hold rule)
      -> waypoints along a smooth head path, each with a pose and a head velocity (passed through, no hold)
      -> end stop (static goal, official hold rule)

The path is defined for the **head center** and built from 1..n_seg_max segments joined with continuous tangents:
  * bend segments: cubic Bezier curves (curvature set by bend_deg), orientation slerped toward a random rotation;
  * arc segments: the tool turns rigidly about a pivot behind the head (by default on the handle line, like a wrist),
    so the head sweeps a circular arc and the motion is rotation-dominated (hammer swings, flips).
The speed along the path is a smooth window (quintic smoothstep ramps from rest and back to rest, asymmetric, a
plateau in between), capped by an acceleration limit. Every waypoint's goal velocity is the path tangent times the
speed there, so consecutive goals are dynamically consistent by construction (the v1/v2 diagnosis: the policy reached
the lead-in moving backwards and sideways because the goal sequence was not).

Geometry ranges are set so that DexToolBench's task trajectories are covered with room to spare (DexToolBench is a
coverage floor, not the target; finetune_velocity/README section 14).
Conventions as in the env: quaternions xyzw, tool frame origin at the handle center with +x toward the head.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from isaacgymenvs.tasks.simtoolreal.velocity_goals import (
    any_perpendicular,
    cone_sample,
    quat_from_axis_angle_xyzw,
    quat_mul_xyzw,
    quat_rotate_xyzw,
)

_SMOOTHSTEP_SSP_MAX = 1.1948  # max over [0, 1] of S(x) S'(x) for S = 6x^5 - 15x^4 + 10x^3


@dataclass
class TrajParams:
    v_max: float  # m/s, peak head speed of a chunk is drawn in [peak_frac * v_max, v_max]
    bend_deg: float  # bend segments: max angle between start tangent and chord, and between chord and end tangent
    rot_max_deg: float  # bend segments: max orientation change per segment
    n_seg_max: int  # segments per chunk drawn uniformly from 1..n_seg_max
    cone_deg: float  # start direction within this angle of the head's face axis (as in v1/v2)
    p_arc: float = 0.0  # probability that a segment is an arc (rigid rotation about a pivot)
    arc_max_deg: float = 0.0  # arc segments: max swept angle
    arc_radius: Tuple[float, float] = (0.08, 0.30)  # m, pivot-to-head distance of arc segments
    arc_pivot_on_handle: float = 0.7  # probability that the arc pivot lies on the handle line (wrist-like swing)
    seg_len: Tuple[float, float] = (0.06, 0.30)  # m, chord (bend) or arc length of each segment
    peak_frac: Tuple[float, float] = (0.5, 1.0)
    ramp_up: Tuple[float, float] = (0.15, 0.6)  # fraction of the chunk's path used to speed up from rest
    ramp_down: Tuple[float, float] = (0.15, 0.4)  # fraction used to slow down to rest (short = strike-like)
    a_max: float = 3.0  # m/s^2 along the path
    spacing: float = 0.03  # m of arc length between waypoints (widened if the chunk would exceed k_max waypoints)
    k_max: int = 48  # waypoints per chunk (the last one is the end stop)
    down_frac: float = 0.25
    down_max_z: float = -0.85
    volume_margin: float = 0.10  # m the moving waypoints may leave the target volume; the end stop must stay inside
    min_head_z: float = 0.58  # head center of every waypoint above this (table top ~0.53)
    tries: int = 8
    fine: int = 24  # samples per segment for the arc-length parametrisation


def quat_slerp_xyzw(q0: Tensor, q1: Tensor, t: Tensor) -> Tensor:
    """Slerp between unit quaternions (..., 4) at t (...,) in [0, 1] (shortest path)."""
    dot = (q0 * q1).sum(-1)
    q1 = torch.where((dot < 0).unsqueeze(-1), -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_t = torch.sin(theta)
    small = sin_t < 1e-4
    w0 = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_t.clamp(min=1e-4))
    w1 = torch.where(small, t, torch.sin(t * theta) / sin_t.clamp(min=1e-4))
    q = w0.unsqueeze(-1) * q0 + w1.unsqueeze(-1) * q1
    return q / q.norm(dim=-1, keepdim=True)


def random_rotation_xyzw(m: int, max_deg: float, device, generator: Optional[torch.Generator] = None) -> Tensor:
    """Rotations about uniformly random axes by angles uniform in [0, max_deg]."""
    axis = torch.randn(m, 3, device=device, generator=generator)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    ang = torch.rand(m, device=device, generator=generator) * math.radians(max_deg)
    return quat_from_axis_angle_xyzw(axis, ang)


def smoothstep(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def _uniform(lo_hi, m, dev, generator):
    return lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(m, device=dev, generator=generator)


def _unit(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _bend_segment(head, tangent, q_cur, L, params, generator):
    """Cubic Bezier from `head` with start tangent `tangent`. Returns points (m, F+1, 3), quats, end tangent."""
    m, dev, F = head.shape[0], head.device, params.fine
    chord = cone_sample(tangent, params.bend_deg, generator) if params.bend_deg > 0 else tangent
    t_end = cone_sample(chord, params.bend_deg, generator) if params.bend_deg > 0 else chord
    p0, p3 = head, head + L.unsqueeze(-1) * chord
    p1, p2 = p0 + (L / 3.0).unsqueeze(-1) * tangent, p3 - (L / 3.0).unsqueeze(-1) * t_end
    u = torch.linspace(0.0, 1.0, F + 1, device=dev).view(1, -1, 1)
    a, b, c, d = (p.unsqueeze(1) for p in (p0, p1, p2, p3))
    pts = (1 - u) ** 3 * a + 3 * (1 - u) ** 2 * u * b + 3 * (1 - u) * u ** 2 * c + u ** 3 * d
    arc = torch.cat([torch.zeros(m, 1, device=dev), (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).cumsum(dim=1)], dim=1)
    frac = arc / arc[:, -1:].clamp(min=1e-9)
    q_end = quat_mul_xyzw(random_rotation_xyzw(m, params.rot_max_deg, dev, generator), q_cur)
    quats = quat_slerp_xyzw(q_cur.unsqueeze(1).expand(m, F + 1, 4), q_end.unsqueeze(1).expand(m, F + 1, 4), frac)
    return pts, quats, t_end


def _arc_segment(head, tangent, q_cur, L, params, generator):
    """Rigid rotation of the tool about a pivot: the head moves on a circle of radius rho, tangent to `tangent`."""
    m, dev, F = head.shape[0], head.device, params.fine
    rho = _uniform(params.arc_radius, m, dev, generator)
    theta = torch.minimum(L / rho, torch.full_like(L, math.radians(params.arc_max_deg)))
    # radius direction n (pivot -> head), perpendicular to the tangent: along the handle (wrist-like) or random
    x_w = quat_rotate_xyzw(q_cur, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(m, 3))
    n_handle = x_w - (x_w * tangent).sum(-1, keepdim=True) * tangent
    n_rand = any_perpendicular(tangent)
    phi = (2.0 * torch.rand(m, device=dev, generator=generator) - 1.0) * math.pi
    n_rand = torch.cos(phi).unsqueeze(-1) * n_rand + torch.sin(phi).unsqueeze(-1) * torch.cross(tangent, n_rand, dim=-1)
    use_handle = (torch.rand(m, device=dev, generator=generator) < params.arc_pivot_on_handle) & (
        n_handle.norm(dim=-1) > 0.2
    )
    n = _unit(torch.where(use_handle.unsqueeze(-1), n_handle, n_rand))
    axis = torch.cross(n, tangent, dim=-1)  # rotating n about this axis turns it toward the tangent
    center = head - rho.unsqueeze(-1) * n
    u = torch.linspace(0.0, 1.0, F + 1, device=dev).view(1, -1)
    ang = theta.unsqueeze(-1) * u  # (m, F+1)
    pts = center.unsqueeze(1) + rho.view(-1, 1, 1) * (
        torch.cos(ang).unsqueeze(-1) * n.unsqueeze(1) + torch.sin(ang).unsqueeze(-1) * tangent.unsqueeze(1)
    )
    q_rot = quat_from_axis_angle_xyzw(axis.unsqueeze(1).expand(m, F + 1, 3), ang)
    quats = quat_mul_xyzw(q_rot, q_cur.unsqueeze(1).expand(m, F + 1, 4))
    t_end = -torch.sin(theta).unsqueeze(-1) * n + torch.cos(theta).unsqueeze(-1) * tangent
    return pts, quats, t_end


def sample_motion_chunks(
    set_pos: Tensor,
    set_quat: Tensor,
    head_offset: Tensor,
    head_axis: Tensor,
    head_axis_symmetric: Tensor,
    volume_min: Tensor,
    volume_max: Tensor,
    params: TrajParams,
    generator: Optional[torch.Generator] = None,
):
    """Sample one motion chunk per env, starting at rest from the set pose.

    set_pos (N, 3), set_quat (N, 4): the chunk's set pose (tool frame), an official delta goal.
    head_offset (N, 3), head_axis (N, 3), head_axis_symmetric (N,): as in velocity_goals.sample_velocity_goals.

    Returns a dict (first feasible of `tries` candidates per env):
      feasible (N,) bool
      set_quat (N, 4)   possibly turned so the head faces a downward start direction
      wp_pos (N, K, 3), wp_quat (N, K, 4)   tool-frame waypoint poses (K = k_max; entries past n_wp repeat the end)
      wp_vel (N, K, 3)  head velocity at each waypoint (0 at the end stop)
      wp_tan (N, K, 3)  unit path tangent at each waypoint (defined even where the speed is 0)
      wp_head (N, K, 3) head center at each waypoint
      wp_s (N, K)       arc length from the set pose's head center
      n_wp (N,) long    number of valid waypoints; waypoint n_wp - 1 is the end stop
      length (N,)       head path length (m)
      is_down (N,) bool
      n_arc (N,) long   number of arc segments
    """
    n = set_pos.shape[0]
    t = params.tries
    dev = set_pos.device
    m = n * t
    F = params.fine
    S_max = params.n_seg_max

    def rep(x):
        return x.unsqueeze(1).expand(n, t, *x.shape[1:]).reshape(m, *x.shape[1:])

    p_set, q_set = rep(set_pos), rep(set_quat)
    r_head, axis_l, sym = rep(head_offset), rep(head_axis), rep(head_axis_symmetric)

    # --- start direction (as v1/v2: cone around the head's face axis, or a steep downward strike) ---
    is_down = torch.rand(m, device=dev, generator=generator) < params.down_frac
    axis_w = quat_rotate_xyzw(q_set, axis_l)
    down_c = torch.zeros(m, 3, device=dev)
    down_c[:, 2] = -1.0
    cap = math.degrees(math.acos(min(1.0, max(-1.0, -params.down_max_z))))
    d_down = cone_sample(down_c, cap, generator)
    flip_rand = (torch.rand(m, device=dev, generator=generator) < 0.5) & sym
    flip_down = sym & ((axis_w * d_down).sum(-1) < 0)
    axis_w = torch.where(torch.where(is_down, flip_down, flip_rand).unsqueeze(-1), -axis_w, axis_w)
    d_norm = cone_sample(axis_w, params.cone_deg, generator)
    ang = torch.acos((axis_w * d_down).sum(-1).clamp(-1.0, 1.0))
    excess = (ang - math.radians(params.cone_deg)).clamp(min=0.0)
    rot_axis = torch.cross(axis_w, d_down, dim=-1)
    rot_axis = torch.where((rot_axis.norm(dim=-1) < 1e-6).unsqueeze(-1), any_perpendicular(axis_w), rot_axis)
    q_turned = quat_mul_xyzw(quat_from_axis_angle_xyzw(rot_axis, excess), q_set)
    q_set = torch.where(is_down.unsqueeze(-1), q_turned / q_turned.norm(dim=-1, keepdim=True), q_set)
    tangent = torch.where(is_down.unsqueeze(-1), d_down, d_norm)

    # --- segments with continuous tangents; inactive segments (j >= n_seg) have zero length ---
    n_seg = torch.randint(1, S_max + 1, (m,), device=dev, generator=generator)
    head = p_set + quat_rotate_xyzw(q_set, r_head)
    q_cur = q_set
    pts_all, quats_all = [head.unsqueeze(1)], [q_cur.unsqueeze(1)]
    n_arc = torch.zeros(m, device=dev).long()
    for j in range(S_max):
        active = j < n_seg
        L = _uniform(params.seg_len, m, dev, generator) * active.float()
        b_pts, b_q, b_tend = _bend_segment(head, tangent, q_cur, L, params, generator)
        if params.p_arc > 0 and params.arc_max_deg > 0:
            a_pts, a_q, a_tend = _arc_segment(head, tangent, q_cur, L, params, generator)
            is_arc = (torch.rand(m, device=dev, generator=generator) < params.p_arc) & active
            sel = is_arc.view(-1, 1, 1)
            seg_pts = torch.where(sel, a_pts, b_pts)
            seg_q = torch.where(sel, a_q, b_q)
            t_end = torch.where(is_arc.unsqueeze(-1), a_tend, b_tend)
            n_arc = n_arc + is_arc.long()
        else:
            seg_pts, seg_q, t_end = b_pts, b_q, b_tend
        # inactive: hold the last point / orientation / tangent
        hold = (~active).view(-1, 1, 1)
        seg_pts = torch.where(hold, head.unsqueeze(1).expand(m, F + 1, 3), seg_pts)
        seg_q = torch.where(hold, q_cur.unsqueeze(1).expand(m, F + 1, 4), seg_q)
        pts_all.append(seg_pts[:, 1:])
        quats_all.append(seg_q[:, 1:])
        head, q_cur = seg_pts[:, -1], seg_q[:, -1]
        tangent = torch.where(active.unsqueeze(-1), _unit(t_end), tangent)

    path = torch.cat(pts_all, dim=1)  # (m, S_max*F + 1, 3) head centers
    path_q = torch.cat(quats_all, dim=1)
    arc = torch.cat([torch.zeros(m, 1, device=dev), (path[:, 1:] - path[:, :-1]).norm(dim=-1).cumsum(dim=1)], dim=1)
    total = arc[:, -1]

    # --- waypoints: n_wp equal steps of about `spacing` (no sliver step), the last one at the end of the path ---
    K = params.k_max
    n_wp = torch.ceil(total / params.spacing - 0.5).long().clamp(1, K)
    step = total / n_wp.float()
    k = torch.arange(1, K + 1, device=dev).unsqueeze(0).float()
    s_k = torch.minimum(k * step.unsqueeze(1), total.unsqueeze(1))  # (m, K)
    ge = arc.unsqueeze(1) >= (s_k.unsqueeze(-1) - 1e-9)
    idx = torch.argmax(ge.float(), dim=-1).clamp(min=1)
    rows = torch.arange(m, device=dev).unsqueeze(1).expand(m, K)
    a0, a1 = arc[rows, idx - 1], arc[rows, idx]
    w = ((s_k - a0) / (a1 - a0).clamp(min=1e-9)).clamp(0.0, 1.0)
    x0, x1 = path[rows, idx - 1], path[rows, idx]
    wp_head = x0 + w.unsqueeze(-1) * (x1 - x0)
    wp_quat = quat_slerp_xyzw(path_q[rows, idx - 1], path_q[rows, idx], w)
    tan = _unit(x1 - x0)

    # --- speed: smooth window (quintic ramps from and back to rest), peak capped by the acceleration limit ---
    up = _uniform(params.ramp_up, m, dev, generator) * total
    down = _uniform(params.ramp_down, m, dev, generator) * total
    peak = _uniform(params.peak_frac, m, dev, generator) * params.v_max
    # d(v^2)/ds <= 2 a_max on both ramps (max of S S' is _SMOOTHSTEP_SSP_MAX)
    peak = torch.minimum(peak, torch.sqrt(params.a_max * torch.minimum(up, down) / _SMOOTHSTEP_SSP_MAX))
    s = s_k
    window = smoothstep(s / up.unsqueeze(1).clamp(min=1e-9)) * smoothstep(
        (total.unsqueeze(1) - s) / down.unsqueeze(1).clamp(min=1e-9)
    )
    v = peak.unsqueeze(1) * window
    valid = k <= n_wp.unsqueeze(1).float()
    v = torch.where(valid, v, torch.zeros_like(v))
    v[rows[:, 0], n_wp - 1] = 0.0  # end stop
    wp_vel = v.unsqueeze(-1) * tan

    # --- tool-frame poses, and copies of the end stop past n_wp ---
    wp_pos = wp_head - quat_rotate_xyzw(wp_quat, r_head.unsqueeze(1).expand(m, K, 3))
    end = (n_wp - 1).unsqueeze(1).expand(m, K)
    pad = ~valid
    for x in (wp_pos, wp_quat, wp_head, wp_vel, tan):
        x[pad] = x[rows, end][pad]
    s_k = torch.where(valid, s_k, total.unsqueeze(1).expand(m, K))

    # --- feasibility ---
    vmin, vmax = volume_min.view(1, 1, 3), volume_max.view(1, 1, 3)
    mg = params.volume_margin
    inside = ((wp_pos >= vmin - mg) & (wp_pos <= vmax + mg)).all(-1) | pad
    above = (wp_head[..., 2] >= params.min_head_z) | pad
    end_pos = wp_pos[rows[:, 0], n_wp - 1]
    end_inside = ((end_pos >= volume_min.view(1, 3)) & (end_pos <= volume_max.view(1, 3))).all(-1)
    set_head_ok = (p_set + quat_rotate_xyzw(q_set, r_head))[:, 2] >= params.min_head_z
    ok = inside.all(-1) & above.all(-1) & end_inside & set_head_ok

    ok = ok.view(n, t)
    first = torch.argmax(ok.float(), dim=1)
    pick = torch.arange(n, device=dev) * t + first
    return dict(
        feasible=ok.any(dim=1),
        set_quat=q_set[pick],
        wp_pos=wp_pos[pick],
        wp_quat=wp_quat[pick],
        wp_vel=wp_vel[pick],
        wp_tan=tan[pick],
        wp_head=wp_head[pick],
        wp_s=s_k[pick],
        n_wp=n_wp[pick],
        length=total[pick],
        is_down=is_down[pick],
        n_arc=n_arc[pick],
    )


# ----------------------------------------------------------------------------------------------------------------------
# Passing a moving waypoint (event-driven advance) and its reward terms
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class PassParams:
    eps_kp: float  # m, keypoint distance within which a pass counts as a success (with the velocity condition)
    sigma_pos: float  # m, q_pos = exp(-kp_dist / sigma_pos)
    sigma_v: float  # q_v = exp(-|dv| / (sigma_v * max(|v_g|, v_floor)))
    eps_v: float  # velocity success: |dv| < eps_v * max(|v_g|, v_floor)
    v_floor: float = 0.1  # m/s, so that slow waypoints near the ends are not judged on a tiny |v_g|
    abandon_dist: float = 0.15  # m, head farther than this from the waypoint's head center abandons the block
    reach_dist: float = 0.015  # m, keypoint distance that advances without crossing (kept below half the spacing so
    # the goal never moves on while the tool is still a waypoint or more behind; smoke test 1: eps_kp 6 cm > spacing)


def waypoint_pass(kp_dist: Tensor, head_pos: Tensor, head_vel: Tensor, wp_head: Tensor, wp_tan: Tensor,
                  wp_vel: Tensor, p: PassParams):
    """Per env (moving waypoint is the current goal). All (N,) except vectors (N, 3).

    reached: keypoint distance <= reach_dist. crossed: the head is past the waypoint's normal plane (along the tangent).
    passed = reached | crossed; abandon: head farther than abandon_dist from the waypoint (and not passed).
    success (at a pass): keypoint distance <= eps_kp and |dv| < eps_v * max(|v_g|, v_floor).
    """
    rel = head_pos - wp_head
    along = (rel * wp_tan).sum(-1)
    lateral = (rel - along.unsqueeze(-1) * wp_tan).norm(dim=-1)
    reached = kp_dist <= p.reach_dist
    crossed = along >= 0.0
    passed = reached | crossed
    abandon = (~passed) & (rel.norm(dim=-1) > p.abandon_dist)
    speed_ref = wp_vel.norm(dim=-1).clamp(min=p.v_floor)
    dv = (head_vel - wp_vel).norm(dim=-1)
    q_pos = torch.exp(-kp_dist / p.sigma_pos)
    q_v = torch.exp(-dv / (p.sigma_v * speed_ref))
    success = (kp_dist <= p.eps_kp) & (dv < p.eps_v * speed_ref)
    v_par = (head_vel * wp_tan).sum(-1)
    v_perp = (head_vel - v_par.unsqueeze(-1) * wp_tan).norm(dim=-1)
    return dict(reached=reached, crossed=crossed, passed=passed, abandon=abandon, q_pos=q_pos, q_v=q_v,
                success=success, lateral=lateral, along=along, dv_rel=dv / speed_ref,
                v_par_rel=v_par / speed_ref, v_perp_rel=v_perp / speed_ref)
