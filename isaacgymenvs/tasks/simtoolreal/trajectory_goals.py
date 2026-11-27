"""Smooth random trajectory goals for velocity-conditioned fine-tuning (pure torch, no Isaac Gym; CPU-testable).

A *motion chunk* is what replaces v1/v2's "lead-in + strike":

    set pose (static goal, official hold rule)
      -> waypoints along a smooth head path, each with a pose and a head velocity (passed through, no hold)
      -> end stop (static goal, official hold rule)

The path is defined for the **head center**: 1-3 cubic Bezier segments joined with continuous tangents, a
continuous, acceleration-limited speed profile that starts and ends at rest, and an orientation that is slerped
within each segment. The tool-frame pose of each waypoint (what the official keypoints are built on) is derived from
the head pose. Because every waypoint's velocity is the path tangent times the speed there, consecutive goals are
dynamically consistent by construction (the v1/v2 diagnosis: the policy reached the lead-in moving backwards and
sideways because the goal sequence was not).

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


@dataclass
class TrajParams:
    v_max: float  # m/s, peak head speed
    bend_deg: float  # curvature: max angle between a segment's start tangent and its chord, and chord and end tangent
    rot_max_deg: float  # max orientation change per segment
    n_seg_max: int  # segments per chunk drawn uniformly from 1..n_seg_max
    cone_deg: float  # start direction within this angle of the head's face axis (as in v1/v2)
    seg_len: Tuple[float, float] = (0.10, 0.25)  # m, chord length of each segment
    cruise_frac: Tuple[float, float] = (0.5, 1.0)  # each segment's cruise speed / v_max
    a_max: float = 2.0  # m/s^2, along-path acceleration / deceleration limit
    spacing: float = 0.04  # m of arc length between waypoints (widened if the chunk would exceed k_max waypoints)
    k_max: int = 24  # waypoints per chunk (the last one is the end stop)
    down_frac: float = 0.25  # chunks whose start direction is steep downward (tool turned to face it)
    down_max_z: float = -0.85
    volume_margin: float = 0.10  # m the moving waypoints may leave the target volume; the end stop must stay inside
    min_head_z: float = 0.58  # head center of every waypoint above this (table top ~0.53)
    tries: int = 8
    fine: int = 32  # samples per segment for the arc-length parametrisation


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


def _bezier(p0, p1, p2, p3, u):
    """Cubic Bezier points. p* (M, 3), u (F,) -> (M, F, 3)."""
    u = u.view(1, -1, 1)
    a, b, c, d = (p.unsqueeze(1) for p in (p0, p1, p2, p3))
    return (1 - u) ** 3 * a + 3 * (1 - u) ** 2 * u * b + 3 * (1 - u) * u ** 2 * c + u ** 3 * d


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
      wp_head (N, K, 3) head center at each waypoint
      wp_s (N, K)       arc length from the set pose's head center
      n_wp (N,) long    number of valid waypoints; waypoint n_wp - 1 is the end stop
      length (N,)       head path length (m)
      is_down (N,) bool
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
    # down chunks: turn the set pose (smallest rotation) so the head axis is within the cone of d
    ang = torch.acos((axis_w * d_down).sum(-1).clamp(-1.0, 1.0))
    excess = (ang - math.radians(params.cone_deg)).clamp(min=0.0)
    rot_axis = torch.cross(axis_w, d_down, dim=-1)
    rot_axis = torch.where((rot_axis.norm(dim=-1) < 1e-6).unsqueeze(-1), any_perpendicular(axis_w), rot_axis)
    q_turned = quat_mul_xyzw(quat_from_axis_angle_xyzw(rot_axis, excess), q_set)
    q_set = torch.where(is_down.unsqueeze(-1), q_turned / q_turned.norm(dim=-1, keepdim=True), q_set)
    tangent = torch.where(is_down.unsqueeze(-1), d_down, d_norm)

    # --- segments: cubic Bezier with continuous tangents; orientation slerped per segment ---
    n_seg = torch.randint(1, S_max + 1, (m,), device=dev, generator=generator)
    head = p_set + quat_rotate_xyzw(q_set, r_head)
    q_cur = q_set
    u = torch.linspace(0.0, 1.0, F + 1, device=dev)
    pts, quats, seg_cruise = [head.unsqueeze(1)], [q_cur.unsqueeze(1)], []
    for j in range(S_max):
        active = (j < n_seg).float().unsqueeze(-1)
        L = params.seg_len[0] + (params.seg_len[1] - params.seg_len[0]) * torch.rand(m, device=dev, generator=generator)
        L = L.unsqueeze(-1) * active  # inactive segments have zero length
        chord = cone_sample(tangent, params.bend_deg, generator) if params.bend_deg > 0 else tangent
        t_end = cone_sample(chord, params.bend_deg, generator) if params.bend_deg > 0 else chord
        p0, p3 = head, head + L * chord
        p1, p2 = p0 + (L / 3.0) * tangent, p3 - (L / 3.0) * t_end
        seg_pts = _bezier(p0, p1, p2, p3, u)  # (m, F+1, 3)
        q_end = quat_mul_xyzw(random_rotation_xyzw(m, params.rot_max_deg, dev, generator), q_cur)
        q_end = torch.where(active.bool(), q_end, q_cur)
        # slerp by arc-length fraction within the segment
        seg_arc = torch.cat(
            [torch.zeros(m, 1, device=dev), (seg_pts[:, 1:] - seg_pts[:, :-1]).norm(dim=-1).cumsum(dim=1)], dim=1
        )
        frac = seg_arc / seg_arc[:, -1:].clamp(min=1e-9)
        seg_q = quat_slerp_xyzw(q_cur.unsqueeze(1).expand(m, F + 1, 4), q_end.unsqueeze(1).expand(m, F + 1, 4), frac)
        pts.append(seg_pts[:, 1:])
        quats.append(seg_q[:, 1:])
        cf = params.cruise_frac
        seg_cruise.append((cf[0] + (cf[1] - cf[0]) * torch.rand(m, device=dev, generator=generator)) * params.v_max)
        head, tangent, q_cur = p3, torch.where(active.bool(), t_end, tangent), q_end

    path = torch.cat(pts, dim=1)  # (m, S_max*F + 1, 3) head centers
    path_q = torch.cat(quats, dim=1)
    arc = torch.cat([torch.zeros(m, 1, device=dev), (path[:, 1:] - path[:, :-1]).norm(dim=-1).cumsum(dim=1)], dim=1)
    total = arc[:, -1]
    cruise = torch.stack(seg_cruise, dim=1)  # (m, S_max)

    # --- waypoints at (about) equal arc length; the last one is the end of the path ---
    K = params.k_max
    # n_wp equal steps of about `spacing` (never a sliver of a last step), at most K
    n_wp = torch.ceil(total / params.spacing - 0.5).long().clamp(1, K)
    spacing = total / n_wp.float()
    k = torch.arange(1, K + 1, device=dev).unsqueeze(0).float()
    s_k = torch.minimum(k * spacing.unsqueeze(1), total.unsqueeze(1))  # (m, K)
    ge = arc.unsqueeze(1) >= (s_k.unsqueeze(-1) - 1e-9)  # (m, K, P)
    idx = torch.argmax(ge.float(), dim=-1).clamp(min=1)  # first fine sample at or past s_k
    rows = torch.arange(m, device=dev).unsqueeze(1).expand(m, K)
    a0, a1 = arc[rows, idx - 1], arc[rows, idx]
    w = ((s_k - a0) / (a1 - a0).clamp(min=1e-9)).clamp(0.0, 1.0)
    x0, x1 = path[rows, idx - 1], path[rows, idx]
    wp_head = x0 + w.unsqueeze(-1) * (x1 - x0)
    wp_quat = quat_slerp_xyzw(path_q[rows, idx - 1], path_q[rows, idx], w)
    tan = x1 - x0
    tan = tan / tan.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    # --- speed profile: per-segment cruise speed, then acceleration limits forward from rest and backward to rest ---
    seg_id = ((idx - 1) // F).clamp(max=S_max - 1)
    v = cruise[rows, seg_id]
    valid = k <= n_wp.unsqueeze(1).float()
    v = torch.where(valid, v, torch.zeros_like(v))
    v[rows[:, 0], n_wp - 1] = 0.0  # end stop
    two_a = 2.0 * params.a_max
    prev_v, prev_s = torch.zeros(m, device=dev), torch.zeros(m, device=dev)
    for i in range(K):  # forward: from rest at the set pose
        lim = torch.sqrt(prev_v ** 2 + two_a * (s_k[:, i] - prev_s).clamp(min=0.0))
        v[:, i] = torch.minimum(v[:, i], lim)
        prev_v, prev_s = v[:, i], s_k[:, i]
    for i in range(K - 2, -1, -1):  # backward: to rest at the end stop
        lim = torch.sqrt(v[:, i + 1] ** 2 + two_a * (s_k[:, i + 1] - s_k[:, i]).clamp(min=0.0))
        v[:, i] = torch.minimum(v[:, i], lim)
    wp_vel = v.unsqueeze(-1) * tan

    # --- tool-frame poses, and copies of the end stop past n_wp ---
    wp_pos = wp_head - quat_rotate_xyzw(wp_quat, r_head.unsqueeze(1).expand(m, K, 3))
    end = (n_wp - 1).unsqueeze(1).expand(m, K)
    pad = ~valid
    for x in (wp_pos, wp_quat, wp_head, wp_vel):
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
        wp_head=wp_head[pick],
        wp_s=s_k[pick],
        n_wp=n_wp[pick],
        length=total[pick],
        is_down=is_down[pick],
    )
