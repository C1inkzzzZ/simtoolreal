"""CPU tests for the trajectory-goal sampler (no Isaac Gym, no GPU).

  python finetune_velocity/tests/test_trajectories.py
"""

import importlib.util
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Register under the package names so trajectory_goals' imports resolve without importing isaacgymenvs.
vg = load("isaacgymenvs.tasks.simtoolreal.velocity_goals", "isaacgymenvs/tasks/simtoolreal/velocity_goals.py")
tg = load("isaacgymenvs.tasks.simtoolreal.trajectory_goals", "isaacgymenvs/tasks/simtoolreal/trajectory_goals.py")

VMIN = torch.tensor([-0.35, -0.2, 0.6])
VMAX = torch.tensor([0.35, 0.2, 0.95])


def _inputs(n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = VMIN + (VMAX - VMIN) * torch.rand(n, 3, generator=g)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    off = torch.zeros(n, 3)
    off[:, 0] = 0.12
    axis = torch.zeros(n, 3)
    axis[:, 1] = 1.0
    sym = torch.zeros(n, dtype=torch.bool)
    sym[: n // 2] = True
    return g, pos, q, off, axis, sym


def _params(level):
    def lerp(a, b):
        return a + (b - a) * level

    return tg.TrajParams(v_max=lerp(0.35, 0.6), bend_deg=lerp(0.0, 75.0), rot_max_deg=lerp(0.0, 120.0),
                         n_seg_max=int(round(lerp(1, 5))), cone_deg=lerp(30.0, 180.0),
                         p_arc=lerp(0.0, 0.4), arc_max_deg=lerp(0.0, 150.0))


def _check(level):
    g, pos, q, off, axis, sym = _inputs()
    P = _params(level)
    out = tg.sample_motion_chunks(pos, q, off, axis, sym, VMIN, VMAX, P, generator=g)
    f = out["feasible"]
    assert f.float().mean() > 0.6, f.float().mean()
    nw = out["n_wp"][f]
    H, V, Q, S, Pp = out["wp_head"][f], out["wp_vel"][f], out["wp_quat"][f], out["wp_s"][f], out["wp_pos"][f]
    m, K = V.shape[0], V.shape[1]
    rows = torch.arange(m)
    k = torch.arange(K).unsqueeze(0)
    valid = k < nw.unsqueeze(1)
    speed = V.norm(dim=-1)

    # speed limits, rest at the end, acceleration limit between waypoints and from the set pose
    assert speed.max() <= P.v_max + 1e-4
    assert speed[rows, nw - 1].max() < 1e-6
    assert (speed[:, 0] ** 2 <= 2 * P.a_max * S[:, 0] + 1e-4).all()
    dv2 = (speed[:, 1:] ** 2 - speed[:, :-1] ** 2).abs()
    ds = S[:, 1:] - S[:, :-1]
    assert (dv2 <= 2 * P.a_max * ds * 1.05 + 1e-4).all(), (dv2 - 2 * P.a_max * ds).max()

    # waypoints about `spacing` apart along the path (never a sliver)
    step = (H[:, 1:] - H[:, :-1]).norm(dim=-1)
    vstep = valid[:, 1:]
    assert step[vstep].max() <= P.spacing * 1.6 + 1e-4, step[vstep].max()  # chords of a curved path
    assert step[vstep].min() >= P.spacing * 0.5, step[vstep].min()

    # goal velocity points along the path (central difference) at interior moving waypoints
    fwd = H[:, 2:] - H[:, :-2]
    cos = (fwd * V[:, 1:-1]).sum(-1) / (fwd.norm(dim=-1) * V[:, 1:-1].norm(dim=-1)).clamp(min=1e-9)
    interior = valid[:, 2:] & (speed[:, 1:-1] > 1e-3)
    # central-difference chord over two 3 cm steps vs the exact tangent: tight bends (75°) allow ~35° between them
    assert cos[interior].min() > 0.7, cos[interior].min()
    assert cos[interior].mean() > 0.97, cos[interior].mean()

    # poses: unit quaternions, head = tool origin + R r_head
    assert ((Q.norm(dim=-1) - 1).abs() < 1e-4).all()
    r = off[:1].expand(m, 3).unsqueeze(1).expand(m, K, 3)
    assert ((Pp + vg.quat_rotate_xyzw(Q, r) - H).abs() < 1e-4).all()

    # feasibility: moving waypoints within the margin, end stop inside, head above the table
    mg = P.volume_margin
    assert ((Pp >= VMIN - mg - 1e-5) & (Pp <= VMAX + mg + 1e-5)).all()
    end = Pp[rows, nw - 1]
    assert ((end >= VMIN - 1e-5) & (end <= VMAX + 1e-5)).all()
    assert (H[..., 2] >= P.min_head_z - 1e-5).all()
    return out, f, P


def test_level0_straight_slow():
    out, f, P = _check(0.0)
    H, nw = out["wp_head"][f], out["n_wp"][f]
    Q = out["wp_quat"][f]
    # straight line: every waypoint on the line from the first to the end stop; no rotation along the chunk
    rows = torch.arange(H.shape[0])
    d = H[rows, nw - 1] - H[:, 0]
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    rel = H - H[:, :1]
    perp = (rel - (rel * d.unsqueeze(1)).sum(-1, keepdim=True) * d.unsqueeze(1)).norm(dim=-1)
    assert perp.max() < 1e-3, perp.max()
    dq = (Q * Q[:, :1]).sum(-1).abs()
    assert (dq > 1 - 1e-5).all()
    print(f"    level 0: feasible {f.float().mean():.2f}, waypoints {nw.float().mean():.1f}")


def test_level1_curved_fast():
    out, f, P = _check(1.0)
    nw = out["n_wp"][f]
    assert nw.max() > 12  # multi-segment chunks happen
    assert (out["n_arc"][f] > 0).float().mean() > 0.2  # arc (rotation-dominated) segments happen
    peak = out["wp_vel"][f].norm(dim=-1).max(dim=1).values
    print(f"    level 1: feasible {f.float().mean():.2f}, waypoints {nw.float().mean():.1f}, "
          f"mean peak speed {peak.mean():.2f} m/s, down {out['is_down'][f].float().mean():.2f}")


def test_slerp():
    a = vg.quat_from_axis_angle_xyzw(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([0.0]))
    b = vg.quat_from_axis_angle_xyzw(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.pi / 2]))
    mid = tg.quat_slerp_xyzw(a, b, torch.tensor([0.5]))
    ref = vg.quat_from_axis_angle_xyzw(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.pi / 4]))
    assert ((mid - ref).abs() < 1e-5).all()


if __name__ == "__main__":
    tests = [v for k_, v in sorted(globals().items()) if k_.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} tests passed")
