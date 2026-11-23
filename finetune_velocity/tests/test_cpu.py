"""CPU tests for the velocity fine-tuning code (no Isaac Gym, no GPU). Run on the Zaratan login node:

  cd ~/scratch.cmsc848x/simtoolreal && source .venv/bin/activate
  python finetune_velocity/tests/test_cpu.py

The modules are loaded from their files so that importing them does not pull in isaacgym.
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


vg = load("velocity_goals", "isaacgymenvs/tasks/simtoolreal/velocity_goals.py")
ec = load("expand_checkpoint", "finetune_velocity/expand_checkpoint.py")


def angle_deg(a, b):
    c = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
    return torch.rad2deg(torch.acos(c.clamp(-1, 1)))


def random_quat(n, g):
    q = torch.randn(n, 4, generator=g)
    return q / q.norm(dim=-1, keepdim=True)


def test_quaternions():
    g = torch.Generator().manual_seed(0)
    q1, q2 = random_quat(500, g), random_quat(500, g)
    v = torch.randn(500, 3, generator=g)
    lhs = vg.quat_rotate_xyzw(vg.quat_mul_xyzw(q1, q2), v)
    rhs = vg.quat_rotate_xyzw(q1, vg.quat_rotate_xyzw(q2, v))
    assert torch.allclose(lhs, rhs, atol=1e-5)
    # 90 deg about z maps x to y
    q = vg.quat_from_axis_angle_xyzw(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([math.pi / 2]))
    assert torch.allclose(vg.quat_rotate_xyzw(q, torch.tensor([[1.0, 0.0, 0.0]])), torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_cone_sample():
    g = torch.Generator().manual_seed(1)
    c = torch.randn(4000, 3, generator=g)
    c = c / c.norm(dim=-1, keepdim=True)
    for deg in (10.0, 30.0, 90.0):
        d = vg.cone_sample(c, deg, g)
        assert torch.allclose(d.norm(dim=-1), torch.ones(4000), atol=1e-5)
        assert angle_deg(d, c).max() <= deg + 1e-3
    d = vg.cone_sample(c, 180.0, g)
    assert d.mean(0).norm() < 0.08  # roughly uniform on the sphere


def test_velocity_match():
    vgoal = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
    vhead = torch.tensor([[0.4, 0.0, 0.0], [0.3, 0.0, 0.0]])
    q, dv, speed = vg.velocity_match(vhead, vgoal, 0.25)
    assert abs(q[0].item() - 1.0) < 1e-6
    assert abs(q[1].item() - math.exp(-1.0)) < 1e-5  # |dv| = 0.25 |v_g| -> 1/e
    assert torch.allclose(speed, torch.tensor([0.4, 0.4]))


def test_sample_velocity_goals():
    g = torch.Generator().manual_seed(2)
    n = 4000
    vmin = torch.tensor([-0.35, -0.2, 0.6])
    vmax = torch.tensor([0.35, 0.2, 0.95])
    pos = vmin + (vmax - vmin) * torch.rand(n, 3, generator=g)
    quat = random_quat(n, g)
    head_offset = torch.zeros(n, 3)
    head_offset[:, 0] = 0.12
    axis = torch.zeros(n, 3)
    axis[:, 1] = 1.0
    sym = torch.zeros(n, dtype=torch.bool)
    sym[: n // 2] = True
    params = vg.VelocityGoalParams(cone_deg=30.0, speed_lo=0.3, speed_hi=0.6)
    out = vg.sample_velocity_goals(pos, quat, head_offset, axis, sym, vmin, vmax, params, generator=g)
    f = out["feasible"]
    assert 0.2 < f.float().mean() <= 1.0, f.float().mean()
    d = out["direction"][f]
    v = out["v_goal"][f]
    q = out["quat"][f]
    p = pos[f]
    lead = out["leadin_pos"][f]
    speed = v.norm(dim=-1)
    assert speed.min() >= 0.3 - 1e-5 and speed.max() <= 0.6 + 1e-5
    assert torch.allclose(v / speed.unsqueeze(-1), d, atol=1e-5)
    assert torch.allclose(lead, p - 0.12 * d, atol=1e-6)
    assert torch.allclose(q.norm(dim=-1), torch.ones(len(q)), atol=1e-5)
    # head axis within the cone of d (either face for symmetric tools)
    ax_w = vg.quat_rotate_xyzw(q, axis[f])
    ang = angle_deg(ax_w, d)
    ang = torch.where(sym[f], torch.minimum(ang, 180.0 - ang), ang)
    assert ang.max() <= 30.0 + 1e-2, ang.max()
    # all three points inside the volume, head above min_head_z
    follow = p + 0.08 * d
    head_w = vg.quat_rotate_xyzw(q, head_offset[f])
    for pt in (lead, p, follow):
        assert ((pt >= vmin - 1e-6) & (pt <= vmax + 1e-6)).all()
        assert (pt[:, 2] + head_w[:, 2] >= 0.58 - 1e-6).all()
    # downward strikes: steep; fewer than down_frac among the feasible ones (they fail the height check more often)
    down = out["is_down"][f]
    assert (d[down][:, 2] <= -0.85 + 1e-5).all()
    assert 0.02 < down.float().mean() < 0.5, down.float().mean()
    print(f"    feasible {f.float().mean():.2f}, down among feasible {down.float().mean():.2f}")
    # non-down goals keep the sampled orientation
    assert torch.allclose(q[~down], quat[f][~down], atol=1e-6)


def test_curriculum():
    c = vg.VelocityCurriculum(level=0.0, step=0.05, min_samples=100, min_steps_between=10)
    assert c.decide(10, 50, 10, 10, step_count=100) is None  # block not full
    assert c.decide(60, 100, 90, 100, step_count=100) == "up" and abs(c.level - 0.05) < 1e-9
    assert c.decide(60, 100, 90, 100, step_count=105) is None  # too soon
    assert c.decide(30, 100, 90, 100, step_count=200) == "stay"
    assert c.decide(60, 100, 30, 100, step_count=300) == "stay"  # static guard
    assert c.decide(10, 100, 90, 100, step_count=400) == "down" and c.level == 0.0
    f = vg.VelocityCurriculum(level=0.6, fixed=True, min_samples=1)
    assert f.decide(100, 100, 100, 100, step_count=10**6) is None and f.level == 0.6
    assert abs(vg.lerp([1.0, 0.25], 0.5) - 0.625) < 1e-9


def _fake_checkpoint(obs=140, state=162, p=32, hidden=64):
    """Same layout as the released checkpoint: LSTM before the actor MLP, MLP critic, SAPG extra params."""
    g = torch.Generator().manual_seed(3)
    actor = {
        "running_mean_std.running_mean": torch.randn(obs, generator=g),
        "running_mean_std.running_var": torch.rand(obs, generator=g) + 0.5,
        "running_mean_std.count": torch.tensor(1e9),
        "a2c_network.extra_params": torch.randn(6, p, generator=g),
        "a2c_network.rnn.rnn.weight_ih_l0": torch.randn(4 * hidden, obs + p, generator=g) * 0.1,
        "a2c_network.rnn.rnn.weight_hh_l0": torch.randn(4 * hidden, hidden, generator=g) * 0.1,
        "a2c_network.rnn.rnn.bias_ih_l0": torch.randn(4 * hidden, generator=g) * 0.1,
        "a2c_network.rnn.rnn.bias_hh_l0": torch.randn(4 * hidden, generator=g) * 0.1,
        "a2c_network.mu.weight": torch.randn(29, hidden, generator=g),
    }
    critic = {
        "model.running_mean_std.running_mean": torch.randn(state, generator=g),
        "model.running_mean_std.running_var": torch.rand(state, generator=g) + 0.5,
        "model.running_mean_std.count": torch.tensor(1e9),
        "model.a2c_network.extra_params": torch.randn(6, p, generator=g),
        "model.a2c_network.actor_mlp.0.weight": torch.randn(hidden, state + p, generator=g) * 0.1,
        "model.a2c_network.actor_mlp.0.bias": torch.randn(hidden, generator=g),
        "model.a2c_network.value.weight": torch.randn(1, hidden, generator=g),
    }
    return {0: {"model": actor, "assymetric_vf_nets": critic, "optimizer": {"state": {}}, "epoch": 5000, "frame": 1}}


def _actor_out(sd, obs, block):
    """Normalize, append the block's extra params, one LSTM step from zero state, mu head."""
    x = (obs - sd["running_mean_std.running_mean"]) / torch.sqrt(sd["running_mean_std.running_var"] + 1e-5)
    x = torch.clamp(x, -5, 5)
    x = torch.cat([x, sd["a2c_network.extra_params"][block]], dim=-1)
    hidden = sd["a2c_network.rnn.rnn.weight_hh_l0"].shape[1]
    lstm = torch.nn.LSTM(x.shape[-1], hidden, batch_first=True)
    with torch.no_grad():
        for k in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"):
            getattr(lstm, k).copy_(sd[f"a2c_network.rnn.rnn.{k}"])
        h, _ = lstm(x.unsqueeze(1))
    return h[:, 0] @ sd["a2c_network.mu.weight"].T


def _critic_out(sd, state, block):
    x = (state - sd["model.running_mean_std.running_mean"]) / torch.sqrt(sd["model.running_mean_std.running_var"] + 1e-5)
    x = torch.cat([torch.clamp(x, -5, 5), sd["model.a2c_network.extra_params"][block]], dim=-1)
    h = torch.nn.functional.elu(x @ sd["model.a2c_network.actor_mlp.0.weight"].T + sd["model.a2c_network.actor_mlp.0.bias"])
    return h @ sd["model.a2c_network.value.weight"].T


def test_expand_checkpoint_is_exact():
    ckpt = _fake_checkpoint()
    new, log = ec.expand_checkpoint(ckpt, 9, 11)
    ec.verify(ckpt[0], new[0], 9, 11)
    assert "optimizer" not in new[0] and new[0]["epoch"] == 0
    g = torch.Generator().manual_seed(4)
    obs = torch.randn(16, 140, generator=g)
    state = torch.randn(16, 162, generator=g)
    block = torch.randint(0, 6, (16,), generator=g)
    a_old = _actor_out(ckpt[0]["model"], obs, block)
    c_old = _critic_out(ckpt[0]["assymetric_vf_nets"], state, block)
    for scale in (0.0, 1.0, 10.0):  # whatever the new dims contain
        obs_new = torch.cat([obs, scale * torch.randn(16, 9, generator=g)], dim=-1)
        state_new = torch.cat([state, scale * torch.randn(16, 11, generator=g)], dim=-1)
        assert torch.allclose(_actor_out(new[0]["model"], obs_new, block), a_old, atol=1e-5)
        assert torch.allclose(_critic_out(new[0]["assymetric_vf_nets"], state_new, block), c_old, atol=1e-5)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} tests passed")
