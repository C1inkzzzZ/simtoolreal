"""Compare the geometry of sampled motion chunks with DexToolBench's task trajectories (histograms + summary).

python finetune_velocity/tools/compare_dextoolbench.py finetune_velocity/figures/geometry_vs_dextoolbench.png
"""

import importlib.util
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vg = _load("isaacgymenvs.tasks.simtoolreal.velocity_goals", "isaacgymenvs/tasks/simtoolreal/velocity_goals.py")
tg = _load("isaacgymenvs.tasks.simtoolreal.trajectory_goals", "isaacgymenvs/tasks/simtoolreal/trajectory_goals.py")
import glob
import json

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEAD = 0.10


def qrot(q, v):
    qv = q[..., :3]
    w = q[..., 3:4]
    t = 2 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def qang(a, b):
    return 2 * np.degrees(np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))


def stats(p, q):
    """p (T,3) tool origin, q (T,4): per-step head distance, rotation, turning angle; head/origin path ratio."""
    h = p + qrot(q, np.broadcast_to([HEAD, 0, 0], p.shape))
    dh = np.diff(h, axis=0)
    dhn = np.linalg.norm(dh, axis=1)
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
    rot = qang(q[1:], q[:-1])
    keep = dhn > 1e-4
    tv = dh[keep] / dhn[keep, None]
    turn = np.degrees(np.arccos(np.clip((tv[1:] * tv[:-1]).sum(1), -1, 1))) if len(tv) > 1 else np.array([0.0])
    return dict(
        step=dhn * 100,
        rot=rot,
        turn=turn,
        ratio=dhn.sum() / max(dp.sum(), 1e-9),
        length=dhn.sum(),
        rotcm=np.array([rot.sum() / max(dhn.sum() * 100, 1e-9)]),
    )


# DexToolBench
dtb = []
for f in sorted(glob.glob(os.path.join(ROOT, "dextoolbench/trajectories") + "/*/*/*.json")):
    G = np.array(json.load(open(f))["goals"])
    st = stats(G[:, :3], G[:, 3:7])
    st["name"] = f.split("trajectories/")[1][:-5]
    # same 3 cm spacing as the sampler for the turning angle (removes jitter between closely spaced demo goals)
    h = G[:, :3] + qrot(G[:, 3:7], np.broadcast_to([HEAD, 0, 0], G[:, :3].shape))
    s_ = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(h, axis=0), axis=1))])
    sg = np.arange(0, s_[-1], 0.03)
    hr = np.stack([np.interp(sg, s_, h[:, j]) for j in range(3)], 1)
    d_ = np.diff(hr, axis=0)
    d_ = d_ / np.maximum(np.linalg.norm(d_, axis=1, keepdims=True), 1e-9)
    st["turn"] = np.degrees(np.arccos(np.clip((d_[1:] * d_[:-1]).sum(1), -1, 1)))
    dtb.append(st)


def params(L, **kw):
    lerp = lambda a, b: a + (b - a) * L
    d = dict(
        v_max=lerp(0.35, 0.6),
        bend_deg=lerp(0.0, float(os.environ.get("BEND_END", 90))),
        rot_max_deg=lerp(0.0, float(os.environ.get("ROT_END", 150))),
        n_seg_max=int(round(lerp(1, int(os.environ.get("NSEG_END", 6))))),
        cone_deg=lerp(30.0, 180.0),
        p_arc=lerp(0.0, 0.4),
        arc_max_deg=lerp(0.0, float(os.environ.get("ARC_END", 180))),
    )
    d.update(kw)
    return tg.TrajParams(**d)


g = torch.Generator().manual_seed(5)
n = 1500
vmin = torch.tensor([-0.35, -0.2, 0.6], dtype=torch.float32)
vmax = torch.tensor([0.35, 0.2, 0.95], dtype=torch.float32)
pos = vmin + (vmax - vmin) * torch.rand(n, 3, generator=g)
q = torch.randn(n, 4, generator=g)
q = q / q.norm(dim=-1, keepdim=True)
off = torch.zeros(n, 3)
off[:, 0] = HEAD
axis = torch.zeros(n, 3)
axis[:, 1] = 1.0
sym = torch.ones(n, dtype=torch.bool)
extra = eval("dict(" + (sys.argv[2] if len(sys.argv) > 2 else "") + ")")
samp = {}
for L in (0.5, 1.0):
    out = tg.sample_motion_chunks(pos, q, off, axis, sym, vmin, vmax, params(L, **extra), generator=g)
    f = np.asarray(out["feasible"])
    P = np.asarray(out["wp_pos"])
    Q = np.asarray(out["wp_quat"])
    NW = np.asarray(out["n_wp"])
    sq = np.asarray(out["set_quat"])
    sp = np.asarray(pos)
    S = []
    for i in np.where(f)[0]:
        k = NW[i]
        pp = np.concatenate([sp[i : i + 1], P[i, :k]])
        qq = np.concatenate([sq[i : i + 1], Q[i, :k]])
        S.append(stats(pp, qq))
    V = np.linalg.norm(np.asarray(out["wp_vel"])[f], axis=-1)
    samp[L] = (S, f.mean(), V.max(1), np.asarray(out["n_arc"])[f])


def cat(S, k):
    return np.concatenate([s[k] for s in S])


def per(S, k):
    return np.array([s[k] for s in S])


fig, axs = plt.subplots(1, 5, figsize=(20, 3.8))
keys = [
    ("step", "head step between waypoints (cm)", cat, np.linspace(0, 6, 31)),
    ("rotcm", "orientation change per cm of head travel (deg/cm), per trajectory", cat, np.linspace(0, 12, 25)),
    ("turn", "path turning angle per 3 cm (deg)", cat, np.linspace(0, 180, 37)),
    ("ratio", "head path / handle path, per trajectory", per, np.linspace(0.6, 2.2, 33)),
    ("length", "head path length per trajectory/chunk (m)", per, np.linspace(0, 1.5, 31)),
]
for ax, (k, title, fn, bins) in zip(axs, keys):
    ax.hist(
        np.clip(fn(dtb, k), bins[0], bins[-1]),
        bins=bins,
        density=True,
        alpha=0.5,
        label="DexToolBench (24 tasks)",
        color="tab:gray",
    )
    for L, c in ((0.5, "tab:blue"), (1.0, "tab:orange")):
        ax.hist(
            np.clip(fn(samp[L][0], k), bins[0], bins[-1]),
            bins=bins,
            density=True,
            histtype="step",
            lw=1.8,
            color=c,
            label=f"sampler level {L}",
        )
    ax.set_title(title, fontsize=10)
axs[0].legend(fontsize=8)
fig.suptitle("Trajectory geometry: sampler vs DexToolBench task trajectories (head offset 10 cm for both)", fontsize=11)
plt.tight_layout()
plt.savefig(sys.argv[1], dpi=110)
for name, arr in (("DexToolBench", dtb),) + tuple((f"level {L}", samp[L][0]) for L in samp):
    print(
        f"{name:13s} rot/cm med {np.median(cat(arr, 'rotcm')):.2f} p90 {np.percentile(cat(arr, 'rotcm'), 90):.2f} | step {np.median(cat(arr, 'step')):.1f} cm  rot/step med {np.median(cat(arr, 'rot')):.1f} p90 {np.percentile(cat(arr, 'rot'), 90):.1f}  "
        f"turn med {np.median(cat(arr, 'turn')):.1f} p90 {np.percentile(cat(arr, 'turn'), 90):.1f}  ratio med {np.median(per(arr, 'ratio')):.2f} p90 {np.percentile(per(arr, 'ratio'), 90):.2f}  "
        f"length med {np.median(per(arr, 'length')):.2f} max {per(arr, 'length').max():.2f}"
    )
for L in samp:
    S, fe, pk, na = samp[L]
    print(
        f"level {L}: feasible {fe:.2f}, peak speed mean {pk.mean():.2f} p90 {np.percentile(pk, 90):.2f}, chunks with an arc {np.mean(na > 0):.2f}"
    )
