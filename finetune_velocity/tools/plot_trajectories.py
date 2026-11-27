"""Plot sampled motion chunks (head paths and speed profiles) at curriculum levels 0, 0.5, 1.

python finetune_velocity/tools/plot_trajectories.py finetune_velocity/figures/trajectory_chunks.png
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
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

g = torch.Generator().manual_seed(3)
n = 400
vmin = torch.tensor([-0.35, -0.2, 0.6], dtype=torch.float32)
vmax = torch.tensor([0.35, 0.2, 0.95], dtype=torch.float32)
pos = vmin + (vmax - vmin) * torch.rand(n, 3, generator=g)
q = torch.randn(n, 4, generator=g)
q = q / q.norm(dim=-1, keepdim=True)
off = torch.zeros(n, 3)
off[:, 0] = 0.12
axis = torch.zeros(n, 3)
axis[:, 1] = 1.0
sym = torch.ones(n, dtype=torch.bool)


def params(L):
    lerp = lambda a, b: a + (b - a) * L
    return tg.TrajParams(
        v_max=lerp(0.35, 0.6),
        bend_deg=lerp(0.0, 90.0),
        rot_max_deg=lerp(0.0, 150.0),
        n_seg_max=int(round(lerp(1, 6))),
        cone_deg=lerp(30.0, 180.0),
        p_arc=lerp(0.0, 0.4),
        arc_max_deg=lerp(0.0, 180.0),
    )


levels = [0.0, 0.5, 1.0]
fig = plt.figure(figsize=(17, 10))
stats = []
for c, L in enumerate(levels):
    P = params(L)
    out = tg.sample_motion_chunks(pos, q, off, axis, sym, vmin, vmax, P, generator=g)
    f = np.asarray(out["feasible"])
    H = np.asarray(out["wp_head"])
    V = np.linalg.norm(np.asarray(out["wp_vel"]), axis=-1)
    S = np.asarray(out["wp_s"])
    NW = np.asarray(out["n_wp"])
    setq = np.asarray(out["set_quat"])
    sethead = np.asarray(pos) + np.asarray(vg.quat_rotate_xyzw(torch.tensor(setq), off))
    ids = np.where(f)[0][:10]
    ax = fig.add_subplot(2, 3, c + 1, projection="3d")
    for i in ids:
        k = NW[i]
        xs = np.concatenate([sethead[i : i + 1], H[i, :k]])
        vv = np.concatenate([[0.0], V[i, :k]])
        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], color="0.75", lw=0.8)
        sc = ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], c=vv, cmap="viridis", vmin=0, vmax=0.6, s=10)
        ax.scatter(*sethead[i], marker="s", color="k", s=14)
        ax.scatter(*H[i, k - 1], marker="X", color="r", s=20)
        # velocity arrows
        ax.quiver(
            H[i, :k, 0],
            H[i, :k, 1],
            H[i, :k, 2],
            *[np.asarray(out["wp_vel"])[i, :k, j] * 0.15 for j in range(3)],
            color="tab:orange",
            lw=0.6,
        )
    ax.set_title(f"level {L:.1f}: head paths", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax2 = fig.add_subplot(2, 3, c + 4)
    for i in ids:
        k = NW[i]
        ax2.plot(np.concatenate([[0.0], S[i, :k]]), np.concatenate([[0.0], V[i, :k]]), marker="o", ms=3, lw=1)
    ax2.set_xlabel("arc length from set pose (m)")
    ax2.set_xlim(0, 0.8)
    ax2.set_ylabel("goal head speed |v_g| (m/s)")
    ax2.set_title(
        f"level {L:.1f}: v_max {P.v_max:.2f} m/s, 1-{P.n_seg_max} segments,\nbend {P.bend_deg:.0f}°, arcs {P.p_arc:.0%} (up to {P.arc_max_deg:.0f}°)",
        fontsize=10,
    )
    ax2.set_ylim(0, 0.65)
    ax2.grid(alpha=0.3)
    stats.append((L, f.mean(), NW[f].mean(), np.asarray(out["length"])[f].mean(), V[f].max(1).mean()))
fig.subplots_adjust(right=0.9, hspace=0.35, wspace=0.25)
cax = fig.add_axes([0.92, 0.55, 0.012, 0.32])
fig.colorbar(sc, cax=cax, label="|v_g| (m/s)")
fig.suptitle(
    "Sampled motion chunks, 10 per level. Top: head paths (black square = set pose, red X = end stop, dots = waypoints colored by |v_g|, arrows = v_g). Bottom: |v_g| along the path.",
    fontsize=11,
)
plt.savefig(sys.argv[1], dpi=110)
for s in stats:
    print("level %.1f feasible %.2f  waypoints %.1f  length %.3f m  mean peak speed %.3f m/s" % s)
