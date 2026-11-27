"""Run the full DexToolBench evaluation (24 object/task pairs) for one policy, then summarize it.

Same protocol as dextoolbench/run_all_evals_isaacgym.py (10 episodes per pair, fixed goal trajectories, success = %
of trajectory goals reached), but with the policy, run name and extra overrides as arguments, and a summary at the end.

Velocity fine-tuned policies: pass --fixed-level <final vel/level of training> so the arm runs at the speed the policy
was trained with. Fixed goal trajectories turn velocity goals off automatically (all goals static, v_g = 0).

Examples:
  python finetune_velocity/run_dextoolbench_eval.py --name pretrained --config pretrained_policy/config.yaml \
      --checkpoint pretrained_policy/model.pth
  python finetune_velocity/run_dextoolbench_eval.py --name v2 --config runs/31_vel_ft_v2/config.yaml \
      --checkpoint runs/31_vel_ft_v2/nn/last_31_vel_ft_v2_ep_3500_rew__10311.79_.pth --fixed-level 0.10
  python finetune_velocity/run_dextoolbench_eval.py --summarize evals/ft_v2 evals/ft_pretrained
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "dextoolbench" / "eval_isaacgym.py"

OBJECTS = {
    "hammer": ["claw_hammer", "mallet_hammer"],
    "spatula": ["flat_spatula", "spoon_spatula"],
    "eraser": ["flat_eraser", "handle_eraser"],
    "screwdriver": ["long_screwdriver", "short_screwdriver"],
    "marker": ["sharpie_marker", "staples_marker"],
    "brush": ["blue_brush", "red_brush"],
}
TASKS = {
    "hammer": ["swing_down", "swing_side"],
    "spatula": ["serve_plate", "flip_over"],
    "eraser": ["wipe_smile", "wipe_c"],
    "screwdriver": ["spin_vertical", "spin_horizontal"],
    "marker": ["draw_smile", "write_c"],
    "brush": ["sweep_forward", "sweep_right"],
}


def summarize(run_dir: Path) -> str:
    per_cat = defaultdict(list)
    lines = []
    for f in sorted(run_dir.glob("*/*/*/eval.json")):
        cat, obj, task = f.parts[-4], f.parts[-3], f.parts[-2]
        d = json.loads(f.read_text())
        per_cat[cat].append(d["avg_goal_pct"])
        lines.append(f"  {cat:12s} {obj:18s} {task:16s} {d['avg_goal_pct']:6.1f}%  ({len(d['episode_goal_pcts'])} ep)")
    n = sum(len(v) for v in per_cat.values())
    out = [f"{run_dir}: {n}/24 pairs"]
    for cat in OBJECTS:
        if per_cat[cat]:
            v = per_cat[cat]
            out.append(f"  {cat:12s} {sum(v) / len(v):6.1f}%  ({len(v)} pairs)")
    allv = [x for v in per_cat.values() for x in v]
    if allv:
        out.append(f"  {'overall':12s} {sum(allv) / len(allv):6.1f}%")
    return "\n".join(out + ["  per pair:"] + lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="output goes to evals/ft_<name>/")
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--fixed-level", type=float, default=None,
                    help="velocity fine-tuned policies: curriculum level to freeze (sets the arm speed)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--summarize", nargs="*", default=None, help="only print summaries of these eval folders")
    args = ap.parse_args()

    if args.summarize is not None:
        for d in args.summarize:
            print(summarize(Path(d)))
        return

    assert args.name and args.config and args.checkpoint, "--name, --config and --checkpoint are required"
    out_root = Path("evals") / f"ft_{args.name}"
    pairs = [(c, o, t) for c in OBJECTS for o in OBJECTS[c] for t in TASKS[c]]
    for i, (cat, obj, task) in enumerate(pairs):
        out = out_root / cat / obj / task
        if (out / "eval.json").exists():
            print(f"[{i + 1}/{len(pairs)}] {cat}/{obj}/{task}: done already, skipping")
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--object_category", cat, "--object_name", obj, "--task_name", task,
            "--checkpoint_path", args.checkpoint, "--config_path", args.config,
            "--output_dir", str(out), "--num_episodes", str(args.episodes),
            "--downsample_factor", "1", "--policy_name", args.name,
        ]
        if args.fixed_level is not None:
            cmd += ["--extra_overrides", f"task.env.velocity.curriculum.fixedLevel={args.fixed_level}"]
        t0 = time.time()
        print(f"[{i + 1}/{len(pairs)}] {cat}/{obj}/{task}", flush=True)
        r = subprocess.run(cmd)
        # Isaac Gym often segfaults at teardown after writing eval.json: judge by the file, not the exit code.
        if not (out / "eval.json").exists():
            print(f"  FAILED (exit {r.returncode}), no eval.json", flush=True)
        else:
            print(f"  {json.loads((out / 'eval.json').read_text())['avg_goal_pct']:.1f}% goals, "
                  f"{time.time() - t0:.0f} s", flush=True)

    summary = summarize(out_root)
    (out_root / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
