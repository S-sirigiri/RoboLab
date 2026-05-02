#!/usr/bin/env python
"""Offline analysis for collision-reporter ``.npz`` files.

Run after a rollout::

    cd src/RoboLab
    .venv/bin/python scripts/analyze_collisions.py output/<run_name>

Recursively finds every ``collisions_run_*.npz`` and prints:
  * fraction of trajectory in collision (per env, per run, aggregated)
  * top obstacles by per-step contribution (which body got closest to which
    obstacle, when in collision)
  * total run-time spent in collision

Pass ``--csv <path>`` to dump a per-step CSV for plotting.
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import numpy as np


def load_run(npz_path: str) -> dict:
    """Lazy-load a single collisions_run_*.npz. Returns a dict of arrays
    plus the static metadata keys."""
    with np.load(npz_path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def _per_env_summary(data: dict) -> list[dict]:
    """Aggregate per-env stats from one run's records."""
    env_ids = data["env_ids"]
    in_coll = data["in_collision"]
    min_clear = data["min_clearance_m"]
    rows = []
    for eid in sorted(np.unique(env_ids)):
        mask = env_ids == eid
        n_total = int(mask.sum())
        n_coll = int(in_coll[mask].sum())
        rows.append(
            {
                "env_id": int(eid),
                "steps_total": n_total,
                "steps_in_collision": n_coll,
                "fraction_in_collision": n_coll / n_total if n_total else 0.0,
                "min_clearance_m": float(np.min(min_clear[mask])) if n_total else float("nan"),
                "mean_clearance_m": float(np.mean(min_clear[mask])) if n_total else float("nan"),
            }
        )
    return rows


def _top_offending_bodies(data: dict, top_k: int = 5) -> list[tuple[str, int]]:
    """Which robot body got into collision most often (across all envs/steps)."""
    body_clr = data["body_clearance_m"]  # (N, B)
    body_names = list(data["body_names"])
    in_coll_mask = body_clr < 0
    counts = in_coll_mask.sum(axis=0).astype(int)
    order = np.argsort(-counts)
    return [(body_names[i], int(counts[i])) for i in order[: top_k] if counts[i] > 0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a single .npz, or a directory to recurse for collisions_run_*.npz")
    parser.add_argument("--csv", help="Optional per-step CSV dump path")
    parser.add_argument("--top-bodies", type=int, default=5)
    args = parser.parse_args()

    if os.path.isfile(args.path):
        files = [args.path]
    else:
        files = sorted(glob(os.path.join(args.path, "**", "collisions_run_*.npz"), recursive=True))
    if not files:
        print(f"No collisions_run_*.npz found under {args.path}", file=sys.stderr)
        return 1

    print(f"Found {len(files)} collision-log file(s).\n")
    overall_total = 0
    overall_coll = 0
    overall_min_clear = float("inf")
    csv_lines: list[str] = []
    if args.csv:
        csv_lines.append("file,env_id,step,in_collision,num_links_in_collision,min_clearance_m,grasped_count,grasped_names\n")

    for path in files:
        data = load_run(path)
        rel = os.path.relpath(path, os.path.commonpath([args.path, path]) if os.path.isdir(args.path) else os.path.dirname(path))
        print(f"=== {rel} ===")
        print(f"  obstacles: {list(data['obstacle_names'])}")
        print(f"  body_radius={float(data['body_radius_m']):.3f} m, "
              f"obstacle_pad={float(data['obstacle_padding_m']):.3f} m, "
              f"records={len(data['steps'])}")
        rows = _per_env_summary(data)
        for r in rows:
            print(
                f"  env{r['env_id']:>2}: "
                f"{r['steps_in_collision']:>4} / {r['steps_total']:>4} steps "
                f"({r['fraction_in_collision']*100:5.1f}%)  "
                f"min_clearance={r['min_clearance_m']*1000:+7.2f} mm  "
                f"mean={r['mean_clearance_m']*1000:+7.2f} mm"
            )
            overall_total += r["steps_total"]
            overall_coll += r["steps_in_collision"]
            overall_min_clear = min(overall_min_clear, r["min_clearance_m"])

        offenders = _top_offending_bodies(data, top_k=args.top_bodies)
        if offenders:
            print(f"  top offending bodies (steps in collision):")
            for name, count in offenders:
                print(f"    {name:<40} {count}")
        print()

        if args.csv:
            steps = data["steps"]
            env_ids = data["env_ids"]
            in_coll = data["in_collision"]
            n_links = data["num_links_in_collision"]
            min_clear = data["min_clearance_m"]
            grasped_count = data["grasped_count"]
            grasped_names = data["grasped_names"]
            for i in range(len(steps)):
                csv_lines.append(
                    f"{rel},{int(env_ids[i])},{int(steps[i])},{int(in_coll[i])},"
                    f"{int(n_links[i])},{float(min_clear[i]):.6f},"
                    f"{int(grasped_count[i])},\"{grasped_names[i]}\"\n"
                )

    print("=== overall ===")
    print(
        f"  {overall_coll} / {overall_total} steps in collision "
        f"({(overall_coll/overall_total*100 if overall_total else 0):5.1f}%)  "
        f"min_clearance={overall_min_clear*1000:+7.2f} mm"
    )

    if args.csv:
        with open(args.csv, "w") as f:
            f.writelines(csv_lines)
        print(f"\nPer-step CSV written: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
