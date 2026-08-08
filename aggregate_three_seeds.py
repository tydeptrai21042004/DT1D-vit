#!/usr/bin/env python3
"""Aggregate run_summary.json files into mean ± sample standard deviation."""
import argparse, json, math
from pathlib import Path


def mean_std(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="output directory containing seed0/seed1/seed2 run summaries")
    args = p.parse_args()
    files = sorted(Path(args.root).rglob("run_summary.json"))
    rows = []
    for f in files:
        data = json.loads(f.read_text())
        if data.get("seed") in (0, 1, 2):
            rows.append((f, data))
    by_seed = {int(d["seed"]): d for _, d in rows}
    missing = [s for s in (0, 1, 2) if s not in by_seed]
    if missing:
        raise SystemExit(f"Missing seed summaries: {missing}")
    vals = [100.0 * float(by_seed[s]["test_top1"]) for s in (0, 1, 2)]
    mean, std = mean_std(vals)
    print("seeds: 0,1,2")
    print("test Acc@1 (%): " + ", ".join(f"{v:.3f}" for v in vals))
    print(f"mean ± std: {mean:.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()
