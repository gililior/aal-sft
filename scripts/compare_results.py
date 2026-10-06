#!/usr/bin/env python3
"""Side-by-side table of evaluate.py runs.

    python scripts/compare_results.py results/*
"""
import json
import os
import sys

BUCKETS = ["2-3", "4-5", "6-7", "8-9"]


def main(dirs):
    rows = []
    for d in dirs:
        p = os.path.join(d, "summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))
        rows.append((os.path.basename(d.rstrip("/")), s))
    if not rows:
        print("no summary.json found")
        return
    w = max(len(n) for n, _ in rows) + 2
    print("success % (Δ tool calls vs better of L*/TTT, successful runs) per minimal-DFA size")
    print("run".ljust(w) + "".join(b.rjust(16) for b in BUCKETS) + "overall".rjust(10))
    for name, s in rows:
        cells = []
        for b in BUCKETS:
            v = s["buckets"].get(b)
            if not v:
                cells.append("-".rjust(16))
                continue
            d = v["mean_delta_calls_successful"]
            cells.append(f"{v['success_rate']:5.1f} ({'-' if d is None else f'{d:+.1f}'})".rjust(16))
        print(name.ljust(w) + "".join(cells) + f"{s['overall_success']:9.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
