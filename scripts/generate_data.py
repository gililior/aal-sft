#!/usr/bin/env python3
"""Generate SFT data from L* / TTT trajectories.

Example:
    python scripts/generate_data.py --per-n 400 --scaffold both --out data/v1

Outputs (per scaffold):
    data/v1/<scaffold>/{train,val}.chat.jsonl    # {"messages":[...], ...metadata}  (HF / OpenAI style)
    data/v1/<scaffold>/{train,val}.gemini.jsonl  # {"contents":[...]}                (Vertex Gemini SFT)
    data/v1/stats.json
Eval set held out: the paper's instances (seeds 1-20 for every n in 2..9) and
every DFA equivalent to one of them.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EVAL_N = range(2, 10)
EVAL_SEEDS = range(1, 21)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _fp_job(args):
    n, seed = args
    from aal_sft.oracle_session import build_target, dfa_fingerprint
    try:
        return n, seed, dfa_fingerprint(_quiet(build_target, n, seed))
    except Exception as e:  # sampler can fail for tiny n / odd seeds
        return n, seed, f"ERR:{e}"


def _traj_job(args):
    n, seed, scaffolds, teacher, ratio = args
    from aal_sft.oracle_session import build_target
    from aal_sft.teacher import make_trajectory
    out = []
    target = _quiet(build_target, n, seed)
    for sc in scaffolds:
        try:
            out.append(_quiet(make_trajectory, n, seed, scaffold=sc, teacher=teacher,
                              budget_ratio=ratio, target=target))
        except Exception as e:
            out.append({"error": repr(e), "n_states": n, "seed": seed, "scaffold": sc})
    return out


def to_gemini(rec):
    return {"contents": [
        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
        for m in rec["messages"]
    ]}


def approx_tokens(rec):
    return sum(len(m["content"]) + len(m.get("reasoning_content", "")) for m in rec["messages"]) // 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=9)
    ap.add_argument("--per-n", type=int, default=300, help="target unique DFAs per state count")
    ap.add_argument("--seed-start", type=int, default=100_000, help="first training seed (eval uses 1-20)")
    ap.add_argument("--max-attempts-factor", type=int, default=30)
    ap.add_argument("--scaffold", default="stateless",
                    help="comma list of: stateless, stateful, think (open thinking models), "
                         "thought (visible <THOUGHT>, for Gemini); 'both' = stateless,stateful")
    ap.add_argument("--teacher", choices=["best", "lstar", "ttt"], default="best",
                    help="'best' = whichever of L*/TTT uses fewer queries on that DFA")
    ap.add_argument("--budget-ratio", type=float, default=2.0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="data/v1")
    args = ap.parse_args()

    from aal_sft.oracle_session import SCAFFOLDS
    scaffolds = ["stateless", "stateful"] if args.scaffold == "both" else args.scaffold.split(",")
    for sc in scaffolds:
        if sc not in SCAFFOLDS:
            ap.error(f"unknown scaffold {sc}")
    os.makedirs(args.out, exist_ok=True)

    with ProcessPoolExecutor(args.workers) as ex:
        # 1. Held-out eval fingerprints
        eval_fp = {}
        for n, s, fp in ex.map(_fp_job, [(n, s) for n in EVAL_N for s in EVAL_SEEDS]):
            eval_fp[fp] = (n, s)
        print(f"eval set: {len(eval_fp)} distinct DFAs from {len(EVAL_N) * len(EVAL_SEEDS)} instances")

        # 2. Pick unique, decontaminated training DFAs
        chosen = defaultdict(list)
        drop = Counter()
        seen = set(eval_fp)
        for n in range(args.n_min, args.n_max + 1):
            seeds = range(args.seed_start, args.seed_start + args.per_n * args.max_attempts_factor)
            for chunk_start in range(0, len(seeds), args.per_n * 2):
                chunk = seeds[chunk_start: chunk_start + args.per_n * 2]
                for _, s, fp in ex.map(_fp_job, [(n, s) for s in chunk], chunksize=16):
                    if len(chosen[n]) >= args.per_n:
                        break
                    if fp.startswith("ERR"):
                        drop["sampler_error"] += 1
                    elif fp in eval_fp:
                        drop["equivalent_to_eval"] += 1
                    elif fp in seen:
                        drop["duplicate"] += 1
                    else:
                        seen.add(fp)
                        chosen[n].append(s)
                if len(chosen[n]) >= args.per_n:
                    break
            print(f"n={n}: {len(chosen[n])} unique DFAs")
        print("dropped:", dict(drop))

        # 3. Trajectories
        jobs = [(n, s, scaffolds, args.teacher, args.budget_ratio) for n in chosen for s in chosen[n]]
        recs = [r for group in ex.map(_traj_job, jobs, chunksize=8) for r in group]

    errors = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    if errors:
        print(f"WARNING: {len(errors)} trajectories failed replay; first: {errors[0]}")

    # 4. Split by DFA (same DFA never in both train and val)
    rng = random.Random(0)
    keys = sorted({(r["n_states"], r["seed"]) for r in recs})
    rng.shuffle(keys)
    val_keys = set(keys[: int(len(keys) * args.val_frac)])

    stats = {"args": vars(args), "dropped": dict(drop), "replay_errors": len(errors), "scaffolds": {}}
    for sc in scaffolds:
        d = os.path.join(args.out, sc)
        os.makedirs(d, exist_ok=True)
        rs = [r for r in recs if r["scaffold"] == sc]
        for split in ("train", "val"):
            part = [r for r in rs if ((r["n_states"], r["seed"]) in val_keys) == (split == "val")]
            rng.shuffle(part)
            with open(os.path.join(d, f"{split}.chat.jsonl"), "w") as f:
                for r in part:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if sc != "think":  # Gemini SFT has no hidden reasoning channel
                with open(os.path.join(d, f"{split}.gemini.jsonl"), "w") as f:
                    for r in part:
                        f.write(json.dumps(to_gemini(r), ensure_ascii=False) + "\n")
        by_n = defaultdict(list)
        for r in rs:
            by_n[r["n_states"]].append(r)
        stats["scaffolds"][sc] = {
            "examples": len(rs),
            "val_examples": sum((r["n_states"], r["seed"]) in val_keys for r in rs),
            "assistant_turns": sum(r["teacher_calls"] for r in rs),
            "teacher_share": dict(Counter(r["teacher"] for r in rs)),
            "approx_tokens_max": max(approx_tokens(r) for r in rs),
            "approx_tokens_total": sum(approx_tokens(r) for r in rs),
            "per_n": {
                n: {
                    "count": len(v),
                    "mean_teacher_calls": round(mean(r["teacher_calls"] for r in v), 1),
                    "mean_lstar_calls": round(mean(r["lstar_calls"] for r in v), 1),
                    "mean_ttt_calls": round(mean(r["ttt_calls"] for r in v), 1),
                    "ttt_share": round(sum(r["teacher"] == "TTT" for r in v) / len(v), 3),
                    "mean_approx_tokens": int(mean(approx_tokens(r) for r in v)),
                }
                for n, v in sorted(by_n.items())
            },
        }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats["scaffolds"], indent=2))


if __name__ == "__main__":
    main()
