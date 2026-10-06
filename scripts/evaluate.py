#!/usr/bin/env python3
"""Evaluate a model on the paper's benchmark with the released harness.

Instances: seeds 1-20 for every minimal-DFA size 2..9 (160), budget = 2x the
better of L*/TTT, deterministic short counterexamples (paper defaults).

    # sanity check (no API calls; must give 100% and Δ=0)
    python scripts/evaluate.py --backend teacher --out results/teacher

    # tuned Gemini on Vertex
    python scripts/evaluate.py --backend vertex --model projects/.../endpoints/123 \
        --project MY_PROJECT --scaffold stateless --out results/gemini-sft

    # open model served by vLLM
    python scripts/evaluate.py --backend openai --model aal --base-url http://localhost:8000/v1 \
        --out results/qwen-sft
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BUCKETS = [(2, 3), (4, 5), (6, 7), (8, 9)]


def make_model(a, n, seed):
    from aal_sft import models
    if a["backend"] == "teacher":
        return models.TeacherReplay(n, seed, a["scaffold"], a["replay_teacher"])
    if a["backend"] == "vertex":
        return models.VertexGemini(a["model"], a["project"], a["location"],
                                   thinking_level=a["thinking_level"] or None,
                                   temperature=a["temperature"])
    if a["backend"] == "openai":
        extra = json.loads(a["extra_body"]) if a["extra_body"] else None
        return models.OpenAICompatible(a["model"], a["base_url"], a["api_key"],
                                       temperature=a["temperature"], extra_body=extra,
                                       max_tokens=a["max_tokens"])
    raise ValueError(a["backend"])


def run_instance(a, n, seed):
    from aal_sft.oracle_session import build_target, game_prompt_for, strategy_results, tool_budget
    from aal_sft.upstream import load
    U = load()
    import output_paths
    from llm_runtime import LLM_interactive_game

    output_paths.set_output_dir(tempfile.mkdtemp(prefix="aal_eval_"))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        mdfa = build_target(n, seed)
        res = strategy_results(mdfa)
        budget = tool_budget(mdfa, a["budget_ratio"])
        tools = [c() for c in U.utils.get_tools()]
        game = U.game_format.GameFormat(dfa=mdfa, tools=tools, hints=["vocabulary"],
                                        max_tool_calls=budget,
                                        game_prompt=game_prompt_for(a["scaffold"]))
        model = make_model(a, n, seed)
        bridge = LLM_interactive_game(game=game)
        err = None
        try:
            bridge.run_with_chat_session(model, verbose=False, max_steps=a["max_steps"])
        except Exception as e:
            err = repr(e)
    best = min(res["LStarStrategy"].total_queries, res["TTTStrategy"].total_queries)
    turns = sum(m["role"] == "assistant" for m in model.messages)
    rec = {
        "n_states": n, "seed": seed, "scaffold": a["scaffold"],
        "success": bool(getattr(bridge, "_reached_optimal", False)),
        "tool_calls": int(bridge._call_counter),
        "model_turns": turns,
        "invalid_turns": turns - int(bridge._call_counter),
        "budget": budget,
        "lstar_calls": res["LStarStrategy"].total_queries,
        "ttt_calls": res["TTTStrategy"].total_queries,
        "delta_vs_best_classic": int(bridge._call_counter) - best,
        "mq_duplicates": len(bridge.mq_duplicate_steps),
        "eq_duplicates": len(bridge.eq_duplicate_steps),
        "eq_contradicts_mq": len({s for s, _, _ in bridge.eq_contradicts_previous_mq}),
        "usage": model.usage,
        "error": err,
    }
    return rec, {"messages": model.messages, "reasoning": model.reasoning}


def bootstrap_ci(xs, iters=100_000, seed=0):
    import numpy as np
    x = np.asarray(xs, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 2
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(iters, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(recs):
    out = {}
    for lo, hi in BUCKETS:
        b = [r for r in recs if lo <= r["n_states"] <= hi]
        if not b:
            continue
        succ = [r["success"] for r in b]
        ok = [r for r in b if r["success"]]
        ci = bootstrap_ci(succ)
        out[f"{lo}-{hi}"] = {
            "n": len(b),
            "success_rate": round(100 * sum(succ) / len(b), 1),
            "success_ci95": [round(100 * ci[0], 1), round(100 * ci[1], 1)],
            "mean_delta_calls_successful": (round(sum(r["delta_vs_best_classic"] for r in ok) / len(ok), 2)
                                            if ok else None),
            "invalid_turn_rate": round(sum(r["invalid_turns"] for r in b) / max(1, sum(r["model_turns"] for r in b)), 4),
            "errors": sum(r["error"] is not None for r in b),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["teacher", "vertex", "openai"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--thinking-level", default="MINIMAL",
                    help="Vertex: MINIMAL (tuned models) / HIGH (paper baselines) / '' to omit")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--extra-body", default=None, help="JSON passed to OpenAI-compatible requests")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--scaffold", choices=["stateless", "stateful", "think", "thought"], default="stateless",
                    help="think = stateless prompt for models with native thinking; thought = visible <THOUGHT>")
    ap.add_argument("--replay-teacher", choices=["best", "lstar", "ttt"], default="best",
                    help="teacher backend only")
    ap.add_argument("--max-tokens", type=int, default=16384, help="openai backend: max output tokens per turn")
    ap.add_argument("--n-states", default="2-9")
    ap.add_argument("--seeds", default="1-20")
    ap.add_argument("--budget-ratio", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = lambda s: range(int(s.split("-")[0]), int(s.split("-")[-1]) + 1)  # noqa: E731
    a = vars(args)
    os.makedirs(os.path.join(args.out, "transcripts"), exist_ok=True)
    res_path = os.path.join(args.out, "results.jsonl")
    done = set()
    recs = []
    if os.path.exists(res_path):  # resume
        for line in open(res_path):
            r = json.loads(line)
            recs.append(r)
            done.add((r["n_states"], r["seed"]))
    jobs = [(n, s) for n in rng(args.n_states) for s in rng(args.seeds) if (n, s) not in done]
    print(f"{len(jobs)} instances to run ({len(done)} already done)")

    with ProcessPoolExecutor(args.workers) as ex, open(res_path, "a") as f:
        futs = {ex.submit(run_instance, a, n, s): (n, s) for n, s in jobs}
        for fut in as_completed(futs):
            n, s = futs[fut]
            rec, msgs = fut.result()
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            with open(os.path.join(args.out, "transcripts", f"n{n}_s{s}.json"), "w") as tf:
                json.dump(msgs, tf, ensure_ascii=False)
            print(f"n={n} seed={s} success={rec['success']} calls={rec['tool_calls']}/{rec['budget']}"
                  f" Δ={rec['delta_vs_best_classic']}" + (f" ERROR {rec['error']}" if rec["error"] else ""),
                  flush=True)

    summary = {"config": {k: v for k, v in a.items() if k != "api_key"}, "buckets": summarize(recs),
               "overall_success": round(100 * sum(r["success"] for r in recs) / len(recs), 1)}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["buckets"], indent=2))
    print("overall success:", summary["overall_success"], "%")


if __name__ == "__main__":
    main()
