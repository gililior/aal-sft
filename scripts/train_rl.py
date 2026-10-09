#!/usr/bin/env python3
"""GRPO on the DFA game: vLLM rollouts against the paper's oracle, LoRA updates.

Each step:
  1. rank 0 saves the current adapter and hot-loads it into the vLLM server,
  2. rank 0 plays `--dfas-per-step` DFAs x `--group-size` episodes (threads -> vLLM),
  3. reward = 1 if the final equivalence query is correct, else 0; advantage =
     reward - group mean (Dr. GRPO), given to EVERY turn of the episode,
  4. all ranks run one on-policy gradient step on those turns.

Launch (see scripts/slurm/rl.sbatch): vLLM on one GPU with
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True, this script with torchrun on the others.
Rerunning with the same --out resumes from the last finished step.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "in_proj_qkv", "in_proj_z", "out_proj"]
EVAL_N, EVAL_SEEDS = range(2, 10), range(1, 21)


# ---------------------------------------------------------------------------
# vLLM client
def vllm_generate_fn(url, model_name, tok, max_new_tokens, temperature, top_p):
    import requests
    stop_texts = ["<|im_end|>", "<|endoftext|>"]

    def generate(prompt_ids):
        body = {"model": model_name, "prompt": prompt_ids, "max_tokens": max_new_tokens,
                "temperature": temperature, "top_p": top_p, "return_token_ids": True,
                "skip_special_tokens": False}
        for attempt in range(6):
            try:
                r = requests.post(f"{url}/v1/completions", json=body, timeout=1800)
                r.raise_for_status()
                ch = r.json()["choices"][0]
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(5 * 2 ** attempt)
        text = ch.get("text") or ""
        for s in stop_texts:
            text = text.replace(s, "")
        ids = ch.get("token_ids")
        if ids is None:  # older vLLM: re-tokenize (rarely differs from the sampled ids)
            ids = tok(text, add_special_tokens=False)["input_ids"]
        return {"text": text, "token_ids": ids, "finish_reason": ch.get("finish_reason")}
    return generate


def vllm_load_adapter(url, name, path):
    import requests
    last = None
    for inplace in (True, False):
        r = requests.post(f"{url}/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": os.path.abspath(path), "load_inplace": inplace},
                          timeout=600)
        if r.ok:
            return
        last = r.text
    raise RuntimeError(f"vLLM refused the adapter: {last}")


def wait_for_vllm(url, minutes=20):
    import requests
    for _ in range(minutes * 12):
        try:
            if requests.get(f"{url}/v1/models", timeout=10).ok:
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM at {url} did not come up")


# ---------------------------------------------------------------------------
def eval_fingerprints():
    from aal_sft.oracle_session import build_target, dfa_fingerprint
    fps = set()
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        for n in EVAL_N:
            for s in EVAL_SEEDS:
                fps.add(dfa_fingerprint(build_target(n, s)))
    return fps


def sample_targets(rng, k, max_n, window, eval_fps, seed_lo):
    """k training DFAs: sizes uniform over the curriculum window, never equivalent to an eval DFA."""
    from aal_sft.oracle_session import build_target, dfa_fingerprint
    out = []
    lo = max(2, max_n - window + 1)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        while len(out) < k:
            n = rng.randint(lo, max_n)
            seed = rng.randrange(seed_lo, seed_lo + 10 ** 8)
            try:
                t = build_target(n, seed)
            except Exception:
                continue
            if dfa_fingerprint(t) in eval_fps:
                continue
            out.append((n, seed, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--init-adapter", default=None, help="start from this LoRA adapter (e.g. per-turn SFT)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vllm-url", default="http://localhost:8000")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dfas-per-step", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--turns-per-episode", type=int, default=0, help="0 = train on every turn")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--budget-ratio", type=float, default=2.0)
    ap.add_argument("--max-invalid", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=24576, help="skip longer turn samples")
    ap.add_argument("--curriculum-start", type=int, default=4, help="largest DFA size at the start")
    ap.add_argument("--curriculum-window", type=int, default=4, help="sizes sampled: max_n-window+1 .. max_n")
    ap.add_argument("--promote-at", type=float, default=0.5, help="success at max_n needed to add a size")
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seed-start", type=int, default=1_000_000)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from aal_sft import hf_utils
    from aal_sft.rl_env import build_samples, play_episode

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
        cpu = dist.new_group(backend="gloo", timeout=timedelta(hours=6))
        barrier = lambda: dist.barrier(group=cpu)  # noqa: E731
    else:
        barrier = lambda: None  # noqa: E731
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    os.makedirs(args.out, exist_ok=True)
    state_dir = os.path.join(args.out, "state")
    tok = AutoTokenizer.from_pretrained(args.model)
    eot = hf_utils.end_of_turn_id(tok)

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.init_adapter:
        c = hf_utils.adapter_lora_config(args.init_adapter)
        lcfg = LoraConfig(r=c["r"], lora_alpha=c["lora_alpha"], target_modules=c["target_modules"],
                          lora_dropout=0.0, task_type="CAUSAL_LM")
    else:
        lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=DEFAULT_TARGETS,
                          lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    resume = os.path.exists(os.path.join(state_dir, "state.json"))
    if resume:
        hf_utils.load_adapter_for_training(model, os.path.join(state_dir, "adapter"))
    elif args.init_adapter:
        hf_utils.load_adapter_for_training(model, args.init_adapter)
    lm_weight = hf_utils.detach_lm_head(model)
    model.to(dev)
    if rank == 0:
        model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    ddp = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp = DDP(model, device_ids=[local], find_unused_parameters=False)

    st = {"step": 0, "max_n": args.curriculum_start, "recent": []}
    if resume:
        st = json.load(open(os.path.join(state_dir, "state.json")))
        opt.load_state_dict(torch.load(os.path.join(state_dir, "optim.pt"), map_location=dev))
        log(f"resumed at step {st['step']} (max_n={st['max_n']})")

    eval_fps = eval_fingerprints() if rank == 0 else None
    if rank == 0:
        wait_for_vllm(args.vllm_url)
        generate = vllm_generate_fn(args.vllm_url, "policy", tok, args.max_new_tokens, args.temperature, args.top_p)
    log_path = os.path.join(args.out, "rl_log.jsonl")
    batch_path = os.path.join(args.out, "batch.pkl")

    def save_adapter(path, for_vllm):
        if os.path.exists(path):
            shutil.rmtree(path)
        model.save_pretrained(path)
        if for_vllm:
            hf_utils.fix_adapter_names_for_vllm(path, args.model)

    while st["step"] < args.steps:
        step = st["step"]
        rng = random.Random(args.seed * 1_000_003 + step)
        t0 = time.time()
        # ---------------- rollouts (rank 0) ----------------
        if rank == 0:
            pol = os.path.join(args.out, "policy", f"step_{step}")
            save_adapter(pol, for_vllm=True)
            vllm_load_adapter(args.vllm_url, "policy", pol)
            old = os.path.join(args.out, "policy", f"step_{step - 2}")
            if os.path.exists(old):
                shutil.rmtree(old)
            targets = sample_targets(rng, args.dfas_per_step, st["max_n"], args.curriculum_window,
                                     eval_fps, args.train_seed_start)
            jobs = [(n, s, t) for (n, s, t) in targets for _ in range(args.group_size)]
            with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                eps = list(ex.map(lambda j: play_episode(generate, tok, j[0], j[1], target=j[2],
                                                         budget_ratio=args.budget_ratio,
                                                         max_invalid=args.max_invalid), jobs))
            groups = [eps[i:i + args.group_size] for i in range(0, len(eps), args.group_size)]
            samples, skipped = build_samples(groups, eot, args.max_new_tokens, args.turns_per_episode,
                                             args.max_len, rng)
            # balance ranks: longest first, round-robin
            samples.sort(key=lambda s: -len(s["input_ids"]))
            with open(batch_path, "wb") as f:
                pickle.dump(samples, f)
            t_roll = time.time() - t0
            by_n = defaultdict(list)
            for e in eps:
                by_n[e.n_states].append(e)
            n_turns = sum(len(e.turns) for e in eps)
            stats = {
                "step": step, "max_n": st["max_n"],
                "success": round(sum(e.success for e in eps) / len(eps), 3),
                "success_by_n": {n: round(sum(e.success for e in v) / len(v), 3) for n, v in sorted(by_n.items())},
                "mean_calls_success": (round(sum(e.calls for e in eps if e.success) /
                                             max(1, sum(e.success for e in eps)), 1)),
                "invalid_turn_rate": round(sum(e.invalid for e in eps) / max(1, n_turns), 3),
                "turns": n_turns,
                "mean_gen_tokens": round(sum(len(t.completion_ids) for e in eps for t in e.turns) / max(1, n_turns)),
                "truncated_turns": sum(t.finish_reason == "length" for e in eps for t in e.turns),
                "groups_with_signal": sum(len({e.reward for e in g}) > 1 for g in groups),
                "samples": len(samples), "skipped_long": skipped,
                "errors": sum(e.error is not None for e in eps),
                "rollout_s": round(t_roll),
            }
        barrier()
        # ---------------- update (all ranks) ----------------
        t1 = time.time()
        samples = pickle.load(open(batch_path, "rb"))
        mine = samples[rank::world]
        if not mine:  # every rank must take part in the synced backward
            mine = [{"input_ids": [eot, eot], "n_prompt": 1, "coef": 0.0}]
        ddp.train()
        for i, s in enumerate(mine):
            sync = (i == len(mine) - 1)
            ctx = ddp.no_sync() if (world > 1 and not sync) else contextlib.nullcontext()
            with ctx:
                ids = torch.tensor([s["input_ids"]], device=dev)
                out = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                hidden = out.logits[0, s["n_prompt"] - 1:-1]       # lm_head is Identity: hidden states
                target = ids[0, s["n_prompt"]:]
                ce = hf_utils.chunked_ce_sum(hidden, lm_weight, target)
                (ce * (s["coef"] * world)).backward()             # DDP averages over ranks
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
        opt.step()
        opt.zero_grad(set_to_none=True)
        barrier()
        # ---------------- bookkeeping (rank 0) ----------------
        if rank == 0:
            at_max = [s_ for n, s_ in stats["success_by_n"].items() if n == st["max_n"]]
            st["recent"] = (st["recent"] + at_max)[-2:]
            promoted = False
            if st["max_n"] < 9 and len(st["recent"]) >= 2 and min(st["recent"]) >= args.promote_at:
                st["max_n"] += 1
                st["recent"] = []
                promoted = True
            st["step"] = step + 1
            stats.update({"train_s": round(time.time() - t1), "grad_norm": round(gnorm, 4), "promoted": promoted})
            with open(log_path, "a") as f:
                f.write(json.dumps(stats) + "\n")
            log(f"step {step}: success {stats['success']:.2f} {stats['success_by_n']} | "
                f"invalid {stats['invalid_turn_rate']:.2f} | gen {stats['mean_gen_tokens']} tok/turn | "
                f"{stats['samples']} samples | rollout {stats['rollout_s']}s train {stats['train_s']}s"
                + (f" | curriculum -> {st['max_n']} states" if promoted else ""))
            tmp = state_dir + ".tmp"
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            os.makedirs(tmp)
            model.save_pretrained(os.path.join(tmp, "adapter"))
            torch.save(opt.state_dict(), os.path.join(tmp, "optim.pt"))
            json.dump(st, open(os.path.join(tmp, "state.json"), "w"))
            if os.path.exists(state_dir):
                shutil.rmtree(state_dir)
            os.rename(tmp, state_dir)
            if st["step"] % args.snapshot_every == 0:
                save_adapter(os.path.join(args.out, "snapshots", f"step_{st['step']}"), for_vllm=True)
        barrier()
        if rank != 0:
            st = json.load(open(os.path.join(state_dir, "state.json")))

    if rank == 0:
        save_adapter(os.path.join(args.out, "final"), for_vllm=True)
        log(f"done: {st['step']} steps; adapter in {os.path.join(args.out, 'final')}")
    barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
