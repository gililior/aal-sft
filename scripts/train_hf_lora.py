#!/usr/bin/env python3
"""LoRA SFT of an open-weights chat model on the teacher trajectories.

Loss is applied to assistant turns only (the model learns to choose queries and
hypotheses, not to predict oracle answers).

Thinking data (`think` scaffold, assistant turns carry reasoning_content) is
trained per turn, matching inference where earlier reasoning is dropped:

    pip install "transformers>=4.46" "peft>=0.13" accelerate datasets torch
    python scripts/train_hf_lora.py --model Qwen/Qwen3.5-4B \
        --data data/think_v1/lstar/think --out runs/qwen3.5-4b-lstar
    python scripts/train_hf_lora.py --model Qwen/Qwen3.5-4B \
        --data data/think_v1/ttt/think --out runs/qwen3.5-4b-ttt

Single GPU (40-80GB) for 4B with bf16 + gradient checkpointing.
Serve for evaluation with vLLM (the reasoning parser keeps thinking out of the
visible transcript, exactly as in training):
    vllm serve Qwen/Qwen3.5-4B --language-model-only --enable-lora --max-lora-rank 32 \
        --reasoning-parser qwen3 --lora-modules aal=runs/qwen3.5-4b-ttt/final --max-model-len 40960
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys


# Attention + MLP projections, plus Qwen3.5 / Qwen3-Next Gated DeltaNet projections
# (in_proj_qkv, in_proj_z, out_proj), which hold 3 of every 4 token-mixing layers.
DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "in_proj_qkv", "in_proj_z", "out_proj"]


def fix_adapter_names_for_vllm(adapter_dir, base_model):
    """Rename adapter weights to the multimodal checkpoint's naming when needed.

    AutoModelForCausalLM loads Qwen3.5 (a vision-language checkpoint) as its text-only
    class, so PEFT saves `model.layers.*`. vLLM serves the checkpoint as the VL class and
    expects `model.language_model.layers.*`; unmatched LoRA weights would be ignored.
    """
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(base_model)
    if getattr(cfg, "vision_config", None) is None:
        return False
    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    sd = load_file(path)
    if any(".language_model." in k for k in sd):
        return False
    out = {k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers."): v
           for k, v in sd.items()}
    save_file(out, path, metadata={"format": "pt"})
    print(f"renamed {len(out)} adapter tensors to model.language_model.* for vLLM")
    return True


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def _common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def tokenize_last_turn(tok, messages, template_kwargs, max_len):
    """One per-turn example: loss on the final assistant turn (reasoning + action) only."""
    def ids(msgs, add_gen):
        return list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_gen, **template_kwargs))

    full = ids(messages, False)
    if len(full) > max_len:
        return None
    # Start of the target = end of the prompt. Use the common prefix so a token
    # merge at the boundary can't break masking.
    start = _common_prefix(full, ids(messages[:-1], True))
    labels = [-100] * start + full[start:]
    return {"input_ids": full, "labels": labels}


def template_renders_reasoning(tok, template_kwargs) -> bool:
    probe = [{"role": "user", "content": "hi"},
             {"role": "assistant", "reasoning_content": "PROBE_REASONING_42", "content": "ok"}]
    return "PROBE_REASONING_42" in tok.apply_chat_template(probe, tokenize=False, **template_kwargs)


def end_of_turn_id(tok, template_kwargs) -> int:
    """The token that closes a chat turn (<|im_end|>, <|eot_id|>, <end_of_turn>, ...)."""
    ids = list(tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=True,
                                       add_generation_prompt=False, **template_kwargs))
    while ids and tok.decode([ids[-1]]).strip() == "":
        ids.pop()
    return ids[-1]


def _span_start(full, hist_ids, hist_gen_ids):
    """Index in `full` where the assistant's own tokens begin.

    The generation prompt is hist_gen_ids[len(common prefix with hist_ids):], e.g.
    <|im_start|> assistant \n [<think> \n]. Mid-conversation turns don't always
    repeat all of it (Qwen3.5 adds <think> only when generating), so walk it token
    by token and stop at the first mismatch.
    """
    k = _common_prefix(full, hist_ids)
    gen = hist_gen_ids[_common_prefix(hist_ids, hist_gen_ids):]
    j = 0
    while k < len(full) and j < len(gen) and full[k] == gen[j]:
        k += 1
        j += 1
    return k


def tokenize_with_mask(tok, messages, template_kwargs, max_len, eot=None):
    """input_ids for the full chat, labels = -100 except on assistant tokens.

    A turn's span starts after its role header and runs through the template's
    end-of-turn token. (Rendering "history up to this turn" is not reliable: Qwen3
    and Qwen3.5 render the final assistant turn differently from earlier ones.)
    """
    def ids(msgs, add_gen):
        return list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_gen, **template_kwargs))

    full = ids(messages, False)
    if len(full) > max_len:
        return None
    if eot is None:
        eot = end_of_turn_id(tok, template_kwargs)
    labels = [-100] * len(full)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        hist = messages[:i]
        start = _span_start(full, ids(hist, False), ids(hist, True))
        try:
            end = full.index(eot, start) + 1
        except ValueError:
            end = len(full)
        labels[start:end] = full[start:end]
    return {"input_ids": full, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--batch-tokens", type=int, default=65536, help="approx tokens per optimizer step")
    ap.add_argument("--template-kwargs", default="{}", help='e.g. \'{"enable_thinking": false}\'')
    ap.add_argument("--target-modules", default=",".join(DEFAULT_TARGETS),
                    help="LoRA targets; names missing from a model are skipped (the Gated DeltaNet "
                         "projections exist only in hybrid models such as Qwen3.5)")
    ap.add_argument("--limit", type=int, default=None, help="debug: use only N trajectories")
    ap.add_argument("--max-val", type=int, default=300, help="cap on validation examples")
    ap.add_argument("--mq-turns-per-traj", type=int, default=8,
                    help="thinking data: MQ turns sampled per trajectory (all EQ turns always kept; 0 = all turns)")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    tkw = json.loads(args.template_kwargs)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from aal_sft.hf_examples import inline_reasoning, per_turn_examples

    native_reasoning = None

    def build(split):
        nonlocal native_reasoning
        p = os.path.join(args.data, f"{split}.chat.jsonl")
        if not os.path.exists(p):
            return []
        recs = load_jsonl(p)[: args.limit]
        thinking = any("reasoning_content" in m for m in recs[0]["messages"])
        if not thinking:
            eot = end_of_turn_id(tok, tkw)
            out = [tokenize_with_mask(tok, r["messages"], tkw, args.max_len, eot) for r in recs]
        else:
            if native_reasoning is None:
                native_reasoning = template_renders_reasoning(tok, tkw)
                print("template renders reasoning_content natively:", native_reasoning)
            rng = random.Random(0 if split == "train" else 1)
            out = []
            for r in recs:
                for ex in per_turn_examples(r["messages"], args.mq_turns_per_traj, rng):
                    ex = ex if native_reasoning else inline_reasoning(ex)
                    out.append(tokenize_last_turn(tok, ex, tkw, args.max_len))
        kept = [x for x in out if x is not None]
        print(f"{split}: {len(kept)}/{len(out)} examples within {args.max_len} tokens "
              f"({'per-turn thinking' if thinking else 'full trajectory'} mode, {len(recs)} trajectories)")
        return kept

    train, val = build("train"), build("val")
    random.Random(1).shuffle(val)
    val = val[: args.max_val]
    mean_len = sum(len(x["input_ids"]) for x in train) / len(train)
    grad_accum = max(1, round(args.batch_tokens / mean_len))
    print(f"mean length {mean_len:.0f} tokens -> grad_accum {grad_accum} (batch size 1)")

    class DS(torch.utils.data.Dataset):
        def __init__(self, xs):
            self.xs = xs

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, i):
            return self.xs[i]

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (L - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch]),
        }

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa")
    if "qwen3_5" in str(getattr(model.config, "model_type", "")) or "next" in str(model.config.model_type):
        try:
            import fla  # noqa: F401
        except ImportError:
            print("WARNING: flash-linear-attention is not installed; Gated DeltaNet layers will use the "
                  "slow pure-PyTorch path. pip install flash-linear-attention", flush=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=[m for m in args.target_modules.split(",") if m],
    ))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        eval_strategy="steps" if val else "no",
        eval_steps=max(1, math.ceil(len(train) / grad_accum / 4)),
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        group_by_length=False,
    )
    random.Random(0).shuffle(train)
    trainer = Trainer(model=model, args=targs, train_dataset=DS(train),
                      eval_dataset=DS(val) if val else None, data_collator=collate)
    trainer.train()
    if trainer.is_world_process_zero():
        final = os.path.join(args.out, "final")
        model.save_pretrained(final)
        tok.save_pretrained(final)
        fix_adapter_names_for_vllm(final, args.model)
        print("saved LoRA adapter to", final)


if __name__ == "__main__":
    main()
