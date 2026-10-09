"""Helpers shared by SFT (scripts/train_hf_lora.py) and RL (scripts/train_rl.py).

Imports torch / transformers lazily so the module can be imported without them.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List


def chat_ids(tok, msgs, add_gen, template_kwargs=None) -> List[int]:
    """Token ids of a rendered chat as a plain list, on any transformers version
    (v5's apply_chat_template returns a dict by default, v4 a list)."""
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_gen, **(template_kwargs or {}))
    if hasattr(out, "keys"):
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    out = list(out)
    if out and isinstance(out[0], (list, tuple)):
        out = list(out[0])
    return [int(x) for x in out]


def end_of_turn_id(tok, template_kwargs=None) -> int:
    """The token that closes a chat turn (<|im_end|>, <|eot_id|>, <end_of_turn>, ...)."""
    ids = chat_ids(tok, [{"role": "user", "content": "x"}], False, template_kwargs)
    while ids and tok.decode([ids[-1]]).strip() == "":
        ids.pop()
    return ids[-1]


# ---------------------------------------------------------------------------
# Adapter naming: AutoModelForCausalLM loads Qwen3.5 (a vision-language checkpoint)
# as its text-only class, so PEFT saves `model.layers.*`; vLLM serves the VL class
# and expects `model.language_model.layers.*`.
_TEXT = "base_model.model.model.layers."
_VL = "base_model.model.model.language_model.layers."


def is_vl_checkpoint(base_model) -> bool:
    from transformers import AutoConfig
    return getattr(AutoConfig.from_pretrained(base_model), "vision_config", None) is not None


def fix_adapter_names_for_vllm(adapter_dir, base_model) -> bool:
    from safetensors.torch import load_file, save_file
    if not is_vl_checkpoint(base_model):
        return False
    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    sd = load_file(path)
    if any(".language_model." in k for k in sd):
        return False
    out = {k.replace(_TEXT, _VL): v for k, v in sd.items()}
    save_file(out, path, metadata={"format": "pt"})
    print(f"renamed {len(out)} adapter tensors to model.language_model.* for vLLM", flush=True)
    return True


def load_adapter_for_training(peft_model, adapter_dir) -> int:
    """Load a saved LoRA adapter (either naming) into a text-only PEFT model.
    Returns the number of tensors loaded; raises if none matched."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    sd = {k.replace(_VL, _TEXT): v for k, v in sd.items()}
    res = set_peft_model_state_dict(peft_model, sd)
    unexpected = list(getattr(res, "unexpected_keys", []) or [])
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} adapter tensors did not match the model, e.g. {unexpected[:3]}")
    n = len(sd)
    print(f"loaded {n} adapter tensors from {adapter_dir}", flush=True)
    return n


def adapter_lora_config(adapter_dir) -> Dict:
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    return {"r": cfg["r"], "lora_alpha": cfg["lora_alpha"], "target_modules": cfg["target_modules"]}


# ---------------------------------------------------------------------------
def detach_lm_head(peft_model):
    """Replace the base model's lm_head by Identity (forward then returns final hidden
    states) and return the frozen output-projection weight, for chunked_ce_sum."""
    import torch
    base = peft_model.get_base_model()
    w = base.lm_head.weight.detach()
    base.lm_head = torch.nn.Identity()
    return w


def chunked_ce_sum(hidden, weight, target, chunk=2048):
    """sum_i CE(hidden_i @ weight.T, target_i), computed in row chunks with activation
    checkpointing so no [rows x vocab] tensor outlives its chunk (in forward or backward)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def ce(h, w, t):
        return F.cross_entropy((h @ w.t()).float(), t, reduction="sum")

    w = weight.to(device=hidden.device, dtype=hidden.dtype)   # no-op when already there
    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hidden.size(0), chunk):
        total = total + checkpoint(ce, hidden[i:i + chunk], w, target[i:i + chunk], use_reentrant=False)
    return total
