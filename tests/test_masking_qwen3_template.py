"""Check loss masking against the real Qwen3 chat template (char-level fake tokenizer).

Needs a Qwen3 template file, e.g. llama.cpp's models/templates/Qwen-Qwen3-0.6B.jinja:
    python tests/test_masking_qwen3_template.py path/to/Qwen-Qwen3-0.6B.jinja data/think_v1/ttt/think
"""
import importlib.util
import json
import os
import random
import sys

import jinja2

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from aal_sft.hf_examples import per_turn_examples  # noqa: E402

spec = importlib.util.spec_from_file_location("train", os.path.join(ROOT, "scripts", "train_hf_lora.py"))
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class CharTok:
    """apply_chat_template with the real template; one 'token' per character."""

    def __init__(self, template: str, force_think_prompt: bool):
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
        self.t = env.from_string(template)
        self.force = force_think_prompt  # Qwen3-*-Thinking-2507 always opens <think> in the prompt

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kw):
        s = self.t.render(messages=messages, add_generation_prompt=add_generation_prompt, **kw)
        if add_generation_prompt and self.force:
            s += "<think>\n"
        if not tokenize:
            return s
        ids = self.encode(s)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)} if self.return_dict else ids

    return_dict = os.environ.get("RETURN_DICT") == "1"   # transformers v5 behaviour

    # as in the real Qwen3 / Qwen3.5 vocabularies, these are single tokens
    SPECIAL = {"<|im_start|>": 0x110000, "<|im_end|>": 0x110001, "<think>": 0x110002, "</think>": 0x110003}
    INV = {v: k for k, v in SPECIAL.items()}

    def encode(self, s):
        out, i = [], 0
        while i < len(s):
            for sp, sid in self.SPECIAL.items():
                if s.startswith(sp, i):
                    out.append(sid)
                    i += len(sp)
                    break
            else:
                out.append(ord(s[i]))
                i += 1
        return out

    def decode(self, ids):
        return "".join(CharTok.INV.get(i) or chr(i) for i in ids)


def labeled_text(ex):
    return CharTok.decode(None, [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])


def main(template_path, data_dir):
    tmpl = open(template_path).read()
    rec = json.loads(open(os.path.join(data_dir, "train.chat.jsonl")).readline())
    for force in (False, True):
        tok = CharTok(tmpl, force)
        assert train.template_renders_reasoning(tok, {}), "template drops reasoning_content"
        exs = per_turn_examples(rec["messages"], 4, random.Random(0))
        for msgs in exs:
            ex = train.tokenize_last_turn(tok, msgs, {}, 10 ** 9)
            lab = labeled_text(ex)
            last = msgs[-1]
            # the supervised span is exactly: [<think>\n] reasoning </think> action <|im_end|>
            assert last["reasoning_content"] in lab and last["content"] in lab, lab[:200]
            assert lab.rstrip().endswith("<|im_end|>"), lab[-50:]
            assert "<|im_start|>" not in lab and "TOOL_RESULT" not in lab, "prompt leaked into labels"
            # earlier turns' reasoning must not appear anywhere in the input
            prompt = CharTok.decode(None, [t for t, l in zip(ex["input_ids"], ex["labels"]) if l == -100])
            for m in msgs[:-1]:
                assert "reasoning_content" not in m
            for m in rec["messages"]:
                r = m.get("reasoning_content")
                if r and len(r) > 40:
                    assert r not in prompt, "an earlier turn's reasoning leaked into the prompt"
        print(f"force_think_prompt={force}: {len(exs)} per-turn examples OK; supervised span starts:",
              repr(labeled_text(train.tokenize_last_turn(tok, exs[0], {}, 10 ** 9))[:60]))

    # full-trajectory mode (stateless data) on the same template.
    srec = json.loads(open(os.path.join(ROOT, "data", "v1", "stateless", "train.chat.jsonl")).readline())
    ex = train.tokenize_with_mask(CharTok(tmpl, False), srec["messages"], {}, 10 ** 9)
    lab = labeled_text(ex)
    n_assist = sum(m["role"] == "assistant" for m in srec["messages"])
    assert lab.count("<TOOL_ACTION>") == n_assist, (lab.count("<TOOL_ACTION>"), n_assist)
    assert "TOOL_RESULT" not in lab and "<|im_start|>" not in lab
    assert lab.count("<|im_end|>") == n_assist
    print(f"full-trajectory mode: {n_assist} assistant turns supervised, no oracle text in labels: OK")


if __name__ == "__main__":
    main(*sys.argv[1:3])
