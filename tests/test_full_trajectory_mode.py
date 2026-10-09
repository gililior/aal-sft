"""Full-trajectory mode with templates/qwen_keep_reasoning.jinja.

1. One trajectory = one example; the loss covers every model turn (reasoning and
   action) and nothing from the prompt or the oracle.
2. The eval client's history (reasoning folded into content, as models.py does)
   renders exactly like the training data, so the tuned model sees at inference
   the same format it was trained on.
3. The chunked, gathered loss equals the standard shifted causal-LM loss.

    python tests/test_full_trajectory_mode.py data/think_v1/ttt/think
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("t", os.path.join(ROOT, "tests", "test_masking_qwen3_template.py"))
tmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tmod)
train, CharTok, labeled = tmod.train, tmod.CharTok, tmod.labeled_text

TEMPLATE = open(os.path.join(ROOT, "templates", "qwen_keep_reasoning.jinja")).read()


def spans(ex):
    out, cur = [], []
    for t, l in zip(ex["input_ids"], ex["labels"]):
        if l != -100:
            cur.append(t)
        elif cur:
            out.append(CharTok.decode(None, cur))
            cur = []
    if cur:
        out.append(CharTok.decode(None, cur))
    return out


def main(data_dir):
    for return_dict in (False, True):
        CharTok.return_dict = return_dict
        tok = CharTok(TEMPLATE, False)
        n_checked = 0
        for line in open(os.path.join(data_dir, "train.chat.jsonl")):
            rec = json.loads(line)
            msgs = rec["messages"]
            ex = train.tokenize_with_mask(tok, msgs, {}, 10 ** 9)
            assistants = [m for m in msgs if m["role"] == "assistant"]
            sp = spans(ex)
            assert len(sp) == len(assistants), (len(sp), len(assistants))
            for s, m in zip(sp, assistants):
                # span = reasoning \n</think>\n\n action <|im_end|>   (the "<think>\n" itself is
                # part of the generation prompt at inference, so it is not a target)
                assert s.startswith(m["reasoning_content"].strip()), s[:120]
                assert s.endswith(m["content"].strip() + "<|im_end|>"), s[-120:]
                assert "</think>" in s and "<|im_start|>" not in s and "TOOL_RESULT" not in s
            lab = labeled(ex)
            assert "TOOL_RESULT" not in lab and "DFA vocabulary" not in lab
            # every model turn's reasoning is also present in the input (kept in context)
            text = CharTok.decode(None, ex["input_ids"])
            assert all(m["reasoning_content"].strip() in text for m in assistants)

            # 2. eval-time history renders identically
            eval_hist = []
            for m in msgs:
                if m["role"] == "assistant":
                    eval_hist.append({"role": "assistant", "content":
                                      f"<think>\n{m['reasoning_content'].strip()}\n</think>\n\n{m['content'].strip()}"})
                else:
                    eval_hist.append({"role": m["role"], "content": m["content"]})
            a = tok.apply_chat_template(msgs, tokenize=False)
            b = tok.apply_chat_template(eval_hist, tokenize=False)
            assert a == b, "eval history renders differently from training data"
            # and the generation prompt opens <think>, as the training spans assume
            assert tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True) \
                .endswith("<|im_start|>assistant\n<think>\n")
            n_checked += 1
            if n_checked == 40:
                break
        print(f"return_dict={return_dict}: {n_checked} trajectories OK "
              f"(every model turn supervised, reasoning kept in context, eval history identical)")

    # 3. chunked gathered loss == standard loss
    rng = np.random.default_rng(0)
    V, d = 64, 16
    W = rng.normal(size=(V, d))
    for _ in range(100):
        L = int(rng.integers(5, 60))
        hid = rng.normal(size=(L, d))
        labels = rng.integers(0, V, L)
        labels[rng.random(L) < 0.5] = -100
        if (labels[1:] == -100).all():
            labels[-1] = 3

        def ce(lg, tg):
            lse = np.log(np.exp(lg).sum(1))
            return lse - lg[np.arange(len(tg)), tg]
        logits = hid @ W.T
        m = labels[1:] != -100
        standard = ce(logits[:-1][m], labels[1:][m]).mean()
        h, t = hid[:-1][m], labels[1:][m]
        total = sum(ce(h[i:i + 7] @ W.T, t[i:i + 7]).sum() for i in range(0, len(t), 7))
        assert abs(standard - total / len(t)) < 1e-10
    print("chunked gathered loss == standard causal-LM loss on 100 random cases")


if __name__ == "__main__":
    main(sys.argv[1])
