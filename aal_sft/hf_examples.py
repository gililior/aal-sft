"""Build training examples for open-weights models (no ML dependencies here).

For `think` trajectories (assistant turns carry `reasoning_content`) thinking
models only ever see their *current* turn's reasoning at inference, because
chat templates drop reasoning from earlier turns. So each training example is
one turn: the visible history so far (earlier reasoning removed) followed by
the target turn with its reasoning, and loss is applied to that turn only.
"""
from __future__ import annotations

import random
from typing import Dict, List


def _is_eq(msg: Dict) -> bool:
    return "evaluate_dfa_candidate" in msg.get("content", "")


def per_turn_examples(messages: List[Dict], k_mq: int, rng: random.Random) -> List[List[Dict]]:
    """All EQ turns plus k_mq sampled MQ turns (k_mq <= 0: every turn)."""
    idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    eq = [i for i in idx if _is_eq(messages[i])]
    mq = [i for i in idx if not _is_eq(messages[i])]
    if k_mq > 0 and len(mq) > k_mq:
        mq = rng.sample(mq, k_mq)
    out = []
    for t in sorted(eq + mq):
        hist = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages[:t]]
        out.append(hist + [dict(messages[t])])
    return out


def inline_reasoning(msgs: List[Dict]) -> List[Dict]:
    """Fallback for templates that ignore `reasoning_content`: put it in <think> tags."""
    out = [dict(m) for m in msgs]
    last = out[-1]
    r = last.pop("reasoning_content", None)
    if r:
        last["content"] = f"<think>\n{r}\n</think>\n\n{last['content']}"
    return out
