"""RL episodes: the model plays the paper's DFA game against the oracle.

Format = Qwen's native chat template with thinking on: each turn the model sees the
paper's stateless prompt plus every earlier query and oracle answer, but not its own
earlier reasoning (prior assistant turns are rendered content-only, as Qwen's template
does). Every turn's exact prompt and sampled token ids are recorded for training.

Tool-call parsing, oracle answers and the follow-up messages are the paper harness's
own (same functions as llm_tool_handling / llm_prompt_building), via OracleSession.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .hf_utils import chat_ids
from .oracle_session import STATELESS, OracleSession, build_target, tool_budget
from .upstream import load

U = load()
import llm_tool_handling  # noqa: E402  (importable after load())

# the harness prints a line per equivalence query; thousands of RL episodes don't need it
U.tools.print = lambda *a, **k: None


@dataclass
class Turn:
    prompt_ids: List[int]
    completion_ids: List[int]
    finish_reason: str
    valid: bool                      # produced a tool call the oracle accepted


@dataclass
class Episode:
    n_states: int
    seed: int
    budget: int
    success: bool = False
    calls: int = 0
    invalid: int = 0
    turns: List[Turn] = field(default_factory=list)
    error: Optional[str] = None
    seconds: float = 0.0

    @property
    def reward(self) -> float:
        return 1.0 if self.success else 0.0


def split_reasoning(text: str):
    """Completion after the '<think>\\n' generation prompt -> (reasoning, visible content).
    No '</think>' = thinking never finished (e.g. hit max tokens) -> no visible content."""
    if "</think>" in text:
        r, c = text.split("</think>", 1)
        return r.strip(), c.strip()
    return text.strip(), ""


def parse_tool_call(content: str):
    calls = llm_tool_handling.extract_tool_calls({"content": content})   # the harness's own parser
    return calls[0] if calls else None


def play_episode(generate: Callable[[List[int]], Dict[str, Any]], tok, n_states: int, seed: int,
                 *, budget_ratio: float = 2.0, max_invalid: int = 10, target=None) -> Episode:
    """generate(prompt_ids) -> {"token_ids": [...], "text": str, "finish_reason": str}."""
    t0 = time.time()
    mdfa = target if target is not None else build_target(n_states, seed)
    budget = tool_budget(mdfa, budget_ratio)
    ep = Episode(n_states=n_states, seed=seed, budget=budget)
    sess = OracleSession(mdfa, budget, scaffold=STATELESS)
    messages = [{"role": "user", "content": sess.initial_prompt()}]
    try:
        while not sess.solved and not sess.budget_exhausted() and ep.invalid <= max_invalid:
            prompt_ids = chat_ids(tok, messages, True)
            out = generate(prompt_ids)
            _, content = split_reasoning(out["text"])
            call = parse_tool_call(content)
            reply = sess.call(call[0], call[1]) if call else {"error": "NO_TOOL_CALLS_DETECTED"}
            valid = "tool_outputs" in reply
            ep.invalid += 0 if valid else 1
            ep.turns.append(Turn(prompt_ids, list(out["token_ids"]), out.get("finish_reason") or "", valid))
            messages.append({"role": "assistant", "content": content})   # earlier reasoning dropped
            messages.append({"role": "user", "content": sess.followup_text(reply)})
    except Exception as e:  # a failed request ends the episode as a failure
        ep.error = repr(e)
    ep.success = bool(sess.solved)
    ep.calls = sess.call_counter
    ep.seconds = time.time() - t0
    return ep


# ---------------------------------------------------------------------------
def group_advantages(rewards: List[float]) -> List[float]:
    """GRPO advantage without std normalization (Dr. GRPO): reward minus group mean."""
    m = sum(rewards) / len(rewards)
    return [r - m for r in rewards]


def build_samples(groups: List[List[Episode]], eot_id: int, max_new_tokens: int,
                  turns_per_episode: int = 0, max_len: int = 32768, rng=None):
    """Turn episodes into weighted training samples.

    Each episode's advantage is given to every one of its turns (or to a random subset of
    `turns_per_episode`, rescaled so the estimate stays unbiased). Loss for a sample =
    coef * sum of the token cross-entropies of its completion; coef = advantage * scale /
    (max_new_tokens * number of episodes): a constant normalizer (Dr. GRPO), so long
    and short turns are not reweighted.
    """
    import random
    rng = rng or random.Random(0)
    n_eps = sum(len(g) for g in groups)
    samples, skipped = [], 0
    for g in groups:
        for ep, adv in zip(g, group_advantages([e.reward for e in g])):
            if adv == 0 or not ep.turns:
                continue
            idx = list(range(len(ep.turns)))
            scale = 1.0
            if turns_per_episode and len(idx) > turns_per_episode:
                idx = rng.sample(idx, turns_per_episode)
                scale = len(ep.turns) / turns_per_episode
            for i in idx:
                t = ep.turns[i]
                comp = list(t.completion_ids)
                if t.finish_reason == "stop" and (not comp or comp[-1] != eot_id):
                    comp.append(eot_id)          # learn to end the turn
                if not comp or len(t.prompt_ids) + len(comp) > max_len:
                    skipped += 1
                    continue
                samples.append({
                    "input_ids": list(t.prompt_ids) + comp,
                    "n_prompt": len(t.prompt_ids),
                    "coef": adv * scale / (max_new_tokens * n_eps),
                })
    return samples, skipped
