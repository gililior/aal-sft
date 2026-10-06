"""Turn L* / TTT runs into chat trajectories in the paper's harness format."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .oracle_session import (
    STATEFUL,
    STATELESS,
    THINK,
    THOUGHT,
    OracleSession,
    build_target,
    strategy_results,
    tool_action_json,
    tool_budget,
)

STRATEGY_KEYS = {"lstar": "LStarStrategy", "ttt": "TTTStrategy"}
DISPLAY = {"LStarStrategy": "L*", "TTTStrategy": "TTT"}


class ReplayMismatch(RuntimeError):
    pass


# --------------------------------------------------------------------------
def _word_from_repr(w: Optional[str]) -> str:
    """Upstream logs words space-separated ("a b b"); the model writes "abb"."""
    if not w or w == "ε":
        return ""
    return "".join(w.split())


def hypothesis_to_candidate(dfa, alphabet: List[str]) -> Dict[str, Any]:
    """automata-lib DFA -> evaluate_dfa_candidate JSON, states renamed q0.. in BFS order."""
    order = [dfa.initial_state]
    seen = {dfa.initial_state}
    q = deque(order)
    while q:
        s = q.popleft()
        for a in alphabet:
            t = dfa.transitions.get(s, {}).get(a)
            if t is not None and t not in seen:
                seen.add(t)
                order.append(t)
                q.append(t)
    name = {s: f"q{i}" for i, s in enumerate(order)}
    trans = [
        [name[s], a, name[dfa.transitions[s][a]]]
        for s in order
        for a in alphabet
        if a in dfa.transitions.get(s, {})
    ]
    return {
        "states": [name[s] for s in order],
        "alphabet": list(alphabet),
        "start_state": name[dfa.initial_state],
        "accept_states": [name[s] for s in order if s in dfa.final_states],
        "transitions": trans,
    }


@dataclass
class Step:
    tool: str  # "MQ" | "EQ"
    tool_input: Dict[str, Any]
    expected: Any  # MQ: bool ; EQ: counterexample word (model form) or None if correct
    candidate: Optional[Dict[str, Any]] = None


def history_to_steps(history, alphabet: List[str]) -> List[Step]:
    steps: List[Step] = []
    for item in history:
        kind, word, result = item[0], item[1], item[2]
        if kind == "MQ":
            steps.append(Step("MQ", {"word": _word_from_repr(word)}, bool(result)))
        elif kind == "EQ":
            cand = hypothesis_to_candidate(item[4], alphabet)
            exp = None if result is True else _word_from_repr(word)
            steps.append(Step("EQ", {"candidate_dfa": cand}, exp, cand))
        else:
            raise ValueError(f"unknown history item {kind}")
    return steps


# --------------------------------------------------------------------------
# Stateful scaffold: synthesised GOAL / WORKING_MEMORY / THOUGHT blocks.
# The algorithms don't produce natural-language state, so these are
# deterministic summaries of what the learner has observed so far.
# --------------------------------------------------------------------------
GOAL_TEXT = (
    "Reconstruct the hidden target DFA exactly while using the available DFA-learning "
    "tool-call budget efficiently."
)


def _fmt_word(w: str) -> str:
    return w if w else "ε"


def _fmt_hyp(c: Dict[str, Any]) -> str:
    delta = ", ".join(f"{s}-{a}->{t}" for s, a, t in c["transitions"])
    acc = ",".join(c["accept_states"]) or "none"
    return f"{len(c['states'])} states, start {c['start_state']}, accept {{{acc}}}; {delta}"


class StatefulRenderer:
    def __init__(self, algo_name: str, budget: int):
        self.algo = algo_name
        self.budget = budget
        self.accepted: List[str] = []
        self.rejected: List[str] = []
        self.last_hyp: Optional[Dict[str, Any]] = None
        self.last_cex: Optional[str] = None
        self.n_eq = 0

    def observe(self, step: Step, public_out: Dict[str, Any]):
        if step.tool == "MQ":
            (self.accepted if public_out["accepted"] else self.rejected).append(step.tool_input["word"])
        else:
            self.n_eq += 1
            self.last_hyp = step.candidate
            w = public_out.get("witness_word")
            if not public_out.get("optimal"):
                self.last_cex = _word_from_repr(w)

    def render(self, step: Step, calls_used: int) -> str:
        mem = [
            f"Strategy: {self.algo} (active automata learning with membership and equivalence queries).",
            f"Calls used: {calls_used}/{self.budget}. Equivalence queries so far: {self.n_eq}.",
            "Accepted: [" + ", ".join(_fmt_word(w) for w in self.accepted) + "]",
            "Rejected: [" + ", ".join(_fmt_word(w) for w in self.rejected) + "]",
        ]
        if self.last_hyp is not None:
            mem.append("Last hypothesis: " + _fmt_hyp(self.last_hyp))
        if self.last_cex is not None:
            mem.append(f"Last counterexample: {_fmt_word(self.last_cex)}")
        if step.tool == "MQ":
            w = _fmt_word(step.tool_input["word"])
            if self.last_cex is not None and self.n_eq > 0:
                thought = (
                    f"Continue the {self.algo} refinement after counterexample {_fmt_word(self.last_cex)}: "
                    f"the label of {w} is not yet known and is needed to separate or place states."
                )
            else:
                thought = (
                    f"Build the initial {self.algo} hypothesis: the label of {w} is not yet known "
                    f"and is needed to determine state acceptance and transitions."
                )
        else:
            n = len(step.candidate["states"])
            thought = (
                f"All observations are consistent with a closed, {n}-state hypothesis. "
                f"Submit it as an equivalence query."
            )
        return "\n".join(mem), thought

    def assistant_text(self, step: Step, calls_used: int) -> str:
        mem, thought = self.render(step, calls_used)
        action = tool_action_json(_tool_name(step), step.tool_input)
        return (
            f"<GOAL>{GOAL_TEXT}</GOAL>\n"
            f"<WORKING_MEMORY>\n{mem}\n</WORKING_MEMORY>\n"
            f"<THOUGHT>{thought}</THOUGHT>\n"
            f"<TOOL_ACTION>\n{action}\n</TOOL_ACTION>"
        )


def _tool_name(step: Step) -> str:
    return "is_word_in_language" if step.tool == "MQ" else "evaluate_dfa_candidate"


def stateless_assistant_text(step: Step) -> str:
    return f"<TOOL_ACTION>\n{tool_action_json(_tool_name(step), step.tool_input)}\n</TOOL_ACTION>"


# --------------------------------------------------------------------------
def choose_strategy(results: Dict[str, Any], teacher: str) -> str:
    if teacher in STRATEGY_KEYS:
        return STRATEGY_KEYS[teacher]
    if teacher != "best":
        raise ValueError(teacher)
    # cheaper total queries; ties go to TTT (fewer redundant queries by design)
    return min(results, key=lambda k: (results[k].total_queries, 0 if k == "TTTStrategy" else 1))


def make_trajectory(
    n_states: int,
    seed: int,
    *,
    scaffold: str = STATELESS,
    teacher: str = "best",
    budget_ratio: float = 2.0,
    target=None,
    enforce_budget: bool = True,
) -> Dict[str, Any]:
    mdfa = target if target is not None else build_target(n_states, seed)
    results = strategy_results(mdfa)
    key = choose_strategy(results, teacher)
    res = results[key]
    notes = None
    if scaffold in (THINK, THOUGHT):
        from .reasoning import run_with_reasons
        traced, notes = run_with_reasons(mdfa, "lstar" if key == "LStarStrategy" else "ttt")
        if [tuple(h[:3]) for h in traced.history] != [tuple(h[:3]) for h in res.history]:
            raise ReplayMismatch("traced run diverged from the reference run")
    alphabet = sorted(mdfa.input_symbols, key=str)
    steps = history_to_steps(res.history, alphabet)
    if len(steps) != res.total_queries:
        raise ReplayMismatch(f"history length {len(steps)} != total_queries {res.total_queries}")

    budget = tool_budget(mdfa, budget_ratio)
    if res.total_queries > budget:
        if enforce_budget:
            raise ReplayMismatch(f"teacher needs {res.total_queries} calls > budget {budget}")
        budget = res.total_queries  # render anyway (used by the teacher-replay eval mock)
    sess = OracleSession(mdfa, budget, scaffold=scaffold)
    renderer = StatefulRenderer(DISPLAY[key], budget) if scaffold == STATEFUL else None

    messages = [{"role": "user", "content": sess.initial_prompt()}]
    for i, st in enumerate(steps):
        if scaffold == STATEFUL:
            msg = {"role": "assistant", "content": renderer.assistant_text(st, sess.call_counter)}
        elif scaffold == THINK:
            msg = {"role": "assistant", "reasoning_content": notes[i], "content": stateless_assistant_text(st)}
        elif scaffold == THOUGHT:
            msg = {"role": "assistant",
                   "content": f"<THOUGHT>\n{notes[i]}\n</THOUGHT>\n{stateless_assistant_text(st)}"}
        else:
            msg = {"role": "assistant", "content": stateless_assistant_text(st)}
        messages.append(msg)

        reply = sess.call(_tool_name(st), st.tool_input)
        out = sess.public_output(reply)
        if out is None:
            raise ReplayMismatch(f"step {i}: tool error {reply}")
        if st.tool == "MQ":
            if out["accepted"] != st.expected:
                raise ReplayMismatch(f"step {i}: MQ {st.tool_input} oracle={out['accepted']} log={st.expected}")
        else:
            got = None if out["optimal"] else _word_from_repr(out["witness_word"])
            if got != st.expected:
                raise ReplayMismatch(f"step {i}: EQ cex oracle={got!r} log={st.expected!r}")
        if renderer:
            renderer.observe(st, out)
        if sess.solved:
            if i != len(steps) - 1:
                raise ReplayMismatch("solved before the end of the teacher trajectory")
            break
        messages.append({"role": "user", "content": sess.followup_text(reply)})

    if not sess.solved:
        raise ReplayMismatch("teacher trajectory did not end with a correct EQ")

    return {
        "id": f"n{n_states}_s{seed}_{scaffold}_{DISPLAY[key]}",
        "n_states": n_states,
        "seed": seed,
        "scaffold": scaffold,
        "teacher": DISPLAY[key],
        "teacher_calls": res.total_queries,
        "teacher_mq": res.mq_queries,
        "teacher_eq": res.eq_queries,
        "lstar_calls": results["LStarStrategy"].total_queries,
        "ttt_calls": results["TTTStrategy"].total_queries,
        "budget": budget,
        "messages": messages,
    }
