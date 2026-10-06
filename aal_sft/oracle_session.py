"""An oracle session that produces exactly the text the paper's chat harness sends.

It reuses the upstream pieces the runtime itself calls:
  * GameFormat.get_game_prompt()        -> initial user message
  * IsWordInLanguageTool / EvaluateDFACandidateTool.invoke() -> oracle answers
  * the canonicalisation + `_candidate_obj` / `html` / `knowledge_state` stripping
    done in llm_tool_handling.handle_model_request
  * llm_prompt_building.build_followup_text() -> the <TOOL_RESULT> user message

so a trajectory rendered through this class is token-identical to what a model
sees when evaluated with the released harness (stateless scaffold).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional

from .upstream import load

U = load()

import llm_prompt_building  # noqa: E402  (importable after load())

STATELESS = "stateless"
STATEFUL = "stateful"
THINK = "think"      # stateless prompt; reasoning goes in the model's native thinking channel
THOUGHT = "thought"  # visible reasoning: <THOUGHT> block before <TOOL_ACTION> (for Gemini SFT)
SCAFFOLDS = (STATELESS, STATEFUL, THINK, THOUGHT)

_THOUGHT_POLICY = (
    "OUTPUT POLICY (MANDATORY):\n"
    "- Every response consists of exactly two blocks and nothing else, in this order:\n"
    "  1. <THOUGHT>...</THOUGHT> containing your reasoning: what you know so far, the state of your learning "
    "procedure, and why the next action is the right one.\n"
    "  2. <TOOL_ACTION>...</TOOL_ACTION> containing a single JSON object with exactly:\n"
    '     {"tool_name": "<tool_name>", "input": { ... }}\n'
    "- Never include call_count (it is added externally).\n"
    "- Do not output text outside the two blocks.\n"
    "\n"
    "Example:\n"
    "<THOUGHT>\n...\n</THOUGHT>\n"
    "<TOOL_ACTION>\n"
    '{ "tool_name": "is_word_in_language", "input": { "word": "ab" } }\n'
    "</TOOL_ACTION>\n"
    "# IMPORTANT: The <TOOL_ACTION> block must include exactly one tool call. Each block represents a single tool "
    "invocation, so multiple tool calls—either to the same tool or to different tools—are not allowed within "
    "the same block.\n"
)

# --------------------------------------------------------------------------
# Stateful scaffold prompt. Not in the public repo; reconstructed from the
# paper (Appendix A.1.2): same rules as the released stateless prompt, with the
# STRICT OUTPUT POLICY replaced by the four-block structured output policy.
# --------------------------------------------------------------------------
_STATEFUL_POLICY = (
    "STRUCTURED OUTPUT POLICY (MANDATORY):\n"
    "- You MUST output exactly four blocks and nothing else, in this exact order:\n"
    "  1. <GOAL>...</GOAL>\n"
    "  2. <WORKING_MEMORY>...</WORKING_MEMORY>\n"
    "  3. <THOUGHT>...</THOUGHT>\n"
    "  4. <TOOL_ACTION>...</TOOL_ACTION>\n"
    "- The <GOAL> block must restate the overall task objective that must remain fixed throughout the interaction: "
    "reconstruct the hidden target DFA exactly while using the available DFA-learning tool-call budget efficiently. "
    "Keep it concise and do not place reasoning or a tool call in this block.\n"
    "- The <WORKING_MEMORY> block must contain a compact, up-to-date summary of the information and intermediate "
    "state that should be preserved for future steps. Update it on every turn as a complete replacement for the "
    "previous working-memory summary.\n"
    "- The <THOUGHT> block must contain the reasoning used to choose the single next action.\n"
    "- The <TOOL_ACTION> block must contain a single JSON object with exactly:\n"
    '  {"tool_name": "<tool_name>", "input": { ... }}\n'
    "- Never include call_count; it is added externally.\n"
    "- Exactly one block of each type is allowed. Do not output markdown fences, comments, headings, or text "
    "outside the four blocks.\n"
    "- Tool documentation later in the prompt may show a standalone <TOOL_ACTION> snippet only to illustrate that "
    "tool's JSON arguments. Those snippets are not complete response examples. In every actual response, you must "
    "still output all four required blocks.\n"
    "# IMPORTANT: The <TOOL_ACTION> block must include exactly one tool call. Each block represents a single tool "
    "invocation, so multiple tool calls--either to the same tool or to different tools--are not allowed within "
    "the same block.\n"
)


def game_prompt_for(scaffold: str) -> str:
    base = U.constants.GAME_PROMPT
    if scaffold in (STATELESS, THINK):
        return base
    head, sep, _ = base.partition("STRICT OUTPUT POLICY (MANDATORY):")
    if not sep:
        raise RuntimeError("Upstream GAME_PROMPT changed; cannot build scaffold prompt")
    if scaffold == STATEFUL:
        return "Goal: " + head + _STATEFUL_POLICY
    if scaffold == THOUGHT:
        return head + _THOUGHT_POLICY
    raise ValueError(f"unknown scaffold {scaffold!r}")


# --------------------------------------------------------------------------
# Targets and budgets, built exactly as upstream main.build_game() does.
# --------------------------------------------------------------------------
def build_target(n_states: int, seed: int, alphabet_size: int = 2):
    raw = U.dfa_factory.make_random_dfa(n_states=n_states, alphabet_size=alphabet_size, seed=seed)
    return U.dfa_class.MinimalDFA.from_dfa(
        raw,
        run_strategy=True,  # runs L* and TTT; results in .strategy_results
        minimal_counterexample=False,  # "deterministic short counterexample" (paper default)
        counterexample_max_extra_len=3,
        counterexample_mode=0,
    )


def strategy_results(mdfa) -> Dict[str, Any]:
    return dict(mdfa.strategy_results[-1])


def tool_budget(mdfa, ratio: float = 2.0) -> int:
    best = min(r.total_queries for run in mdfa.strategy_results for r in run.values())
    return int(Decimal(str(best * ratio)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def dfa_fingerprint(dfa) -> str:
    return U.dfa_class._stable_dfa_fingerprint(dfa)


# --------------------------------------------------------------------------
class _Owner:
    """Just enough of LLM_interactive_game for build_followup_text()."""

    @staticmethod
    def _json_default(o):
        return llm_prompt_building.json_default(None, o)

    def _strip_knowledge_state_deep(self, obj):
        return llm_prompt_building.strip_knowledge_state_deep(self, obj)


@dataclass
class OracleSession:
    target: Any
    max_tool_calls: int
    scaffold: str = STATELESS
    call_counter: int = 0
    solved: bool = False
    knowledge_state: Dict[str, set] = field(
        default_factory=lambda: {"words_accepted_by_dfa": set(), "words_rejected_by_dfa": set()}
    )

    def __post_init__(self):
        tools = [cls() for cls in U.utils.get_tools()]
        self.game = U.game_format.GameFormat(
            dfa=self.target,
            tools=tools,
            hints=["vocabulary"],
            max_tool_calls=self.max_tool_calls,
            game_prompt=game_prompt_for(self.scaffold),
        )
        self._tools = {t.tool_name: t for t in tools}
        self._owner = _Owner()
        self.alphabet = sorted(self.target.input_symbols, key=str)

    # ---- messages -------------------------------------------------------
    def initial_prompt(self) -> str:
        p = self.game.get_game_prompt()
        return p.replace("{MAX_CALLS}", str(self.max_tool_calls))

    def budget_exhausted(self) -> bool:
        return self.call_counter >= self.max_tool_calls

    def call(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one tool call. Returns the reply dict the runtime would build.

        Mirrors llm_tool_handling.handle_model_request: invalid calls return
        {"error": "NO_TOOL_CALLS_DETECTED"} and do not consume budget.
        """
        if self.budget_exhausted():
            return {"tool_outputs": [{
                "tool_name": "tool_budget",
                "call_count": self.call_counter,
                "error": "MAX_TOOL_CALLS_REACHED",
                "output": {"max_tool_calls": self.max_tool_calls},
            }]}
        tool = self._tools.get(U.utils.normalize_tool_name(tool_name))
        if tool is None:
            return {"error": "NO_TOOL_CALLS_DETECTED"}
        call_count = self.call_counter + 1
        out = tool.invoke({
            "tool_name": tool.tool_name,
            "call_count": call_count,
            "input": tool_input if isinstance(tool_input, dict) else {},
            "knowledge_state": self.knowledge_state,
        })
        if isinstance(out, dict) and out.get("error") is not None:
            return {"error": "NO_TOOL_CALLS_DETECTED"}
        self.call_counter = call_count
        if isinstance(out.get("knowledge_state"), dict):
            self.knowledge_state = out["knowledge_state"]
        payload = out.get("output") or {}
        if tool.tool_name == "is_word_in_language":
            payload["word"] = U.utils.canonical_word(payload.get("word"), self.target.input_symbols)
        else:
            w = payload.get("witness_word")
            payload["witness_word"] = (
                U.utils.canonical_word(w, self.target.input_symbols) if isinstance(w, str) else None
            )
            payload.pop("_candidate_obj", None)
            if payload.get("optimal") is True:
                self.solved = True
        return {"tool_outputs": [out]}

    def followup_text(self, reply: Dict[str, Any]) -> str:
        return llm_prompt_building.build_followup_text(self._owner, reply)

    @staticmethod
    def public_output(reply: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The oracle answer as the model sees it (html / knowledge_state stripped)."""
        outs = reply.get("tool_outputs") or []
        if not outs:
            return None
        o = outs[0].get("output") or {}
        return {k: v for k, v in o.items() if k not in ("html", "knowledge_state")}


def tool_action_json(tool_name: str, tool_input: Dict[str, Any]) -> str:
    return json.dumps({"tool_name": tool_name, "input": tool_input}, ensure_ascii=False)
