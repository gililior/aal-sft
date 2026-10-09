"""Chat model adapters for the upstream runtime: send(text, step) -> {"content": str}.

The runtime sends only the newest user message each turn, so adapters keep the
conversation history themselves.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class _History:
    model_name = "model"

    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.usage: List[Dict[str, Any]] = []
        self.reasoning: List[Optional[str]] = []
        self.keep_reasoning = False   # put each turn's reasoning back into the history

    def send(self, text: str, step: Optional[int] = None) -> Dict[str, Any]:
        self.messages.append({"role": "user", "content": text})
        out = self._generate()
        hist = out
        r = self.reasoning[-1] if (self.keep_reasoning and self.reasoning) else None
        if r and r.strip():
            # same rendering as training in full-trajectory mode (templates/qwen_keep_reasoning.jinja)
            hist = f"<think>\n{r.strip()}\n</think>\n\n{(out or '').strip()}"
        self.messages.append({"role": "assistant", "content": hist})
        return {"content": out}

    def _generate(self) -> str:
        raise NotImplementedError


def _retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # rate limits, 5xx
            if i == tries - 1:
                raise
            wait = min(180, 10 * 2 ** i)
            print(f"model call failed ({type(e).__name__}: {e}); retrying in {wait}s", flush=True)
            time.sleep(wait)


class VertexGemini(_History):
    """Base or tuned Gemini on Vertex. For a tuned model pass its endpoint
    ("projects/.../locations/.../endpoints/...") as `model`."""

    def __init__(self, model: str, project: str, location: str = "us-central1",
                 thinking_level: Optional[str] = "MINIMAL", temperature: Optional[float] = None,
                 max_output_tokens: int = 8192):
        super().__init__()
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.model_name = model
        cfg: Dict[str, Any] = {"max_output_tokens": max_output_tokens}
        if temperature is not None:
            cfg["temperature"] = temperature
        if thinking_level:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        self.config = types.GenerateContentConfig(**cfg)

    def _generate(self) -> str:
        t = self.types
        contents = [t.Content(role="user" if m["role"] == "user" else "model",
                              parts=[t.Part(text=m["content"])]) for m in self.messages]
        r = _retry(lambda: self.client.models.generate_content(
            model=self.model_name, contents=contents, config=self.config))
        um = getattr(r, "usage_metadata", None)
        if um is not None:
            self.usage.append({"in": um.prompt_token_count, "out": um.candidates_token_count,
                               "thoughts": getattr(um, "thoughts_token_count", None)})
        return r.text or ""


class OpenAICompatible(_History):
    """Any OpenAI-compatible chat endpoint, e.g. vLLM serving the LoRA model."""

    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY",
                 temperature: Optional[float] = None, max_tokens: int = 8192,
                 extra_body: Optional[Dict[str, Any]] = None, keep_reasoning: bool = False):
        super().__init__()
        self.keep_reasoning = keep_reasoning
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model
        self.kw: Dict[str, Any] = {"max_tokens": max_tokens}
        if temperature is not None:
            self.kw["temperature"] = temperature
        if extra_body:
            self.kw["extra_body"] = extra_body

    def _generate(self) -> str:
        r = _retry(lambda: self.client.chat.completions.create(
            model=self.model_name, messages=self.messages, **self.kw))
        if r.usage:
            self.usage.append({"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens})
        # With vLLM's --reasoning-parser the thinking arrives separately; send() puts it
        # back into the history only when keep_reasoning is set (full-trajectory mode).
        msg = r.choices[0].message
        # vLLM puts it in `reasoning_content` (older) or `reasoning` (newer)
        self.reasoning.append(getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None))
        return msg.content or ""


class TeacherReplay(_History):
    """Replays the L*/TTT teacher trajectory. Must score 100% with Δ calls = 0;
    used to sanity-check the evaluation pipeline."""

    model_name = "teacher-replay"

    def __init__(self, n_states: int, seed: int, scaffold: str, teacher: str = "best"):
        super().__init__()
        from .teacher import make_trajectory
        # The harness enforces the real budget; an over-budget teacher just fails there.
        traj = make_trajectory(n_states, seed, scaffold=scaffold, teacher=teacher, enforce_budget=False)
        self.script = [m["content"] for m in traj["messages"] if m["role"] == "assistant"]

    def _generate(self) -> str:
        i = sum(m["role"] == "assistant" for m in self.messages)
        return self.script[i] if i < len(self.script) else ""
