"""RL episode loop against the real oracle, with a scripted policy (no GPU needed).

    python tests/test_rl_env.py path/to/Qwen3.5-4B.jinja
"""
import contextlib
import importlib.util
import io
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("t", os.path.join(ROOT, "tests", "test_masking_qwen3_template.py"))
tmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tmod)
CharTok = tmod.CharTok

from aal_sft.oracle_session import build_target  # noqa: E402
from aal_sft.rl_env import build_samples, group_advantages, play_episode  # noqa: E402
from aal_sft.teacher import make_trajectory  # noqa: E402

EOT = CharTok.SPECIAL["<|im_end|>"]


def scripted(tok, traj, garbage_first=0):
    turns = [m for m in traj["messages"] if m["role"] == "assistant"]
    state = {"i": 0, "g": garbage_first, "prompts": []}

    def generate(prompt_ids):
        state["prompts"].append(CharTok.decode(None, prompt_ids))
        if state["g"] > 0:
            state["g"] -= 1
            text = "hmm, let me think\n</think>\n\nI am not sure what to query."
        else:
            m = turns[state["i"]]
            state["i"] += 1
            text = f"{m['reasoning_content']}\n</think>\n\n{m['content']}"
        return {"text": text, "token_ids": tok.encode(text) + [EOT], "finish_reason": "stop"}
    return generate, state


def main(template_path):
    tok = CharTok(open(template_path).read(), False)
    quiet = contextlib.redirect_stdout(io.StringIO())
    for n, seed in [(2, 3), (4, 7), (6, 11), (9, 17)]:
        with quiet:
            target = build_target(n, seed)
            traj = make_trajectory(n, seed, scaffold="think", teacher="ttt", target=target)
        gen, st = scripted(tok, traj)
        with quiet:
            ep = play_episode(gen, tok, n, seed, target=build_target(n, seed))
        assert ep.error is None, ep.error
        assert ep.success and ep.calls == traj["teacher_calls"] and ep.invalid == 0, (ep.success, ep.calls, ep.invalid)
        # prompts show earlier queries/answers but never earlier reasoning
        reasoning = [m["reasoning_content"] for m in traj["messages"] if m["role"] == "assistant"]
        last = st["prompts"][-1]
        assert last.endswith("<|im_start|>assistant\n<think>\n")
        assert last.count("<TOOL_RESULT>") == len(reasoning) - 1
        assert not any(r in p for p in st["prompts"] for r in reasoning if len(r) > 40)
        # the oracle messages are the harness's own, as in the teacher data
        user_msgs = [m["content"].strip() for m in traj["messages"] if m["role"] == "user"]
        assert all(u.strip() in last for u in user_msgs)
        print(f"n={n} seed={seed}: solved in {ep.calls} calls ({len(ep.turns)} turns), no reasoning leak")

    # invalid outputs: counted, don't consume budget, game continues
    with quiet:
        traj = make_trajectory(3, 5, scaffold="think", teacher="ttt")
    gen, _ = scripted(tok, traj, garbage_first=2)
    with quiet:
        ep = play_episode(gen, tok, 3, 5)
    assert ep.success and ep.invalid == 2 and ep.calls == traj["teacher_calls"]
    assert [t.valid for t in ep.turns[:3]] == [False, False, True]
    print(f"2 garbage turns: invalid=2, still solved in {ep.calls} calls")

    # an episode that only produces garbage stops after max_invalid + 1 turns, unsolved
    def junk(prompt_ids):
        t = "no idea\n</think>\n\nnothing"
        return {"text": t, "token_ids": tok.encode(t) + [EOT], "finish_reason": "stop"}
    with quiet:
        ep = play_episode(junk, tok, 3, 5, max_invalid=10)
    assert not ep.success and ep.reward == 0 and len(ep.turns) == 11
    print("all-garbage episode: stops after 11 turns, reward 0")

    # advantages and samples
    assert group_advantages([1, 0, 0, 1]) == [0.5, -0.5, -0.5, 0.5]
    eps_ok = [ep_ for ep_ in [play_episode(scripted(tok, traj)[0], tok, 3, 5) for _ in range(2)]]
    eps_bad = [play_episode(junk, tok, 3, 5) for _ in range(2)]
    samples, skipped = build_samples([eps_ok + eps_bad], EOT, max_new_tokens=2048, rng=random.Random(0))
    n_turns = sum(len(e.turns) for e in eps_ok + eps_bad)
    assert len(samples) == n_turns and skipped == 0
    pos = [s for s in samples if s["coef"] > 0]
    neg = [s for s in samples if s["coef"] < 0]
    assert len(pos) == sum(len(e.turns) for e in eps_ok) and len(neg) == sum(len(e.turns) for e in eps_bad)
    assert all(s["input_ids"][-1] == EOT and s["n_prompt"] < len(s["input_ids"]) for s in samples)
    same, _ = build_samples([eps_ok], EOT, 2048)
    assert same == []   # all-equal group: no gradient
    sub, _ = build_samples([eps_ok + eps_bad], EOT, 2048, turns_per_episode=3, rng=random.Random(0))
    assert len(sub) == 12
    print(f"samples: every turn of every episode weighted by its advantage ({len(pos)} +, {len(neg)} -); "
          f"all-equal groups skipped; subsampling rescales")


if __name__ == "__main__":
    main(sys.argv[1])
