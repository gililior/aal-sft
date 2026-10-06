"""Check that our rendered trajectories equal what the upstream runtime sends.

We run upstream LLM_interactive_game.run_with_chat_session with a scripted
"model" that replays our assistant turns, capture every text the runtime sends
to the model, and compare with our user turns. Stateless scaffold only (the
stateful prompt is not part of the released runtime).
"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aal_sft.oracle_session import build_target, game_prompt_for, tool_budget  # noqa: E402
from aal_sft.teacher import make_trajectory  # noqa: E402
from aal_sft.upstream import load  # noqa: E402

U = load()


def run_upstream(traj, mdfa):
    import output_paths
    from llm_runtime import LLM_interactive_game

    output_paths.set_output_dir(tempfile.mkdtemp(prefix="aal_up_"))
    tools = [c() for c in U.utils.get_tools()]
    game = U.game_format.GameFormat(
        dfa=mdfa, tools=tools, hints=["vocabulary"],
        max_tool_calls=tool_budget(mdfa), game_prompt=game_prompt_for("stateless"),
    )
    script = [m["content"] for m in traj["messages"] if m["role"] == "assistant"]
    sent = []

    class Scripted:
        model_name = "scripted-teacher"

        def send(self, text, step=None):
            sent.append(text)
            return {"content": script[len(sent) - 1]}

    bridge = LLM_interactive_game(game=game)
    with contextlib.redirect_stdout(io.StringIO()):
        bridge.run_with_chat_session(Scripted(), verbose=False)
    return sent, bridge


def test_matches(cases=((2, 3), (3, 101), (5, 11), (7, 4), (9, 17))):
    for n, seed in cases:
        mdfa = build_target(n, seed)
        with contextlib.redirect_stdout(io.StringIO()):
            traj = make_trajectory(n, seed, target=mdfa)
        ours = [m["content"] for m in traj["messages"] if m["role"] == "user"]
        sent, bridge = run_upstream(traj, build_target(n, seed))
        assert len(sent) == len(ours), (n, seed, len(sent), len(ours))
        for i, (a, b) in enumerate(zip(ours, sent)):
            assert a == b, f"n={n} seed={seed} message {i} differs:\nOURS:\n{a[-400:]}\nUPSTREAM:\n{b[-400:]}"
        assert bridge._reached_optimal, (n, seed)
        print(f"ok n={n} seed={seed}: {len(sent)} messages identical, solved in {traj['teacher_calls']} calls")


if __name__ == "__main__":
    test_matches()
