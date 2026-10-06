"""Import the paper's code (upstream_app/) with visualization side effects disabled.

The upstream runtime writes an HTML report on every equivalence query and draws
DFAs with pyvis. None of that affects what the model sees (the `html` key is
stripped from TOOL_RESULT before it is sent), so we no-op it for speed and to
avoid the pyvis dependency.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

UPSTREAM_DIR = Path(__file__).resolve().parent.parent / "upstream_app"


def _install_pyvis_stub() -> None:
    if "pyvis" in sys.modules:
        return
    pyvis = types.ModuleType("pyvis")
    network = types.ModuleType("pyvis.network")

    class Network:  # minimal no-op stand-in
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return lambda *a, **k: None

    network.Network = Network
    pyvis.network = network
    sys.modules["pyvis"] = pyvis
    sys.modules["pyvis.network"] = network


def load():
    """Make upstream modules importable and return them as a namespace."""
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    os.environ.setdefault("MPLBACKEND", "Agg")
    _install_pyvis_stub()

    import L_star
    import TTT
    import dfa_class
    import dfa_factory
    import tools
    import game_format
    import constants
    import utils

    noop = lambda *a, **k: ""  # noqa: E731
    # The EQ tool copies the drawn hypothesis file next to itself, so drawing
    # must return a real path. Point it at one empty file in a private temp dir.
    import tempfile

    draw_dir = Path(tempfile.mkdtemp(prefix="aal_draw_"))
    draw_file = draw_dir / "dfa.html"
    draw_file.write_text("")
    draw_stub = lambda *a, **k: str(draw_file)  # noqa: E731

    L_star.write_lstar_comparison_html = noop
    TTT.write_ttt_comparison_html = noop
    TTT.draw_DFA_html = draw_stub
    dfa_class.draw_DFA_html = draw_stub
    dfa_class.draw_DFA_html_option2 = draw_stub
    tools.write_llm_comparison_html = noop
    tools.read_if_path = noop

    return types.SimpleNamespace(
        L_star=L_star,
        TTT=TTT,
        dfa_class=dfa_class,
        dfa_factory=dfa_factory,
        tools=tools,
        game_format=game_format,
        constants=constants,
        utils=utils,
    )
