"""Explain every query L* / TTT makes, from the algorithm's actual internal state.

We run the upstream strategies (AALpy 2.0.0 L* with Rivest-Schapire processing,
and the paper's discrimination-tree "TTT") with tracing hooks. Whenever a query
really reaches the oracle (i.e. is appended to the strategy's history), we walk
the Python call stack to find which step of the algorithm issued it and read
that step's local state (observation table, discrimination tree, counterexample
analysis variables). That yields a faithful rationale for each query.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

from .upstream import load

U = load()

from aalpy.learning_algs.deterministic.ObservationTable import ObservationTable  # noqa: E402

Word = Tuple[Any, ...]


def w2s(w) -> str:
    if w is None:
        return "?"
    s = "".join(str(x) for x in w)
    return s if s else "ε"


def _stack(start: int = 2) -> List[Tuple[str, Dict[str, Any]]]:
    f = sys._getframe(start)
    out = []
    while f is not None:
        out.append((f.f_code.co_name, f.f_locals))
        f = f.f_back
    return out


def _find(stack, name, pred=None):
    for n, loc in stack:
        if n == name and (pred is None or pred(loc)):
            return loc
    return None


# ==========================================================================
# L* (AALpy)
# ==========================================================================
def _bit(v) -> str:
    return "1" if v else "0"


def render_table(t: ObservationTable, highlight: Optional[Word] = None) -> str:
    E = t.E
    head = "E = [" + ", ".join(w2s(e) for e in E) + "]"
    lines = [head, "S rows:"]

    def row(s):
        vals = list(t.T.get(s, ()))
        cells = [_bit(v[0] if isinstance(v, tuple) else v) for v in vals]
        cells += ["?"] * (len(E) - len(cells))
        mark = "  <-" if highlight is not None and s == highlight else ""
        return f"  {w2s(s):>8} | {' '.join(cells)}{mark}"

    for s in t.S:
        lines.append(row(s))
    lines.append("S·A rows:")
    for s in t.s_dot_a():
        lines.append(row(s))
    return "\n".join(lines)


def _lstar_mq_reason(stack, word: Word) -> str:
    rs = _find(stack, "rs_cex_processing")
    if rs is not None:
        cex = rs["cex"]
        if "mid" not in rs or "cex_out" not in rs:
            return (f"Counterexample {w2s(cex)} was returned. Before analysing it, I need its true label, "
                    f"so query {w2s(word)}.")
        mid, lo, hi = rs["mid"], rs["lower"], rs["upper"]
        acc = rs["s_bracket"]
        lab = "accepted" if rs["cex_out"][-1] else "rejected"
        return (f"Rivest–Schapire analysis of counterexample {w2s(cex)} (true label: {lab}). "
                f"Binary search over split points, current range [{lo}, {hi}], trying split {mid}: "
                f"the prefix {w2s(cex[:mid])} leads my hypothesis to the state with access word {w2s(acc)}. "
                f"Replace the prefix by that access word and keep the suffix {w2s(cex[mid:])}: "
                f"if {w2s(word)} has the same label as the counterexample, the error lies to the right, "
                f"otherwise to the left. That pins down a new distinguishing suffix for E.")
    ut = _find(stack, "update_obs_table")
    if ut is not None:
        table: ObservationTable = ut["self"]
        s, e = ut.get("s"), ut.get("e")
        body = render_table(table, highlight=s)
        run = _find(stack, "run_Lstar")
        if ut.get("e_set"):
            why = (f"The counterexample analysis produced new suffix(es) {[w2s(x) for x in ut['e_set']]} for E. "
                   f"Fill the new column for every row: cell T[{w2s(s)}][{w2s(e)}] is unknown")
        elif ut.get("s_set"):
            closing = run.get("rows_to_close") if run else None
            moved = ", ".join(w2s(r) for r in closing) if closing else "?"
            why = (f"The table was not closed: row(s) {moved} in S·A matched no row in S, so they moved to S "
                   f"and their one-letter extensions need rows. Cell T[{w2s(s)}][{w2s(e)}] is unknown")
        else:
            why = f"Initialising the table (S = {{ε}}, E = {{ε}}). Cell T[{w2s(s)}][{w2s(e)}] is unknown"
        return f"L* observation table:\n{body}\n{why}, so query {w2s(s)}·{w2s(e)} = {w2s(word)}."
    if _find(stack, "counterexample_successfully_processed") is not None:
        return f"Check whether the previous counterexample is still misclassified: query {w2s(word)}."
    return f"Query {w2s(word)}."


def _lstar_eq_reason(stack, hyp) -> str:
    run = _find(stack, "run_Lstar")
    t: ObservationTable = run["observation_table"]
    n = len(hyp.states)
    return (f"L* observation table:\n{render_table(t)}\n"
            f"Every S·A row equals some S row, so the table is closed. Each distinct S row is a state "
            f"({n} states; accepting where the ε column is 1), and transitions follow the S·A rows. "
            f"Submit this hypothesis as an equivalence query.")


# ==========================================================================
# Paper's discrimination-tree learner ("TTT")
# ==========================================================================
def render_tree(node) -> str:
    if node is None:
        return "?"
    if node.is_leaf():
        return f"q{node.state}"
    return f"[{w2s(node.disc or ())}? no→{render_tree(node.left)} | yes→{render_tree(node.right)}]"


def _ttt_state(lr) -> str:
    acc = ", ".join(f"q{q}={w2s(lr.access[q])}" for q in sorted(lr.access))
    return (f"Discrimination-tree learner state: {lr.n_states} state(s).\n"
            f"Access words: {acc}\n"
            f"Tree: {render_tree(lr.dt.root)}")


def _sift_path(lr, u: Word, upto) -> str:
    """Answers already known on the way from the root to `upto` when sifting u."""
    steps = []
    node = lr.dt.root
    while node is not None and not node.is_leaf() and node is not upto:
        v = node.disc or ()
        known = lr.sul._lru.get(tuple(u + v))
        if known is None:
            break
        steps.append(f"{w2s(u + v)} {'accepted' if known else 'rejected'}")
        node = node.right if known else node.left
    return "; ".join(steps)


def _ttt_mq_reason(stack, word: Word) -> str:
    lr_loc = _find(stack, "run", lambda l: type(l.get("self")).__name__ == "TTTLearner")
    if lr_loc is None:
        lr_loc = _find(stack, "_build_hypothesis") or _find(stack, "_refine")
    lr = lr_loc["self"]
    head = _ttt_state(lr)

    g = _find(stack, "g")
    if g is not None:
        rd = _find(stack, "_rs_decompose")
        w, i = rd["w"], g["i"]
        q = rd["q_after"][i]
        return (f"{head}\nAnalysing counterexample {w2s(w)} (Rivest–Schapire): for each split i, run the prefix "
                f"{w2s(w[:i])} through the hypothesis (reaches q{q}), replace it by q{q}'s access word "
                f"{w2s(lr.access[q])} and keep the suffix {w2s(w[i:])}. The first split where the answer flips "
                f"exposes a wrong transition. Query {w2s(word)}.")

    pick = _find(stack, "_pick_progress_discriminator")
    if pick is not None:
        qold, qnew, a = pick["qold"], pick["qnew"], pick["a"]
        cand = pick.get("cand")
        if _find(stack, "sift") is not None or _find(stack, "path_signature") is not None:
            what = (f"tentatively splitting q{qold} on {w2s(pick.get('cand2', cand))} and re-sifting "
                    f"{w2s(lr.access[pick['q_u']] + (a,))} to check the split changes where it lands")
        elif _find(stack, "minimize_suffix") is not None:
            what = f"shortening candidate discriminator {w2s(cand)} to the shortest suffix that still separates them"
        else:
            what = f"testing candidate discriminator {w2s(cand)}: does it separate the two access words?"
        return (f"{head}\nRefining after the counterexample: new state q{qnew} with access word "
                f"{w2s(lr.access[qnew])} must be split off from q{qold} (access {w2s(lr.access[qold])}). "
                f"Searching for a discriminator — {what}. Query {w2s(word)}.")

    if _find(stack, "split_leaf") is not None:
        sl = _find(stack, "split_leaf")
        return (f"{head}\nSplitting leaf q{sl['qold']} into q{sl['qold']} / q{sl['qnew']} with discriminator "
                f"{w2s(sl['disc'])}: record which side each access word falls on. Query {w2s(word)}.")
    if _find(stack, "minimize_suffix") is not None:
        ms = _find(stack, "minimize_suffix")
        return (f"{head}\nShortening discriminator {w2s(ms['disc'])} between access words {w2s(ms['u1'])} and "
                f"{w2s(ms['u2'])}: find the shortest suffix that still separates them. Query {w2s(word)}.")

    sift = _find(stack, "sift")
    bh = _find(stack, "_build_hypothesis")
    if sift is not None and bh is not None:
        u = sift["u"]
        q, a = bh.get("q"), bh.get("a")
        known = _sift_path(lr, u, sift["node"])
        known = f" Known so far: {known}." if known else ""
        return (f"{head}\nBuilding the hypothesis: transition q{q} --{a}--> is found by sifting "
                f"q{q}'s access word + {a} = {w2s(u)} through the tree.{known} At the node with discriminator "
                f"{w2s(sift['v'])} I need to know whether {w2s(word)} is accepted.")
    if bh is not None:
        q = bh.get("q")
        return (f"{head}\nBuilding the hypothesis: state q{q} is accepting iff its access word "
                f"{w2s(lr.access[q])} is accepted. Query {w2s(word)}.")
    if _find(stack, "_target_accepts") is not None:
        return (f"{head}\nCounterexample {w2s(word)} was returned. I need its true label to analyse it "
                f"(and later to check whether the refined hypothesis still gets it wrong). Query {w2s(word)}.")
    return f"{head}\nQuery {w2s(word)}."


def _ttt_eq_reason(stack, hyp) -> str:
    lr = _find(stack, "run", lambda l: type(l.get("self")).__name__ == "TTTLearner")["self"]
    return (f"{_ttt_state(lr)}\nEvery state's acceptance and every transition is determined by sifting "
            f"through the tree, giving a {lr.n_states}-state hypothesis. Submit it as an equivalence query.")


# ==========================================================================
# Tracing
# ==========================================================================
class _Tracer:
    def __init__(self):
        self.notes: List[str] = []
        self.active = None  # "lstar" | "ttt"


TRACER = _Tracer()


def _wrap_mq(cls, meth, kind):
    orig = getattr(cls, meth)

    def wrapped(self, word, *a, **k):
        before = len(self.history) if self.history is not None else 0
        out = orig(self, word, *a, **k)
        if TRACER.active == kind and self.history is not None and len(self.history) > before:
            st = _stack(2)
            w = tuple(word) if word is not None else ()
            note = _lstar_mq_reason(st, w) if kind == "lstar" else _ttt_mq_reason(st, w)
            TRACER.notes.append(note)
        return out

    wrapped._aal_wrapped = True
    setattr(cls, meth, wrapped)


def _wrap_eq(cls, kind):
    orig = cls.find_cex

    def wrapped(self, hypothesis):
        if TRACER.active == kind:
            st = _stack(2)
            TRACER.notes.append(_lstar_eq_reason(st, hypothesis) if kind == "lstar"
                                else _ttt_eq_reason(st, hypothesis))
        return orig(self, hypothesis)

    wrapped._aal_wrapped = True
    cls.find_cex = wrapped


def _install():
    if getattr(U.L_star.CountingSUL.query, "_aal_wrapped", False):
        return
    _wrap_mq(U.L_star.CountingSUL, "query", "lstar")
    _wrap_mq(U.TTT.CountingSUL, "membership", "ttt")
    _wrap_eq(U.L_star.MinimalDFAEqOracle, "lstar")
    _wrap_eq(U.TTT.MinimalDFAEqOracle, "ttt")


def run_with_reasons(mdfa, which: str):
    """Run L* ('lstar') or TTT ('ttt') on mdfa; return (result, notes) aligned with result.history."""
    _install()
    strat = U.L_star.LStarStrategy() if which == "lstar" else U.TTT.TTTStrategy()
    TRACER.notes, TRACER.active = [], which
    try:
        res = strat.run(mdfa)
    finally:
        TRACER.active = None
    if len(TRACER.notes) != len(res.history):
        raise RuntimeError(f"{which}: {len(TRACER.notes)} notes for {len(res.history)} queries")
    return res, list(TRACER.notes)
