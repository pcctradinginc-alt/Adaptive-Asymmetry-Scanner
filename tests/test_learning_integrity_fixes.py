"""Regressionstests Master-Auftrag (2026-10-10): Lern-/Evidenz-Integrität.
* Schatten-Outcomes aus Näherungen (Aktien-Fallback, Delta-Approx) werden gezeigt, aber nie gelernt.
* Entscheidungs-Ledger des Adapters: je (Tag, Ticker) genau eine Entscheidung (Neustart = keine Pseudo-Stichprobe).
* Ein Fehler in der Schatten-Auswertung bricht den Feedback-Lauf nicht ab (save_history + spätere Resolver laufen)."""
from __future__ import annotations

import ast
import inspect
from datetime import date, datetime, timedelta

import feedback
from modules import production_intelligence_adapter as pia
from modules.atomic_io import read_jsonl
from tests.test_shadow_lifecycle import closes_fn, live_fn, snap

D0 = date(2026, 8, 1)


def _run(hist, today, tmp_path, method):
    return feedback.evaluate_shadow_trades(hist, datetime.combine(today, datetime.min.time()),
                                           ledger_dir=tmp_path / "ledger", archive_path=tmp_path / "archive.jsonl",
                                           price_history_fn=closes_fn(), live_outcome_fn=live_fn(0.3, method))


def _due():
    return D0 + timedelta(days=int(feedback.cfg.learning.close_after_days) + 1)


def test_shadow_approximation_is_shown_but_never_learned(tmp_path):
    hist = {"shadow_trades": [snap(1)]}
    _run(hist, _due(), tmp_path, "stock_fallback")
    st = hist["shadow_trades"][0]
    assert st.get("outcome") is not None                # Legacy-Horizont fällig, Live-Fenster getroffen
    assert st["outcome_method"] == "stock_fallback"
    assert not [k for k in (hist.get("processed_outcomes") or {}) if k.startswith("shadow:")]


def test_shadow_reliable_outcome_is_learned_exactly_once(tmp_path):
    hist = {"shadow_trades": [snap(1)]}
    _run(hist, _due(), tmp_path, "option_quote")
    st = hist["shadow_trades"][0]
    assert st.get("outcome") is not None
    keys = [k for k in (hist.get("processed_outcomes") or {}) if k.startswith("shadow:")]
    assert len(keys) == 1 and keys[0].endswith(":feature_stats_external")


def test_decision_ledger_one_decision_per_day_and_ticker(tmp_path):
    rec = {"decision_id": "a1", "date": "2026-10-12", "ticker": "AAPL", "final_production_decision": "TRADE"}
    assert pia.write_ledger([rec], tmp_path) == 1
    assert pia.write_ledger([{**rec, "decision_id": "a2", "final_production_decision": "ABSTAIN"}], tmp_path) == 0
    assert pia.write_ledger([{**rec, "decision_id": "b1", "ticker": "MSFT"}], tmp_path) == 1
    rows = read_jsonl(tmp_path / "2026-10.jsonl")
    assert [(r["ticker"], r["decision_id"]) for r in rows] == [("AAPL", "a1"), ("MSFT", "b1")]   # erste gilt


def test_shadow_evaluation_failure_cannot_abort_feedback_run():
    """Statisch: der Aufruf in main() steht in try/except (vorher einziger ungeschützter Resolver)."""
    tree = ast.parse(inspect.getsource(feedback.main))
    guarded = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(sub, ast.Call) and getattr(sub.func, "id", None) == "evaluate_shadow_trades":
                    guarded = bool(node.handlers)
    assert guarded
