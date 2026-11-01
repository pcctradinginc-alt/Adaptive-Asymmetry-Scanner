"""Performance- und Lernqualitäts-Audit 2026-11-01: Monatsreport trennt nach Evidenzqualität.

Synthetische, kontrollierte Trades – sie prüfen Zählung, Klassen, Statistik und Kennzeichnung, nie Performance.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

import monthly_report as mr
from modules import outcomes

ROOT = Path(__file__).resolve().parents[1]


def trade(ticker, entry, close, outcome, *, method="option_quote", strategy="LONG_CALL", strike=100.0,
          expiry="2026-12-18", reconstructed=None, sector_etf=None):
    t = {"ticker": ticker, "entry_date": entry, "close_date": close, "outcome": outcome, "strategy": strategy,
         "option": {"strike": strike, "expiry": expiry}}
    if method:
        t["outcome_method"] = method
    if reconstructed:
        t["outcome_method_reconstructed"] = reconstructed
    if sector_etf:
        t["sector_etf"] = sector_etf
    return t


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


# ── Zählung und Deduplizierung ───────────────────────────────────────────────

def test_two_different_contracts_same_ticker_and_day_are_not_deduplicated():
    """Reproduziert 116 vs. 117: UNH 2026-04-11 Put 310 und Call 330 sind zwei Trades."""
    put = trade("UNH", "2026-04-11", "2026-05-26", -0.8315, method=None, strategy="LONG_PUT", strike=310.0,
                expiry="2026-08-21")
    call = trade("UNH", "2026-04-11", "2026-05-26", 2.5884, method=None, strategy="LONG_CALL", strike=330.0,
                 expiry="2026-08-21")
    assert mr.dedup_closed_trades([put, call]) == [put, call]


def test_identical_trades_are_counted_once():
    t = trade("LLY", "2026-04-11", "2026-05-26", 0.8166, method=None, strike=980.0)
    assert mr.dedup_closed_trades([t, dict(t), dict(t)]) == [t]
    assert mr.overall_stats(mr.dedup_closed_trades([t, dict(t)]))["n"] == 1


def test_repository_history_counts_all_distinct_closed_trades():
    h = json.loads((ROOT / "outputs" / "history.json").read_text())
    closed = h["closed_trades"]
    assert len(mr.dedup_closed_trades(closed)) == len({mr.trade_identity(t) for t in closed})
    q = h.get("closed_trades_quarantine") or []
    assert not any(t in closed for t in q) or all(t.get("_quarantine") for t in q)   # Kopien nur in Quarantäne


# ── Evidenzklassen ──────────────────────────────────────────────────────────

def _mixed():
    return [trade("A", "2026-09-01", "2026-09-20", -0.4, method="spread_quote", strategy="BULL_CALL_SPREAD"),
            trade("B", "2026-09-02", "2026-09-21", 0.5, method="option_quote"),
            trade("C", "2026-04-11", "2026-05-26", 2.0, method=None),                          # UNKNOWN
            trade("D", "2026-04-12", "2026-05-27", 3.0, method=None, reconstructed="delta_approx"),
            trade("E", "2026-04-13", "2026-05-28", -1.0, method="delta_approx")]                # APPROXIMATED


def test_reliable_and_unknown_are_never_mixed_and_reconstructed_is_never_reliable():
    ev = mr.evidence_report(_mixed(), "2026-09", memory={})
    assert ev["class_counts"] == {"RELIABLE": 2, "UNKNOWN": 1, "RECONSTRUCTED": 1, "APPROXIMATED": 1}
    assert ev["reliable"]["n_evaluable"] == 2 and ev["reliable"]["mean"] == pytest.approx(0.05)
    assert ev["classes"]["RECONSTRUCTED"]["n_evaluable"] == 1 and ev["classes"]["UNKNOWN"]["mean"] == 2.0
    assert outcomes.outcome_class(_mixed()[3]) == "RECONSTRUCTED" and not outcomes.is_reliable_outcome(_mixed()[3])
    assert ev["explorative"]["n_evaluable"] == 5
    html = _text(mr.build_evidence_html(ev, "2026-09"))
    a_section = html.split("Alle Kennzahlen je Evidenzklasse")[0]
    assert "+300.0%" not in a_section and "+200.0%" not in a_section            # keine UNKNOWN-/RECONSTRUCTED-Werte


def test_missing_outcomes_are_never_zero_returns():
    ts = [trade("A", "2026-09-01", "2026-09-20", 0.2), trade("B", "2026-09-02", "2026-09-21", None),
          trade("C", "2026-09-03", "2026-09-22", -0.4)]
    s = mr.evidence_stats(ts)
    assert s["n_trades"] == 3 and s["n_evaluable"] == 2 and s["n_missing_outcome"] == 1
    assert s["mean"] == pytest.approx(-0.1) and s["win_rate"] == 0.5


def test_mean_median_profit_factor_and_ex_top3_are_correct():
    outs = [0.6, -0.2, -0.4, 0.1, -1.0, 0.3]
    s = mr.evidence_stats([trade(f"T{i}", "2026-09-01", "2026-09-30", o) for i, o in enumerate(outs)])
    assert s["mean"] == pytest.approx(sum(outs) / 6) and s["median"] == pytest.approx((-0.2 + 0.1) / 2)
    assert s["profit_factor"] == pytest.approx(1.0 / 1.6) and s["mean_ex_top3"] == pytest.approx((-0.2 - 0.4 - 1.0) / 3)
    assert s["total_loss_rate"] == pytest.approx(1 / 6)


def test_total_loss_definition_is_identical_in_every_report_statistic():
    ts = [trade("A", "2026-10-01", "2026-10-20", -0.96), trade("B", "2026-10-02", "2026-10-21", -0.5),
          trade("C", "2026-10-03", "2026-10-22", -1.0), trade("D", "2026-10-04", "2026-10-23", -0.94)]
    assert outcomes.TOTAL_LOSS_THRESHOLD == -0.95
    assert mr.month_stats(ts, "2026-10")["total_losses"] == mr.overall_stats(ts)["total_losses"] == 2
    assert mr.rolling_stats(ts)["total_loss_rate"] == pytest.approx(2 / 4)
    assert mr.evidence_stats(ts)["total_loss_rate"] == pytest.approx(2 / 4)


# ── Statistik: kleine Stichproben und abhängige (überlappende) Trades ──────────

def test_small_samples_never_claim_significance_or_a_confidence_interval():
    s = mr.evidence_stats([trade(f"T{i}", "2026-09-01", "2026-09-30", 0.5) for i in range(7)])
    assert s["status"] == mr.INSUFFICIENT and s["ci90"] is None and "n=7" in s["reason"]
    html = _text(mr.build_evidence_html(mr.evidence_report([trade("A", "2026-09-01", "2026-09-20", 0.9)],
                                                           "2026-09", memory={}), "2026-09"))
    assert "INSUFFICIENT_EVIDENCE" in html and "signifikant" not in html


def test_overlapping_trades_do_not_produce_an_iid_confidence_interval():
    """104 von 117 Trades lagen in EINEM 45-Tage-Fenster: der alte iid-Bootstrap gab ein enges KI aus."""
    same_window = [trade(f"T{i}", "2026-04-11", "2026-05-26", (-1) ** i * 0.3) for i in range(60)]
    g = mr.grouped_ci(same_window)
    assert g["ci"] is None and g["independent_windows"] == 1 and "1 unabhängige" in g["reason"]
    html = mr.build_rolling_stats_html(same_window)
    assert "unabhängige Haltefenster" in html and " bis " not in _text(html)
    spread = [trade(f"T{i}", (date(2026, 1, 1) + timedelta(days=50 * (i % 6))).isoformat(), "2026-12-01",
                    0.1 + 0.01 * i) for i in range(30)]
    g2 = mr.grouped_ci(spread)
    assert g2["independent_windows"] == 6 and g2["ci"] is not None and g2["ci"][0] < g2["ci"][1]


def test_month_comparison_is_never_significant_without_independent_windows():
    ts = [trade(f"T{i}", "2026-04-11", f"2026-06-{(i % 28) + 1:02d}", 0.9 if i >= 30 else -0.9) for i in range(60)]
    html = mr.build_rolling_stats_html(ts, {"win_rate": 0.0, "n": 30}, {"win_rate": 1.0, "n": 30})
    assert "kein Nachweis einer Veränderung" in html and "signifikant besser" not in html


def test_spy_is_context_only_never_an_options_edge():
    cur = {"n": 5, "win_rate": 0.2, "wins": 1, "mean": 0.3, "median": 0.1, "total_losses": 0}
    html = mr.build_html("2026-10", cur, None, None, {"days": 0}, [], 0.02, None, None, None)
    assert "SPY +2.0%" in html and "Edge" not in html and "kein Options-Alpha" in html


# ── Vorschläge: explorativ, keine Testslots, keine Produktionsänderung ─────────

def _gate_history(method):
    ts = []
    for i in range(40):
        t = trade(f"G{i}", (date(2026, 1, 5) + timedelta(days=7 * i)).isoformat(), "2026-12-01",
                  0.5 if i % 2 else -0.5, method=method)
        t["deep_analysis"] = {"impact": 6 if i % 2 else 4, "surprise": 3}
        ts.append(t)
    return {"closed_trades": ts, "shadow_trades": []}


def test_explorative_gate_suggestions_get_no_challenger_template_and_change_nothing(tmp_path):
    cfg_before = (ROOT / "config.yaml").read_bytes()
    challengers_before = (ROOT / "challengers.yaml").read_bytes()
    h = _gate_history(method=None)                                                  # nur UNKNOWN
    s = {"gate": "Impact-Floor", "field": "impact", "mode": "min", "current": 4, "suggested": 6, "gain_pp": 50.0,
         "current_stats": {"n": 40, "win_rate": 0.5, "mean": 0.0},
         "suggested_stats": {"n": 20, "win_rate": 1.0, "mean": 0.5}}
    se = mr.suggestion_evidence(h, s)
    assert se["classes"] == {"UNKNOWN": 20} and se["reliable_n"] == 0 and not se["template_allowed"]
    html = mr.build_tuning_html([s], h)
    assert "EXPLORATIV – keine Challenger-Vorlage" in html and "registered_on:" not in html
    rel = _gate_history(method="option_quote")                                      # 20 RELIABLE, viele Fenster
    se2 = mr.suggestion_evidence(rel, s)
    assert se2["reliable_n"] == 20 and se2["windows"] >= mr.MIN_INDEPENDENT_WINDOWS and se2["template_allowed"]
    assert "registered_on:" in mr.build_tuning_html([s], rel)
    assert (ROOT / "config.yaml").read_bytes() == cfg_before                       # Schwellen unverändert
    assert (ROOT / "challengers.yaml").read_bytes() == challengers_before          # nichts registriert
    assert mr.current_thresholds()["impact"] == 4 and mr.current_thresholds()["surprise"] == 3


def test_slot_analysis_is_marked_explorative_with_verified_breakdown():
    ev = mr.evidence_report(_mixed(), "2026-09", memory={})
    html = mr.build_slot_html([{"name": "Spreads vs. Long Calls", "n": 5, "ready": True, "finding": "x"}], ev)
    assert mr.EXPLORATIVE_BANNER in html and "BULL_CALL_SPREAD: n=1" in html


# ── Forschungsfortschritt: fehlend bleibt fehlend ────────────────────────────

def test_research_progress_marks_missing_files_and_fields_never_zero(tmp_path):
    rp = mr.research_progress(tmp_path, tmp_path / "missing.json")
    assert rp["horizons"] == {20: None, 5: None} and len(rp["missing"]) == 4
    html = _text(mr.build_research_html(rp))
    assert "fehlt" in html and "Fehlende Forschungsdateien" in html and " 0 " not in html
    (tmp_path / "status_5d.json").write_text(json.dumps({
        "verified_labels": 359, "label_dates": 8, "model_instances": {"count": 1},
        "forward": {"valid_decisions": 0, "valid_predictions": 0, "verdict": "NEED_MORE_DATA",
                    "vs_reference": {"verdict": "NEED_MORE_DATA"}}}))
    rp2 = mr.research_progress(tmp_path, tmp_path / "missing.json")
    h5 = rp2["horizons"][5]
    assert h5["verified_labels"] == 359 and h5["model_instances"] == 1 and h5["learning_state"] is None
    assert "359" in mr.build_research_html(rp2)


def test_loss_causes_separate_observed_plausible_supported_and_unchecked(tmp_path):
    p = tmp_path / "fa.json"
    p.write_text(json.dumps({"reliable": {"n": 6, "primary": {"signal_wrong": {"n": 2}}, "labels": {}},
                             "unreliable": {"n": 71}}))
    html = _text(mr.build_loss_causes_html(p))
    for part in ("Beobachtete Tatsache", "Plausible Ursache", "Statistisch unterstützt", "Ungeprüft / Datenlücke"):
        assert part in html
    assert "keine – n=6" in html and "71 Verluste" in html
    assert "fehlt" in mr.build_loss_causes_html(tmp_path / "none.json")


# ── Vollständiger Report mit kontrollierten Daten, E-Mail-Versand ─────────────

def test_full_monthly_report_with_controlled_data_and_missing_research(tmp_path, monkeypatch):
    hist = {"closed_trades": _mixed() + [trade("UNH", "2026-04-11", "2026-05-26", -0.8, method=None,
                                               strategy="LONG_PUT", strike=310.0),
                                         trade("UNH", "2026-04-11", "2026-05-26", 2.5, method=None, strike=330.0)],
            "shadow_trades": []}
    (tmp_path / "history.json").write_text(json.dumps(hist))
    monkeypatch.setattr(mr, "HISTORY_PATH", tmp_path / "history.json")
    monkeypatch.setattr(mr, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(mr, "OPTION_LEARNER_DIR", tmp_path / "ol")
    monkeypatch.setattr(mr, "ALPHA_VALIDATION_PATH", tmp_path / "none.json")
    monkeypatch.setattr(mr, "FAILURE_ANALYSIS_PATH", tmp_path / "none.json")
    monkeypatch.setattr(mr, "TRADE_MEMORY_PATH", tmp_path / "none.jsonl")
    monkeypatch.setattr(mr, "spy_return", lambda key: 0.01)

    class _Today(date):
        @classmethod
        def today(cls):
            return date(2026, 10, 1)
    monkeypatch.setattr(mr, "date", _Today)
    sent = {}
    monkeypatch.setattr(mr, "_send_smtp", lambda subject, html: sent.update(subject=subject, html=html))
    mr.main()
    subj, html = sent["subject"], _text(sent["html"])
    assert "verifiziert n=2 (INSUFFICIENT_EVIDENCE)" in subj and "explorativ n=7" in subj and "Win-Rate" not in subj
    assert html.index("A. Verifizierte Performance") < html.index("B. Historische Performance") \
        < html.index("C. Forschungs- und Lernfortschritt")
    assert mr.EXPLORATIVE_BANNER in html and "Gesamt (alle closed Trades, alle Evidenzklassen)" in html
    assert "Win-Rate 57% (4/7)" in html                                           # beide UNH-Kontrakte gezählt
    assert "Fehlende Forschungsdateien" in html and "Edge" not in html
