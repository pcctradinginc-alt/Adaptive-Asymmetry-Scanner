"""Expectation Alpha (SHADOW): Darstellung im Montagsbericht (Abschnitt 21) und in den beiden Tagesmails.

Nur Rendering: offline, synthetische Eingaben in tmp_path, kein SMTP, kein Netzwerk. Geprüft wird, dass SHADOW-Daten
nie wie eine Handelsempfehlung aussehen (nur Aggregate in der Mail, kein Ticker) und dass fehlende/kaputte Dateien
nichts brechen."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from modules import email_reporter as er
from modules.reporter import compute_exit_rules
from reports import weekly

TODAY = date(2026, 10, 12)
EA_TITLE = "EXPECTATION ALPHA – SHADOW (Research, keine Handelsempfehlung)"


# ── Hilfen ──────────────────────────────────────────────────────────────────
def _stat(n, status="OK", **kw):
    return {"n": n, "independent_dates": n, "status": status, **kw}


def _ok(n, mean, **kw):
    return _stat(n, "OK", mean=mean, median=mean / 2, hit_rate=0.56, mae=-0.04, mfe=0.09,
                 ci95=[mean - 0.01, mean + 0.01], **kw)


def _evaluation() -> dict:
    need = _stat(7, "NEED_MORE_DATA")
    return {
        "generated": "2026-10-11T06:00:00+00:00", "mode": "SHADOW – Research, keine Handelsempfehlung",
        "population": {"n": 41, "dates": 12, "event_clusters": 30, "errors": 1,
                       "by_status": {"TRADE": 9, "WAIT": 14, "ABSTAIN": 17, "ERROR": 1},
                       "by_group": {"A": 5, "B": 6}},
        "primary_horizon": 60,
        "groups": {"20": {"A": need}, "60": {"A": _ok(52, 0.0312), "B": need, "C": _ok(31, -0.0125), "D": need,
                                              "E": need, "X": need}},
        "values": {
            "abstention": {"all": _ok(40, 0.01), "kept": _ok(33, 0.0185), "delta": 0.0085},
            "wait": {"paired": _ok(34, 0.0112), "n_resolved": 20, "no_entry_share": 0.35},
            "expression": {"thesis_quality": _ok(33, 0.014), "expression_quality": need,
                           "rule_vs_default": _ok(32, -0.002)},
            "kill_management": _ok(40, 0.0061)},
        "contracts": [
            {"key": "EA001", "title": "t", "state": "FORWARD_COLLECTING", "n": 12, "independent_dates": 9,
             "span_days": 21, "regimes": ["risk_on", "risk_off"], "delta": 0.0123, "ci": [-0.004, 0.03],
             "next_requirement": "NEED_MORE_DATA: 21 von 120 Signaltagen", "forward_start": "2026-10-01"},
            {"key": "EA002", "title": "t", "state": "FORWARD_COLLECTING", "n": 0, "independent_dates": 0,
             "span_days": 0, "regimes": [], "delta": None, "ci": [None, None], "next_requirement": None,
             "forward_start": "2026-10-01"}],
        "failure_classes": {"THESIS_WRONG": 4, "TIMING_WRONG": 2},
        "data_status": {"runs": 9, "error_runs": 1, "missing_domains": {"labor": 3}, "stale_components": {"cpi": 2},
                        "median_runtime_seconds": 41.5}}


def _run(day: int, **kw) -> dict:
    r = {"date": f"2026-10-{day:02d}", "candidate_count": 10 + day, "enriched_count": 10 + day,
         "status_counts": {"TRADE": 2, "WAIT": 3, "ABSTAIN": 4, "ERROR": 1}, "groups": {"A": 1},
         "gap_z": {"growth": 0.8, "labor": None}, "missing_counts": {"labor": "NO_DATA"}, "stale_components": [],
         "regime_uncertainty": 0.35, "errors": [{"where": "prices", "error": "x"}], "run_errors": [],
         "runtime_seconds": 38.25}
    r.update(kw)
    return r


def _w(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _make_ea(root: Path, n_runs: int = 9):
    d = root / "outputs" / "expectation_alpha"
    _w(d / "evaluation.json", _evaluation())
    lines = [json.dumps(_run(day)) for day in range(1, n_runs + 1)]
    lines.insert(2, "{kaputt")                                    # unlesbare Zeile wird übersprungen
    (d / "runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _section(data: dict):
    return dict((n, b) for n, _t, b in weekly.intelligence_sections(data))[21]


def _flat(blocks) -> str:
    out = []
    for b in blocks:
        if b[0] in ("para", "note"):
            out.append(b[1])
        elif b[0] == "kv":
            out += [f"{k}: {v}" for k, v in b[1]]
        elif b[0] == "list":
            out += list(b[1])
        elif b[0] == "table":
            out += [" | ".join(b[1])] + [" | ".join(map(str, r)) for r in b[2]]
    return "\n".join(out)


def _renderable(blocks):
    """render_html/render_md erwarten Strings (kv-Werte, Listeneinträge)."""
    for b in blocks:
        assert b[0] in ("para", "note", "kv", "table", "list")
        if b[0] in ("para", "note"):
            assert isinstance(b[1], str)
        elif b[0] == "kv":
            assert all(isinstance(k, str) and isinstance(v, str) for k, v in b[1])
        elif b[0] == "list":
            assert all(isinstance(i, str) for i in b[1])
        elif b[0] == "table":
            assert all(len(r) == len(b[1]) for r in b[2])


# ── Montagsbericht: Abschnitt 21 ────────────────────────────────────────────
def test_section_title_registered_and_last():
    assert weekly.SECTION_TITLES[21] == "EXPECTATION ALPHA (SHADOW – Research, keine Handelsempfehlung)"
    assert max(weekly.SECTION_TITLES) == 21


def test_section_21_no_data_when_files_missing(tmp_path):
    data = weekly.collect(tmp_path, TODAY, state_path=tmp_path / "st.json")
    assert data["ea"] == {} and data["ea_runs"] == []
    assert weekly.expectation_alpha_section(data) == [("para", weekly.NO_DATA)]
    secs = {n: (t, b) for n, t, b in weekly.intelligence_sections(data)}
    assert secs[21] == (weekly.SECTION_TITLES[21], [("para", weekly.NO_DATA)])
    txt = weekly.render_text(data)
    assert f"A21. {weekly.SECTION_TITLES[21]}" in txt
    weekly.render_html(data)
    weekly.render_md(data)


def test_collect_reads_ea_files_and_keeps_last_seven_runs(tmp_path):
    _make_ea(tmp_path, n_runs=9)
    data = weekly.collect(tmp_path, TODAY, state_path=tmp_path / "st.json")
    assert data["ea"]["primary_horizon"] == 60
    assert [r["date"] for r in data["ea_runs"]] == [f"2026-10-0{d}" for d in range(3, 10)]


def test_section_21_renders_population_run_groups_values_contracts(tmp_path):
    _make_ea(tmp_path)
    data = weekly.collect(tmp_path, TODAY, state_path=tmp_path / "st.json")
    blocks = _section(data)
    _renderable(blocks)
    t = _flat(blocks)
    # Population + letzter Lauf (nur 2026-10-09 = ea_runs[-1])
    assert "Population n: 41" in t and "Signaltage: 12" in t and "Ereignis-Cluster: 30" in t
    assert "Shadow-TRADE 9" in t and "WAIT 14" in t and "ABSTAIN 17" in t
    assert "Letzter EA-Lauf (2026-10-09)" in t and "Kandidaten: 19" in t and "Angereichert: 19" in t
    assert "Shadow-TRADE / WAIT / ABSTAIN / ERROR: 2 / 3 / 4 / 1" in t
    assert "labor (NO_DATA)" in t and "1 Kontext (prices)" in t and "Laufzeit: 38.25 s" in t
    # NEWS × CONTEXT im Primärhorizont (60), nicht im 20er-Horizont
    tables = [b for b in blocks if b[0] == "table"]
    grp = tables[0]
    assert grp[1] == ["Gruppe", "n", "Status", "Mittel", "Median", "Treffer", "MAE", "MFE", "CI95"]
    rows = {r[0]: r for r in grp[2]}
    assert list(rows) == ["A", "B", "C", "D", "E", "X"]
    assert rows["A"] == ["A", "52", "OK", "+3.1%", "+1.6%", "56.0%", "-4.0%", "+9.0%", "[+2.1%, +4.1%]"]
    assert rows["C"][3] == "-1.2%"
    assert rows["B"] == ["B", "7", "NEED_MORE_DATA"] + ["–"] * 6
    # gepaarte Werte
    assert "Abstention Δ" in t and "+0.9%" in t
    assert "WAIT gepaart" in t and "Anteil ohne Einstieg 35.0%" in t
    assert "Thesis-Qualität" in t and "Expression-Qualität" in t and "Regel vs. Default" in t
    assert "Kill-Management" in t
    assert "Expression-Qualität (gewählt minus Median): n 7, Status NEED_MORE_DATA, Δ –" in t
    # Verträge
    con = tables[1]
    assert con[1] == ["Vertrag", "Zustand", "n", "Tage", "Spanne", "Regime (Anzahl)", "Δ", "CI", "nächste Anforderung"]
    assert con[2][0] == ["EA001", "FORWARD_COLLECTING", "12", "9", "21 T", "2", "+1.2%", "[-0.4%, +3.0%]",
                         "NEED_MORE_DATA: 21 von 120 Signaltagen"]
    assert con[2][1][5:] == ["0", "n/a", "–", "–"]
    # Fehlerklassen + Datenstatus
    assert "THESIS_WRONG 4" in t and "TIMING_WRONG 2" in t
    assert "Datenstatus (letzte 9 Läufe)" in t and "labor 3" in t and "cpi 2" in t and "41.5 s" in t
    # alle drei Renderer
    txt, md, html = weekly.render_text(data), weekly.render_md(data), weekly.render_html(data)
    for out in (txt, md, html):
        assert "EXPECTATION ALPHA" in out and "EA001" in out
    assert "<th>Gruppe</th>" in html


def test_section_21_never_calls_anything_a_recommendation(tmp_path):
    _make_ea(tmp_path)
    data = weekly.collect(tmp_path, TODAY, state_path=tmp_path / "st.json")
    blocks = _section(data)
    t = _flat(blocks)
    assert "Shadow-TRADE" in t
    assert "SHADOW" in t and "EA001–EA007" in t and "PromotionController" in t and "NONE" in t
    assert "Champion" in t and "Sizing" in t
    # "Empfehlung" nur in der Verneinung ("keine Handelsempfehlung")
    assert "Empfehlung" not in t.replace("keine Handelsempfehlung", "")
    assert "empfehl" not in t.replace("keine Handelsempfehlung", "").lower()
    assert t.count("keine Handelsempfehlung") >= 1
    assert "keine Handelsempfehlung" in weekly.SECTION_TITLES[21]
    # TRADE nie nackt, immer als Shadow-TRADE
    assert not re.search(r"(?<!Shadow-)\bTRADE\b", t)
    assert not re.search(r"(?<!Shadow-)\bTRADE\b", weekly.render_text(data).split("A21.")[1])


def test_section_21_only_runs_or_only_evaluation(tmp_path):
    only_eval = tmp_path / "e"
    _w(only_eval / "outputs" / "expectation_alpha" / "evaluation.json", _evaluation())
    blocks = _section(weekly.collect(only_eval, TODAY, state_path=tmp_path / "s1.json"))
    _renderable(blocks)
    t = _flat(blocks)
    assert "Population n: 41" in t and f"Letzter EA-Lauf (runs.jsonl): {weekly.NO_DATA}" in t
    only_runs = tmp_path / "r"
    d = only_runs / "outputs" / "expectation_alpha"
    d.mkdir(parents=True)
    (d / "runs.jsonl").write_text(json.dumps(_run(5)) + "\n", encoding="utf-8")
    blocks = _section(weekly.collect(only_runs, TODAY, state_path=tmp_path / "s2.json"))
    _renderable(blocks)
    t = _flat(blocks)
    assert "Kandidaten: 15" in t and f"EA-Ledger (evaluation.json): {weekly.NO_DATA}" in t
    assert "Fehlerklassen (Primärhorizont, regelbasiert): " + weekly.NO_DATA in t
    assert f"Datenstatus: {weekly.NO_DATA}" in t and f"Verträge EA001–EA007: {weekly.NO_DATA}" in t


@pytest.mark.parametrize("ea, runs", [
    ({"population": {"n": 3}, "groups": [], "values": {"abstention": None, "wait": []}, "contracts": [None, 7, {}]},
     [{"status_counts": None, "errors": "x", "missing_counts": [], "run_errors": None}]),
    ({"primary_horizon": 60, "groups": {"60": {"A": None, "B": {"status": "OK"}}}, "values": {"kill_management": 3},
      "contracts": [{"key": "EA009", "regimes": 4, "ci": "x", "span_days": "n"}], "data_status": [], "failure_classes": []},
     []),
])
def test_section_21_survives_malformed_input(ea, runs):
    blocks = weekly.expectation_alpha_section({"ea": ea, "ea_runs": runs})
    _renderable(blocks)
    assert blocks and blocks[0][0] == "note"


# ── E-Mail: _ea_shadow_html ─────────────────────────────────────────────────
def _summary(**kw) -> dict:
    s = {"mode": "shadow", "enabled": True, "production_influence": "NONE", "candidate_count": 14,
         "enriched_count": 13, "status_counts": {"TRADE": 2, "WAIT": 3, "ABSTAIN": 7, "ERROR": 1},
         "shadow_trade_count": 2, "wait_count": 3, "abstain_count": 7, "error_count": 1,
         "groups": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 0, "X": 3},
         "gap_z": {"growth": 1.2345, "labor": None, "inflation": -0.5}, "regime_uncertainty": 0.4321,
         "errors": [{"where": "prices", "error": "x"}], "run_errors": ["LAUFZEITBUDGET_UEBERSCHRITTEN"],
         "missing_counts": {"labor": "NO_DATA"}, "runtime_seconds": 12.3,
         # Ticker stehen in unbeteiligten Schlüsseln: dürfen NIE in der Mail erscheinen
         "candidate_errors": [{"ticker": "ZQXW", "error": "kaputt"}],
         "observations": [{"observation_id": "o1", "ticker": "QQQX"}],
         "tickers": ["ZQXW", "QQQX", "WWWY"]}
    s.update(kw)
    return s


def test_shadow_html_empty_for_missing_or_disabled():
    assert er._ea_shadow_html(None) == ""
    assert er._ea_shadow_html({}) == ""
    assert er._ea_shadow_html({"mode": "off", "enabled": False}) == ""
    assert er._ea_shadow_html("kaputt") == ""


def test_shadow_html_error_summary_escapes_error():
    h = er._ea_shadow_html({"mode": "shadow", "enabled": True, "status": "ERROR",
                            "error": "ValueError: <script>alert(1)</script> & co"})
    assert EA_TITLE in h
    assert "EA-Fehler: ValueError: &lt;script&gt;alert(1)&lt;/script&gt; &amp; co" in h
    assert "<script>" not in h
    assert "Kandidaten" not in h


def test_shadow_html_shows_only_aggregates_never_tickers():
    h = er._ea_shadow_html(_summary())
    assert EA_TITLE in h
    assert "Kandidaten: 14" in h and "angereichert 13" in h
    assert "Shadow-TRADE 2" in h and "WAIT 3" in h and "ABSTAIN 7" in h and "ERROR 1" in h
    assert "A 1" in h and "B 2" in h and "C 3" in h and "D 4" in h and "E 0" in h and "X 3" in h
    assert "growth 1.23" in h and "labor n/v" in h and "inflation -0.50" in h
    assert "Regime-Unsicherheit: 0.43" in h
    assert "Anzahl Fehler: 3" in h                                # 1 Kontext + 1 Lauf + 1 Kandidat
    for tk in ("ZQXW", "QQQX", "WWWY"):
        assert tk not in h
    assert "kaputt" not in h and "o1" not in h
    # neutral/grau, nicht grün
    assert "#94a3b8" in h and "#16a34a" not in h and "#f0fdf4" not in h and "#22c55e" not in h


def test_shadow_html_escapes_strings_and_handles_odd_values():
    h = er._ea_shadow_html(_summary(gap_z={"<b>x</b>": None, "y": float("nan"), "z": "str"},
                                    regime_uncertainty=None, status_counts=None, groups="x",
                                    candidate_count="<i>1</i>"))
    assert "&lt;b&gt;x&lt;/b&gt; n/v" in h and "y n/v" in h and "z n/v" in h
    assert "<b>x</b>" not in h and "<i>" not in h and "&lt;i&gt;1&lt;/i&gt;" in h
    assert "Regime-Unsicherheit: n/v" in h
    assert "Shadow-TRADE 2" in h                                  # Fallback auf shadow_trade_count
    # Zusammenfassung ohne Detailfelder -> trotzdem kein Absturz
    assert EA_TITLE in er._ea_shadow_html({"mode": "shadow", "enabled": True})


# ── E-Mail: Einbindung in beide Tagesmails ──────────────────────────────────
@pytest.fixture
def outbox(monkeypatch):
    sent = []
    monkeypatch.setattr(er, "_send_smtp", lambda subject, html: sent.append((subject, html)))
    monkeypatch.setattr(er, "_external_context_html", lambda: "<!--ext-->")
    monkeypatch.setattr(er, "_v2_recommendation_html", lambda today: "<!--v2-->")
    return sent


def _proposal(ticker="ABC", score=90):
    p = {"ticker": ticker, "strategy": "LONG_CALL", "trade_score": {"total": score, "grade": "B",
                                                                      "best_argument_for": "dafür",
                                                                      "best_argument_against": "dagegen"},
         "deep_analysis": {"catalyst_confidence": 7, "time_to_materialization": "4-8 Wochen"},
         "simulation": {"hit_rate": 0.62}, "mc_hit_rate": 0.62,
         "option": {"strike": 105, "expiry": "2026-12-18", "dte": 77, "bid": 4.0, "ask": 4.2},
         "roi_analysis": {"delta": 0.55, "theta_daily_pct": 0.02, "vega_loss": 0.05, "breakeven": 109.2,
                          "breakeven_pct": 0.04},
         "implied_move_pct": 8.0, "model_move_pct": 14.0, "edge_vs_implied": 6.0}
    p["exit_rules"] = compute_exit_rules(p)
    return p


def test_status_mail_contains_shadow_block_only_with_summary(outbox):
    er.send_status_email({"trades": 0, "vix": 16.4, "expectation_alpha": _summary()}, "2026-10-12")
    subj, h = outbox[-1]
    assert "Kein Trade" in subj and EA_TITLE in h and "Shadow-TRADE 2" in h
    assert h.index("<!--ext-->") < h.index(EA_TITLE)               # nach dem externen Kontext
    for tk in ("ZQXW", "QQQX", "WWWY"):
        assert tk not in h
    for stats in ({"trades": 0}, {"trades": 0, "expectation_alpha": None},
                  {"trades": 0, "expectation_alpha": {"mode": "off", "enabled": False}}):
        er.send_status_email(stats, "2026-10-12")
        assert "EXPECTATION ALPHA" not in outbox[-1][1]
    # Fehlerfall steht sichtbar in der Mail
    er.send_status_email({"trades": 0, "expectation_alpha": {"mode": "shadow", "enabled": True, "status": "ERROR",
                                                            "error": "OSError: <x>"}}, "2026-10-12")
    assert "EA-Fehler: OSError: &lt;x&gt;" in outbox[-1][1]


def test_no_trade_mail_via_send_email_carries_shadow_block(outbox):
    er.send_email([_proposal(score=5)], "2026-10-12", {"universe": 500, "expectation_alpha": _summary()})
    subj, h = outbox[-1]
    assert "Kein Trade" in subj and EA_TITLE in h


def test_trade_mail_contains_shadow_block_only_with_summary(outbox):
    er.send_email([_proposal()], "2026-10-12", {"expectation_alpha": _summary()})
    subj, h = outbox[-1]
    assert "Trade Empfehlung" in subj and "ABC" in h
    assert EA_TITLE in h and "Shadow-TRADE 2" in h
    assert h.index("<!--v2-->") < h.index(EA_TITLE) < h.index("<!--ext-->")   # nach V2-Block, vor Externem Kontext
    for tk in ("ZQXW", "QQQX", "WWWY"):
        assert tk not in h
    shadow = er._ea_shadow_html(_summary())
    assert "ABC" not in shadow
    for stats in ({}, {"expectation_alpha": None}, {"expectation_alpha": {"mode": "off", "enabled": False}}):
        er.send_email([_proposal()], "2026-10-12", stats)
        assert "EXPECTATION ALPHA" not in outbox[-1][1]


def test_build_trade_email_stays_backward_compatible(outbox):
    h = er._build_trade_email([_proposal()], "2026-10-12")          # alter Aufruf ohne ea
    assert "ABC" in h and "EXPECTATION ALPHA" not in h
    h = er._build_trade_email([_proposal()], "2026-10-12", ea=_summary())
    assert EA_TITLE in h
    assert "EXPECTATION ALPHA" not in er._build_status_email({"trades": 0}, "2026-10-12")
