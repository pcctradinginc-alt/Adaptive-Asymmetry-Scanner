"""Champion-Protokoll und gemeinsames Fehlerbudget für Challenger/Promotion (Aktionsplan 2026-11-09, Punkte 1 und 7).

Geprüft:
  * die entscheidungsrelevante Konfiguration ist gepinnt – jede Gate-/Prompt-Änderung bricht das Protokoll
    (neue Protokollversion nötig, kein Tuning auf gesehenen Daten);
  * Bewertung nur mit Ledger-Zuteilung, nur verlässliche Netto-Outcomes ab Start, Tagesmittel, eingefrorene Prüfungen;
  * neue Challenger/Verträge/Champion erhalten Alpha genau einmal aus dem Ledger (idempotent, fail closed),
    Altbestände bleiben unverändert (Altfamilie, nie gelockert).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import yaml

from modules import challenger as ch
from modules import champion_protocol as cp
from modules import ledger_registration as lr
from modules import option_learner as ol
from modules import promotion_controller as pc

NOW = datetime(2026, 11, 9, 21, 0, tzinfo=timezone.utc)


def test_decision_config_is_pinned_by_the_registered_protocol():
    proto = cp.load_protocol()
    assert cp.decision_config_sha() == proto["decision_config_sha256"], (
        "Entscheidungs-Konfiguration (Gates, Filter, Optionen, Pipeline, Modelle, Kosten, Haltedauer, "
        "economics-first, Prompts) geändert: neue Champion-Protokollversion registrieren (version, start_date, "
        "decision_config_sha256) – keine Gate-Änderung innerhalb eines laufenden Protokolls.")
    assert proto["start_date"] > proto["registered_on"] and proto["looks_at_n"] == sorted(proto["looks_at_n"])


def _proto(tmp_path, sha, start="2026-11-10"):
    p = tmp_path / "proto.yaml"
    p.write_text(yaml.safe_dump({"version": "champion-test", "registered_on": "2026-11-09", "start_date": start,
                                 "looks_at_n": [5, 10], "max_duration_days": 365,
                                 "decision_config_sha256": sha}))
    return p


def _hist(tmp_path, ys, start=date(2026, 11, 10), method="option_quote"):
    trades = []
    for i, y in enumerate(ys):
        d = date.fromordinal(start.toordinal() + i)
        trades.append({"ticker": f"T{i}", "entry_date": d.isoformat(), "close_date": d.isoformat(),
                       "outcome": y, "outcome_net": y, "outcome_method": method})
    trades.append({"ticker": "OLD", "entry_date": "2026-11-01", "close_date": "2026-11-05", "outcome_net": 5.0,
                   "outcome_method": "option_quote"})                         # vor Start: zählt nie
    p = tmp_path / "history.json"
    p.write_text(json.dumps({"closed_trades": trades}))
    return p


def _ledger_with(tmp_path, test_id, sub, h):
    lp = tmp_path / "research" / "alpha_ledger.jsonl"
    assert ol.alpha_allocate(test_id, sub, h, NOW, path=lp) is not None
    return lp


def test_protocol_needs_allocation_breaks_on_config_change_and_freezes_looks(tmp_path):
    sha = cp.decision_config_sha()
    proto = _proto(tmp_path, sha)
    hist = _hist(tmp_path, [0.3, 0.2, 0.25, 0.4, 0.35, 0.3, 0.2, 0.3, 0.25, 0.35])
    kw = {"protocol_path": proto, "history_path": hist, "out": tmp_path / "out"}
    assert cp.evaluate(date(2026, 11, 30), alpha_ledger_path=tmp_path / "none.jsonl", **kw)["status"] \
        == "AWAITING_ALLOCATION"                                               # ohne Zuteilung keine Bestätigung
    lp = _ledger_with(tmp_path, lr.champion_test_id(cp.load_protocol(proto)), "champion", "h")
    res = cp.evaluate(date(2026, 11, 30), alpha_ledger_path=lp, **kw)
    assert res["reliable_trades"] == 10 and res["look_alpha"] == pytest.approx(res["ledger_alpha"] / 2)
    assert res["looks"][0]["n_trades"] == 5 and res["status"] in ("CONFIRMED_POSITIVE", "CONTINUE")
    frozen = json.loads((tmp_path / "out" / "look_1.json").read_text())
    _hist(tmp_path, [-0.9] * 10)                                               # spätere Daten ändern Prüfung 1 nie
    again = cp.evaluate(date(2026, 12, 30), alpha_ledger_path=lp, **kw)
    assert again["looks"][0]["test"] == frozen["test"]
    broken = cp.evaluate(date(2026, 11, 30), alpha_ledger_path=lp,
                         **{**kw, "protocol_path": _proto(tmp_path, "abc")})
    assert broken["status"] == "PROTOCOL_BROKEN"


def test_only_reliable_net_outcomes_after_start_count_and_days_are_one_observation(tmp_path):
    h = json.loads(_hist(tmp_path, [0.1, -0.2], method="option_quote").read_text())
    h["closed_trades"].append({"ticker": "X", "entry_date": "2026-11-10", "close_date": "2026-11-20",
                               "outcome_net": 0.5, "outcome_method": "underlying_proxy"})   # nicht verlässlich
    h["closed_trades"].append({"ticker": "Y", "entry_date": "2026-11-10", "close_date": "2026-11-21",
                               "outcome_net": 0.3, "outcome_method": "spread_quote"})
    tr = cp.reliable_trades(h, date(2026, 11, 10))
    assert [t["ticker"] for t in tr] == ["T0", "T1", "Y"]
    assert cp.day_means(tr) == pytest.approx([0.2, -0.2])                      # 10.11.: (0,1 + 0,3)/2
    t = cp.one_sided_test([0.1, 0.2, 0.15, 0.12], 0.01)
    assert t["p_value"] < 0.01 and t["lower"] > 0
    assert cp.one_sided_test([-0.1, -0.2, -0.15], 0.05)["upper"] < 0


def test_ledger_registration_allocates_new_tests_once_and_keeps_legacy_untouched(tmp_path):
    reg = tmp_path / "challengers.yaml"
    reg.write_text(yaml.safe_dump({"challengers": [
        {"id": "old", "registered_on": "2026-09-26", "start_date": "2026-09-28", "rule": [], "metric": "m",
         "status": "active"},
        {"id": "new", "registered_on": "2026-11-09", "start_date": "2026-11-10", "rule": [], "metric": "m",
         "status": "active"}]}))
    proto = _proto(tmp_path, "x")
    paths = {"registry_path": reg, "contracts_path": tmp_path / "none.yaml", "champion_path": proto}
    pend = lr.pending_registrations(**paths)
    assert [p[0] for p in pend] == ["CHALLENGER:new", "CHAMPION:champion-test"]
    lp = tmp_path / "r" / "alpha_ledger.jsonl"
    first = lr.allocate_all(NOW, path=lp, **paths)
    assert [r["status"] for r in first] == ["NEW", "NEW"]
    assert [r["status"] for r in lr.allocate_all(NOW, path=lp, **paths)] == ["ALLOCATED", "ALLOCATED"]
    d = yaml.safe_load(reg.read_text())
    d["challengers"][1]["min_n"] = 5                                           # Vorregistrierung nachträglich geändert
    reg.write_text(yaml.safe_dump(d))
    assert lr.allocate_all(NOW, path=lp, **paths)[0]["status"] == "SPEC_CHANGED"
    st = ol.alpha_ledger_state(lp, persist=False)
    assert len(st["allocations"]) == 2 and st["summary"]["legacy_families_outside_budget"]


def test_challenger_alpha_regimes_never_loosen_the_legacy_family(tmp_path):
    new = {"id": "new", "registered_on": "2026-11-09", "start_date": "2026-11-10", "rule": [], "metric": "m",
           "max_duration_days": 90}
    old = {**new, "id": "old", "registered_on": "2026-09-26", "start_date": "2026-09-28"}
    lp = tmp_path / "r" / "alpha_ledger.jsonl"
    r = ch.evaluate(new, [], date(2026, 12, 1), 2, lp)
    assert r["alpha"] is None and r["alpha_regime"] == "awaiting_ledger_allocation" and r["verdict"] == "running"
    ol.alpha_allocate(lr.challenger_test_id(new), "challenger", lr.spec_hash(lr.challenger_spec(new)), NOW, path=lp)
    r = ch.evaluate(new, [], date(2026, 12, 1), 2, lp)
    alloc = lr.ledger_alpha(lr.challenger_test_id(new), lp)
    assert r["alpha_regime"] == "alpha-ledger-v1" and r["alpha"] == pytest.approx(alloc["alpha"] / 3)
    r_old = ch.evaluate(old, [], date(2026, 12, 1), 2, lp)                     # nur noch 2 aktive Altfälle
    assert r_old["alpha"] == pytest.approx(0.10 / (3 * 3)) and r_old["alpha_regime"].startswith("legacy")


def test_new_promotion_contracts_get_ledger_alpha_only(tmp_path):
    policy = {"statistics": {"alpha_family": 0.05, "planned_looks": 12}}
    c_old = {"hypothesis_id": "H", "production_class": "X", "registered_at": "2026-10-03T12:00:00Z"}
    c_new = {**c_old, "registered_at": "2026-11-10T12:00:00Z"}
    lp = tmp_path / "r" / "alpha_ledger.jsonl"
    old = pc.family_alpha(c_old, [], policy, lp)
    assert old["alpha_effective"] == pytest.approx(0.05 / 12) and old["alpha_regime"].startswith("legacy")
    import modules.hypothesis_contract as hc
    orig_key, orig_hash = hc.key, hc.spec_hash
    try:
        hc.key, hc.spec_hash = (lambda c: "K1"), (lambda c: "S1")
        assert pc.family_alpha(c_new, [], policy, lp)["alpha_effective"] == 0.0    # ohne Zuteilung: keine Bestätigung
        ol.alpha_allocate("PROMOTION:K1", "promotion_contract", "S1", NOW, path=lp)
        got = pc.family_alpha(c_new, [], policy, lp)
        assert got["alpha_regime"] == "alpha-ledger-v1"
        assert got["alpha_effective"] == pytest.approx(min(0.05 / 12, got["ledger_alpha"] / 12))
    finally:
        hc.key, hc.spec_hash = orig_key, orig_hash


def test_real_registry_routes_new_challengers_and_the_champion_through_the_ledger():
    ids = [p[0] for p in lr.pending_registrations()]
    assert "CHALLENGER:no_mc_gates" in ids and "CHALLENGER:mismatch_directional_news" in ids
    assert any(i.startswith("CHAMPION:") for i in ids)
    assert not any(i in ("CHALLENGER:final_mc_dte_shadow", "CHALLENGER:trade_score_61") for i in ids)
    cfg = ol.alpha_ledger_config()
    subs = {s for c in cfg["classes"].values() for s in c["subsystems"]}
    assert {"challenger", "promotion_contract", "champion"} <= subs
    assert Path(cp.PROTOCOL).exists()
