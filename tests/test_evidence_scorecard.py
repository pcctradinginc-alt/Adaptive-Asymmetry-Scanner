"""Evidence-Scorecard und echte Options-Populationen (2026-10-11). Synthetische Daten prüfen nur die Mechanik."""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from modules import evidence_ledger as el

D0 = date(2026, 1, 5)


def _row(i, *, stage="EARLY", opt=None, strat=None, direction="BULLISH", day_off=0, late=False):
    d = D0 + timedelta(days=day_off)
    oc = {"ret_5d": 0.01}
    if opt is not None:
        oc.update({"real_opt_ret_5d": opt, "real_opt_mark_date_5d": (d + timedelta(days=6)).isoformat()})
    if strat is not None:
        oc.update({"real_strat_ret_5d": strat, "real_strat_mark_date_5d": (d + timedelta(days=6)).isoformat()})
    if late:
        oc["real_opt_late_5d"] = True
    status, rs = {"CHAMPION": ("proposed", None), "LATE": ("rejected", "options_design"),
                  "EARLY": ("rejected", "deep_analysis")}[stage]
    return {"signal_id": f"s{i}", "ticker": f"T{i % 7}", "direction": direction, "date": d.isoformat(),
            "signal_timestamp": f"{d.isoformat()}T15:00:00+00:00", "status": status, "reject_stage": rs,
            "features": {"prescreen_category": "catalyst", "vix_structure": "contango"}, "outcomes": oc,
            "real_option": {"symbol": f"X{i}", "ask": 2.0, "mid": 1.9, "entry_date": d.isoformat(),
                            "expiry": "2027-01-15", "entry_pending": False},
            "real_strategy": {"strategy": "BULL_CALL_SPREAD", "net_debit_entry": 1.0, "entry_date": d.isoformat(),
                              "expiry": "2027-01-15", "entry_pending": False, "legs": [{}, {}]}}


def _write(tmp_path, rows):
    (tmp_path / "2026-01.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_option_records_are_separate_populations_with_commission_and_timing(tmp_path):
    _write(tmp_path, [_row(1, opt=0.5, strat=-0.2, direction="BEARISH")])
    recs = {r["population"]: r for r in el.from_candidate_ledger(tmp_path, horizons=(5,))}
    assert set(recs) == {"CANDIDATE", "CANDIDATE_OPTION", "CANDIDATE_STRATEGY"}
    o, s = recs["CANDIDATE_OPTION"], recs["CANDIDATE_STRATEGY"]
    assert o["outcome"] == 0.5 and o["outcome_net"] == pytest.approx(0.5 - 2 * 0.65 / 200)   # Kommission, kein Flip
    assert s["outcome_net"] == pytest.approx(-0.2 - 4 * 0.65 / 100)                          # 2 Legs × 2 Seiten
    assert o["price_type"] == "OBSERVED_QUOTE" and o["quality"] == "RELIABLE"
    assert o["label_available_at"] == (D0 + timedelta(days=7)).isoformat()                    # Tag nach dem Mark
    assert o["meta"]["stage"] == "EARLY" and o["meta"]["signal_type"] == "catalyst"
    meta_lbl = el.labeled(el.from_candidate_ledger(tmp_path, horizons=(5,)), as_of=D0 + timedelta(days=30),
                          population="CANDIDATE", horizon=5)
    assert {r["population"] for r in meta_lbl} == {"CANDIDATE"}               # Meta-Label sieht nur den Proxy


def test_late_marks_never_count_as_reliable(tmp_path):
    _write(tmp_path, [_row(1, opt=0.1, late=True)])
    o = [r for r in el.from_candidate_ledger(tmp_path, horizons=(5,)) if r["population"] == "CANDIDATE_OPTION"][0]
    assert o["quality"] == "LATE_MARK"
    assert not el.labeled([o], as_of=D0 + timedelta(days=60))                 # nie in der Default-Evidenz


def test_scorecard_need_more_data_and_cells(tmp_path):
    _write(tmp_path, [_row(i, opt=0.1, stage="LATE" if i % 2 else "EARLY", day_off=i % 3) for i in range(10)])
    sc = el.scorecard(el.from_candidate_ledger(tmp_path), as_of=D0 + timedelta(days=60))
    c = sc["cells"]["CANDIDATE_OPTION_5d|ALL"]
    assert c["n"] == 10 and c["n_signal_days"] == 3 and c["verdict"] == "NEED_MORE_DATA" and "ci" not in c
    assert sc["cells"]["CANDIDATE_OPTION_5d|LATE_STAGE"]["n"] == 5
    assert sc["baselines"]["CANDIDATE_OPTION_5d|LATE_STAGE_vs_EARLY"]["verdict"] == "NEED_MORE_DATA"
    assert sc["alpha_per_cell"] == pytest.approx(0.05 / sc["n_cells"], abs=1e-6)  # Bonferroni über alle Zellen


def test_scorecard_detects_clear_effect_with_enough_independent_days(tmp_path):
    rows = []
    for i in range(120):
        late = i % 2 == 0
        rows.append(_row(i, opt=(0.30 if late else -0.30) + (i % 5) * 0.01, stage="LATE" if late else "EARLY",
                         day_off=i % 40))
    _write(tmp_path, rows)
    sc = el.scorecard(el.from_candidate_ledger(tmp_path), as_of=D0 + timedelta(days=90))
    assert sc["cells"]["CANDIDATE_OPTION_5d|LATE_STAGE"]["verdict"] == "POSITIVE"
    b = sc["baselines"]["CANDIDATE_OPTION_5d|LATE_STAGE_vs_EARLY"]
    assert b["verdict"] == "POSITIVE" and b["delta_mean_net"] > 0.5


def test_scorecard_respects_point_in_time(tmp_path):
    _write(tmp_path, [_row(i, opt=0.1, day_off=i) for i in range(5)])
    early = el.scorecard(el.from_candidate_ledger(tmp_path), as_of=D0 + timedelta(days=7))
    assert early["cells"].get("CANDIDATE_OPTION_5d|ALL", {"n": 0})["n"] == 1      # nur bis as_of bekannte Labels


def test_mark_coverage_counts_due_marked_pending(tmp_path):
    rows = [_row(1, opt=0.1), _row(2)]                                          # zweiter: fällig, nicht markiert
    _write(tmp_path, rows)
    cov = el.option_mark_coverage(tmp_path, as_of=D0 + timedelta(days=9))
    assert cov["real_opt_ret_5d"] == {"due": 2, "marked": 1, "pending": 1, "late_marks": 0, "unavailable": 0,
                                      "oldest_pending_days": 4}


def test_mark_coverage_counts_terminal_status_as_unavailable_not_pending(tmp_path):
    """Mark im Endzustand (candidate_ledger._give_up_marks, z. B. Kontrakt ohne Quote) ist nicht mehr offen."""
    gone = _row(2)
    gone["outcomes"]["real_opt_mark_status_5d"] = "NO_QUOTE"
    _write(tmp_path, [_row(1, opt=0.1), gone])
    cov = el.option_mark_coverage(tmp_path, as_of=D0 + timedelta(days=9))
    assert cov["real_opt_ret_5d"] == {"due": 2, "marked": 1, "pending": 0, "late_marks": 0, "unavailable": 1,
                                      "oldest_pending_days": None}


def _closed(tid, *, method="option_quote", outcome=0.3, strategy="LONG_CALL", ask=2.0, evaluation=None):
    t = {"ticker": tid, "entry_date": "2026-09-01", "close_date": "2026-09-10", "strategy": strategy,
         "outcome": outcome, "outcome_method": method, "outcome_reliable": True,
         "option": {"ask": ask, "bid": ask - 0.1, "strike": 100.0, "expiration": "2026-12-18"},
         "deep_analysis": {"direction": "BULLISH"}}
    if evaluation:
        t["outcome_evaluation"] = evaluation
    return t


def test_champion_trade_net_only_for_evaluable_trades():
    ok = {"status": "OK", "reasons": [], "flags": [], "tier": "OBSERVED"}
    bad = {"status": "DATA_BAD", "reasons": ["ENTRY_QUOTE_UNUSABLE"], "flags": [], "tier": "NOT_EVALUABLE"}
    recs = {r["entity"]: r for r in el.from_champion_trades({"closed_trades": [
        _closed("OK1", evaluation=ok),                                             # 1 Leg: 2*0.65/(2.0*100)
        _closed("SPR", evaluation=ok, strategy="BULL_CALL_SPREAD"),                # 2 Legs
        _closed("MOD", method="delta_approx"),                                     # modelliert -> UNUSABLE
        _closed("BAD", evaluation=bad)]})}                                         # Methode ok, Bewertung DATA_BAD
    assert recs["OK1"]["outcome_net"] == pytest.approx(0.3 - 1.3 / 200)
    assert recs["SPR"]["outcome_net"] == pytest.approx(0.3 - 2.6 / 200)
    assert recs["OK1"]["quality"] == "RELIABLE"
    assert recs["MOD"]["outcome_net"] is None and recs["BAD"]["outcome_net"] is None   # nie Netto ohne Bewertung OK
    assert recs["BAD"]["quality"] == "DATA_BAD"                                        # nie RELIABLE
