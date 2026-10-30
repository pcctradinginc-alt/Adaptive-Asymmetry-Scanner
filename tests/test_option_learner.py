import hashlib
import json
import random
from datetime import date, datetime, timedelta, timezone

import pytest

from modules import market_snapshot as ms
from modules import option_learner as ol
from modules.atomic_io import append_jsonl, read_jsonl

H = 5


def _trading_days(start, n):
    out, d = [], start
    while len(out) < n:
        if ms.is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def _next_trading_day(d):
    while not ms.is_trading_day(d):
        d += timedelta(days=1)
    return d


def _quote(symbol, day, bid, ask):
    ts = f"{day}T16:00:00+00:00"
    rid = hashlib.sha256(f"{symbol}{day}{bid}{ask}".encode()).hexdigest()[:20]
    return {"record_id": rid, "occ_symbol": symbol, "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4),
            "quote_time": ts, "retrieved_at": ts, "quality_status": "OK", "quality_flags": []}


def _row(day, i, sentiment, ret=None, quotes=None, ask=10.0, h=H):
    ticker = f"T{i}"
    symbol = f"{ticker}270319C00100000"
    legs = [{"symbol": symbol, "side": "long", "bid": ask - 0.2, "ask": ask, "mid": ask - 0.1}]
    r = {"date": day, "ticker": ticker, "signal_id": f"{day}-{ticker}", "entry_price": 100.0,
         "features": {"sentiment_score": sentiment},
         "real_option": {"symbol": symbol, "strike": 100.0, "dte": 160, "iv": 0.3, "delta": 0.5,
                         "open_interest": 500},
         "real_strategy": {"strategy": "LONG_CALL", "legs": legs, "entry_pending": False, "entry_date": day,
                           "net_debit_entry": ask, "net_mid_entry": ask - 0.1, "expiry": "2027-03-19"},
         "outcomes": {}}
    if ret is not None:
        mark = _next_trading_day(date.fromisoformat(day) + timedelta(days=h)).isoformat()
        bid = round(ask * (1 + ret), 2)
        r["outcomes"] = {f"real_strat_ret_{h}d": round(bid / ask - 1, 4), f"real_strat_mark_date_{h}d": mark}
        quotes += [_quote(symbol, day, ask - 0.2, ask), _quote(symbol, mark, bid, bid + 0.1)]
    return r


def _history(days, signal=True, seed=3, h=H):
    """10 Kandidaten je Tag; mit signal: positives Sentiment -> +30 %, negatives -> -30 % (plus Rauschen)."""
    rng, rows, quotes = random.Random(seed), [], []
    for d in days:
        for i in range(10):
            s = rng.uniform(-1, 1)
            base = (0.3 if s > 0 else -0.3) if signal else rng.choice((0.3, -0.3))
            rows.append(_row(d.isoformat(), i, s, base + rng.uniform(-0.05, 0.05), quotes, h=h))
    return rows, quotes


def test_walk_forward_validates_but_without_forward_evidence_nothing_is_recommended(monkeypatch, tmp_path):
    days = _trading_days(date(2026, 1, 5), 40)
    rows, quotes = _history(days)
    today = _trading_days(days[-1] + timedelta(days=10), 1)[0].isoformat()
    rows += [_row(today, 1, 0.8), _row(today, 2, -0.8)]
    monkeypatch.setattr(ol.red, "load", lambda root, as_of: rows)
    monkeypatch.setattr(ol.red, "load_quotes", lambda root, labelled, as_of: quotes)
    sent = []
    monkeypatch.setattr("modules.mailer.send_mail", lambda subject, html, text: sent.append(subject) or {"status": "sent"})

    res = ol.recommend(date.fromisoformat(today), horizon=H, out=tmp_path)
    assert res["walk_forward"]["verdict"] == "VALIDATED"
    assert res["forward"]["verdict"] == "NEED_MORE_DATA"                 # leeres Journal: nichts prospektiv belegt
    assert res["recommendations"] == [] and res["release"]["allowed"] is False
    assert res["watchlist"][0]["ticker"] == "T1"
    assert ol.mail(res) is None and sent == []
    md = (tmp_path / "recommendations" / f"{today}_{H}d.md").read_text()
    assert "UNVALIDIERT" in md and "Frühindikator" in md


def _stats(verdict, **kw):
    s = {"n": 400, "dates": 25, "independent_periods": 6, "mean_all": 0.01, "ic_mean": 0.2, "ic_t": 3.0,
         "n_selected": 120, "selected_dates": 25, "mean_selected": 0.08, "selected_ci95": [0.02, 0.14],
         "coverage": 1.0, "experiment_id": ol.experiment_id(), "representativeness": {"warnings": []},
         "by_expression": {}, "verdict": verdict}
    s.update(kw)
    return s


def _gated(monkeypatch, tmp_path, fw, horizon=20):
    """recommend mit vorgegebenem Walk-Forward (VALIDATED) und Forward-Bericht; zählt versandte Mails."""
    days = _trading_days(date(2026, 1, 5), 40)
    rows, quotes = _history(days, h=horizon)
    today = _trading_days(days[-1] + timedelta(days=40), 1)[0].isoformat()
    rows += [_row(today, 1, 0.8, h=horizon), _row(today, 2, -0.8, h=horizon)]
    monkeypatch.setattr(ol.red, "load", lambda root, as_of: rows)
    monkeypatch.setattr(ol.red, "load_quotes", lambda root, labelled, as_of: quotes)
    monkeypatch.setattr(ol, "walk_forward", lambda *a, **k: _stats("VALIDATED"))
    monkeypatch.setattr(ol, "forward", lambda *a, **k: fw)
    sent = []
    monkeypatch.setattr("modules.mailer.send_mail", lambda subject, html, text: sent.append(subject) or {"status": "sent"})
    res = ol.recommend(date.fromisoformat(today), horizon=horizon, out=tmp_path)
    ol.mail(res)
    return res, sent


@pytest.mark.parametrize("fw", [
    _stats("NEED_MORE_DATA", dates=4, n_selected=12, independent_periods=1),
    _stats("INCONCLUSIVE", mean_selected=0.03, selected_ci95=[-0.02, 0.09]),
    _stats("FORWARD_NEGATIVE", mean_selected=-0.06, selected_ci95=[-0.11, -0.01]),
    None,                                                                # Forward-Bericht fehlt
    {},
], ids=["need_more_data", "inconclusive", "negative", "missing", "empty"])
def test_validated_walk_forward_without_positive_forward_gives_no_recommendation_and_no_mail(monkeypatch, tmp_path,
                                                                                               fw):
    res, sent = _gated(monkeypatch, tmp_path, fw)
    assert res["recommendations"] == [] and res["release"]["allowed"] is False
    assert res["watchlist"] and sent == []
    assert "UNVALIDIERT" in (tmp_path / "recommendations" / f"{res['date']}_20d.md").read_text()


def test_only_validated_and_forward_positive_releases_positive_predictions(monkeypatch, tmp_path):
    res, sent = _gated(monkeypatch, tmp_path, _stats("FORWARD_POSITIVE"))
    assert res["release"] == {"allowed": True, "reasons": []}
    assert [r["ticker"] for r in res["recommendations"]] == ["T1"]       # nur positiver Netto-Erwartungswert
    assert all(r["pred_net"] > 0 for r in res["recommendations"]) and res["watchlist"] == []
    assert len(sent) == 1


def test_five_day_model_is_an_early_indicator_without_trade_mail(monkeypatch, tmp_path):
    res, sent = _gated(monkeypatch, tmp_path, _stats("FORWARD_POSITIVE"), horizon=5)
    assert res["recommendations"] and res["mail_eligible"] is False
    assert sent == []


def test_release_gate_fails_closed_on_contradictory_or_incomplete_reports():
    ok = _stats("FORWARD_POSITIVE")
    assert ol.release_gate(_stats("VALIDATED"), ok)["allowed"] is True
    contradictory = [
        (_stats("VALIDATED", dates=5), ok),                              # Verdikt passt nicht zu den Zahlen
        (_stats("VALIDATED", ic_t=1.2), ok),
        (_stats("VALIDATED"), _stats("FORWARD_POSITIVE", independent_periods=2)),
        (_stats("VALIDATED"), _stats("FORWARD_POSITIVE", selected_ci95=None)),
        (_stats("VALIDATED"), _stats("FORWARD_POSITIVE", mean_selected=-0.01)),
        (_stats("VALIDATED"), _stats("FORWARD_POSITIVE", coverage=0.6)),              # fehlende Ergebnisse
        (_stats("VALIDATED", coverage=None), ok),
        (_stats("VALIDATED"), _stats("FORWARD_POSITIVE", experiment_id="alt")),      # andere Experimentversion
        (_stats("VALIDATED"), {"verdict": "FORWARD_POSITIVE"}),          # Kennzahlen fehlen
        (None, ok),
    ]
    for wf, fw in contradictory:
        gate = ol.release_gate(wf, fw)
        assert gate["allowed"] is False and gate["reasons"]


def test_mail_rechecks_the_gate_even_for_a_forged_result():
    forged = {"date": "2026-11-02", "horizon": 20, "walk_forward": _stats("VALIDATED"),
              "forward": _stats("INCONCLUSIVE", selected_ci95=[-0.01, 0.1]),
              "recommendations": [{"ticker": "X", "pred_net": 0.2}], "watchlist": []}
    assert ol.mail(forged) is None
    assert ol.mail({**forged, "forward": None}) is None


def test_no_relation_means_no_recommendation_only_watchlist(monkeypatch, tmp_path):
    days = _trading_days(date(2026, 1, 5), 40)
    rows, quotes = _history(days, signal=False)
    today = _trading_days(days[-1] + timedelta(days=10), 1)[0].isoformat()
    rows += [_row(today, 1, 0.8), _row(today, 2, -0.8)]
    monkeypatch.setattr(ol.red, "load", lambda root, as_of: rows)
    monkeypatch.setattr(ol.red, "load_quotes", lambda root, labelled, as_of: quotes)
    res = ol.recommend(date.fromisoformat(today), horizon=H, out=tmp_path)
    assert res["walk_forward"]["verdict"] != "VALIDATED"
    assert res["recommendations"] == []
    assert len(res["watchlist"]) == 2
    assert ol.mail(res) is None                                          # keine tägliche Nullmeldung


def test_labels_without_archived_quotes_are_not_learned():
    days = _trading_days(date(2026, 1, 5), 12)
    rows, quotes = _history(days)
    assert len(ol.dataset(rows, quotes, date(2026, 12, 31), H)) == 120
    assert ol.dataset(rows, [], date(2026, 12, 31), H) == []


def test_walk_forward_trains_only_on_labels_known_before_the_test_day(monkeypatch):
    days = _trading_days(date(2026, 1, 5), 30)
    rows, quotes = _history(days)
    cands = ol.candidates(rows, quotes, AS_OF, H)
    real_fit, seen = ol.fit, []

    def spy(train):
        seen.append(max(d["label_day"] for d in train))
        return real_fit(train)
    monkeypatch.setattr(ol, "fit", spy)
    tested = sorted({c["date"] for c in cands})
    ol.walk_forward(cands, AS_OF, H)
    first_test = len(tested) - len(seen)
    for test_day, newest_label in zip(tested[first_test:], seen):
        assert newest_label < date.fromisoformat(test_day)


def test_journal_is_append_only(monkeypatch, tmp_path):
    days = _trading_days(date(2026, 1, 5), 15)
    rows, quotes = _history(days)
    today = _trading_days(days[-1] + timedelta(days=10), 1)[0].isoformat()
    rows += [_row(today, 1, 0.8)]
    monkeypatch.setattr(ol.red, "load", lambda root, as_of: rows)
    monkeypatch.setattr(ol.red, "load_quotes", lambda root, labelled, as_of: quotes)
    now = datetime(2026, 3, 1, 19, tzinfo=timezone.utc)
    assert ol.recommend(date.fromisoformat(today), horizon=H, out=tmp_path, now=now)["journaled"] == 3
    assert ol.recommend(date.fromisoformat(today), horizon=H, out=tmp_path, now=now)["journaled"] == 0
    assert [(j["kind"], j.get("role")) for j in read_jsonl(tmp_path / ol.JOURNAL)] == [
        ("prediction", None), ("decision", "reference"), ("decision", "learning")]


def test_contract_features_use_decision_time_quotes_not_the_fill():
    r = _row("2026-03-02", 1, 0.1)
    r["real_strategy"]["snapshot_legs"] = [{"symbol": r["real_strategy"]["legs"][0]["symbol"],
                                            "bid": 9.0, "ask": 11.0, "mid": 10.0}]
    f = ol.contract_features(r)
    assert abs(f["c_entry_spread"] - 0.1) < 1e-9                         # 11 / 10 - 1 aus dem Snapshot
    assert f["c_is_call"] == 1.0 and f["c_is_spread"] == 0.0


def _spread_row(day, i, sentiment):
    r = _row(day, i, sentiment)
    long_sym, short_sym = f"T{i}270319C00100000", f"T{i}270319C00110000"
    r["real_option"].update({"bid": 9.8, "ask": 10.0, "mid": 9.9, "entry_date": day, "entry_pending": False})
    r["real_strategy"].update({"strategy": "BULL_CALL_SPREAD", "net_debit_entry": 6.0, "net_mid_entry": 5.8,
                               "legs": [{"symbol": long_sym, "side": "long", "bid": 9.8, "ask": 10.0, "mid": 9.9},
                                        {"symbol": short_sym, "side": "short", "bid": 4.0, "ask": 4.2, "mid": 4.1}]})
    return r


def test_expressions_compare_strategy_and_single_without_duplicates():
    spread = _spread_row("2026-03-02", 1, 0.1)
    spread["outcomes"] = {"real_opt_ret_5d": 0.1, "real_strat_ret_5d": -0.2}
    exprs = {e["expression"]: e for e in ol.expressions(spread)}
    assert set(exprs) == {"spread", "single"}
    assert exprs["single"]["row"]["outcomes"] == {"real_strat_ret_5d": 0.1}       # Einzel-Option-Label übernommen
    assert exprs["spread"]["row"]["outcomes"] == {"real_strat_ret_5d": -0.2}
    same = _row("2026-03-02", 2, 0.1)                                              # Strategie = Einzel-Option
    same["real_option"]["symbol"] = same["real_strategy"]["legs"][0]["symbol"]
    assert [e["expression"] for e in ol.expressions(same)] == ["single"]


def test_evaluate_writes_manifest_history_learning_curve_and_report_card(monkeypatch, tmp_path):
    days = _trading_days(date(2026, 1, 5), 40)
    rows, quotes = _history(days)
    monkeypatch.setattr(ol.red, "load", lambda root, as_of: rows)
    monkeypatch.setattr(ol.red, "load_quotes", lambda root, labelled, as_of: quotes)
    early = days[25] + timedelta(days=1)
    first = ol.evaluate(early, horizon=H, out=tmp_path)
    assert first["manifest"]["all_labels_verified"] and first["manifest"]["trains_model"]
    assert first["feature_coverage"]["sentiment_score"] == 1.0 and first["feature_coverage"]["news_novelty"] == 0.0
    assert first["vs_previous"] is None
    ol.evaluate(early, horizon=H, out=tmp_path)                                    # gleicher Stand: kein neuer Eintrag
    later = ol.evaluate(date(2026, 12, 31), horizon=H, out=tmp_path)
    history = read_jsonl(tmp_path / f"status_history_{H}d.jsonl")
    assert [h["as_of"] for h in history] == [early.isoformat(), "2026-12-31"]
    assert later["vs_previous"]["previous_as_of"] == early.isoformat()
    assert later["vs_previous"]["delta_labels"] > 0
    assert later["manifest"]["labels_sha256"] != first["manifest"]["labels_sha256"]
    assert later["baselines"]["beats_trade_all"] is True
    card = ol.report_card(tmp_path, horizons=(H,))
    assert len(card["rows"]) == len(ol.CARD_QUESTIONS)
    answers = {r["key"]: r[f"{H}d"][0] for r in card["rows"]}
    assert answers["learns"] == "Ja" and answers["validated"] == "Ja" and answers["self_limit"] == "Ja"
    assert answers["proven"] == "Nein"                                             # ohne Forward-Bestätigung nie
    assert (tmp_path / "report_card.md").exists()


def test_recommend_journals_every_expression_and_recommends_one_per_candidate(monkeypatch, tmp_path):
    days = _trading_days(date(2026, 1, 5), 40)
    rows, quotes = _history(days)
    today = _trading_days(days[-1] + timedelta(days=10), 1)[0].isoformat()
    rows += [_spread_row(today, 1, 0.9)]
    monkeypatch.setattr(ol.red, "load", lambda root, as_of: rows)
    monkeypatch.setattr(ol.red, "load_quotes", lambda root, labelled, as_of: quotes)
    monkeypatch.setattr(ol, "forward", lambda *a, **k: _stats("FORWARD_POSITIVE"))
    res = ol.recommend(date.fromisoformat(today), horizon=H, out=tmp_path)
    assert res["journaled"] == 4 and res["decisions"] == 1          # 2 Formen + Entscheidung + Referenz-Entscheidung
    assert len(res["recommendations"]) == 1
    assert res["recommendations"][0]["expression"] in ("spread", "single")


def test_monday_report_shows_report_card():
    from reports import weekly
    assert "KEINE DATEN" in weekly.option_learner_blocks({})[0][1]
    card = {"as_of": "2026-11-07", "horizons": [20], "rows": [
        {"key": "learns", "question": "Lernt das System?", "20d": ["Ja", "359 Labels"]}]}
    blocks = weekly.option_learner_blocks({"option_learner_card": card})
    assert blocks[1][0] == "table" and blocks[1][2] == [["Lernt das System?", "Ja – 359 Labels"]]


def test_contract_variant_is_its_own_expression():
    r = _row("2026-03-02", 1, 0.1)
    r["real_option_alt"] = {"symbol": "T1261218C00100000", "bid": 4.9, "ask": 5.0, "mid": 4.95,
                            "entry_date": "2026-03-02", "entry_pending": False, "dte": 60}
    r["outcomes"] = {"real_alt_ret_5d": 0.05, "real_strat_ret_5d": -0.1}
    exprs = {e["expression"]: e for e in ol.expressions(r)}
    assert set(exprs) == {"single", "single_alt"}
    assert exprs["single_alt"]["row"]["outcomes"] == {"real_strat_ret_5d": 0.05}
    assert ol.contract_features(exprs["single_alt"]["row"])["c_dte"] == 60


def test_gate_check_reports_outcome_per_reject_reason():
    data = [{"date": f"2026-03-{d:02d}", "gate": "options_design_roi_or_edge", "y": 0.1} for d in range(1, 13)
            for _ in range(2)] + [{"date": "2026-03-02", "gate": "earnings_gate", "y": -0.2}]
    g = ol.by_gate(data)
    assert g["options_design_roi_or_edge"]["sufficient"] is True and g["options_design_roi_or_edge"]["mean"] == 0.1
    assert g["earnings_gate"]["sufficient"] is False


def test_volatility_pricing_features_from_decision_time():
    r = _row("2026-03-02", 1, 0.1)
    r["real_strategy"]["iv_rank"] = 30.0
    r["real_strategy"]["iv_rank_components"] = {"rv_score": 25.0, "term_score": 40.0}
    f = ol.contract_features(ol.expressions(r)[0]["row"])
    assert (f["c_iv_rank"], f["c_rv_score"], f["c_term_score"]) == (30.0, 25.0, 40.0)


AS_OF = date(2026, 12, 31)


def _fw_case(n_dates, exp, start=date(2026, 3, 2), per_day_models=False, y=0.2, seed=5):
    """Forward-Fall: je Tag 4 Kandidaten mit geprüftem Label (Netto ~ y) und am Signaltag festgehaltener Prognose
    und TRADE-Entscheidung der Experimentversion `exp` aus registrierten, vorher trainierten Modellinstanzen."""
    rng, cands, journal, instances = random.Random(seed), [], [], []
    for k, d in enumerate(_trading_days(start, n_dates)):
        mid = f"{exp}-m{k if per_day_models else 0}"
        if not any(i["model_id"] == mid for i in instances):
            instances.append({"model_id": mid, "experiment_id": exp, "train_cutoff": d.isoformat(),
                              "max_label_day": (d - timedelta(days=1)).isoformat()})
        for i in range(4):
            sid, contract = f"{d}-{exp}-{i}", [f"T{i}270319C00100000"]
            cands.append({"date": d.isoformat(), "signal_id": sid, "ticker": f"T{i}", "sample": "directional",
                          "gate": "x", "exprs": [
                              {"expression": "single", "contract": contract, "strategy": "LONG_CALL", "x": {},
                               "ask": 1.0, "mid": 0.95, "status": "VERIFIED", "y": y + rng.uniform(-0.05, 0.05),
                               "label_day": d + timedelta(days=H + 1)}]})
            entry = {"date": d.isoformat(), "signal_id": sid, "contract": contract, "horizon": H, "pred_net": 0.1,
                     "experiment_id": exp, "model_id": mid, "created_at": f"{d}T16:30:00+00:00"}
            journal += [{**entry, "kind": "prediction"},
                        {**entry, "kind": "decision", "role": "learning", "action": "TRADE"}]
    return cands, journal, instances


def test_experiment_version_changes_with_features_label_costs_training_or_selection(monkeypatch):
    base = ol.experiment_id()
    assert base == ol.experiment_id()                                             # deterministisch
    for name, value in (("LEDGER_FEATURES", ol.LEDGER_FEATURES[:-1]), ("COMMISSION_PER_LEG_SIDE", 1.0),
                        ("RIDGE_ALPHA", 1.0), ("WINSOR", (0.05, 0.95)), ("ALLOWED_STRATEGIES", ("LONG_CALL",)),
                        ("SOURCES", ol.SOURCES[:2])):
        with monkeypatch.context() as m:
            m.setattr(ol, name, value)
            assert ol.experiment_id() != base, name
    with monkeypatch.context() as m:                                              # Evidenzschwellen: gleiche Version
        m.setattr(ol, "MIN_OOS_DATES", 40)
        assert ol.experiment_id() == base


def test_mixed_experiment_versions_never_form_a_joint_forward_positive():
    old_cands, old_journal, old_inst = _fw_case(25, "OLD")
    new_cands, new_journal, new_inst = _fw_case(5, "NEW", start=date(2026, 4, 20))
    cands, journal, inst = old_cands + new_cands, old_journal + new_journal, old_inst + new_inst
    assert ol.forward(old_journal, old_cands, H, old_inst, AS_OF, exp="OLD")["verdict"] == "FORWARD_POSITIVE"
    fw = ol.forward(journal, cands, H, inst, AS_OF, exp="NEW")
    assert fw["verdict"] == "NEED_MORE_DATA" and fw["valid_decisions"] == 20 and fw["valid_predictions"] == 20
    assert fw["excluded"] == {"OTHER_EXPERIMENT": 200}                            # gekennzeichnet, nicht gemischt
    assert fw["journal_by_experiment"] == {"NEW": 40, "OLD": 200}
    legacy = [{k: v for k, v in j.items() if k not in ("experiment_id", "model_id")} for j in old_journal]
    assert ol.forward(legacy + new_journal, cands, H, inst, AS_OF, exp="NEW")["excluded"] == {"OTHER_EXPERIMENT": 200}


def test_retraining_within_one_experiment_forms_one_prospective_learning_curve():
    cands, journal, inst = _fw_case(25, "EXP", per_day_models=True)
    fw = ol.forward(journal, cands, H, inst, AS_OF, exp="EXP")
    assert fw["verdict"] == "FORWARD_POSITIVE" and fw["model_instances"] == 25 and fw["excluded"] == {}
    # gespeicherte Entscheidungen zählen so, wie sie damals festgehalten wurden – nie neu berechnet
    assert fw["n_selected"] == 100
    flipped = [dict(j, action="ABSTAIN") if j["kind"] == "decision" else j for j in journal]
    assert ol.forward(flipped, cands, H, inst, AS_OF, exp="EXP")["n_selected"] == 0


def test_late_unregistered_or_future_trained_predictions_never_count():
    cands, journal, inst = _fw_case(25, "EXP")
    j0 = journal[1]                                                               # Entscheidung des 1. Kandidaten
    bad = [dict(j0, signal_id="late", created_at="2026-03-03T09:00:00+00:00", date="2026-03-02"),
           dict(j0, created_at="2026-03-02T23:30:00-05:00"),                     # UTC bereits der Folgetag
           dict(j0, model_id="nie-registriert"),
           dict(j0, created_at=None),
           dict(j0, pred_net=-5.0)]                                             # zweite Fassung: erste gilt
    inst = inst + [{"model_id": "future", "experiment_id": "EXP", "train_cutoff": "2026-03-02",
                    "max_label_day": "2026-03-02"}]
    bad.append(dict(journal[3], model_id="future"))
    fw = ol.forward(journal + bad, cands, H, inst, AS_OF, exp="EXP")
    assert fw["excluded"] == {"DUPLICATE": 1, "LATE_PREDICTION": 2, "NO_TIMESTAMP": 1,
                              "TRAINED_ON_FUTURE": 1, "UNKNOWN_MODEL_INSTANCE": 1}
    assert fw["valid_decisions"] == 100 and fw["verdict"] == "FORWARD_POSITIVE"


def test_recommend_journals_experiment_and_registered_model_instance(monkeypatch, tmp_path):
    days = _trading_days(date(2026, 1, 5), 15)
    rows, quotes = _history(days)
    today = _trading_days(days[-1] + timedelta(days=10), 1)[0]
    rows += [_row(today.isoformat(), 1, 0.8)]
    monkeypatch.setattr(ol.red, "load", lambda root, as_of: rows)
    monkeypatch.setattr(ol.red, "load_quotes", lambda root, labelled, as_of: quotes)
    now = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc).replace(hour=17)
    res = ol.recommend(today, horizon=H, out=tmp_path, now=now)
    ol.recommend(today, horizon=H, out=tmp_path, now=now)
    journal, registry = read_jsonl(tmp_path / ol.JOURNAL), read_jsonl(tmp_path / ol.INSTANCES)
    assert len(registry) == 1 and len(journal) == 3                               # append-only, keine Dubletten
    inst, entry = registry[0], journal[0]
    assert entry["experiment_id"] == inst["experiment_id"] == ol.experiment_id() == res["experiment_id"]
    assert entry["model_id"] == inst["model_id"] == res["model_id"]
    assert inst["max_label_day"] < today.isoformat() and inst["n_train_labels"] > 0 and inst["labels_sha256"]
    assert ol._invalid(entry, ol.experiment_id(), {inst["model_id"]: inst}) is None


class _PrefersSpread:
    """Modell-Attrappe: sagt für den Spread stets mehr voraus als für die Einzel-Option."""

    def predict(self, xs):
        return [0.1 if x.get("c_is_spread") else 0.05 for x in xs]


class _PrefersSingle:
    def predict(self, xs):
        return [0.05 if x.get("c_is_spread") else 0.1 for x in xs]


def _two_form_cands(days, rule, h=H, seed=11, n=8, sample="research_census", status=None):
    """Je Tag n Kandidaten mit zwei Formen: Spread (Standardform, zuerst) und Einzel-Option.
    rule(sentiment, rng) -> (y_spread, y_single); status(i) -> Prüfstatus der Einzel-Option."""
    rng, out = random.Random(seed), []
    for d in days:
        for i in range(n):
            s = rng.uniform(-1, 1)
            y_spread, y_single = rule(s, rng)
            single_status = (status or (lambda i: "VERIFIED"))(i)
            ld = d + timedelta(days=h + 1)
            out.append({"date": d.isoformat(), "signal_id": f"{d}-{i}", "ticker": f"T{i}", "sample": sample,
                        "gate": "x", "exprs": [
                            {"expression": "spread", "contract": [f"T{i}C1", f"T{i}C2"],
                             "strategy": "BULL_CALL_SPREAD", "x": {"sentiment_score": s, "c_is_spread": 1.0},
                             "ask": 1.0, "mid": 0.9, "status": "VERIFIED", "y": y_spread, "label_day": ld},
                            {"expression": "single", "contract": [f"T{i}C1"], "strategy": "LONG_CALL",
                             "x": {"sentiment_score": s, "c_is_spread": 0.0}, "ask": 2.0, "mid": 1.9,
                             "status": single_status, "y": y_single if single_status == "VERIFIED" else None,
                             "label_day": ld if single_status == "VERIFIED" else None}]})
    return out


def _noise(rng):
    return rng.uniform(-0.05, 0.05)


def test_the_hindsight_best_variant_is_never_selected_and_cannot_validate_the_choice(monkeypatch):
    days = _trading_days(date(2026, 1, 5), 40)
    cands = _two_form_cands(days, lambda s, rng: (-0.3 + _noise(rng), 0.2 + _noise(rng)))
    monkeypatch.setattr(ol, "fit", lambda train: _PrefersSpread())
    wf = ol.walk_forward(cands, AS_OF, H)
    assert set(wf["chosen_forms"]) == {"spread"}                     # Einzel-Option war im Nachhinein besser
    assert wf["mean_selected"] < -0.2
    assert wf["by_expression"]["single"]["mean_all"] > 0.15          # profitable Einzelform ...
    assert wf["verdict"] == "NOT_VALIDATED"                          # ... validiert nicht die Auswahlregel
    assert ol.release_gate(wf, _stats("FORWARD_POSITIVE"))["allowed"] is False
    assert wf["vs_default"]["mean_diff"] == 0                        # gewählt = Standardform (Spread)


def test_the_decision_uses_predictions_only_never_outcomes():
    cand = _two_form_cands([date(2026, 3, 2)], lambda s, rng: (0.5, -0.5), n=1)[0]
    swapped = {**cand, "exprs": [dict(e, y=-(e["y"] or 0)) for e in cand["exprs"]]}
    assert ol.decide(cand, [0.1, 0.2]) == ol.decide(swapped, [0.1, 0.2])
    assert ol.decide(cand, [0.1, 0.2])["expression"] == "single"
    assert ol.decide(cand, [-0.1, -0.2])["action"] == "ABSTAIN"
    puts = {**cand, "exprs": [dict(e, strategy="LONG_PUT") for e in cand["exprs"]]}
    assert ol.decide(puts, [0.3, 0.3]) == {"action": "ABSTAIN", "reason": "NO_ALLOWED_EXPRESSION",
                                           "expression": None, "contract": None, "pred": None}


def test_a_learned_choice_is_validated_and_compared_with_the_default_form():
    days = _trading_days(date(2026, 1, 5), 40)
    cands = _two_form_cands(days, lambda s, rng: ((0.1 if s > 0 else -0.1) - 0.05 + _noise(rng),
                                                  (0.3 if s > 0 else -0.3) + 0.05 + _noise(rng)))
    wf = ol.walk_forward(cands, AS_OF, H)
    assert wf["verdict"] == "VALIDATED" and wf["coverage"] == 1.0
    assert set(wf["chosen_forms"]) == {"single"}
    assert wf["abstained"] > 0                                       # negative Prognosen: kein Handel
    assert wf["vs_default"]["mean_diff"] > 0.2 and wf["vs_default"]["ci95"][0] > 0
    rep = wf["representativeness"]
    assert set(rep["by_sample"]) == {"research_census"}
    assert any("Forschungsstichprobe" in w for w in rep["warnings"])
    assert any("bullische" in w for w in rep["warnings"]) and any("Marktphase" in w for w in rep["warnings"])


def test_missing_quotes_of_the_chosen_variant_never_count_as_profit(monkeypatch):
    days = _trading_days(date(2026, 1, 5), 40)
    cands = _two_form_cands(days, lambda s, rng: (0.4 + _noise(rng), 0.2 + _noise(rng)),
                            status=lambda i: "EXIT_NOT_ARCHIVED" if i < 2 else "VERIFIED")
    monkeypatch.setattr(ol, "fit", lambda train: _PrefersSingle())
    wf = ol.walk_forward(cands, AS_OF, H)
    assert set(wf["chosen_forms"]) == {"single"}                     # der Spread ersetzt die Lücke nicht
    assert wf["unevaluable"] == {"EXIT_NOT_ARCHIVED": wf["trades"] // 4}
    assert wf["coverage"] == 0.75 and wf["verdict"] == "INSUFFICIENT_COVERAGE"
    assert wf["mean_selected_unevaluable_as_loss"] < 0 < wf["mean_selected"]


def test_one_candidate_per_ticker_and_day_and_only_due_outcomes():
    days = _trading_days(date(2026, 1, 5), 40)
    cands = _two_form_cands(days, lambda s, rng: (0.1, 0.2))
    twin = [dict(c, signal_id=c["signal_id"] + "-b") for c in cands]   # zweites Signal derselben Aktie
    model = _PrefersSingle()
    recs = [{"cand": c, "dec": ol.decide(c, model.predict([e["x"] for e in c["exprs"]])), "preds": {}}
            for c in cands + twin]
    assert ol.assess(recs, H)["n_selected"] == len(cands)
    last = date.fromisoformat(cands[-1]["date"])
    assert not ol._due(cands[-1]["date"], last + timedelta(days=H + ol.DUE_GRACE_DAYS), H)
    assert ol._due(cands[-1]["date"], last + timedelta(days=H + ol.DUE_GRACE_DAYS + 1), H)
    from modules import candidate_ledger
    assert ol.DUE_GRACE_DAYS >= candidate_ledger.LATE_MARK_DAYS


def test_forward_evaluates_the_journaled_choice_not_a_later_better_one():
    days = _trading_days(date(2026, 3, 2), 25)
    cands = _two_form_cands(days, lambda s, rng: (-0.2 + _noise(rng), 0.3 + _noise(rng)))
    inst = [{"model_id": "m", "experiment_id": "E", "train_cutoff": "2026-03-02", "max_label_day": "2026-03-01"}]
    journal = [{"kind": "decision", "role": "learning", "date": c["date"], "signal_id": c["signal_id"],
                "horizon": H, "action": "TRADE", "contract": c["exprs"][0]["contract"], "pred_net": 0.1,
                "experiment_id": "E", "model_id": "m", "created_at": f"{c['date']}T16:00:00+00:00"} for c in cands]
    fw = ol.forward(journal, cands, H, inst, AS_OF, exp="E")
    assert set(fw["chosen_forms"]) == {"spread"} and fw["verdict"] == "FORWARD_NEGATIVE"
    early = ol.forward(journal, cands, H, inst, date(2026, 3, 20), exp="E")
    assert early["pending_decisions"] > 0 and early["n"] < fw["n"]   # noch nicht fällig: nicht bewertet


def test_candidates_carry_label_status_of_every_form():
    r = _spread_row("2026-03-02", 1, 0.1)
    r["outcomes"] = {"real_strat_ret_5d": 0.1, "real_strat_mark_date_5d": "2026-03-09"}
    c = ol.candidates([r], [], AS_OF, H)[0]
    assert [(e["expression"], e["status"]) for e in c["exprs"]] == [("spread", "ENTRY_NOT_ARCHIVED"),
                                                                    ("single", "NO_LABEL")]
    assert c["sample"] == "directional" and c["exprs"][0]["ask"] == 6.0      # 10.0 Ask - 4.0 Bid


def _pairs(days, learn_good=True, n=6, seed=3, same=False):
    """Gepaarte Entscheidungen: die lernende Version wählt die Einzel-Option (+20 %), die Referenz den Spread
    (−10 %) – oder umgekehrt; same=True: beide wählen gleich."""
    cands = _two_form_cands(days, lambda s, rng: (-0.1 + _noise(rng), 0.2 + _noise(rng)), n=n, seed=seed)
    single = {"action": "TRADE", "contract": None}
    out = []
    for c in cands:
        good = dict(single, contract=c["exprs"][1]["contract"])
        bad = dict(single, contract=c["exprs"][0]["contract"])
        learn, ref = (good, bad) if learn_good else (bad, good)
        out.append({"cand": c, "learning": learn, "reference": learn if same else ref})
    return out


def test_paired_comparison_has_predefined_verdicts():
    days = _trading_days(date(2026, 3, 2), 30)
    better = ol.compare_policies(_pairs(days), H)
    assert better["verdict"] == "LEARNING_BETTER" and better["decisions_differ"] == better["candidates"] == 180
    assert better["learning_minus_reference"]["mean"] > 0.25 and better["learning_minus_reference"]["ci95"][0] > 0
    assert ol.compare_policies(_pairs(days, learn_good=False), H)["verdict"] == "LEARNING_WORSE"
    same = ol.compare_policies(_pairs(days, same=True), H)
    assert same["verdict"] == "NO_SIGNIFICANT_DIFFERENCE" and same["learning_minus_reference"]["mean"] == 0
    few = ol.compare_policies(_pairs(days[:6]), H)                     # deutlich besser, aber zu wenig Tage
    assert few["verdict"] == "NEED_MORE_DATA"


def test_paired_comparison_uses_same_budget_baselines_and_no_artificial_certainty():
    days = _trading_days(date(2026, 3, 2), 30)
    pairs = _pairs(days)
    base = ol.compare_policies(pairs, H)
    # "alles handeln" = Standardform (Spread) je Kandidat; "nie handeln" = 0; gleiche Kandidaten und Tage
    assert base["daily_mean"]["no_trade"] == 0.0 and base["daily_mean"]["trade_all"] < 0
    assert base["daily_mean"]["trade_all"] == base["daily_mean"]["reference"]
    assert ol._baselines(base)["beats_no_trade"] and ol._baselines(base)["beats_trade_all"]
    # mehrere Kontrakte/Signale derselben Aktie am selben Tag erhöhen die Sicherheit nicht
    twins = [dict(p, cand=dict(p["cand"], signal_id=p["cand"]["signal_id"] + "-b")) for p in pairs]
    assert ol.compare_policies(pairs + twins, H) == base
    # ABSTAIN zählt als 0 bei gleichem Budget; ein ungeprüftes Ergebnis schließt den Kandidaten für alle aus
    c = pairs[0]["cand"]
    hole = dict(c, exprs=[c["exprs"][0], dict(c["exprs"][1], status="EXIT_NOT_ARCHIVED", y=None)])
    res = ol.compare_policies([dict(pairs[0], cand=hole)], H)
    assert res["candidates"] == 0 and res["excluded"] == {"EXIT_NOT_ARCHIVED": 1}
    abstain = ol.compare_policies([dict(pairs[0], learning={"action": "ABSTAIN"})], H)
    assert abstain["daily_mean"]["learning"] == 0.0


def _ledger_env(monkeypatch, n_days):
    days = _trading_days(date(2026, 1, 5), n_days)
    rows, quotes = _history(days)
    monkeypatch.setattr(ol.red, "load", lambda root, as_of: rows)
    monkeypatch.setattr(ol.red, "load_quotes", lambda root, labelled, as_of: quotes)
    return days, rows


def test_reference_is_frozen_while_the_learning_version_retrains(monkeypatch, tmp_path):
    days, rows = _ledger_env(monkeypatch, 40)
    day1, day2 = days[20], days[39]
    rows += [_row(day1.isoformat(), 1, 0.8), _row(day2.isoformat(), 1, 0.8)]
    def at(d):
        return datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).replace(hour=17)
    first = ol.recommend(day1, horizon=H, out=tmp_path, now=at(day1))
    path = tmp_path / ol.REFERENCE_DIR / f"{H}d_{ol.experiment_id()}.json"
    frozen = path.read_text()
    second = ol.recommend(day2, horizon=H, out=tmp_path, now=at(day2))
    assert path.read_text() == frozen                                   # nie überschrieben
    assert first["reference_model_id"] == second["reference_model_id"] == first["model_id"]
    assert second["model_id"] != first["model_id"]                     # lernende Version hat nachtrainiert
    ref = [j for j in read_jsonl(tmp_path / ol.JOURNAL) if j.get("role") == "reference"]
    assert {j["date"] for j in ref} == {day1.isoformat(), day2.isoformat()}
    assert {j["model_id"] for j in ref} == {first["model_id"]}
    model, meta = ol.reference(tmp_path, H)                             # gespeicherte Gewichte = damaliges Modell
    train = [d for d in ol.dataset(rows, ol.red.load_quotes(None, rows, day1), day1, H) if d["label_day"] < day1]
    assert ol.instance(model, train, day1)["model_id"] == meta["model_id"]


def test_new_experiment_version_starts_a_new_reference_cohort(monkeypatch, tmp_path):
    days, rows = _ledger_env(monkeypatch, 30)
    day = days[25]
    rows += [_row(day.isoformat(), 1, 0.8)]
    ol.recommend(day, horizon=H, out=tmp_path)
    old = sorted(p.name for p in (tmp_path / ol.REFERENCE_DIR).iterdir())
    monkeypatch.setattr(ol, "COMMISSION_PER_LEG_SIDE", 1.0)            # Kostenmodell geändert
    ol.recommend(day, horizon=H, out=tmp_path)
    new = sorted(p.name for p in (tmp_path / ol.REFERENCE_DIR).iterdir())
    assert len(old) == 1 and len(new) == 2 and old[0] in new


def test_forward_learning_progress_needs_journaled_pairs_and_reports_need_more_data():
    days = _trading_days(date(2026, 3, 2), 25)
    cands = _two_form_cands(days, lambda s, rng: (-0.1 + _noise(rng), 0.2 + _noise(rng)), n=4)
    inst = [{"model_id": m, "experiment_id": "E", "train_cutoff": "2026-03-02", "max_label_day": "2026-03-01"}
            for m in ("learn", "ref")]
    journal = []
    for c in cands:
        e = {"date": c["date"], "signal_id": c["signal_id"], "horizon": H, "action": "TRADE", "pred_net": 0.1,
             "experiment_id": "E", "kind": "decision", "created_at": f"{c['date']}T16:00:00+00:00"}
        journal += [dict(e, role="learning", model_id="learn", contract=c["exprs"][1]["contract"]),
                    dict(e, role="reference", model_id="ref", contract=c["exprs"][0]["contract"])]
    fw = ol.forward(journal, cands, H, inst, AS_OF, exp="E")
    assert fw["vs_reference"]["verdict"] == "LEARNING_BETTER" and fw["vs_reference"]["reference_models"] == ["ref"]
    assert fw["valid_decisions"] == len(cands)                          # Referenz zählt nicht als Lern-Evidenz
    only_learning = [j for j in journal if j["role"] == "learning"]
    assert ol.forward(only_learning, cands, H, inst, AS_OF, exp="E")["vs_reference"]["verdict"] == "NEED_MORE_DATA"
    early = ol.forward(journal, cands, H, inst, date(2026, 3, 25), exp="E")
    assert early["vs_reference"]["verdict"] == "NEED_MORE_DATA"         # zu wenig fällige Tage: nicht "verbessert"


def test_report_card_and_monday_report_show_learning_progress(monkeypatch, tmp_path):
    _ledger_env(monkeypatch, 40)
    ol.evaluate(date(2026, 12, 31), horizon=H, out=tmp_path)
    card = ol.report_card(tmp_path, horizons=(H,))
    row = next(r for r in card["rows"] if r["key"] == "improves")
    assert row[f"{H}d"][0] == "offen" and "NEED_MORE_DATA" in row[f"{H}d"][1] and "retrospektiv" in row[f"{H}d"][1]
    from reports import weekly
    table = weekly.option_learner_blocks({"option_learner_card": card})[1][2]
    assert any("eingefrorene Referenz" in r[0] for r in table)


def test_only_approved_single_changes_are_valid_variants():
    cfg = ol.variant_config()
    assert cfg["version"] == "olv-1"
    assert ol.validate_overrides({"ridge_alpha": 30}, cfg) == {"ridge_alpha": 30}
    assert ol.validate_overrides({"drop_feature_group": "ml_scores"}, cfg)
    for bad in ({}, {"ridge_alpha": 30.0, "min_pred": 0.02}, {"ridge_alpha": 31.0}, {"learning_rate": 0.1},
                {"drop_feature_group": "everything"}, {"min_pred": -1.0}, "ridge_alpha=30"):
        with pytest.raises(ValueError):
            ol.validate_overrides(bad, cfg)
    with pytest.raises(ValueError):
        ol.validate_overrides({"ridge_alpha": 30.0}, {})                  # ohne Konfiguration nichts freigegeben


def _auto(change, h=H, pid=None):
    return {"proposal_id": pid or f"P-{change}", "horizon": h, "change": change, "auto_experiment": True,
            "gap_ids": ["model:untested_alternatives"], "falsification": {"max_days": 90}}


def test_base_version_is_unchanged_and_each_variant_is_its_own_version():
    assert ol.experiment_id() == ol.experiment_id(None) == ol.experiment_id({}) == "4a0f886d1979"   # Basis gepinnt
    cfg = ol.variant_config()
    params = [ol.variant_params(c, cfg) for c in ({"ridge_alpha": 30.0}, {"min_pred": 0.02},
                                                   {"drop_feature_group": "ml_scores"})]
    assert params[2]["drop"] == sorted(cfg["approved_methods"]["drop_feature_group"]["ml_scores"])
    ids = {ol.experiment_id(p) for p in params}
    assert len(ids) == 3 and ol.experiment_id() not in ids


def test_variant_parameters_change_only_the_approved_part():
    days = _trading_days(date(2026, 1, 5), 20)
    rows, quotes = _history(days)
    data = ol.dataset(rows, quotes, AS_OF, H)
    base, strong = ol.fit(data), ol.fit(data, {"alpha": 100.0})
    assert base.names == strong.names and abs(strong.coef).sum() < abs(base.coef).sum()
    assert "sentiment_score" not in ol.fit(data, {"drop": ["sentiment_score"]}).names
    cand = _two_form_cands([date(2026, 3, 2)], lambda s, rng: (0.1, 0.1), n=1)[0]
    assert ol.decide(cand, [0.01, 0.03])["action"] == "TRADE"
    low = ol.decide(cand, [0.01, 0.03], min_pred=0.05)
    assert low["action"] == "ABSTAIN" and low["reason"] == "PRED_BELOW_MIN"


def test_variant_registration_is_bounded_diverse_and_never_repeats(tmp_path):
    cfg = ol.variant_config()
    now = datetime(2026, 11, 7, 9, tzinfo=timezone.utc)
    props = [_auto({"ridge_alpha": 30.0}), _auto({"ridge_alpha": 100.0}), _auto({"drop_feature_group": "ml_scores"}),
             _auto({"drop_feature_group": "positioning"}), _auto({"min_pred": 0.02}),
             dict(_auto({"ridge_alpha": 3.0}), auto_experiment=False), _auto({"ridge_alpha": 31.0}),
             _auto({"ridge_alpha": 30.0}, h=20)]
    new = ol.register_variants(tmp_path, props, now, cfg)
    by_h = {h: [v["overrides"] for v in new if v["horizon"] == h] for h in (5, 20)}
    assert by_h[5] == [{"ridge_alpha": 30.0}, {"drop_feature_group": "ml_scores"}, {"min_pred": 0.02}]
    assert by_h[20] == [{"ridge_alpha": 30.0}]
    assert all(v["start_date"] == "2026-11-08" and v["base_experiment_id"] == ol.experiment_id() for v in new)
    assert ol.register_variants(tmp_path, props, now, cfg) == []                 # idempotent
    ridge = next(v for v in new if v["horizon"] == 5 and "ridge_alpha" in v["overrides"])
    append_jsonl(tmp_path / ol.VARIANTS, [{"event": "closed", "variant_id": ridge["variant_id"],
                                           "status": "FALSIFIED_NO_EFFECT", "closed_at": "2026-12-01"}])
    again = ol.register_variants(tmp_path, props, now, cfg)
    assert [v["overrides"] for v in again] == [{"ridge_alpha": 100.0}]          # freier Platz, nie ridge 30 erneut
    assert ol.active_variants(tmp_path, 5, date(2026, 11, 7)) == []            # erst ab dem Folgetag aktiv
    assert len(ol.active_variants(tmp_path, 5, date(2026, 11, 8))) == 3


def _variant_env(monkeypatch, tmp_path, register=True):
    days, rows = _ledger_env(monkeypatch, 40)
    day = days[30]
    rows += [_row(day.isoformat(), 1, 0.8)]
    if register:
        ol.register_variants(tmp_path, [_auto({"ridge_alpha": 100.0}), _auto({"min_pred": 0.05})],
                             datetime.combine(day - timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc),
                             ol.variant_config())
    now = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=17)
    return day, now


def test_variants_run_isolated_on_identical_candidates(monkeypatch, tmp_path):
    base_dir, var_dir = tmp_path / "base", tmp_path / "var"
    day, now = _variant_env(monkeypatch, base_dir, register=False)
    plain = ol.recommend(day, horizon=H, out=base_dir, now=now)
    _variant_env(monkeypatch, var_dir)
    with_variants = ol.recommend(day, horizon=H, out=var_dir, now=now)
    assert {k: v for k, v in with_variants.items() if k != "variant_runs"} == \
        {k: v for k, v in plain.items() if k != "variant_runs"}
    assert (var_dir / ol.JOURNAL).read_text() == (base_dir / ol.JOURNAL).read_text()   # Basis-Journal unberührt
    assert sorted(with_variants["variant_runs"].values()) == [10, 10]           # alle Kandidaten des Tages
    for v in ol.variant_registry(var_dir).values():
        entries = read_jsonl(var_dir / ol.VARIANT_DIR / f"{v['variant_id']}.jsonl")
        base_dec = [j for j in read_jsonl(base_dir / ol.JOURNAL) if j["kind"] == "decision" and j["role"] == "learning"]
        assert {(e["date"], e["signal_id"]) for e in entries} == {(j["date"], j["signal_id"]) for j in base_dec}
        assert {e["experiment_id"] for e in entries} == {v["experiment_id"]}
        assert {e["role"] for e in entries} == {"variant"}
        inst = {i["model_id"]: i for i in read_jsonl(var_dir / ol.INSTANCES)}
        assert all(ol._invalid(e, v["experiment_id"], inst) is None for e in entries)


def test_a_failing_variant_never_touches_the_base(monkeypatch, tmp_path):
    day, now = _variant_env(monkeypatch, tmp_path)
    real_fit = ol.fit

    def broken(train, params=None):
        if params:
            raise RuntimeError("Variante defekt")
        return real_fit(train)
    monkeypatch.setattr(ol, "fit", broken)
    res = ol.recommend(day, horizon=H, out=tmp_path, now=now)
    assert res["decisions"] == 10 and all(str(r).startswith("Fehler") for r in res["variant_runs"].values())
    assert not (tmp_path / ol.VARIANT_DIR).exists()


def _variant_case(tmp_path, rule, n_days=30, variant_pick=1, base_pick=0, family=1, single_status=None,
                  drop_base=(), late=False, base_exp=None):
    """Ab 2.3.2026 n_days Handelstage à 4 Kandidaten mit zwei Formen (Spread = Standardform, Einzel-Option);
    rule(tag_index, rng) -> (y_spread, y_single). Basis und Referenz wählen base_pick, die Variante variant_pick.
    Prüfzeitpunkte bei 30/60/90 Tagen ab Start (max_days 90)."""
    days, rng = _trading_days(date(2026, 3, 2), n_days), random.Random(3)
    base = ol.experiment_id()
    inst = [{"model_id": m, "experiment_id": e, "train_cutoff": "2026-03-01", "max_label_day": "2026-02-27"}
            for m, e in (("base", base), ("var", "VEXP"))]
    cands, journal, vj = [], [], []
    for k, d in enumerate(days):
        for i in range(4):
            ys, y1 = rule(k, rng)
            st1 = (single_status or (lambda i: "VERIFIED"))(i)
            ld = d + timedelta(days=H + 1)
            c = {"date": d.isoformat(), "signal_id": f"{d}-{i}", "ticker": f"T{i}", "sample": "research_census",
                 "gate": "x", "exprs": [
                     {"expression": "spread", "contract": [f"T{i}C1", f"T{i}C2"], "strategy": "BULL_CALL_SPREAD",
                      "x": {}, "ask": 1.0, "mid": 0.9, "status": "VERIFIED", "y": ys, "label_day": ld},
                     {"expression": "single", "contract": [f"T{i}C1"], "strategy": "LONG_CALL", "x": {}, "ask": 2.0,
                      "mid": 1.9, "status": st1, "y": y1 if st1 == "VERIFIED" else None,
                      "label_day": ld if st1 == "VERIFIED" else None}]}
            cands.append(c)
            e = {"date": c["date"], "signal_id": c["signal_id"], "ticker": c["ticker"], "horizon": H,
                 "kind": "decision", "action": "TRADE", "created_at": f"{c['date']}T16:00:00+00:00", "pred_net": 0.1}
            if i not in drop_base:
                for role in ("learning", "reference"):
                    journal.append(dict(e, role=role, experiment_id=base, model_id="base",
                                        contract=c["exprs"][base_pick]["contract"]))
            vj.append(dict(e, role="variant", variant_id="OLV-1", experiment_id="VEXP", model_id="var",
                           contract=c["exprs"][variant_pick]["contract"],
                           created_at="2026-12-01T00:00:00+00:00" if late else e["created_at"]))
    reg = [{"event": "registered", "variant_id": "OLV-1", "horizon": H, "base_experiment_id": base_exp or base,
            "experiment_id": "VEXP", "overrides": {"ridge_alpha": 30.0}, "start_date": "2026-03-02",
            "falsification": {"max_days": 90}, "look_days": ol.look_days(90), "proposal_id": "OLP-x",
            "gap_ids": ["model:overfit_risk:5d"]}]
    reg += [dict(reg[0], variant_id=f"OLV-dummy{i}", experiment_id=f"D{i}", base_experiment_id=base)
            for i in range(family - 1)]
    append_jsonl(tmp_path / ol.VARIANTS, reg)
    append_jsonl(tmp_path / ol.VARIANT_DIR / "OLV-1.jsonl", vj)
    return cands, journal, inst


def _eval_variant(tmp_path, as_of, case=None, **kw):
    cands, journal, inst = case or _variant_case(tmp_path, **kw)
    res = ol.evaluate_variants(tmp_path, cands, H, as_of, journal, inst, ol.variant_config(),
                               datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc))
    return res, res["variants"]["OLV-1"]


def _gauss(spread, single, sd=0.1):
    return lambda k, rng: (rng.gauss(spread, sd), rng.gauss(single, sd))


LOOK1, LOOK2, FINAL = date(2026, 4, 20), date(2026, 5, 20), date(2026, 6, 20)   # Prüfung 1 / 2 / 3 auswertbar


def test_better_and_net_profitable_variant_is_proposed_for_adoption(tmp_path):
    _res, v = _eval_variant(tmp_path, LOOK1, rule=_gauss(0.0, 0.3))
    assert (v["look"], v["relative_verdict"], v["economic_verdict"]) == (1, "BETTER_THAN_BASE",
                                                                         "POSITIVE_EXECUTABLE_ALPHA")
    assert v["status"] == "PROPOSE_ADOPTION" and v["change_proposal"]["state"] == "VALID"
    assert v["comparison"]["coverage"] == 1.0 and v["comparison"]["decisions_after_cutoff"] > 0
    assert v["vs_frozen_reference"]["verdict"] == "LEARNING_BETTER"
    md = (tmp_path / ol.CHANGE_DIR / "OLV-1.md").read_text()
    assert md.startswith("> **Aktueller Evidenzstatus: GÜLTIG") and "RIDGE_ALPHA = 30.0" in md


def test_relatively_better_but_net_losing_variant_is_never_an_adoption(tmp_path):
    _res, v = _eval_variant(tmp_path / "a", LOOK1, rule=_gauss(-0.4, -0.1, sd=0.05))
    assert v["relative_verdict"] == "BETTER_THAN_BASE" and v["economic_verdict"] == "NEGATIVE_NET"
    assert v["status"] == "BETTER_THAN_BASE_ONLY" and "change_proposal" not in v
    assert not (tmp_path / "a" / ol.CHANGE_DIR).exists()
    _res, final = _eval_variant(tmp_path / "b", FINAL, rule=_gauss(-0.4, -0.1, sd=0.05), n_days=70)
    assert final["look"] == 3 and final["status"] == "BETTER_THAN_BASE_NOT_PROFITABLE"
    assert ol.variant_registry(tmp_path / "b")["OLV-1"]["status"] == "BETTER_THAN_BASE_NOT_PROFITABLE"
    assert not (tmp_path / "b" / ol.CHANGE_DIR).exists()


def test_missing_quotes_or_base_decisions_block_a_positive_verdict(tmp_path):
    _res, v = _eval_variant(tmp_path / "q", LOOK1, rule=_gauss(0.0, 0.3),
                            single_status=lambda i: "EXIT_NOT_ARCHIVED" if i == 0 else "VERIFIED")
    cmp = v["comparison"]
    assert cmp["coverage"] == 0.75 and cmp["excluded"] == {"EXIT_NOT_ARCHIVED": cmp["due_candidates"] // 4}
    assert v["relative_verdict"] == v["economic_verdict"] == "INSUFFICIENT_COVERAGE"
    assert v["status"] == "RUNNING" and "change_proposal" not in v
    _res, b = _eval_variant(tmp_path / "b", LOOK1, rule=_gauss(0.0, 0.3), drop_base=(0,))
    assert b["comparison"]["excluded"] == {"NO_BASE_DECISION": b["comparison"]["due_candidates"] // 4}
    assert b["comparison"]["coverage"] == 0.75 and b["status"] == "RUNNING"


def test_alpha_is_spent_over_preregistered_looks():
    spend = ol.alpha_spending(0.05)
    assert len(spend) == 3 and abs(sum(spend) - 0.05) < 1e-12 and spend[0] < 0.001 < spend[1] < spend[2]
    assert ol.look_days(90) == [30, 60, 90] and ol.look_days(200) == [67, 133, 200]
    # ein Effekt, der einen einzelnen 95-%-Test besteht, reicht an der frühen Prüfung nicht
    cands = _two_form_cands(_trading_days(date(2026, 3, 2), 30),
                            lambda s, rng: (rng.gauss(0, 0.15), rng.gauss(0.05, 0.15)), n=4, seed=3)
    pairs = [{"cand": c, "learning": {"action": "TRADE", "contract": c["exprs"][1]["contract"]},
              "reference": {"action": "TRADE", "contract": c["exprs"][0]["contract"]}} for c in cands]
    assert ol.compare_policies(pairs, H)["verdict"] == "LEARNING_BETTER"
    assert ol.compare_policies(pairs, H, spend[0], method="normal")["verdict"] == "NO_SIGNIFICANT_DIFFERENCE"


def test_weekly_reevaluation_reuses_the_frozen_look_data(tmp_path):
    case = _variant_case(tmp_path, rule=_gauss(0.0, 0.05), n_days=62)
    _res, early = _eval_variant(tmp_path, date(2026, 4, 10), case=case)
    assert early["look"] == 0 and early["status"] == "RUNNING" and early["relative_verdict"] == "AWAITING_LOOK"
    assert early["next_look"] == {"look": 1, "cutoff": "2026-04-01", "ready_after": "2026-04-16"}
    _res, w1 = _eval_variant(tmp_path, LOOK1, case=case)
    _res, w2 = _eval_variant(tmp_path, date(2026, 5, 10), case=case)          # eine Woche später, gleiche Prüfung
    assert w1["look"] == w2["look"] == 1 and w1["look_cutoff"] == "2026-04-01"
    assert w1["vs_base"] == w2["vs_base"] and w1["comparison"] == w2["comparison"] | {
        "decisions_after_cutoff": w1["comparison"]["decisions_after_cutoff"]}


def test_proposal_shows_its_current_evidence_and_is_withdrawn_when_the_advantage_fades(tmp_path, monkeypatch):
    def fades(k, rng):                                     # Vorteil nur in den ersten ~22 Handelstagen
        return rng.gauss(0.0, 0.05), rng.gauss(0.3 if k < 22 else -0.3, 0.05)
    case = _variant_case(tmp_path, rule=fades, n_days=62)
    _res, first = _eval_variant(tmp_path, LOOK1, case=case)
    assert first["status"] == "PROPOSE_ADOPTION" and first["change_proposal"]["state"] == "VALID"
    path = tmp_path / ol.CHANGE_DIR / "OLV-1"
    original = json.loads(path.with_suffix(".json").read_text())["original_markdown"]
    _res, later = _eval_variant(tmp_path, LOOK2, case=case)
    assert later["look"] == 2 and later["status"] not in ol.ADOPTION
    assert later["change_proposal"]["state"] == "WITHDRAWN"
    rec = json.loads(path.with_suffix(".json").read_text())
    assert rec["original_markdown"] == original and rec["original"]["status"] == "PROPOSE_ADOPTION"
    assert [h["state"] for h in rec["status_history"]] == ["VALID", "WITHDRAWN"]
    md = path.with_suffix(".md").read_text()
    assert md.startswith("> **Aktueller Evidenzstatus: ZURÜCKGEZOGEN") and md.endswith(original)
    monkeypatch.setattr(ol, "experiment_id", lambda params=None: "NEUE_BASIS")
    _res, moved = _eval_variant(tmp_path, LOOK2, case=case)
    assert moved["status"] == "BASE_CHANGED" and moved["change_proposal"]["state"] == "SUPERSEDED"


def test_preregistered_falsification_closes_variants(tmp_path):
    _res, worse = _eval_variant(tmp_path / "w", LOOK1, rule=_gauss(-0.3, 0.0), variant_pick=0, base_pick=1)
    assert worse["status"] == "FALSIFIED_WORSE"
    assert ol.variant_registry(tmp_path / "w")["OLV-1"]["status"] == "FALSIFIED_WORSE"
    _res, interim = _eval_variant(tmp_path / "i", LOOK1, rule=_gauss(0.0, 0.3), variant_pick=0)
    assert interim["relative_verdict"] == "NO_SIGNIFICANT_DIFFERENCE" and interim["status"] == "RUNNING"
    _res, final = _eval_variant(tmp_path / "f", FINAL, rule=_gauss(0.0, 0.3), variant_pick=0, n_days=70)
    assert final["status"] == "FALSIFIED_NO_EFFECT"
    assert "Prüfung 3/3" in ol.variant_registry(tmp_path / "f")["OLV-1"]["close_reason"]
    res, family = _eval_variant(tmp_path / "k", LOOK1, rule=_gauss(0.0, 0.3), family=20)
    assert res["family_size"] == 20 and res["alpha_variant"] == 0.0025
    assert family["alpha_look"] == round(ol.alpha_spending(0.0025)[0], 8)


def test_variant_evidence_counts_only_prospective_entries(tmp_path):
    _res, v = _eval_variant(tmp_path, LOOK1, rule=_gauss(0.0, 0.3), late=True)
    assert v["invalid_entries"] == {"LATE_PREDICTION": 120} and v["comparison"]["due_candidates"] == 0
    assert v["relative_verdict"] == "NEED_MORE_DATA" and v["status"] == "RUNNING"


def test_variant_experiments_and_open_proposals_land_in_research_memory(tmp_path):
    from modules import research_memory as rm
    _res, v = _eval_variant(tmp_path, LOOK1, rule=_gauss(0.0, 0.3))
    status = {"variants": {"variants": {
        "OLV-1": v, "OLV-2": {**v, "variant_id": "OLV-2", "status": "FALSIFIED_WORSE", "overrides": {"min_pred": 0.02}},
        "OLV-3": {**v, "variant_id": "OLV-3", "status": "BETTER_THAN_BASE_ONLY", "overrides": {"min_pred": 0.05}}}}}
    imp = {"proposals": [{"proposal_id": "OLP-d", "kind": "data_source", "change": {"source": "x"},
                          "approval_required": True, "priority": 0.1, "gap_ids": ["model:no_skill:20d"],
                          "cost": {"paid_data": True}},
                         {"proposal_id": "OLP-v", "kind": "variant", "approval_required": False}]}
    paths = {"option_status_5d": tmp_path / "s.json", "option_improvement": tmp_path / "i.json"}
    paths["option_status_5d"].write_text(json.dumps(status))
    paths["option_improvement"].write_text(json.dumps(imp))
    entries = {e["hypothesis_id"]: e for e in rm.collect(paths)}
    assert entries["OLV-1"]["status"] == "FORWARD_VALIDATED" and entries["OLV-1"]["prospective"] is True
    assert entries["OLV-1"]["evidence"]["economic"] == "POSITIVE_EXECUTABLE_ALPHA"
    assert entries["OLV-1"]["spec"]["signal"] == "ridge_alpha=30.0" and entries["OLV-1"]["evidence"]["coverage"] == 1.0
    assert entries["OLV-2"]["status"] == "REJECTED" and entries["OLV-3"]["status"] == "PENDING_FORWARD"
    assert entries["OLP-d"]["status"] == "APPROVAL_REQUIRED" and "OLP-v" not in entries
    memory = tmp_path / "memory.jsonl"
    assert rm.sync(memory, paths) == 4 and rm.sync(memory, paths) == 0      # append-only, keine Dubletten


def test_report_card_shows_detected_gaps_and_running_improvements(monkeypatch, tmp_path):
    _ledger_env(monkeypatch, 40)
    ol.evaluate(date(2026, 12, 31), horizon=H, out=tmp_path)
    (tmp_path / "improvement.json").write_text(json.dumps({
        "gaps": [{"gap_id": "model:overfit_risk:5d", "horizon": H, "finding": "34 Merkmale auf 359 Labels"}],
        "proposals": [{"horizon": H, "auto_experiment": True, "approval_required": False}]}))
    card = {r["key"]: r[f"{H}d"] for r in ol.report_card(tmp_path, horizons=(H,))["rows"]}
    assert card["gaps"][0] == "Ja" and "34 Merkmale" in card["gaps"][1] and "1 automatisch" in card["gaps"][1]
    assert list(card["improvements"]) == ["offen", "noch keine Variante registriert"]


def test_weekly_workflow_runs_research_improvement_after_the_evaluation_without_secrets():
    import yaml as _yaml
    with open(".github/workflows/ml_research.yml", encoding="utf-8") as fh:
        wf = _yaml.safe_load(fh)
    steps = next(iter(wf["jobs"].values()))["steps"]
    ids = [s.get("id") for s in steps]
    step = steps[ids.index("research_improvement")]
    assert ids.index("research_improvement") == ids.index("option_learner") + 1
    assert step["run"].strip() == "python -m modules.research_director --options"
    assert "env" not in step and step.get("continue-on-error") is True and "secrets" not in json.dumps(step)
    assert "steps.research_improvement.outcome" in json.dumps(steps[-1])
