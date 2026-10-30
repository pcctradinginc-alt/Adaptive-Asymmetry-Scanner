import hashlib
import random
from datetime import date, datetime, timedelta, timezone

import pytest

from modules import market_snapshot as ms
from modules import option_learner as ol
from modules.atomic_io import read_jsonl

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
    assert ol.recommend(date.fromisoformat(today), horizon=H, out=tmp_path, now=now)["journaled"] == 2
    assert ol.recommend(date.fromisoformat(today), horizon=H, out=tmp_path, now=now)["journaled"] == 0
    assert [j["kind"] for j in read_jsonl(tmp_path / ol.JOURNAL)] == ["prediction", "decision"]


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
    assert res["journaled"] == 3 and res["decisions"] == 1          # Spread und Einzel-Option + eine Entscheidung
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
    assert len(registry) == 1 and len(journal) == 2                               # append-only, keine Dubletten
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
