import hashlib
import random
from datetime import date, datetime, timedelta, timezone

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


def _row(day, i, sentiment, ret=None, quotes=None, ask=10.0):
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
        mark = _next_trading_day(date.fromisoformat(day) + timedelta(days=H)).isoformat()
        bid = round(ask * (1 + ret), 2)
        r["outcomes"] = {f"real_strat_ret_{H}d": round(bid / ask - 1, 4), f"real_strat_mark_date_{H}d": mark}
        quotes += [_quote(symbol, day, ask - 0.2, ask), _quote(symbol, mark, bid, bid + 0.1)]
    return r


def _history(days, signal=True, seed=3):
    """10 Kandidaten je Tag; mit signal: positives Sentiment -> +30 %, negatives -> -30 % (plus Rauschen)."""
    rng, rows, quotes = random.Random(seed), [], []
    for d in days:
        for i in range(10):
            s = rng.uniform(-1, 1)
            base = (0.3 if s > 0 else -0.3) if signal else rng.choice((0.3, -0.3))
            rows.append(_row(d.isoformat(), i, s, base + rng.uniform(-0.05, 0.05), quotes))
    return rows, quotes


def test_walk_forward_validates_a_real_relation_and_recommends(monkeypatch, tmp_path):
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
    assert [r["ticker"] for r in res["recommendations"]] == ["T1"]       # nur positiver Netto-Erwartungswert
    assert res["watchlist"] == []
    ol.mail(res)
    assert len(sent) == 1
    assert (tmp_path / "recommendations" / f"{today}_{H}d.md").exists()


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
    data = ol.dataset(rows, quotes, date(2026, 12, 31), H)
    real_fit, seen = ol.fit, []

    def spy(train):
        seen.append(max(d["label_day"] for d in train))
        return real_fit(train)
    monkeypatch.setattr(ol, "fit", spy)
    tested = sorted({d["date"] for d in data})
    ol.walk_forward(data)
    first_test = len(tested) - len(seen)
    for test_day, newest_label in zip(tested[first_test:], seen):
        assert newest_label < date.fromisoformat(test_day)


def test_journal_is_append_only_and_forward_ignores_late_predictions(monkeypatch, tmp_path):
    days = _trading_days(date(2026, 1, 5), 15)
    rows, quotes = _history(days)
    today = _trading_days(days[-1] + timedelta(days=10), 1)[0].isoformat()
    rows += [_row(today, 1, 0.8)]
    monkeypatch.setattr(ol.red, "load", lambda root, as_of: rows)
    monkeypatch.setattr(ol.red, "load_quotes", lambda root, labelled, as_of: quotes)
    now = datetime(2026, 3, 1, 19, tzinfo=timezone.utc)
    assert ol.recommend(date.fromisoformat(today), horizon=H, out=tmp_path, now=now)["journaled"] == 1
    assert ol.recommend(date.fromisoformat(today), horizon=H, out=tmp_path, now=now)["journaled"] == 0
    assert len(read_jsonl(tmp_path / ol.JOURNAL)) == 1

    data = ol.dataset(rows, quotes, date(2026, 12, 31), H)
    d = data[0]
    late = [{"date": d["date"], "signal_id": d["signal_id"], "horizon": H, "pred_net": 0.1,
             "created_at": (d["label_day"] + timedelta(days=1)).isoformat()}]
    assert ol.forward(late, data, H)["n"] == 0                          # nach dem Label geschrieben: zählt nie


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
    res = ol.recommend(date.fromisoformat(today), horizon=H, out=tmp_path)
    assert res["journaled"] == 2                                                   # Spread und Einzel-Option
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
