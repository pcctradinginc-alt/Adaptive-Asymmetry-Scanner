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
