import hashlib
from datetime import date, timedelta

from modules import market_snapshot as ms
from modules.research_evidence_discovery import (
    discover,
    eligible,
    provenance,
    quote_index,
)


def quote(symbol, day, bid, ask):
    ts = f"{day}T16:00:00+00:00"                     # 11:00/12:00 New York: inside the mark window
    rid = hashlib.sha256(f"{symbol}{day}{bid}{ask}".encode()).hexdigest()[:20]
    return {"record_id": rid, "occ_symbol": symbol, "bid": bid, "ask": ask,
            "mid": round((bid + ask) / 2, 4), "quote_time": ts, "retrieved_at": ts,
            "quality_status": "OK", "quality_flags": []}


def trading_day_on_or_after(d):
    while not ms.is_trading_day(d):
        d += timedelta(days=1)
    return d


def row(day, ticker, value, ret=None, quotes=None):
    """Candidate row with a long-call strategy; `quotes` receives the matching archive records."""
    r = {"date": day, "ticker": ticker, "features": {"momentum": value}, "outcomes": {}}
    if ret is None:
        return r
    symbol = f"{ticker}270319C00100000"
    mark_day = trading_day_on_or_after(date.fromisoformat(day) + timedelta(days=20)).isoformat()
    exit_bid = round(10.0 * (1 + ret), 2)
    r["real_strategy"] = {"legs": [{"symbol": symbol, "side": "long", "bid": 9.9, "ask": 10.0, "mid": 9.95}],
                          "entry_pending": False, "entry_date": day, "net_debit_entry": 10.0,
                          "expiry": "2027-03-19"}
    r["outcomes"] = {"real_strat_ret_20d": ret, "real_strat_mark_date_20d": mark_day}
    if quotes is not None:
        quotes += [quote(symbol, day, 9.9, 10.0), quote(symbol, mark_day, exit_bid, exit_bid + 0.1)]
    return r


def test_never_substitute_stock_returns_for_options():
    r = row("2026-08-03", "AAPL", 1)
    r["outcomes"]["ret_20d"] = 0.8
    assert not eligible(r, date(2026, 10, 30))
    out = discover([r], "2026-10-30")
    assert out["genuine_option_labels"] == 0
    assert out["missing_or_immature_option_labels"] == 1


def test_immature_option_mark_not_used():
    assert not eligible(row("2026-10-26", "AAPL", 1, 0.5), date(2026, 10, 30))


def test_insufficient_evidence_blocks_discovery():
    quotes = []
    out = discover([row("2026-08-03", "AAPL", 1, 0.2, quotes)], "2026-10-30", quotes=quotes)
    assert out["genuine_option_labels"] == 1
    assert out["status"] == "INSUFFICIENT_EVIDENCE"
    assert out["discoveries"] == []
    assert out["production_influence"] == "NONE"


def test_label_without_archived_quotes_is_not_evidence():
    out = discover([row("2026-08-03", "AAPL", 1, 0.2)], "2026-10-30")
    assert out["genuine_option_labels"] == 0
    assert out["unverified_option_labels"] == 1
    assert out["provenance_counts"] == {"ENTRY_NOT_ARCHIVED": 1}


def test_provenance_rejects_claims_the_archive_does_not_support():
    quotes = []
    r = row("2026-08-03", "AAPL", 1, 0.2, quotes)
    index = quote_index(quotes)
    assert provenance(r, index) == "VERIFIED"
    r["outcomes"]["real_strat_ret_20d"] = 0.25          # claimed return differs from the archived exit bid
    assert provenance(r, index) == "EXIT_PRICE_MISMATCH"
    r["outcomes"]["real_strat_ret_20d"] = 0.2
    r["outcomes"]["real_strat_late_20d"] = True
    assert provenance(r, index) == "LATE_MARK"
    del r["outcomes"]["real_strat_late_20d"]
    r["outcomes"]["real_strat_mark_date_20d"] = "2026-08-10"
    assert provenance(r, index) == "PREMATURE_MARK"
    r["outcomes"]["real_strat_mark_date_20d"] = "2026-08-24"
    assert provenance(r, quote_index(quotes[:1])) == "EXIT_NOT_ARCHIVED"


def test_provenance_rejects_quotes_outside_regular_session():
    quotes = []
    r = row("2026-08-03", "AAPL", 1, 0.2, quotes)
    for q in quotes:                                   # 22:00 UTC: after the close
        q["quote_time"] = q["retrieved_at"] = q["retrieved_at"].replace("T16:", "T22:")
    assert provenance(r, quote_index(quotes)) == "ENTRY_NOT_ARCHIVED"


def test_provenance_uses_exact_records_of_timing_v2_fill():
    quotes = []
    r = row("2026-08-03", "AAPL", 1, 0.2, quotes)
    r["real_strategy"]["entry_quote_record_ids"] = [quotes[0]["record_id"]]
    assert provenance(r, quote_index(quotes)) == "VERIFIED"
    r["real_strategy"]["entry_quote_record_ids"] = ["missing"]
    assert provenance(r, quote_index(quotes)) == "ENTRY_NOT_ARCHIVED"
    r["real_strategy"]["entry_quote_record_ids"] = [quotes[0]["record_id"]]
    r["real_strategy"]["net_debit_entry"] = 11.0
    assert provenance(r, quote_index(quotes)) == "ENTRY_PRICE_MISMATCH"


def test_holdout_can_disagree_with_training_without_promotion():
    rows, quotes = [], []
    day = date(2026, 1, 2)
    for i in range(70):
        day = trading_day_on_or_after(day)
        for j in range(4):
            signal = j % 2
            # Training relation is positive, but late holdout reverses.
            ret = (0.2 if signal else -0.1) if i < 49 else (-0.2 if signal else 0.1)
            rows.append(row(day.isoformat(), f"T{j}", signal, ret, quotes))
        day += timedelta(days=1)
    out = discover(rows, "2026-10-30", quotes=quotes)
    assert out["status"] == "HOLDOUT_OBSERVED"
    assert out["discoveries"][0]["train_delta"] > 0
    assert out["discoveries"][0]["holdout_delta"] < 0
    assert out["discoveries"][0]["promotion_eligible"] is False


def test_deduplicate_same_signal():
    quotes = []
    rows = [row("2026-08-03", "AAPL", 1, 0.2, quotes)] * 3
    assert discover(rows, "2026-10-30", quotes=quotes)["genuine_option_labels"] == 1
