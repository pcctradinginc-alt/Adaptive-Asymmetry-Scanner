"""Audit 2026-11-01: spread short legs are bought back at the ASK.

Live finding: PTC270416C00210000 (short leg of a bull call spread) quoted bid 0 /
ask 0.20 on its 20-day due day. Because every leg had to have bid > 0, the
spread mark slipped to the next day while the long-only mark of the same
candidate stayed on the due day. If the zero bid persists, the mark becomes
LATE or is given up - exactly for losing spreads (survivorship bias).

Long legs (sold at bid) and all defaults (timing_policy, trade fills) are
unchanged. Synthetic fixtures only - no evidence of economic success.
"""
import hashlib
from datetime import date, datetime, timedelta, timezone

import pytest

from modules import candidate_ledger as cl
from modules import option_quotes as oq
from modules.research_evidence_discovery import provenance, quote_index

DUE = date(2026, 10, 28)                       # entry 2026-10-08 + 20 calendar days (PTC case)
IN_WINDOW = datetime(2026, 10, 28, 16, 15, tzinfo=timezone.utc)   # 12:15 New York


def rec(symbol, bid, ask, when=IN_WINDOW, flags=(), status=None):
    ts = when.isoformat()
    flags = sorted(set(flags) | ({"ZERO_BID"} if bid == 0 else set()))
    return {"record_id": hashlib.sha256(f"{symbol}{ts}{bid}{ask}".encode()).hexdigest()[:20],
            "occ_symbol": symbol, "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4),
            "quote_time": ts, "retrieved_at": ts, "multiplier": 100,
            "quality_status": status or ("SUSPECT" if bid == 0 else "OK"), "quality_flags": flags}


# ── usability rule ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("q, long_ok, short_ok", [
    (rec("S", 0.0, 0.2), True, True),                                    # zero bid: buy back at ask / worthless long
    (rec("S", 0.0, 0.8), False, True),                                   # zero bid, large ask: no long exit
    (rec("S", 0.05, 0.1), True, True),
    (rec("S", 0.0, 0.0), False, False),                                  # no ask: nothing to buy back
    (rec("S", 0.0, 0.2, flags=("STALE",)), False, False),
    (rec("S", 0.0, 0.2, status="UNUSABLE"), False, False),
    (rec("S", 0.0, 0.2, when=datetime(2026, 10, 28, 13, 40, tzinfo=timezone.utc)), False, False),  # before window
])
def test_exit_usability_by_leg_side(q, long_ok, short_ok):
    # Audit 2026-11-06: Long-Exit akzeptiert zusätzlich eine wertlose Option (Bid 0, Ask <= ZERO_BID_EXIT_MAX_ASK);
    # Entry-Regel usable_rth_quote unverändert (Bid > 0).
    assert oq.usable_rth_exit_quote(q, "long") is long_ok
    assert oq.usable_rth_quote(q) is (long_ok and (q["bid"] or 0) > 0)
    assert oq.usable_rth_exit_quote(q, "short") is short_ok


def _archive(tmp_path, *records):
    root = tmp_path / "quotes"
    for r in records:
        p = root / (oq.quote_day(r) + ".jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            import json
            fh.write(json.dumps(r) + "\n")
    return root


def test_history_long_exit_opt_in_includes_worthless_long_quotes(tmp_path):
    root = _archive(tmp_path, rec("W", 0.0, 0.05), rec("G", 0.0, 2.0))
    now = IN_WINDOW + timedelta(hours=2)
    assert oq.rth_quote_history(["W", "G"], now, root=root) == {}                # Standard (Entry/timing) unverändert
    assert list(oq.rth_quote_history(["W", "G"], now, root=root, long_exit=True)) == ["W"]


def test_history_default_unchanged_short_symbols_opt_in(tmp_path):
    root = _archive(tmp_path, rec("S", 0.0, 0.2), rec("L", 4.0, 4.4))
    now = IN_WINDOW + timedelta(hours=2)
    default = oq.rth_quote_history(["S", "L"], now, root=root)
    assert "S" not in default and "L" in default                         # timing_policy & co. unchanged
    opt_in = oq.rth_quote_history(["S", "L"], now, root=root, short_symbols=["S"])
    assert [d for d, _ in opt_in["S"]] == [DUE.isoformat()] and "L" in opt_in


# ── candidate ledger: strategy mark on the due day ───────────────────────────

def _spread_row(long_sym="L", short_sym="S"):
    return {"ticker": "PTC", "direction": "BULLISH", "outcomes": {},
            "real_strategy": {"strategy": "BULL_CALL_SPREAD", "entry_pending": False, "entry_date": "2026-10-08",
                              "expiry": "2027-04-16", "net_debit_entry": 3.0, "net_mid_entry": 2.8,
                              "legs": [{"symbol": long_sym, "side": "long", "strike": 195.0},
                                       {"symbol": short_sym, "side": "short", "strike": 210.0}]}}


@pytest.fixture
def archive_mode(monkeypatch, tmp_path):
    def setup(*records):
        root = _archive(tmp_path, *records)
        monkeypatch.setattr(oq, "ROOT", root)
        monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "closed")
        monkeypatch.setattr(cl, "_now_utc", lambda: datetime(2026, 10, 28, 23, 0, tzinfo=timezone.utc))
        return root
    return setup


def test_spread_with_zero_bid_short_leg_is_marked_on_due_day(archive_mode):
    nxt = IN_WINDOW + timedelta(days=1)
    archive_mode(rec("L", 1.0, 1.2), rec("S", 0.0, 0.2),                # due day: short leg bid 0
                 rec("L", 0.9, 1.1, when=nxt), rec("S", 0.05, 0.25, when=nxt))
    row = _spread_row()
    assert cl._fill_real_strategy_marks([row], datetime(2026, 10, 28, 23, 0)) is True  # noqa: DTZ001 – ledger API is naive
    oc = row["outcomes"]
    assert oc["real_strat_mark_date_20d"] == "2026-10-28"                # not shifted to the next day
    assert oc["real_strat_ret_20d"] == pytest.approx(round((1.0 - 0.2) / 3.0 - 1, 4))   # long bid - short ask
    assert oc["real_strat_mark_source_20d"] == "archive_rth" and "real_strat_late_20d" not in oc


def test_worthless_long_leg_is_marked_as_total_loss(archive_mode):
    """Audit 2026-11-06: Long-Leg Bid 0 / Ask 0,10 ist ein beobachteter Exit (Verkauf zum Bid) -> Totalverlust
    auf dem Fälligkeitstag statt verschobenem/aufgegebenem Mark (positive Verzerrung)."""
    archive_mode(rec("L", 0.0, 0.1), rec("S", 0.0, 0.05))
    row = _spread_row()
    cl._fill_real_strategy_marks([row], datetime(2026, 10, 28, 23, 0))  # noqa: DTZ001 – ledger API is naive
    assert row["outcomes"]["real_strat_ret_20d"] == -1.0
    assert row["outcomes"]["real_strat_mark_date_20d"] == "2026-10-28"


def test_zero_bid_long_leg_with_large_ask_is_still_not_used(archive_mode):
    archive_mode(rec("L", 0.0, 1.5), rec("S", 0.05, 0.2))                # fehlende Geldseite, kein Totalverlust-Beleg
    row = _spread_row()
    cl._fill_real_strategy_marks([row], datetime(2026, 10, 28, 23, 0))  # noqa: DTZ001 – ledger API is naive
    assert "real_strat_ret_20d" not in row["outcomes"]


# ── provenance (research_evidence_discovery) ─────────────────────────────────

def _prov_row(claimed):
    entry = "2026-10-14"
    return {"date": entry, "ticker": "PTC",
            "real_strategy": {"entry_pending": False, "entry_date": entry, "net_debit_entry": 3.0,
                              "expiry": "2027-04-16",
                              "legs": [{"symbol": "L", "side": "long", "bid": 3.9, "ask": 4.0},
                                       {"symbol": "S", "side": "short", "bid": 1.0, "ask": 1.1}]},
            "outcomes": {"real_strat_ret_20d": claimed, "real_strat_mark_date_20d": "2026-11-03"}}


def _entry_quotes():
    t = datetime(2026, 10, 14, 16, 0, tzinfo=timezone.utc)
    return [rec("L", 3.9, 4.0, when=t), rec("S", 1.0, 1.1, when=t)]


def test_provenance_verifies_spread_exit_with_zero_bid_short_leg():
    t = datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc)
    quotes = _entry_quotes() + [rec("L", 0.5, 0.6, when=t), rec("S", 0.0, 0.05, when=t)]
    claimed = round((0.5 - 0.05) / 3.0 - 1, 4)
    assert provenance(_prov_row(claimed), quote_index(quotes), 20) == "VERIFIED"


def test_provenance_verifies_worthless_long_exit():
    t = datetime(2026, 11, 3, 17, 0, tzinfo=timezone.utc)
    quotes = _entry_quotes() + [rec("L", 0.0, 0.1, when=t), rec("S", 0.0, 0.05, when=t)]
    assert provenance(_prov_row(-1.0), quote_index(quotes), 20) == "VERIFIED"
    big_ask = _entry_quotes() + [rec("L", 0.0, 1.5, when=t), rec("S", 0.0, 0.05, when=t)]
    assert provenance(_prov_row(-1.0), quote_index(big_ask), 20) == "EXIT_NOT_ARCHIVED"


def test_entry_short_leg_still_needs_a_bid():
    t = datetime(2026, 10, 14, 16, 0, tzinfo=timezone.utc)
    row = _prov_row(0.0)
    ids = [rec("L", 3.9, 4.0, when=t), rec("S", 0.0, 1.1, when=t)]      # short leg cannot be sold at bid 0
    row["real_strategy"]["entry_quote_record_ids"] = [r["record_id"] for r in ids]
    assert provenance(row, quote_index(ids), 20) == "ENTRY_QUOTE_UNUSABLE"


# ── alpha-to-options exits use the same leg-side rule ────────────────────────

def test_alpha_link_exit_and_reverification_accept_zero_bid_short_leg():
    from modules.autonomous_alpha import option_economics as oe
    from modules.autonomous_alpha import option_link as al
    d = {"exit_target_day": "2026-10-28", "exit_deadline_day": "2026-11-04", "decision_day": "2026-10-14",
         "legs": [{"occ_symbol": "L", "side": "long", "bid": 3.9, "ask": 4.0, "quote_record_id": "e1"},
                  {"occ_symbol": "S", "side": "short", "bid": 1.0, "ask": 1.1, "quote_record_id": "e2"}]}
    lq, sq = rec("L", 1.0, 1.2), rec("S", 0.0, 0.2)
    day, legs = al._exit_quotes(d, [lq, sq], IN_WINDOW + timedelta(hours=3))
    assert day == "2026-10-28" and [q["occ_symbol"] for q in legs] == ["L", "S"]
    # Audit 2026-11-06: wertloser Long-Exit (Bid 0, kleines Ask) ist beobachtet; Bid 0 bei großem Ask bleibt offen
    single = {**d, "legs": [d["legs"][0]]}
    assert al._exit_quotes(single, [rec("L", 0.0, 0.1)], IN_WINDOW + timedelta(hours=3))[0] == "2026-10-28"
    assert al._exit_quotes(single, [rec("L", 0.0, 1.5)], IN_WINDOW + timedelta(hours=3)) is None
    t = datetime(2026, 10, 14, 16, 0, tzinfo=timezone.utc)
    e1, e2 = rec("L", 3.9, 4.0, when=t), rec("S", 1.0, 1.1, when=t)
    d["legs"][0]["quote_record_id"], d["legs"][1]["quote_record_id"] = e1["record_id"], e2["record_id"]
    o = {"exit_day": "2026-10-28", "exit_legs": [{"occ_symbol": q["occ_symbol"], "bid": q["bid"], "ask": q["ask"],
                                                  "quote_record_id": q["record_id"]} for q in (lq, sq)]}
    archive = {r["record_id"]: r for r in (e1, e2, lq, sq)}
    assert oe._verify(d, o, archive) == []
