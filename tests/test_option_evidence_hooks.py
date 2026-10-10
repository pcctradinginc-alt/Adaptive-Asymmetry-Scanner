"""Options-Evidenz (Master-Auftrag 2026-10-10): beobachtete Quotes werden an den echten Abrufstellen archiviert
(Kandidaten-Entry/-Marks, Trade-Entry, Exit-Marks); Defaults des Options-Designers nie als beobachtet;
reproduzierbarer Trade-Outcome mit Kostenmodell (brutto/netto) und Quote-Verweisen."""
from __future__ import annotations

from datetime import datetime, timezone

import feedback as fb
import pipeline
from modules import market_snapshot as ms
from modules import option_quotes as oq
from modules.atomic_io import read_jsonl


def test_trade_entry_archives_legs_without_defaulted_greeks():
    p = {"ticker": "AAPL", "strategy": "BULL_CALL_SPREAD",
         "option": {"expiry": "2026-12-18", "strike": 150.0, "bid": 5.0, "ask": 5.4, "open_interest": 900,
                    "implied_vol": 0.30, "iv_source": "default_0.30", "delta": 0.55, "delta_source": "chain",
                    "data_source": "tradier", "option_symbol": "AAPL261218C00150000",
                    "spread_leg": {"strike": 160.0, "bid": 2.0, "ask": 2.3}}}
    ids = pipeline._archive_entry_quotes(p)
    assert len(ids) == 2
    recs = {r["record_id"]: r for r in oq.read("2000-01-01")}
    long_leg = recs[ids[0]]
    assert long_leg["occ_symbol"] == "AAPL261218C00150000" and long_leg["context"] == "trade_entry"
    assert long_leg["iv"] is None and long_leg["delta"] == 0.55          # IV-Default nie als beobachtet
    assert recs[ids[1]]["occ_symbol"] == "AAPL261218C00160000" and recs[ids[1]]["price_type"] == "OBSERVED_QUOTE"


def test_candidate_mark_quotes_are_archived(monkeypatch):
    monkeypatch.setattr(ms, "_use_tradier", lambda: True)
    monkeypatch.setattr(ms, "_fetch_quotes_raw", lambda chunk: [
        {"symbol": "MSFT261218P00400000", "underlying": "MSFT", "bid": 7.1, "ask": 7.5, "volume": 12,
         "open_interest": 300, "bid_date": 1791640000000}])
    out = ms.fetch_option_quotes(["MSFT261218P00400000"])
    assert out["MSFT261218P00400000"]["bid"] == 7.1
    rec = [r for r in oq.read("2000-01-01") if r["context"] == "candidate_mark"]
    assert len(rec) == 1 and rec[0]["underlying"] == "MSFT" and rec[0]["quote_time"] is not None


def test_exit_quote_archived_and_trade_outcome_reproducible(monkeypatch):
    class _R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"options": {"option": [{"option_type": "call", "strike": 150.0, "bid": 7.0, "ask": 7.4,
                                            "symbol": "AAPL261218C00150000", "open_interest": 800}]}}
    monkeypatch.setattr(fb.requests, "get", lambda *a, **k: _R())
    q = fb._tradier_option_quote("AAPL", 150.0, "2026-12-18", "call")
    assert q == {"bid": 7.0, "ask": 7.4, "source": "tradier"}
    trade = {"ticker": "AAPL", "strategy": "LONG_CALL", "entry_date": "2026-10-12", "close_date": "2026-11-20",
             "entry_debit": 5.4, "entry_quote_record_ids": ["e1"],
             "option": {"expiry": "2026-12-18", "strike": 150.0, "option_symbol": "AAPL261218C00150000"}}
    meta = {"method": "option_quote", "price_type": "OBSERVED_QUOTE"}
    rec = fb.record_trade_outcome(trade, (7.0 - 5.4) / 5.4, meta, now=datetime.now(timezone.utc))
    assert rec and rec["reliable"] and rec["price_type"] == "OBSERVED_QUOTE"
    assert rec["exit_quote_record_ids"][0] is not None                      # Verweis auf archivierte Exit-Quote
    assert rec["outcome_net"] < rec["outcome"]                               # Kommission explizit abgezogen
    assert abs((rec["outcome"] - rec["outcome_net"]) - (0.65 * 2) / (5.4 * 100)) < 1e-6
    assert trade["outcome_net"] == rec["outcome_net"] and trade["cost_model"]["version"]
    assert fb.record_trade_outcome(trade, 0.1, meta) is None                 # je trade_id genau einmal
    rows = read_jsonl(oq.ROOT.parent / "trade_outcomes.jsonl")
    assert len(rows) == 1


def test_modeled_outcome_is_recorded_but_never_reliable():
    trade = {"ticker": "X", "strategy": "LONG_PUT", "entry_date": "2026-10-12", "entry_debit": 2.0,
             "option": {"expiry": "2026-12-18", "strike": 50.0}}
    rec = fb.record_trade_outcome(trade, 0.4, {"method": "delta_approx", "price_type": "MODELED"})
    assert rec and rec["reliable"] is False and rec["price_type"] == "MODELED"
