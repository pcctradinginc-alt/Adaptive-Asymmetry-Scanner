from datetime import datetime, timedelta, timezone

from modules import market_snapshot as ms


def test_select_liquid_contract_prefers_tightest_spread_in_window(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test")
    today = datetime.now(timezone.utc).date()
    exps = [(today + timedelta(days=d)).isoformat() for d in (20, 60, 200)]
    monkeypatch.setattr(ms, "_fetch_expirations", lambda ticker: exps)
    seen = []
    chain = [{"symbol": "X_wide", "option_type": "call", "strike": 100, "bid": 4.0, "ask": 5.0},
             {"symbol": "X_tight", "option_type": "call", "strike": 103, "bid": 3.0, "ask": 3.1},
             {"symbol": "X_far", "option_type": "call", "strike": 120, "bid": 1.0, "ask": 1.01},
             {"symbol": "X_put", "option_type": "put", "strike": 100, "bid": 4.0, "ask": 4.01}]
    monkeypatch.setattr(ms, "_fetch_chain", lambda ticker, exp: seen.append(exp) or chain)
    monkeypatch.setattr(ms, "_archive_quote", lambda raw, underlying, context: None)
    c = ms.select_liquid_contract("X", "BULLISH", 100.0)
    assert seen == [exps[1]]                                     # erste Expiry mit 45–90 Tagen
    assert c["symbol"] == "X_tight" and c["dte"] == 60           # engste Spanne im ±5-%-Band, nur Calls
    monkeypatch.delenv("TRADIER_API_KEY")
    assert ms.select_liquid_contract("X", "BULLISH", 100.0) is None
