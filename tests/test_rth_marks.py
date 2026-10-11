"""scripts/rth_marks.py – Options-Marks nur in der regulären Session, idempotent, nie entscheidungswirksam."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import rth_marks  # noqa: E402

from modules import candidate_ledger as cl  # noqa: E402
from modules import market_snapshot as ms  # noqa: E402

NOW = datetime(2026, 10, 9, 19, 5, tzinfo=timezone.utc)
HIST = {"active_trades": [
    {"ticker": "AAPL", "strategy": "BULL_CALL_SPREAD",
     "option": {"option_symbol": "AAPL261120C00200000", "strike": 200, "expiry": "2026-11-20",
                "spread_leg": {"strike": 210, "bid": 1.0, "ask": 1.2}}},
    {"ticker": "XOM", "strategy": "LONG_PUT", "option": {"strike": 100, "expiry": "2026-12-18"}}]}


def test_outside_regular_session_does_nothing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ms, "us_market_session", lambda ts: "post")
    monkeypatch.setattr(cl, "update_outcomes", lambda today: calls.append(today))
    monkeypatch.setattr(ms, "fetch_option_quotes", lambda *a, **k: calls.append("q") or {})
    assert rth_marks.run(NOW, tmp_path / "h.json") == {"session": "post", "skipped": True}
    assert calls == []                                                   # nie Modellpreis, nie Ersatz-Mark


def test_regular_session_marks_ledger_and_archives_active_contracts(monkeypatch, tmp_path):
    hp = tmp_path / "history.json"
    hp.write_text(json.dumps(HIST))
    before = hp.read_bytes()
    seen = {}
    monkeypatch.setattr(ms, "us_market_session", lambda ts: "regular")
    monkeypatch.setattr(cl, "update_outcomes", lambda today: seen.setdefault("today", today))

    def fake_fetch(symbols, context="candidate_mark"):
        seen["symbols"], seen["context"] = list(symbols), context
        return {s: {"bid": 1.0, "ask": 1.1} for s in symbols if not s.startswith("XOM")}
    monkeypatch.setattr(ms, "fetch_option_quotes", fake_fetch)
    res = rth_marks.run(NOW, hp)
    assert seen["today"] == "2026-10-09" and seen["context"] == "trade_mark_rth"
    assert seen["symbols"] == ["AAPL261120C00200000", "AAPL261120C00210000", "XOM261218P00100000"]
    assert res["quotes"] == 2 and res["missing"] == ["XOM261218P00100000"]
    assert hp.read_bytes() == before                                     # history.json unverändert
