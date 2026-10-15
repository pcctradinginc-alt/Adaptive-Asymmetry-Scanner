"""B2: Kennzahlen aus ohnehin geladenen Tradier-Ketten (synthetische Ketten, kein Netzwerk)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from modules import chain_summary as cs

T0 = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc)


def opt(kind, strike, delta, iv, oi, vol):
    return {"option_type": kind, "strike": strike, "open_interest": oi, "volume": vol,
            "greeks": {"delta": delta, "mid_iv": iv}}


CHAIN = [opt("call", 95, 0.70, 0.30, 100, 10), opt("call", 100, 0.52, 0.28, 300, 50),
         opt("put", 100, -0.47, 0.32, 200, 40), opt("put", 95, -0.30, 0.35, 50, None),
         {"option_type": "call", "strike": 110, "greeks": None, "open_interest": None, "volume": 5}]


def test_summarize_atm_iv_ratios_and_missing_values():
    s = cs.summarize("AAA", "2026-11-20", CHAIN, T0)
    assert s["call_oi"] == 400 and s["put_oi"] == 250 and s["call_volume"] == 65 and s["put_volume"] == 40
    assert s["pc_oi_ratio"] == 0.625 and s["atm_iv"] == 0.3 and s["atm_strike"] == 100 and s["dte"] == 37
    assert cs.summarize("AAA", "2026-11-20", [], T0) is None
    s2 = cs.summarize("AAA", "2026-11-20", [{"option_type": "call", "greeks": {}}], T0)
    assert s2["atm_iv"] is None and s2["call_oi"] is None and s2["pc_oi_ratio"] is None      # nie 0


def test_record_once_per_day_and_never_raises(tmp_path):
    assert cs.record("AAA", "2026-11-20", CHAIN, "x", T0, root=tmp_path) is True
    assert cs.record("AAA", "2026-11-20", CHAIN, "x", T0, root=tmp_path) is False            # gleicher Tag
    assert cs.record("AAA", "2026-11-20", None, "x", T0, root=tmp_path) is False
    assert cs.record("AAA", "kaputt", [{"option_type": "call", "open_interest": "x"}], "x", T0, root=tmp_path) is True
    rows = [json.loads(x) for x in (tmp_path / "2026-10-14.jsonl").read_text().splitlines()]
    assert rows[0]["source"] == "x" and rows[1]["dte"] is None


def test_record_without_root_is_disabled_under_pytest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cs.record("ZZZ", "2026-11-20", CHAIN, "x", T0) is False
    assert not (tmp_path / "outputs").exists()


def test_features_oi_change_and_term_structure(tmp_path):
    far = [dict(o, greeks=dict(o["greeks"] or {}, mid_iv=(o["greeks"] or {}).get("mid_iv", 0) + 0.05))
           if o.get("greeks") else o for o in CHAIN]
    cs.record("BBB", "2026-11-20", CHAIN, "x", T0, root=tmp_path)
    later = [dict(o, open_interest=(o["open_interest"] or 0) + 10) if o.get("open_interest") else o for o in CHAIN]
    cs.record("BBB", "2026-11-20", later, "x", T1, root=tmp_path)
    cs.record("BBB", "2027-01-15", far, "x", T1, root=tmp_path)
    f = {(r["date"], r["ticker"]): r for r in cs.features(cs.load(tmp_path))}
    d1 = f[("2026-10-15", "BBB")]
    assert d1["oi_change_1d"]["2026-11-20"] == {"since": "2026-10-14", "call": 20.0, "put": 20.0}
    assert d1["term_slope_iv"] == 0.05 and d1["term_dte"] == [36, 92]
    d0 = f[("2026-10-14", "BBB")]
    assert d0["oi_change_1d"] == {} and d0["term_slope_iv"] is None               # keine Vorgeschichte/1 Laufzeit
