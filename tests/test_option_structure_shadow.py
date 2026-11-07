"""Audit 2026-11-07 – Optionsbewertung (A7 "IV-Rank", A8 Mark-Versatz), nur Shadow.

A8: Options-Marks liegen am ersten Handelstag am/nach dem Fälligkeitstag, ret_{h}d am letzten davor (Live: 295/699
5-T-Labels mit 1–4 T Versatz). Neu: ret_{h}d_at_mark = Underlying exakt am Mark-Tag; ret_{h}d unverändert.
A7: Der "IV-Rank" enthält kein IV-Niveau (Rangkorrelation mit der Kontrakt-IV 0,05, n = 873) und wählt Spreads, die
ausführbar gepaart 5 pp/Tag schlechter abschneiden als das Long-Leg (Mid: kein Unterschied). Neu: IV/RV21 je Zeile,
Spread-Gegenfaktum real_spread_cf für Long-Zeilen, gepaarter Regelvergleich. Keine Produktionsentscheidung geändert.
Synthetische Daten – kein Beleg für wirtschaftlichen Erfolg.
"""
import hashlib
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from modules import candidate_ledger as cl
from modules import option_quotes as oq
from modules import spread_execution as se


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    cl._state.update({"date": None, "entries": {}, "flushed": False})
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes", lambda tickers: {})
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda tickers, period_days: {})
    yield
    cl._state.update({"date": None, "entries": {}, "flushed": False})


def _weekday_bars(start: datetime, days: int, base: float = 100.0, skip=()):
    return [(start + timedelta(days=d), base + d) for d in range(days)
            if (start + timedelta(days=d)).weekday() < 5 and (start + timedelta(days=d)).date() not in skip]


def _row(day: str, **outcomes) -> dict:
    return {"date": day, "ticker": "AAPL", "pipeline_version": "v8.4", "config_hash": "abc", "status": "rejected",
            "reject_stage": "universe", "reject_reason": "x", "direction": None, "features": {},
            "entry_price": 100.0, "outcomes": dict(outcomes)}


def _write(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f"{rows[0]['date'][:7]}.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


def _read(root, month):
    return [json.loads(x) for x in (root / f"{month}.jsonl").read_text().splitlines() if x.strip()]


# ── A8: Underlying zum Mark-Tag der Option ────────────────────────────────────

MONDAY = datetime(2026, 10, 26)


def test_underlying_label_at_option_mark_day_beside_unchanged_ret_5d(monkeypatch, tmp_path):
    """Mo 26.10. + 5 T = Sa -> ret_5d = Freitagsschluss (wie bisher); Options-Mark am Mo 02.11. -> at_mark = Montag."""
    root = tmp_path / "ledger"
    _write(root, [_row("2026-10-26", real_opt_mark_date_5d="2026-11-02")])
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {"AAPL": _weekday_bars(MONDAY, 12)})
    cl.update_outcomes("2026-11-04", root=root)
    oc = _read(root, "2026-10")[0]["outcomes"]
    assert oc["ret_5d"] == pytest.approx(0.04)                    # Fr 30.10. (Index 4), unverändert
    assert oc["ret_5d_at_mark"] == pytest.approx(0.07)            # Mo 02.11. (Index 7)
    assert oc["ret_5d_at_mark_date"] == "2026-11-02"


def test_at_mark_waits_for_a_completed_session_and_never_substitutes(monkeypatch, tmp_path):
    root = tmp_path / "ledger"
    _write(root, [_row("2026-10-26", real_opt_mark_date_5d="2026-11-02")])
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: {"AAPL": _weekday_bars(MONDAY, 12)})
    cl.update_outcomes("2026-11-02", root=root)                   # Mark-Tag = heute: Sitzung läuft noch
    assert "ret_5d_at_mark" not in _read(root, "2026-10")[0]["outcomes"]
    gap = {"AAPL": _weekday_bars(MONDAY, 12, skip={date(2026, 11, 2)})}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: gap)
    cl.update_outcomes("2026-11-04", root=root)                   # kein Schluss genau am Mark-Tag: leer
    assert "ret_5d_at_mark" not in _read(root, "2026-10")[0]["outcomes"]


def test_price_basis_change_also_invalidates_at_mark_labels(monkeypatch, tmp_path):
    root = tmp_path / "ledger"
    _write(root, [_row("2026-10-26", ret_5d=0.04, ret_5d_at_mark=0.07, ret_5d_at_mark_date="2026-11-02",
                       real_opt_mark_date_5d="2026-11-02")])
    # Abspaltung (CTVA-Fall): Historie bis zum Einstieg rückwirkend bereinigt, danach nicht -> Renditen widersprechen
    split = {"AAPL": [(ts, p / 4 if ts <= MONDAY else p) for ts, p in _weekday_bars(MONDAY, 30)]}
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda t, p: split)
    cl.update_outcomes("2026-11-20", root=root)
    row = _read(root, "2026-10")[0]
    assert "ret_5d_at_mark" not in row["outcomes"] and row["outcomes_invalidated"]["ret_5d_at_mark"] == 0.07
    assert "ret_5d_at_mark_date" not in row["outcomes"]["price_basis"]["invalidated"]


# ── A7: Spread-Gegenfaktum und IV/RV ─────────────────────────────────────────

LONG_RAW = {"symbol": "AAPL270319C00200000", "strike": 200.0, "expiry": "2027-03-19", "dte": 133, "iv": 0.30,
            "bid": 10.0, "ask": 10.4, "mid": 10.2}
SHORT_RAW = {"symbol": "AAPL270319C00220000", "strike": 220.0, "bid": 5.0, "ask": 5.4, "mid": 5.2}


@pytest.mark.parametrize("strategy, source, short, expected", [
    ({"strategy": "LONG_CALL"}, "replicated", SHORT_RAW, "BULL_CALL_SPREAD"),
    ({"strategy": "BULL_CALL_SPREAD"}, "replicated", SHORT_RAW, None),        # Struktur liegt schon vor
    ({"strategy": "LONG_CALL"}, "spread_no_liquidity", SHORT_RAW, None),      # kein handelbares Short-Leg
    ({"strategy": "LONG_CALL"}, "replicated", {**SHORT_RAW, "bid": 0}, None),
    ({"strategy": "LONG_CALL"}, "replicated", None, None),
])
def test_spread_counterfactual_only_for_long_rows(monkeypatch, strategy, source, short, expected):
    calls = []
    monkeypatch.setattr(cl.market_snapshot, "select_spread_short_leg",
                        lambda *a: calls.append(a) or (dict(short) if short else None))
    cf = cl._build_spread_counterfactual({"direction": "BULLISH"}, "AAPL", LONG_RAW,
                                         {**strategy, "strategy_source": source})
    if expected is None:
        assert cf is None
        return
    assert cf["strategy"] == expected and cf["strategy_source"] == "counterfactual_shadow"
    assert [g["side"] for g in cf["legs"]] == ["long", "short"] and cf["net_debit_entry"] == pytest.approx(5.4)
    assert cf["net_mid_entry"] == pytest.approx(5.0) and calls == [("AAPL", "2027-03-19", "call", 200.0)]


def test_iv_rv_recorded_point_in_time_without_changing_the_choice(monkeypatch):
    monkeypatch.setattr(cl.market_snapshot, "select_spread_short_leg", lambda *a: None)
    closes = [100 * (1.01 if i % 2 else 0.99) ** (i % 7) for i in range(300)]
    base = cl._build_real_strategy({"direction": "BULLISH", "features": {}}, "AAPL", 100.0, LONG_RAW, closes=closes)
    comp = base["iv_rank_components"]
    assert comp["iv_rv_21"] == pytest.approx(round(0.30 / comp["rv_21_level"], 4))
    assert base["strategy_source"] in ("replicated", "spread_no_liquidity")     # Wahl wie bisher (choose_strategy)
    assert cl._iv_rv(0.3, None) is None and cl._iv_rv(0, 0.2) is None and cl._iv_rv("x", 0.2) is None


IN_WINDOW = datetime(2026, 10, 28, 16, 15, tzinfo=timezone.utc)


def _rec(symbol, bid, ask):
    ts = IN_WINDOW.isoformat()
    return {"record_id": hashlib.sha256(f"{symbol}{ts}".encode()).hexdigest()[:20], "occ_symbol": symbol,
            "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4), "quote_time": ts, "retrieved_at": ts,
            "multiplier": 100, "quality_status": "OK", "quality_flags": []}


def test_counterfactual_is_marked_under_its_own_prefix(monkeypatch, tmp_path):
    root = tmp_path / "quotes"
    root.mkdir()
    (root / "2026-10-28.jsonl").write_text("".join(json.dumps(_rec(s, b, a)) + "\n"
                                                   for s, b, a in (("L", 6.0, 6.2), ("S", 2.0, 2.2))))
    monkeypatch.setattr(oq, "ROOT", root)
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda ts: "closed")
    monkeypatch.setattr(cl, "_now_utc", lambda: datetime(2026, 10, 28, 23, 0, tzinfo=timezone.utc))
    spec = {"strategy": "BULL_CALL_SPREAD", "entry_pending": False, "entry_date": "2026-10-08", "expiry": "2027-04-16",
            "net_debit_entry": 3.0, "net_mid_entry": 2.8, "legs": [{"symbol": "L", "side": "long", "strike": 195.0},
                                                                    {"symbol": "S", "side": "short", "strike": 210.0}]}
    row = {"ticker": "PTC", "direction": "BULLISH", "outcomes": {}, "real_spread_cf": spec,
           "real_strategy": {**spec, "strategy": "LONG_CALL", "legs": spec["legs"][:1], "net_debit_entry": 5.0,
                             "net_mid_entry": 4.9, "entry_pending": True}}
    for key, prefix in cl.STRATEGY_SPECS:
        cl._fill_real_strategy_marks([row], datetime(2026, 10, 28, 23, 0), key=key, prefix=prefix)  # noqa: DTZ001
    oc = row["outcomes"]
    assert oc["real_spread_cf_ret_20d"] == pytest.approx(round((6.0 - 2.2) / 3.0 - 1, 4))
    assert oc["real_spread_cf_mark_date_20d"] == "2026-10-28"
    assert not any(k.startswith("real_strat_") for k in oc)         # real_strategy wartet noch auf den Entry


def test_v2_entry_and_symbols_treat_the_counterfactual_as_multi_leg():
    row = {"real_spread_cf": {"legs": [{"symbol": "S", "side": "short"}, {"symbol": "L", "side": "long"}]}}
    assert cl._v2_symbols(row, "real_spread_cf") == ["L", "S"]
    assert "real_spread_cf" in cl._ENTRY_KEYS


def test_give_up_checks_the_entry_ask_for_single_contracts():
    """Vorher galt nur real_option als Einzelkontrakt: real_option_alt (Ask vorhanden) wurde als NO_ENTRY_QUOTE
    statt NO_QUOTE aufgegeben."""
    entry = datetime(2026, 10, 1)
    row = {"outcomes": {}, "real_option_alt": {"ask": 2.0, "expiry": "2027-01-15"}}
    assert cl._give_up_marks([(row, 5, entry)], "real_alt", "real_option_alt", datetime(2026, 11, 7)) is True
    assert row["outcomes"][cl._mark_status_key("real_alt", 5)] == "NO_QUOTE"


# ── gepaarter Regelvergleich ──────────────────────────────────────────────────

def _census_row(day, i, spread_choice, ivrv, long_ret, spread_ret, mid_long, mid_spread):
    rs = {"strategy": "BULL_CALL_SPREAD" if spread_choice else "LONG_CALL", "replicated_iv_rank": 50.0 + i,
          "iv_rank_components": {"iv_rv_21": ivrv}, "net_debit_entry": 1.2, "net_mid_entry": 1.0}
    oc = {"real_opt_ret_5d": long_ret, "real_opt_ret_mid_5d": mid_long,
          "real_strat_ret_5d": spread_ret if spread_choice else long_ret,
          "real_strat_ret_mid_5d": mid_spread if spread_choice else mid_long}
    row = {"date": day, "ticker": f"T{i}", "signal_id": f"{day}{i}", "real_strategy": rs,
           "real_option": {"iv": 0.2 + 0.01 * i, "ask": 1.05, "mid": 1.0}, "outcomes": oc}
    if not spread_choice:
        row["real_spread_cf"] = {"strategy": "BULL_CALL_SPREAD", "net_debit_entry": 1.2, "net_mid_entry": 1.0}
        oc["real_spread_cf_ret_5d"], oc["real_spread_cf_ret_mid_5d"] = spread_ret, mid_spread
    return row


def test_structure_comparison_pairs_rules_on_identical_rows():
    rows = []
    for d in range(12):
        day = (date(2026, 10, 1) + timedelta(days=d)).isoformat()
        for i in range(4):
            # Spread ausführbar 20 pp schlechter, zum Mid gleich; IV/RV < 1 bei zwei von vier Zeilen
            rows.append(_census_row(day, i, spread_choice=i % 2 == 0, ivrv=0.8 if i < 2 else 1.3,
                                    long_ret=0.0, spread_ret=-0.2, mid_long=0.05, mid_spread=0.05))
    rows.append(dict(rows[0]))                                     # Dublette (Aktie, Tag) zählt einmal
    res = se.structure_comparison(rows, horizons=(5,))
    ex, mid = res["horizons"]["5d"]["executable"], res["horizons"]["5d"]["mid"]
    assert ex["production_minus_long_only"]["n"] == 48 and ex["production_minus_long_only"]["verdict"] == "A_WORSE"
    assert ex["production_minus_long_only"]["mean_daily_diff"] == pytest.approx(-0.1)
    assert ex["ivrv_minus_long_only"]["mean_daily_diff"] == pytest.approx(-0.1)    # Spread bei IV/RV >= 1
    assert mid["production_minus_long_only"]["verdict"] == "NO_SIGNIFICANT_DIFFERENCE"
    assert res["rows_with_ivrv"] == 48 and res["rows_with_spread_counterfactual"] == 24
    assert res["entry_cost_vs_mid_median"] == {"spread": pytest.approx(0.2), "long": pytest.approx(0.05)}
    assert res["iv_rank_vs_contract_iv"]["rank_corr"] == pytest.approx(1.0)


def test_structure_comparison_needs_data_before_any_verdict():
    rows = [_census_row("2026-10-01", i, True, 1.2, 0.0, -0.2, 0.0, 0.0) for i in range(5)]
    res = se.structure_comparison(rows, horizons=(5,))
    assert {v["verdict"] for v in res["horizons"]["5d"]["executable"].values()} == {"NEED_MORE_DATA"}
