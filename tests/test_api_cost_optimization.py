"""API-Kosten-Optimierung 2026-10-09: Request-Dedup (Cache Hit/Miss/Expiry/Corrupt), Freshness-Klassen,
Production-Critical immer frisch, Budget-Tiers (OPTIONAL/RESEARCH zuerst, Produktion nie), Monatsprojektion,
Kostenmatrix, Äquivalenz (gleiche Expiries/Kontrakte mit und ohne Cache), PIT/Universe unverändert."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest
import yaml

from modules import cost_telemetry as ct
from modules import request_cache as rc

UTC = timezone.utc
T0 = datetime(2026, 10, 9, 14, 0, tzinfo=UTC).timestamp()


class _Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _counting_fetch(values):
    calls = []

    def f():
        calls.append(1)
        v = values[len(calls) - 1] if isinstance(values, list) else values
        if isinstance(v, Exception):
            raise v
        return v
    return f, calls


# ── Request-Cache ────────────────────────────────────────────────────────────────────────────────────────
EXP = ("tradier", "markets/options/expirations")


def test_cache_miss_then_hit_for_identical_request():
    f, calls = _counting_fetch({"expirations": {"date": ["2026-11-20"]}})
    clk = _Clock(T0)
    a = rc.cached(*EXP, {"symbol": "AAA", "includeAllRoots": "true"}, f, now=clk)
    b = rc.cached(*EXP, {"includeAllRoots": "true", "symbol": "AAA"}, f, now=clk)   # Parameter-Reihenfolge egal
    assert a == b and len(calls) == 1
    assert rc.stats()["hit"] == 1 and rc.stats()["miss"] == 1


def test_different_symbol_or_params_is_a_miss():
    f, calls = _counting_fetch({"x": 1})
    rc.cached(*EXP, {"symbol": "AAA"}, f, now=_Clock(T0))
    rc.cached(*EXP, {"symbol": "BBB"}, f, now=_Clock(T0))
    rc.cached(*EXP, {"symbol": "AAA", "includeAllRoots": "true"}, f, now=_Clock(T0))
    assert len(calls) == 3


def test_cache_expiry_and_new_trading_day_invalidate():
    ttl = rc.cache_ttl_s(*EXP)
    assert ttl == 21600
    f, calls = _counting_fetch({"x": 1})
    clk = _Clock(T0)
    rc.cached(*EXP, {"symbol": "AAA"}, f, now=clk)
    clk.t = T0 + ttl + 1                                   # max_staleness überschritten
    rc.cached(*EXP, {"symbol": "AAA"}, f, now=clk)
    assert len(calls) == 2 and rc.stats()["expired"] == 1
    late = datetime(2026, 10, 9, 23, 59, tzinfo=UTC).timestamp()
    next_day = datetime(2026, 10, 10, 0, 1, tzinfo=UTC).timestamp()       # 2 min später, aber neuer Tag
    rc.cached(*EXP, {"symbol": "ZZZ"}, f, now=_Clock(late))
    rc.cached(*EXP, {"symbol": "ZZZ"}, f, now=_Clock(next_day))
    assert len(calls) == 4


def test_corrupt_cache_entry_is_refetched_never_forced():
    f, calls = _counting_fetch([{"v": 1}, {"v": 2}])
    rc.cached(*EXP, {"symbol": "AAA"}, f, now=_Clock(T0))
    rc._STORE[rc.request_key(*EXP, {"symbol": "AAA"})] = {"t": "kaputt"}       # beschädigt
    assert rc.cached(*EXP, {"symbol": "AAA"}, f, now=_Clock(T0)) == {"v": 2}
    assert rc.stats()["corrupt"] == 1 and len(calls) == 2


def test_errors_are_never_cached():
    f, calls = _counting_fetch([RuntimeError("503"), {"v": 1}])
    with pytest.raises(RuntimeError):
        rc.cached(*EXP, {"symbol": "AAA"}, f, now=_Clock(T0))
    assert rc.cached(*EXP, {"symbol": "AAA"}, f, now=_Clock(T0)) == {"v": 1}
    assert len(calls) == 2


def test_cached_value_is_a_copy():
    f, _ = _counting_fetch({"expirations": {"date": ["2026-11-20"]}})
    a = rc.cached(*EXP, {"symbol": "AAA"}, f, now=_Clock(T0))
    a["expirations"]["date"].append("MUTIERT")
    b = rc.cached(*EXP, {"symbol": "AAA"}, f, now=_Clock(T0))
    assert b["expirations"]["date"] == ["2026-11-20"]


def test_targeted_invalidation_for_corporate_action():
    f, calls = _counting_fetch({"x": 1})
    for s in ("AAA", "BBB"):
        rc.cached(*EXP, {"symbol": s}, f, now=_Clock(T0))
    assert rc.invalidate("tradier", symbol="AAA") == 1
    rc.cached(*EXP, {"symbol": "AAA"}, f, now=_Clock(T0))
    rc.cached(*EXP, {"symbol": "BBB"}, f, now=_Clock(T0))
    assert len(calls) == 3


@pytest.mark.parametrize("ep", ["markets/quotes", "markets/options/chains"])
def test_production_critical_quotes_and_chains_always_fetched_fresh(ep):
    assert rc.cache_ttl_s("tradier", ep) == 0
    f, calls = _counting_fetch({"x": 1})
    for _ in range(3):
        rc.cached("tradier", ep, {"symbol": "AAA"}, f, now=_Clock(T0))
    assert len(calls) == 3 and rc.stats()["bypass"] == 3


def test_freshness_classes_cover_all_endpoints_and_unknown_is_fresh():
    cfg = yaml.safe_load(open("config/data_freshness.yaml", encoding="utf-8"))
    classes = {"REALTIME_OR_DAILY_CRITICAL", "DAILY", "EVENT_DRIVEN", "WEEKLY", "MONTHLY", "STATIC_OR_SLOW"}
    tiers = set(ct.TIERS)
    for k, v in {**cfg["endpoints"], **cfg["llm_workflows"]}.items():
        assert v["class"] in classes and v["tier"] in tiers, k
    for k, v in cfg["endpoints"].items():
        if v["class"] == "REALTIME_OR_DAILY_CRITICAL":
            assert v["max_staleness_s"] == 0 and not v["cache_in_run"], k
    assert rc.cache_ttl_s("unknown", "endpoint") == 0                          # im Zweifel frisch


def test_pit_sources_are_never_served_from_request_cache():
    """FRED/ALFRED, EIA, CFTC usw.: PIT-/Release-Lag-Logik bleibt im Archiv; kein Request-Cache."""
    cfg = yaml.safe_load(open("config/data_freshness.yaml", encoding="utf-8"))
    for k in cfg["endpoints"]:
        if k.startswith("external/"):
            assert rc.cache_ttl_s(*k.split("/", 1)) == 0


# ── Verdrahtung: Duplicate Request + Äquivalenz ──────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, js):
        self._js = js

    def raise_for_status(self):
        return None

    def json(self):
        return self._js


def _fake_tradier(monkeypatch, modules_):
    calls = []
    exp = {"expirations": {"date": ["2026-10-16", "2027-02-19", "2027-06-18"]}}
    chain = {"options": {"option": [
        {"symbol": "AAA270219C00100000", "option_type": "call", "strike": 100, "bid": 4.0, "ask": 4.4,
         "open_interest": 900, "greeks": {"mid_iv": 0.31, "delta": 0.55}},
        {"symbol": "AAA270219C00110000", "option_type": "call", "strike": 110, "bid": 1.9, "ask": 2.2,
         "open_interest": 700, "greeks": {"mid_iv": 0.30, "delta": 0.35}}]}}

    def get(url, params=None, headers=None, timeout=None):
        calls.append((url.rsplit("/v1/", 1)[-1], json.dumps(params, sort_keys=True)))
        return _Resp(exp if url.endswith("expirations") else chain)
    for m in modules_:
        monkeypatch.setattr(m.requests, "get", get)
    return calls


def test_duplicate_expirations_request_across_modules_fetched_once(monkeypatch):
    from modules import alpha_sources, market_snapshot, options_designer
    calls = _fake_tradier(monkeypatch, [market_snapshot, options_designer, alpha_sources])
    a = market_snapshot._fetch_expirations("AAA")
    b = options_designer._tradier_expirations("AAA")
    assert a == b == ["2026-10-16", "2027-02-19", "2027-06-18"]
    assert sum(1 for c in calls if c[0] == "markets/options/expirations") == 1
    market_snapshot._fetch_chain("AAA", "2027-02-19")
    options_designer._tradier_chain("AAA", "2027-02-19")
    assert sum(1 for c in calls if c[0] == "markets/options/chains") == 2      # Quotes immer frisch


def test_contract_selection_identical_with_and_without_cache(monkeypatch):
    """Finale Entscheidungs-Äquivalenz: gleicher Kontrakt, gleiche Quotes – Cache spart nur den Request."""
    from modules import market_snapshot
    monkeypatch.setattr(market_snapshot, "_use_tradier", lambda: True)
    _fake_tradier(monkeypatch, [market_snapshot])
    with_cache = [market_snapshot.select_contract("AAA", "BULLISH", 90, 103.0) for _ in range(2)]
    monkeypatch.setattr(rc, "cache_ttl_s", lambda *a: 0.0)                      # Baseline: ohne Cache
    rc.reset()
    baseline = market_snapshot.select_contract("AAA", "BULLISH", 90, 103.0)
    strip = (lambda d: {k: v for k, v in d.items() if k != "quote_ts"})
    assert strip(with_cache[0]) == strip(with_cache[1]) == strip(baseline)
    assert baseline["strike"] == 100.0 and baseline["bid"] == 4.0 and baseline["ask"] == 4.4


def test_underlying_quotes_are_batched_with_per_symbol_handling(monkeypatch):
    from modules import market_snapshot
    monkeypatch.setattr(market_snapshot, "_use_tradier", lambda: True)
    chunks = []

    def raw(symbols):
        chunks.append(list(symbols))
        return [{"symbol": s, "bid": 1.0, "ask": 1.1} for s in symbols if s != "BAD"]
    monkeypatch.setattr(market_snapshot, "_fetch_quotes_raw", raw)
    tickers = [f"T{i}" for i in range(249)] + ["BAD"]
    out = market_snapshot.fetch_underlying_quotes(tickers + ["T1"])              # Duplikat im Input
    assert [len(c) for c in chunks] == [100, 100, 50]
    assert len(out) == 249 and "BAD" not in out                                 # Fehler je Symbol, Rest vollständig


def test_release_driven_delta_fetch_exists_for_weekly_sources():
    """Delta statt Full Refresh: EIA/CFTC fragen nur nach neuer Veröffentlichung (is_due, unverändert)."""
    from modules.external.sources import commodities as c
    for cls in (c.EiaConnector, c.CftcCotConnector):
        assert callable(getattr(cls, "is_due", None))


def test_request_cache_hit_rate_is_recorded_per_run(tmp_path):
    f, _ = _counting_fetch({"x": 1})
    for _ in range(3):
        rc.cached(*EXP, {"symbol": "AAA"}, f, now=_Clock(T0))
    rows = ct.flush_api_counts("scanner", ledger_dir=tmp_path)
    rcrow = next(r for r in rows if r["kind"] == "request_cache")
    assert rcrow["hit"] == 2 and rcrow["miss"] == 1 and rcrow["hit_rate"] == pytest.approx(0.667, abs=1e-3)


# ── Budget-Tiers + Projektion ────────────────────────────────────────────────────────────────────────────
def _ledger(tmp_path, per_day: float, days: list[date], wf="deep_analysis", scope="production"):
    p = tmp_path / f"ledger-{days[0]:%Y-%m}.jsonl"
    with p.open("a") as f:
        for d in days:
            f.write(json.dumps({"ts": f"{d.isoformat()}T19:00:00+00:00", "kind": "llm", "workflow": wf, "stage": wf,
                                "scope": scope, "model": "claude-sonnet-4-6", "cost_usd": per_day,
                                "success": True}) + "\n")
    return tmp_path


def test_projection_math():
    rows = [{"ts": f"2026-10-0{d}T19:00:00+00:00", "kind": "llm", "workflow": "deep_analysis", "stage": "deep_analysis",
             "scope": "production", "model": "m", "cost_usd": 1.0} for d in (6, 7, 8)]
    p = ct.projection(rows, date(2026, 10, 9))
    # MTD 3 $ + 1 $/Scanner-Tag x 16 verbleibende Handelstage (9.10. noch ohne Lauf + 15)
    assert p["cost_month_to_date_usd"] == 3.0 and p["remaining_scanner_days"] == 16
    assert p["projected_month_usd"] == 19.0 and p["cost_today_usd"] == 0.0
    assert p["by_module"] == {"deep_analysis": 3.0}
    assert ct.projection([], date(2026, 10, 9))["projected_month_usd"] is None   # ohne Messung nie 0


def test_optional_research_suppressed_under_budget_pressure_production_never(tmp_path, monkeypatch):
    days = [date(2026, 10, d) for d in (6, 7, 8)]
    _ledger(tmp_path, 1.3, days)
    monkeypatch.setattr(ct, "LEDGER_DIR", tmp_path)
    monkeypatch.setattr(ct, "_now", lambda: datetime(2026, 10, 9, 12, tzinfo=UTC))
    monkeypatch.setattr(ct, "_freshness", lambda: {"llm_workflows": {
        "opt_wf": {"tier": "TIER4_OPTIONAL"}, "res_wf": {"tier": "TIER3_RESEARCH"},
        "model_routing_ab": {"tier": "TIER2_IMPORTANT"}, "deep_analysis": {"tier": "TIER1_PRODUCTION_REQUIRED"}}})
    st = ct.budget_status()
    assert st["soft_pressure"] is True and st["projected_month_usd"] > 10
    assert st["deferred_tiers"] == ["TIER4_OPTIONAL"] and st["level"] == "OK"
    assert ct.allow("opt_wf") is False                                       # zuerst OPTIONAL
    assert ct.allow("res_wf") is True                                        # RESEARCH erst an harter Grenze
    assert ct.allow("deep_analysis") is True and ct.allow("model_routing_ab") is True


def test_research_deferred_only_when_hard_budget_exceeded_production_still_runs(tmp_path, monkeypatch):
    _ledger(tmp_path, 9.0, [date(2026, 10, d) for d in (1, 2, 5)])           # 27 $ > 25 $ Monatsgrenze
    monkeypatch.setattr(ct, "LEDGER_DIR", tmp_path)
    monkeypatch.setattr(ct, "_now", lambda: datetime(2026, 10, 9, 12, tzinfo=UTC))
    monkeypatch.setattr(ct, "_freshness", lambda: {"llm_workflows": {"res_wf": {"tier": "TIER3_RESEARCH"}}})
    st = ct.budget_status()
    assert st["level"] == "EXCEEDED" and "TIER3_RESEARCH" in st["deferred_tiers"]
    assert ct.allow("res_wf") is False
    for wf in ("deep_analysis", "prescreening"):                             # Produktion nie gedrosselt
        assert ct.allow(wf) is True
    assert ct.production_warning() is not None                               # aber sichtbar gewarnt


def test_cost_matrix_sorted_by_share_and_honest_about_unknown_plans():
    rows = [
        {"kind": "llm", "run_id": "r1", "workflow": "deep_analysis", "stage": "deep_analysis", "ticker": "A",
         "model": "claude-sonnet-4-6", "cost_usd": 0.013, "input_tokens": 1400, "output_tokens": 1100, "batch": True},
        {"kind": "llm", "run_id": "r1", "workflow": "deep_analysis", "stage": "deep_analysis", "ticker": "A",
         "model": "claude-haiku-4-5-20251001", "cost_usd": 0.005, "stop_reason": "max_tokens", "batch": True},
        {"kind": "llm", "run_id": "r1", "workflow": "prescreening", "stage": "prescreening",
         "model": "claude-haiku-4-5-20251001", "cost_usd": 0.006, "batch": True},
        {"kind": "api", "run_id": "r1", "provider": "finnhub", "requests": 600, "rate_limited": 20},
        {"kind": "api", "run_id": "r1", "provider": "tradier", "requests": 1100},
    ]
    m = ct.cost_matrix(rows)
    assert [r["consumer"] for r in m["rows"][:3]] == ["deep_analysis", "prescreening", "model_routing_ab"]
    shares = [r["share"] for r in m["rows"] if r["provider"] == "anthropic"]
    assert shares == sorted(shares, reverse=True) and abs(sum(shares) - 1.0) < 1e-6
    ab = next(r for r in m["rows"] if r["consumer"] == "model_routing_ab")
    assert ab["tier"] == "TIER2_IMPORTANT" and ab["truncated"] == 1
    fin = next(r for r in m["rows"] if r["provider"] == "finnhub")
    tra = next(r for r in m["rows"] if r["provider"] == "tradier")
    assert fin["cost_per_run_usd"] == 0.0 and fin["rate_limited_429"] == 20
    assert tra["cost_per_run_usd"] is None                                   # Plan unbekannt -> nie geraten


# ── Unverändert: Universe, Gates, Produktionspfad ────────────────────────────────────────────────────────
def test_universe_and_decision_policies_unchanged():
    from modules import universe_v2
    assert universe_v2.v1_unchanged() is True
    pol = yaml.safe_load(open("config/cost_policy.yaml", encoding="utf-8"))
    assert pol["scopes"]["deep_analysis"] == pol["scopes"]["prescreening"] == "production"
    assert pol["budgets"] == {"weekly_llm_usd": 6.0, "monthly_llm_usd": 25.0, "warn_fraction": 0.80,
                              "throttle_order": ["research", "shadow"]}
    assert pol["bearish_prefilter"]["max_pass_loss"] == 0.05 and pol["analysis_cache"]["mode"] == "auto"
    assert pol["model_routing"]["deep_analysis"]["min_gate_agreement"] == 0.92
