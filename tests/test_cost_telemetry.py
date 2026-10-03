"""Tests: Kosten-Telemetrie, Budget-Guards, Analyse-Cache, Monatsreport-Abschnitt KOSTEN & EFFIZIENZ."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from modules import analysis_cache as ac
from modules import cost_telemetry as ct

SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"


def _resp(inp=1000, out=500, cr=0, cw=0, text='{"ok": 1}'):
    return SimpleNamespace(usage=SimpleNamespace(input_tokens=inp, output_tokens=out,
                                                 cache_read_input_tokens=cr, cache_creation_input_tokens=cw),
                           content=[SimpleNamespace(text=text, type="text")], model=SONNET, stop_reason="end_turn")


class _Client:
    def __init__(self, resp=None, exc=None):
        self.calls = []
        self.messages = self
        self._resp, self._exc = resp, exc

    def create(self, **kw):
        self.calls.append(kw)
        if self._exc:
            raise self._exc
        return self._resp


def _llm(ts, model=SONNET, workflow="deep_analysis", cost=None, success=True, **kw):
    usage = {"input_tokens": kw.pop("inp", 1000), "output_tokens": kw.pop("out", 500),
             "cache_read_input_tokens": kw.pop("cr", 0), "cache_creation_input_tokens": 0}
    c = ct.compute_cost(model, usage) if cost is None and success else cost
    return {"ts": ts, "kind": "llm", "workflow": workflow, "stage": workflow, "scope": ct.scope_of(workflow),
            "model": model, "family": ct.model_family(model), **usage, "cost_usd": c, "success": success,
            "cache_savings_usd": ct.cache_savings(model, usage), **kw}


# ── Kostenrechnung ──────────────────────────────────────────────────────────
def test_compute_cost_official_prices():
    # Sonnet 4.6: 3 $/MTok input, 15 $/MTok output
    assert ct.compute_cost(SONNET, {"input_tokens": 1_000_000, "output_tokens": 0}) == pytest.approx(3.0)
    assert ct.compute_cost(SONNET, {"input_tokens": 0, "output_tokens": 1_000_000}) == pytest.approx(15.0)
    assert ct.compute_cost(HAIKU, {"input_tokens": 1_000_000, "output_tokens": 1_000_000}) == pytest.approx(6.0)
    # Batch 50 % auf input/output
    assert ct.compute_cost(SONNET, {"input_tokens": 1_000_000, "output_tokens": 0}, batch=True) == pytest.approx(1.5)


def test_cache_tokens_priced_separately():
    u = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 1_000_000,
         "cache_creation_input_tokens": 1_000_000}
    assert ct.compute_cost(SONNET, u) == pytest.approx(0.30 + 3.75)
    # Ersparnis: Lesen spart 2.70 $, Schreiben kostet 0.75 $ Aufschlag
    assert ct.cache_savings(SONNET, u) == pytest.approx(2.70 - 0.75)


def test_unknown_model_or_missing_usage_is_none_not_zero():
    assert ct.compute_cost("gpt-x", {"input_tokens": 10, "output_tokens": 10}) is None
    assert ct.compute_cost(SONNET, {"input_tokens": None, "output_tokens": 10}) is None
    assert ct.usage_from_response(SimpleNamespace()) == {f: None for f in ct.USAGE_FIELDS}


# ── tracked_create ─────────────────────────────────────────────────────────
def test_tracked_create_records_usage_and_passes_result():
    client = _Client(_resp(inp=2000, out=900, cr=100))
    r = ct.tracked_create(client, workflow="deep_analysis", ticker="AAPL", model=SONNET, max_tokens=10,
                          messages=[])
    assert r is client._resp and client.calls[0]["model"] == SONNET
    rows = ct.load_ledger()
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "AAPL" and row["scope"] == "production" and row["family"] == "SONNET"
    assert row["input_tokens"] == 2000 and row["cache_read_input_tokens"] == 100 and row["success"] is True
    assert row["cost_usd"] == pytest.approx((2000 * 3 + 900 * 15 + 100 * 0.3) / 1e6)


def test_tracked_create_failed_call_recorded_and_reraised():
    client = _Client(exc=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        ct.tracked_create(client, workflow="prescreening", model=HAIKU, messages=[])
    row = ct.load_ledger()[0]
    assert row["success"] is False and row["error"] == "RuntimeError" and row["cost_usd"] is None


def test_telemetry_failure_never_breaks_call(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(ct, "LEDGER_DIR", blocker / "sub")      # nicht beschreibbar
    client = _Client(_resp())
    assert ct.tracked_create(client, workflow="deep_analysis", model=SONNET, messages=[]) is client._resp


# ── Aggregation / Monatsgrenzen ────────────────────────────────────────────
def _write(rows):
    for r in rows:
        ct.record(dict(r))


def test_aggregation_by_family_workflow_scope_week():
    _write([_llm("2026-10-05T13:00:00+00:00"), _llm("2026-10-05T13:01:00+00:00", HAIKU, "prescreening"),
            _llm("2026-10-06T13:00:00+00:00", HAIKU, "shadow_relation"),
            _llm("2026-10-07T03:00:00+00:00", "claude-opus-5-5", "hypothesis_factory"),
            _llm("2026-10-07T13:00:00+00:00", success=False, cost=None)])
    agg = ct.aggregate(ct.load_ledger())
    t = agg["totals"]
    assert t["calls"] == 5 and t["failed"] == 1 and t["unknown_cost_calls"] == 0
    assert agg["by"]["family"]["SONNET"]["calls"] == 2 and agg["by"]["family"]["OPUS"]["calls"] == 1
    assert set(agg["by"]["scope"]) == {"production", "shadow", "research"}
    assert list(agg["by"]["week"]) == ["2026-W41"]
    assert t["cost_usd"] == pytest.approx(sum(v["cost_usd"] for v in agg["by"]["workflow"].values()), abs=1e-3)


def test_month_boundaries_and_previous_month(tmp_path):
    _write([_llm("2025-12-31T23:59:00+00:00"), _llm("2026-01-01T00:00:00+00:00"),
            _llm("2026-01-31T23:00:00+00:00"), _llm("2026-02-01T00:00:00+00:00")])
    assert sorted(p.name for p in ct.LEDGER_DIR.glob("*.jsonl")) == [
        "ledger-2025-12.jsonl", "ledger-2026-01.jsonl", "ledger-2026-02.jsonl"]
    s = ct.month_summary("2026-01", reports_dir=tmp_path)
    one = ct.compute_cost(SONNET, {"input_tokens": 1000, "output_tokens": 500})
    assert s["telemetry_calls"] == 2 and s["cost_usd"] == pytest.approx(2 * one)
    assert s["prev_month"] == "2025-12" and s["prev_cost_usd"] == pytest.approx(one)
    assert s["delta_usd"] == pytest.approx(one) and s["delta_pct"] == pytest.approx(1.0)


def test_missing_telemetry_marks_incomplete(tmp_path):
    (tmp_path / "2026-03-02.json").write_text(json.dumps({"stats": {"prescreened": 5, "analyzed": 3, "trades": 0}}))
    s = ct.month_summary("2026-03", reports_dir=tmp_path)
    assert s["telemetry_calls"] == 0 and s["cost_usd"] is None and s["complete"] is False
    assert s["cost_per_scan"] is None and s["delta_usd"] is None
    _write([_llm("2026-03-02T13:00:00+00:00")])
    (tmp_path / "2026-03-03.json").write_text(json.dumps({"stats": {"prescreened": 5, "trades": 1}}))
    s = ct.month_summary("2026-03", reports_dir=tmp_path)
    assert s["days_without_telemetry"] == 1 and s["complete"] is False
    assert s["scan_runs"] == 2 and s["final_trades"] == 1 and s["cost_per_final_trade"] == s["cost_usd"]


def test_corrupt_ledger_lines_skipped():
    ct.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    (ct.LEDGER_DIR / "ledger-2026-04.jsonl").write_text("not json\n" + json.dumps(_llm("2026-04-02T10:00:00+00:00")) + "\n")
    assert len(ct.load_ledger()) == 1


# ── Budgets / Guards ───────────────────────────────────────────────────────
def _spend(usd, ts="2026-10-06T10:00:00+00:00", workflow="deep_analysis"):
    ct.record({"ts": ts, "kind": "llm", "workflow": workflow, "scope": ct.scope_of(workflow), "model": SONNET,
               "cost_usd": usd, "success": True})


NOW = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)


def test_budget_ok_warn_exceeded_and_throttle_order():
    pol = ct.policy()
    wk = float(pol["budgets"]["weekly_llm_usd"])
    assert ct.budget_status(NOW)["level"] == "OK"
    _spend(wk * 0.85)
    st = ct.budget_status(NOW)
    assert st["level"] == "WARN" and st["throttled_scopes"] == ["research"] and st["warnings"]
    assert ct.allow("hypothesis_factory", NOW) is False      # Research zuerst
    assert ct.allow("shadow_relation", NOW) is True
    assert ct.allow("deep_analysis", NOW) is True            # Produktion nie
    _spend(wk * 0.3)
    st = ct.budget_status(NOW)
    assert st["level"] == "EXCEEDED" and "shadow" in st["throttled_scopes"]
    assert ct.allow("shadow_relation", NOW) is False
    assert ct.allow("deep_analysis", NOW) is True and ct.allow("prescreening", NOW) is True
    assert ct.production_warning(NOW)


def test_budget_week_resets_but_month_accumulates():
    pol = ct.policy()
    _spend(float(pol["budgets"]["weekly_llm_usd"]) * 0.9, ts="2026-10-01T10:00:00+00:00")   # Vorwoche
    st = ct.budget_status(NOW)
    assert st["week_usd"] == 0 and st["month_usd"] > 0


# ── Fremd-API-Zähler ────────────────────────────────────────────────────────
def test_api_counts_flush_quota():
    ct._HOST_MAP.update({"newsapi.org": "newsapi"})
    ct._API_COUNTS["newsapi"]["requests"] += 50
    ct._API_COUNTS["newsapi"]["rate_limited"] += 1
    rows = ct.flush_api_counts("scanner")
    assert rows[0]["provider"] == "newsapi" and rows[0]["quota_usage"] == 0.5 and rows[0]["cost_usd"] is None
    assert not ct._API_COUNTS
    agg = ct.aggregate(ct.load_ledger())
    assert agg["api"]["newsapi"]["requests"] == 50 and agg["totals"]["calls"] == 0
    assert ct._paid_api_cost(agg["api"]) == 0.0
    assert ct._paid_api_cost({"tradier": {"requests": 3}}) is None     # unbekannter Plan -> nicht verfügbar


# ── Analyse-Cache ──────────────────────────────────────────────────────────
def _key(**over):
    base = dict(ticker="AAPL", news=["Apple beats estimates", "Apple beats estimates!"], move_48h=0.0123,
                mc_hit_rate=0.61, eps_deviation="1.0%", earnings_date="2026-10-30", macro_text="Regime X",
                sector="Tech", prescreen_category="earnings", is_mega_cap=True, data_anomaly=False,
                prompt_version="p1", model=SONNET)
    base.update(over)
    return ac.deep_analysis_key(**base)


def test_news_clustering_dedups_near_duplicates():
    assert ac.cluster_headlines(["Apple beats estimates", "APPLE beats estimates!", "FDA approves drug"]) == [
        "Apple beats estimates", "FDA approves drug"]


def test_cache_key_invalidation():
    k = _key()
    assert k == _key(news=["Apple beats estimates"])           # Duplikat ändert nichts
    assert k == _key(move_48h=0.0149)                          # gleicher 1-%-Bucket
    for change in (dict(news=["Apple misses"]), dict(move_48h=0.05), dict(earnings_date="2026-11-01"),
                   dict(macro_text="Regime Y"), dict(prompt_version="p2"), dict(model=HAIKU)):
        assert _key(**change) != k, change


def _set_mode(monkeypatch, mode):
    pol = dict(ct.policy())
    pol["analysis_cache"] = {**pol.get("analysis_cache", {}), "mode": mode}
    monkeypatch.setattr(ct, "_POLICY_CACHE", pol)


def test_cache_observe_never_returns_and_compares(monkeypatch):
    _set_mode(monkeypatch, "observe")
    k = _key()
    assert ac.lookup(k, workflow="deep_analysis") is None
    ac.store(k, {"direction": "BULLISH", "impact": 6, "red_team": {"red_team_verdict": "PASSIERT"}},
             cost_usd=0.02, model=SONNET)
    assert ac.lookup(k, workflow="deep_analysis") is None      # observe: Sonnet läuft trotzdem
    row = ac.compare_observed(k, {"direction": "BULLISH", "impact": 7,
                                  "red_team": {"red_team_verdict": "PASSIERT"}}, workflow="deep_analysis")
    assert row["direction_equal"] and row["verdict_equal"] and row["impact_abs_diff"] == 1
    agg = ct.aggregate(ct.load_ledger())
    assert agg["analysis_cache"]["would_hit"] == 1 and agg["analysis_cache"]["potential_saved_usd"] == 0.02


def test_cache_active_returns_copy_and_ttl(monkeypatch):
    _set_mode(monkeypatch, "active")
    k = _key()
    t0 = datetime(2026, 10, 6, 13, tzinfo=timezone.utc)
    ac.store(k, {"impact": 5}, cost_usd=0.02, model=SONNET, now=t0)
    hit = ac.lookup(k, workflow="deep_analysis", now=t0.replace(hour=20))
    assert hit == {"impact": 5}
    hit["impact"] = 9
    assert ac.lookup(k, workflow="deep_analysis", now=t0.replace(hour=21)) == {"impact": 5}
    assert ac.lookup(k, workflow="deep_analysis", now=datetime(2026, 10, 8, 13, tzinfo=timezone.utc)) is None


def test_cache_activation_report():
    rows = [{"kind": "cache_check", "direction_equal": True, "verdict_equal": True, "impact_abs_diff": 0.5}] * 30
    assert ac.activation_report(rows)["decision"] == "KEEP"
    assert ac.activation_report(rows[:5])["decision"] == "PENDING"
    bad = rows[:25] + [{"kind": "cache_check", "direction_equal": False, "verdict_equal": True,
                        "impact_abs_diff": 3}] * 5
    assert ac.activation_report(bad)["decision"] == "REJECT"


def test_deep_analysis_uses_telemetry_cache_and_prompt_caching(monkeypatch):
    from modules import deep_analysis as da
    _set_mode(monkeypatch, "active")
    obj = da.DeepAnalysis.__new__(da.DeepAnalysis)
    obj._macro = {"data_available": False}
    payload = {"red_team": {"argument_1": "x", "red_team_verdict": "PASSIERT"}, "impact": 5, "surprise": 4,
               "direction": "BULLISH", "time_to_materialization": "2-3 Monate"}
    obj.client = _Client(_resp(text=json.dumps(payload)))
    monkeypatch.setattr(obj, "_get_48h_move", lambda t: 0.02)
    cand = {"ticker": "MSFT", "info": {"currentPrice": 400, "marketCap": 3e12, "sector": "Tech"},
            "news": ["Microsoft signs deal"], "quick_mc": {"hit_rate": 0.6, "n_paths": 1000, "n_days": 30}}
    r1 = obj._analyze(dict(cand))
    r2 = obj._analyze(dict(cand))
    assert r1["impact"] == 5 and r2["impact"] == 5
    assert len(obj.client.calls) == 1                          # zweiter Lauf aus dem Cache
    sys_blocks = obj.client.calls[0]["system"]
    assert sys_blocks[0]["cache_control"] == {"type": "ephemeral"}
    llm = [r for r in ct.load_ledger() if r["kind"] == "llm"]
    assert len(llm) == 1 and llm[0]["workflow"] == "deep_analysis" and llm[0]["ticker"] == "MSFT"


# ── Monatsreport ───────────────────────────────────────────────────────────
def test_monthly_report_section_with_and_without_telemetry(tmp_path, monkeypatch):
    import monthly_report as mr
    empty = mr.build_cost_html("2026-05", ct.month_summary("2026-05", reports_dir=tmp_path))
    assert "KOSTEN &amp; EFFIZIENZ" in empty and "unvollständig" in empty and "nicht verfügbar" in empty
    _write([_llm("2026-04-10T13:00:00+00:00"), _llm("2026-05-10T13:00:00+00:00"),
            _llm("2026-05-10T13:01:00+00:00", HAIKU, "prescreening"),
            _llm("2026-05-11T03:00:00+00:00", "claude-opus-5-5", "hypothesis_factory")])
    (tmp_path / "2026-05-10.json").write_text(json.dumps({"stats": {"prescreened": 2, "analyzed": 1, "trades": 1}}))
    s = ct.month_summary("2026-05", reports_dir=tmp_path)
    html = mr.build_cost_html("2026-05", s)
    assert "unvollständig" not in html and s["complete"] is True
    assert "Sonnet-Calls / Haiku-Calls" in html and "<b>1 / 1</b>" in html
    assert "Kosten je finalem Trade" in html and "Größter Kostenblock" in html
    assert "Produktion / Shadow / Research" in html


def test_monthly_report_never_fails_and_is_in_email(monkeypatch):
    import monthly_report as mr

    def boom(*a, **k):
        raise RuntimeError("telemetry kaputt")
    monkeypatch.setattr(ct, "month_summary", boom)
    sec = mr._safe_cost_html("2026-09")
    assert "unvollständig" in sec
    html = mr.build_html("2026-09", None, None, None, {"days": 0, "zero_days": 0, "top_rejects": [], "top_stops": []})
    assert "KOSTEN &amp; EFFIZIENZ 2026-09" in html
    sent = {}
    monkeypatch.setattr(mr, "_send_smtp", lambda subject, html: sent.update(html=html))
    monkeypatch.setattr(mr, "HISTORY_PATH", type("P", (), {"exists": lambda self: True,
                                                          "read_text": lambda self: '{"closed_trades": []}'})())
    for name in ("funnel_summary",):
        monkeypatch.setattr(mr, name, lambda key: {"days": 0, "zero_days": 0, "top_rejects": [], "top_stops": []})
    monkeypatch.setattr(mr, "spy_return", lambda key: None)
    mr.main()
    assert "KOSTEN &amp; EFFIZIENZ" in sent["html"]
