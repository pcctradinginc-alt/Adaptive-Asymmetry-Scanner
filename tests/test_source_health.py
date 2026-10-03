"""Täglicher Source Health Check: API-Ausfall, Timeout, Rate Limit, falsches Schema, leere
Antwort, veraltete Daten (frequenzabhängig), teilweise fehlende Daten, Cache-Fallback,
Provider-Fallback, Recovery (nicht sofort HEALTHY), Safe Mode, Mail nur bei Änderung,
Scanner liest den Snapshot und nimmt nie Verfügbarkeit an."""
from __future__ import annotations

import json
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from modules import source_health as sh
from modules.external.http import AuthError, FetchError

NOW = datetime(2026, 10, 5, 12, 40, tzinfo=timezone.utc)


def cfg(core=None, alt=None, cache=None):
    base = sh.load_config()
    return {**base, "core_sources": core if core is not None else [], "alt_sources": alt or [],
            "cache_max_age_days": cache or {}, "downstream_extra": {}}


def resp(payload, status=200):
    return types.SimpleNamespace(json=lambda: payload, status=status, content=json.dumps(payload).encode())


def fetch_ok(payload):
    return lambda url, **kw: resp(payload)


def fetch_raise(exc):
    def f(url, **kw):
        raise exc
    return f


JSON_SPEC = {"kind": "http_json", "url": "https://x", "required_keys": ["data"]}


# ── Probes ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("fetch,expect", [
    (fetch_ok({"data": [1]}), {"ok": True, "schema_ok": True, "empty": False}),
    (fetch_raise(FetchError("HTTPSConnectionPool: Max retries exceeded (ConnectionError)")), {"reachable": False}),
    (fetch_raise(FetchError("ReadTimeout: read timed out")), {"timeout": True, "reachable": False}),
    (fetch_raise(FetchError("429 für https://x")), {"rate_limited": True, "http_status": 429}),
    (fetch_raise(AuthError("401 für https://x")), {"auth_ok": False}),
    (fetch_ok({"results": []}), {"schema_ok": False}),
    (fetch_ok({"data": []}), {"empty": True, "ok": False}),
])
def test_probe_classifies_failures(fetch, expect):
    r = sh.run_probe(JSON_SPEC, NOW, fetch=fetch)
    for k, v in expect.items():
        assert r[k] == v, (k, r)


def test_probe_auth_missing_never_calls_and_secret_redacted():
    called = []
    r = sh.run_probe({**JSON_SPEC, "auth_env": "XKEY"}, NOW, fetch=lambda u, **k: called.append(1), env={})
    assert r["auth_missing"] and r["auth_ok"] is False and not called
    r2 = sh.run_probe({**JSON_SPEC, "auth_env": "XKEY", "auth_param": "token"}, NOW, env={"XKEY": "s3cr3t"},
                      fetch=fetch_raise(FetchError("500 für https://x?token=s3cr3t")))
    assert "s3cr3t" not in r2["error"]


def test_probe_uses_bounded_exponential_backoff():
    seen = {}

    def f(url, **kw):
        seen.update(kw)
        return resp({"data": [1]})
    sh.run_probe(JSON_SPEC, NOW, fetch=f)
    assert seen["retries"] == 3 and seen["backoff"] == 2.0


def test_yfinance_probe_stale_partial_and_empty():
    idx = pd.to_datetime(["2026-09-24", "2026-09-25", "2026-09-26"]).tz_localize("UTC")
    yf = types.SimpleNamespace(Ticker=lambda s: types.SimpleNamespace(
        history=lambda period: pd.DataFrame({"Close": [1.0, np.nan, 2.0]}, index=idx), options=()))
    r = sh.run_probe({"kind": "yfinance_history", "symbol": "SPY"}, NOW, yf_module=yf)
    assert r["ok"] and r["missing_rate"] == pytest.approx(0.333, abs=1e-3) and r["latest_observation"].startswith("2026-09-26")
    e = sh.run_probe({"kind": "yfinance_options", "symbol": "SPY"}, NOW, yf_module=yf)
    assert e["empty"] and not e["ok"]


# ── Klassifikation ─────────────────────────────────────────────────────────
def core(sid="market_prices", **kw):
    return {"source_id": sid, "display_name": sid, "criticality": "CRITICAL",
            "probe": {"kind": "yfinance_history", "symbol": "SPY"}, "max_observation_age_days": 5,
            "decisions": ["scanner_candidates"], "features": ["price_history"], **kw}


def snap_for(checks, prev=None, c=None):
    return sh.build_snapshot(checks, prev, c or cfg(core=[core()]), NOW, deps=sh.downstream(c or cfg(core=[core()])))


def chk(sid="market_prices", probe=None, latest="2026-10-02T00:00:00+00:00", **kw):
    p = probe if probe is not None else sh._probe_result(ok=True, reachable=True, auth_ok=True, schema_ok=True,
                                                         empty=False, latency_s=0.5, latest_observation=latest)
    return {"source_id": sid, "kind": "core", "criticality": "CRITICAL", "probe": p, "latest_observation":
            p.get("latest_observation"), "last_success": NOW.isoformat() if p.get("ok") else None,
            "max_observation_age_days": 5, "missing_rate": p.get("missing_rate"), **kw}


def test_status_mapping_for_failure_modes():
    c = cfg(core=[core()])
    healthy = sh.classify(chk(), None, c, NOW)
    assert healthy["status"] == sh.HEALTHY
    stale = sh.classify(chk(latest="2026-09-20T00:00:00+00:00"), None, c, NOW)
    assert stale["status"] == sh.STALE                                                # 15 T > 5 T
    rl = sh.classify(chk(probe=sh._probe_result(reachable=True, rate_limited=True, error="429")), None, c, NOW)
    assert rl["status"] == sh.DEGRADED
    to = sh.classify(chk(probe=sh._probe_result(reachable=False, timeout=True, error="timeout")), None, c, NOW)
    assert to["status"] == sh.DEGRADED and to["failed"]
    down = sh.classify(chk(probe=sh._probe_result(reachable=False, error="conn")), None, c, NOW)
    assert down["status"] == sh.BROKEN
    schema = sh.classify(chk(probe=sh._probe_result(reachable=True, auth_ok=True, schema_ok=False, error="x")), None, c, NOW)
    assert schema["status"] == sh.BROKEN
    empty = sh.classify(chk(probe=sh._probe_result(reachable=True, auth_ok=True, schema_ok=True, empty=True,
                                                   error="leer")), None, c, NOW)
    assert empty["status"] == sh.DEGRADED
    part = sh.classify(chk(probe={**sh._probe_result(ok=True, reachable=True, auth_ok=True, schema_ok=True, empty=False,
                                                     latest_observation="2026-10-02T00:00:00+00:00"),
                                  "missing_rate": 0.4}), None, c, NOW)
    assert part["status"] == sh.DEGRADED and any("Missing" in r for r in part["reasons"])


def test_repeated_failures_escalate_to_broken():
    c = cfg(core=[core()])
    prev = None
    for i in range(3):
        r = sh.classify(chk(probe=sh._probe_result(reachable=True, rate_limited=True, error="429")), prev, c, NOW)
        prev = {**r, "status": r["status"]}
    assert r["consecutive_failures"] == 3 and r["status"] == sh.BROKEN


def test_monthly_source_is_not_stale_after_a_day():
    """Registry-Quelle (monatlich): 40 Tage alte Beobachtung ist normal, 200 Tage nicht."""
    contracts = {"m": {"source_id": "m", "enabled": True, "frequency": "monthly", "expected_update_cadence": "P1M",
                       "criticality": "medium", "license_status": "OK"}}
    rh = {"m": {"status": "PASS", "last_success": "2026-10-05T06:00:00+00:00",
                "latest_observation": (NOW - timedelta(days=40)).isoformat()}}
    c = cfg()
    ok = sh.build_snapshot(sh.gather(c, NOW, probes=False, registry_health=rh, contracts=contracts), None, c, NOW, deps={})
    assert ok["sources"]["m"]["status"] == sh.HEALTHY
    rh["m"]["latest_observation"] = (NOW - timedelta(days=200)).isoformat()
    old = sh.build_snapshot(sh.gather(c, NOW, probes=False, registry_health=rh, contracts=contracts), None, c, NOW, deps={})
    assert old["sources"]["m"]["status"] == sh.STALE


def test_store_checks_duplicates_outliers_and_pit_violation():
    c = cfg()
    df = pd.DataFrame({"series_id": ["a", "a", "b", "c"] * 10, "metric": "m",
                       "value": [1.0] * 39 + [1e9], "available_at": "2026-10-01T00:00:00+00:00",
                       "retrieved_at": "2026-10-02T00:00:00+00:00", "observation_time": "2026-09-30T00:00:00+00:00"})
    st = sh.frame_stats(df, NOW, c)
    assert st["duplicate_rate"] > 0.5 and st["future_timestamps"] == 0
    bad = df.assign(available_at="2026-10-09T00:00:00+00:00")                        # Verfügbarkeit nach Abruf
    st2 = sh.frame_stats(bad, NOW, c)
    r = sh.classify({"source_id": "s", "kind": "alt", "criticality": "NON_CRITICAL", "store": st2,
                     "last_success": NOW.isoformat(), "latest_observation": "2026-09-30T00:00:00+00:00"}, None, c, NOW)
    assert st2["future_timestamps"] == 40 and r["status"] == sh.BROKEN


# ── Recovery ───────────────────────────────────────────────────────────────
def test_recovery_requires_confirmed_passes():
    c = cfg(core=[core()])
    broken = {"status": sh.BROKEN, "consecutive_failures": 4}
    r1 = sh.classify(chk(), broken, c, NOW)
    assert r1["status"] == sh.DEGRADED and r1["recovering"] and any("RECOVERING" in x for x in r1["reasons"])
    r2 = sh.classify(chk(), {**r1}, c, NOW)
    assert r2["status"] == sh.HEALTHY and r2["consecutive_failures"] == 0
    r_fail = sh.classify(chk(latest="2026-09-01T00:00:00+00:00"), {**r1}, c, NOW)
    assert r_fail["status"] == sh.STALE                                               # Rückfall bricht Recovery ab


# ── Fallbacks, Features, Safe Mode ─────────────────────────────────────────
def test_provider_fallback_is_explicit_and_visible():
    c = cfg(core=[core("options_chain", fallback="options_chain_yfinance", decisions=["options_design"],
                       features=["option_quotes"]),
                  core("options_chain_yfinance", fallback_only=True, decisions=["options_design"],
                       features=["option_quotes"])])
    checks = [chk("options_chain", probe=sh._probe_result(auth_ok=False, error="AUTH_MISSING", auth_missing=True),
                  fallback="options_chain_yfinance"),
              chk("options_chain_yfinance", fallback_only=True)]
    s = sh.build_snapshot(checks, None, c, NOW, deps=sh.downstream(c))
    p = s["sources"]["options_chain"]
    assert p["fallback_active"] and p["source_primary"] == "options_chain" and p["source_actual"] == "options_chain_yfinance"
    assert s["fallbacks"][0]["actual"] == "options_chain_yfinance"
    assert "options_design" not in s["safe_mode"]["blocked_decisions"]
    f = s["features"]["option_quotes"]
    assert f["available"] and f["source_actual"] == "options_chain_yfinance" and f["data_quality"] < 1.0
    # Fallback selbst kaputt -> kein improvisierter Ersatz: Pfad blockiert
    checks[1] = chk("options_chain_yfinance", fallback_only=True, probe=sh._probe_result(reachable=False, error="x"))
    s2 = sh.build_snapshot(checks, None, c, NOW, deps=sh.downstream(c))
    assert s2["sources"]["options_chain"]["source_actual"] is None
    assert "options_design" in s2["safe_mode"]["blocked_decisions"] and s2["safe_mode"]["active"]


def test_cache_only_with_explicit_max_age_and_flagged_stale():
    alt = {"source_id": "sec_companyfacts", "kind": "alt", "criticality": "NON_CRITICAL", "probe": None,
           "last_success": (NOW - timedelta(days=20)).isoformat(),
           "latest_observation": (NOW - timedelta(days=12)).isoformat(), "max_success_age_days": 9}
    c = cfg(cache={"sec_xbrl_fundamentals": 30})
    deps = {"sec_companyfacts": {"features": ["xbrl_sue"], "models": [], "hypotheses": ["ALT-XBRL-001"],
                                 "decisions": [], "other": []}}
    s = sh.build_snapshot([alt], None, c, NOW, deps=deps)
    assert s["sources"]["sec_companyfacts"]["status"] == sh.STALE
    f = s["features"]["xbrl_sue"]
    assert f["available"] and f["stale"] and f["data_quality"] == 0.5 and f["cache_max_age_days"] == 30
    s2 = sh.build_snapshot([alt], None, cfg(cache={}), NOW, deps=deps)               # ohne Freigabe: kein Cache
    assert s2["features"]["xbrl_sue"] == {**s2["features"]["xbrl_sue"], "available": False, "data_quality": 0.0}
    assert "xbrl_sue" in s2["safe_mode"]["unavailable_features"]


def test_critical_failure_blocks_path_and_global_safe_mode():
    c = cfg(core=[core()])
    s = sh.build_snapshot([chk(probe=sh._probe_result(reachable=False, error="down"))], None, c, NOW,
                          deps=sh.downstream(c))
    sm = s["safe_mode"]
    assert sm["active"] and "scanner_candidates" in sm["blocked_decisions"]
    assert "keine Promotion neuer Hypothesen" in sm["effects"]


def test_multiple_important_failures_trigger_global_safe_mode():
    c = cfg()
    checks = [{"source_id": f"i{i}", "kind": "alt", "criticality": "IMPORTANT", "probe":
               sh._probe_result(reachable=False, error="down"), "last_success": None, "latest_observation": None}
              for i in range(2)]
    checks.append({"source_id": "n", "kind": "alt", "criticality": "NON_CRITICAL", "probe": None,
                   "last_success": NOW.isoformat(), "latest_observation": NOW.isoformat()})
    deps = {"i0": {"features": [], "models": [], "hypotheses": ["H0"], "decisions": [], "other": []}}
    s = sh.build_snapshot(checks, None, c, NOW, deps=deps)
    assert s["safe_mode"]["active"] and any("unabhängige wichtige" in r for r in s["safe_mode"]["global_reasons"])
    assert s["safe_mode"]["disabled_signals"] == ["H0"]
    one = sh.build_snapshot(checks[1:], None, c, NOW, deps={})
    assert not any("unabhängige" in r for r in one["safe_mode"]["global_reasons"])


def test_research_only_source_never_critical():
    assert sh._effective_criticality("CRITICAL", []) == "IMPORTANT"
    assert sh._effective_criticality("CRITICAL", ["risk_gates"]) == "CRITICAL"


# ── Verbraucher ────────────────────────────────────────────────────────────
def test_snapshot_reader_never_assumes_availability(tmp_path):
    p = tmp_path / "s.json"
    assert sh.load_snapshot(p, NOW)["unknown"]
    p.write_text(json.dumps({"generated": (NOW - timedelta(hours=40)).isoformat(), "sources": {}}))
    assert sh.load_snapshot(p, NOW)["unknown"]                                       # veraltet = unbekannt
    p.write_text("{kaputt")
    assert sh.load_snapshot(p, NOW)["unknown"]
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps({"active": False, "reasons": []}))
    sm = sh.effective_safe_mode(snapshot_path=tmp_path / "none.json", meta_path=meta, now=NOW)
    assert sm["active"] and not sm["known"] and "DATA HEALTH unbekannt" in sm["reasons"][0]


def test_scanner_preflight_probes_core_live_without_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "SNAPSHOT", tmp_path / "none.json")
    c = cfg(core=[core()])
    ok = sh.scanner_preflight(NOW, cfg=c, live=lambda cc, now, **kw: [chk()])
    assert ok["proceed"] and ok["source"] == "live_core_probe" and ok["safe_mode"]   # Intelligence trotzdem aus
    bad = sh.scanner_preflight(NOW, cfg=c, live=lambda cc, now, **kw: [chk(probe=sh._probe_result(reachable=False,
                                                                                                    error="x"))])
    assert not bad["proceed"] and "scanner_candidates" in bad["blocked_decisions"]


def test_feature_store_marks_unavailable_from_failure_date_only():
    from modules.alt_data.feature_store import apply_health
    df = pd.DataFrame({"date": pd.to_datetime(["2026-09-25", "2026-10-02", "2026-10-09"]), "ticker": "A",
                       "xbrl_sue": [1.0, 2.0, 3.0], "alt_xbrl_available": 1.0})
    s = {"features": ["xbrl_sue"], "availability_col": "alt_xbrl_available", "contracts": ["sec_companyfacts"]}
    health = {"generated": "2026-10-05T12:40:00+00:00",
              "sources": {"sec_companyfacts": {"status": "BROKEN", "failed_since": "2026-10-01T12:00:00+00:00"}},
              "features": {"xbrl_sue": {"available": False, "stale": None}}}
    out = apply_health(df.copy(), s, health)
    assert out["xbrl_sue"].tolist()[0] == 1.0 and out["xbrl_sue"].iloc[1:].isna().all()   # nie 0, Historie bleibt
    assert out["alt_xbrl_available"].tolist() == [1.0, 0.0, 0.0]
    health["features"]["xbrl_sue"] = {"available": True, "stale": True}
    cached = apply_health(df.copy(), s, health)
    assert cached["xbrl_sue"].tolist() == [1.0, 2.0, 3.0] and cached["alt_xbrl_available_stale"].tolist() == [0, 1, 1]


# ── Historie, Report, Mail ─────────────────────────────────────────────────
def test_mail_only_on_relevant_change_and_history(tmp_path):
    c = cfg(core=[core()])
    sent = []
    send = lambda subj, html, text, dry_run=False: sent.append(subj) or {"status": "sent"}
    good = lambda d: [chk(latest=(NOW + timedelta(days=d - 1)).isoformat())]
    run = lambda checks, now: _run(c, checks, now, send, tmp_path)
    r1 = run(good(0), NOW)
    assert r1["mail"]["status"] == "no_change" and not sent                           # neu + gesund: keine Mail
    r2 = run(good(1), NOW + timedelta(days=1))
    assert r2["mail"]["status"] == "no_change" and not sent                           # unverändert gesund: keine Mail
    r3 = run([chk(probe=sh._probe_result(reachable=False, error="down"))], NOW + timedelta(days=2))
    assert sent and "degradation" in sent[-1] and "safe_mode_on" in sent[-1]
    r4 = run(good(3), NOW + timedelta(days=3))                                         # erst RECOVERING
    assert r4["snapshot"]["sources"]["market_prices"]["status"] == sh.DEGRADED
    r5 = run(good(4), NOW + timedelta(days=4))
    assert r5["snapshot"]["sources"]["market_prices"]["status"] == sh.HEALTHY and "recovery" in sent[-1]
    hist = [json.loads(x) for x in (tmp_path / sh.HISTORY.name).read_text().splitlines()]
    assert [h["status"] for h in hist] == ["HEALTHY", "HEALTHY", "BROKEN", "DEGRADED", "HEALTHY"]
    inst = sh.instability(tmp_path / sh.HISTORY.name, now=NOW + timedelta(days=4))
    assert inst["market_prices"]["flips"] == 3
    rep = (tmp_path / sh.REPORT_MD.name).read_text()
    assert "Daily Data Health Report" in rep and "Safe Mode" in rep


def _run(c, checks, now, send, out):
    """check() mit vorgegebenen Checks (statt Netz)."""
    orig = sh.gather
    sh.gather = lambda *a, **k: checks
    try:
        return sh.check(now=now, cfg=c, probes=False, send=send, out_dir=out)
    finally:
        sh.gather = orig


def test_config_core_sources_and_workflow():
    c = sh.load_config()
    ids = {x["source_id"] for x in c["core_sources"]}
    assert {"market_prices", "vix_level", "options_chain"} <= ids
    fb = {x["source_id"]: x.get("fallback") for x in c["core_sources"]}
    assert fb["options_chain"] == "options_chain_yfinance" and fb["vix_level"] == "vix_fred"
    wf = open(".github/workflows/source_health.yml").read()
    assert "modules.source_health check" in wf and "cron" in wf
