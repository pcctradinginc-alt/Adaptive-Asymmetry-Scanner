"""Expectation Alpha – Integration:
* EA aus vs. SHADOW: identische Champion-Pfade in pipeline.main() (gleiche Kandidaten an Stufe 5, gleiche
  Stats/Rejects), EA schreibt nur seinen eigenen Ledger;
* statische Isolation: pipeline.py nutzt EA nur über die SHADOW-Hooks, EA-Module schreiben keine
  Produktionsdateien;
* Promotion: EA-Verträge nur mit EA-Evidenz ab forward_start, höchstens FORWARD_VALIDATED mit Einfluss NONE,
  REJECT bei Gegenrichtung, gepaarte Verträge, der Adapter wendet EA-Verträge nie an.
"""
from __future__ import annotations

import ast
import copy
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import tests.ea_fixtures as fx
from modules import hypothesis_contract as hc
from modules import promotion_controller as pc
from modules import production_intelligence_adapter as pia
from modules.expectation_alpha import ledger as eal

ROOT = Path(__file__).resolve().parent.parent
UTC = timezone.utc


# ── pipeline.main(): Champion unverändert ───────────────────────────────────
def _run_pipeline(tmp_path, monkeypatch, mode: str) -> dict:
    import pipeline
    from modules import cost_telemetry, email_reporter, model_routing, research_memory, source_health, system_state
    from modules.expectation_alpha import config as ec, data as ed
    work = tmp_path / mode
    work.mkdir()
    monkeypatch.chdir(work)
    seen: dict = {"mismatch_in": None, "status": []}
    cfg = copy.deepcopy(ec.load())
    cfg["mode"] = mode
    monkeypatch.setattr(ec, "load", lambda path=None: copy.deepcopy(cfg))
    today = datetime.now(UTC).date()                     # pipeline.main() entscheidet zur echten Uhrzeit
    px = fx.make_px(end=(today + timedelta(days=20)).isoformat(), years=7)
    ar, cm = fx.make_archive(end=today.isoformat()), fx.make_commodity(end=today.isoformat())
    monkeypatch.setattr(ed, "_default_archive", lambda c: ar)
    monkeypatch.setattr(ed, "_default_prices", lambda c, t: px)
    monkeypatch.setattr(ed, "_default_commodity", lambda c, t: cm)
    monkeypatch.setattr(pipeline, "send_status_email", lambda stats, today, health=None: seen["status"].append(stats))
    monkeypatch.setattr(email_reporter, "send_email", lambda p, t, s=None: None)
    monkeypatch.setattr(cost_telemetry, "install_http_counter", lambda: False)
    monkeypatch.setattr(source_health, "scanner_preflight",
                        lambda: {"proceed": True, "blocked_decisions": [], "safe_mode": False, "data_quality": 1.0,
                                 "reasons": [], "source": "test", "fallbacks": []})
    monkeypatch.setattr(system_state, "current", lambda **k: {"state_version": "t1"})
    monkeypatch.setattr(system_state, "safe_mode_view",
                        lambda s: {"state_version": "t1", "active": False, "reasons": [], "drift_level": "NORMAL",
                                   "known": True})
    monkeypatch.setattr(pipeline, "build_health_report", lambda **k: {"status": "test"})

    class _Gates:
        last_vix = 18.0

        def global_ok(self):
            return True
    monkeypatch.setattr(pipeline, "RiskGates", lambda: _Gates())
    monkeypatch.setattr(pipeline, "get_macro_context", lambda: {"macro_regime": "test"})

    class _Designer:
        def __init__(self, **k):
            pass

        def _sector_momentum_ok(self, c):
            return True

        def _get_iv_rank(self, t):
            return None
    monkeypatch.setattr(pipeline, "OptionsDesigner", _Designer)
    tickers = [("AAPL", "Technology"), ("XOM", "Energy"), ("JPM", "Financial Services"), ("PG", "Consumer Defensive"),
               ("NVDA", "Technology"), ("CAT", "Industrials")]

    class _Ingest:
        def __init__(self, history=None):
            pass

        def run(self):
            return [{"ticker": t, "market_cap": 1e11, "avg_volume": 5e6, "info": {"sector": s}} for t, s in tickers]
    monkeypatch.setattr(pipeline, "DataIngestion", _Ingest)
    monkeypatch.setattr(pipeline, "score_candidate", lambda c: {"sentiment_score": 0.1})
    monkeypatch.setattr(pipeline, "enrich_with_sentiment_drift", lambda c, h: c)

    class _Pre:
        failed_tickers: list = []

        def run(self, cands):
            return list(cands)
    monkeypatch.setattr(pipeline, "Prescreener", _Pre)
    monkeypatch.setattr(pipeline, "enrich_with_alpha_sources", lambda c: c)
    monkeypatch.setattr(pipeline, "validate_candidate_data", lambda c: c)
    monkeypatch.setattr(pipeline, "attach_external_context_stage", lambda s: (s, {}))

    class _Miro:
        def _get_market_params(self, t):
            return None, 0.0, None

        def run_for_dte(self, c, days_to_expiry=None, min_hit_rate=None):
            return {"simulation": {"hit_rate": 0.9}}
    monkeypatch.setattr(pipeline, "MirofishSimulation", _Miro)
    monkeypatch.setattr(model_routing, "apply_bearish_prefilter", lambda c, allowed: (c, []))
    dirs = {"AAPL": "BULLISH", "XOM": "BULLISH", "JPM": "BEARISH", "PG": "BULLISH", "NVDA": "BULLISH", "CAT": "BULLISH"}
    scores = {"AAPL": (6, 5), "XOM": (3, 2), "JPM": (6, 6), "PG": (5, 3), "NVDA": (7, 6), "CAT": (4, 2)}

    class _DA:
        skipped_for_time: list = []

        def run(self, cands, deadline=None):
            out = []
            for c in cands:
                imp, sur = scores[c["ticker"]]
                out.append({**c, "deep_analysis": {"direction": dirs[c["ticker"]], "impact": imp, "surprise": sur,
                                                   "time_to_materialization": "4-8 Wochen",
                                                   "catalyst": f"Event {c['ticker']}"}})
            return out
    monkeypatch.setattr(pipeline, "DeepAnalysis", _DA)
    monkeypatch.setattr(pipeline, "refresh_catalyst_relevance", lambda a: None)
    monkeypatch.setattr(pipeline, "run_shadow_relation_analysis", lambda a, deadline=None: None)
    monkeypatch.setattr(research_memory, "load", lambda *a, **k: {})
    monkeypatch.setattr(research_memory, "review_candidate", lambda *a, **k: None)

    class _MM:
        data_missing: list = []

        def run(self, analyses):
            seen["mismatch_in"] = copy.deepcopy(analyses)
            return []                                    # Lauf endet deterministisch nach Stufe 5
    monkeypatch.setattr(pipeline, "MismatchScorer", _MM)
    pipeline.reject_stats.clear()
    pipeline.main()
    seen["rejects"] = {k: dict(v) for k, v in pipeline.reject_stats.items()}
    seen["work"] = work
    return seen


def test_pipeline_champion_identical_with_ea_off_and_shadow(tmp_path, monkeypatch):
    off = _run_pipeline(tmp_path, monkeypatch, "off")
    on = _run_pipeline(tmp_path, monkeypatch, "shadow")
    assert off["mismatch_in"] is not None and on["mismatch_in"] == off["mismatch_in"]   # gleiche Kandidaten an Stufe 5
    assert on["rejects"] == off["rejects"]
    s_off = {k: v for k, v in off["status"][-1].items() if k != "expectation_alpha"}
    s_on = {k: v for k, v in on["status"][-1].items() if k != "expectation_alpha"}
    assert s_on == s_off
    ea = on["status"][-1]["expectation_alpha"]
    assert ea["enabled"] and ea["production_influence"] == "NONE" and ea["enriched_count"] == 6
    assert "observations" not in ea
    assert not (off["work"] / "outputs" / "expectation_alpha").exists()
    rows = eal.read_rows(on["work"] / "outputs" / "expectation_alpha")
    assert len(rows) == 6 and {r["ticker"] for r in rows} == {"AAPL", "XOM", "JPM", "PG", "NVDA", "CAT"}
    ds = [json.loads(x) for x in (on["work"] / "outputs" / "expectation_alpha" / "downstream.jsonl")
          .read_text().splitlines()]
    assert len(ds) == 6 and all(d["champion_decision"] in ("NO_TRADE", "UNKNOWN") for d in ds)


def test_pipeline_ea_exception_never_breaks_scan(tmp_path, monkeypatch):
    import modules.expectation_alpha as ea_mod

    def boom(*a, **k):
        raise RuntimeError("EA kaputt")
    monkeypatch.setattr(ea_mod, "enrich_candidates", boom)
    res = _run_pipeline(tmp_path, monkeypatch, "shadow")
    assert res["mismatch_in"] is not None
    assert res["status"][-1]["expectation_alpha"]["status"] == "ERROR"


# ── statische Isolation ─────────────────────────────────────────────────────
def test_pipeline_uses_ea_only_via_shadow_hooks():
    """EA wird erst am Laufende ausgewertet (nach allen Champion-Entscheidungen, ohne Champion-Laufzeitbudget);
    bei Stufe 4 wird nur eine tiefe Kopie eingefroren. Rückgabe nur in stats."""
    src = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fins = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_expectation_alpha_finalize"]
    assert len(fins) == 1
    inside = {id(n) for n in ast.walk(fins[0])}
    ea_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "_ea"]
    assert {n.func.attr for n in ea_calls} == {"enrich_candidates", "record_downstream"}
    assert all(id(n) in inside for n in ea_calls)                  # nirgends sonst
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_expectation_alpha_finalize"]
    snap = [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "save_stats_snapshot"]
    assert len(calls) == 1 and len(snap) == 1 and id(calls[0]) in {id(n) for n in ast.walk(snap[0])}
    assert "_ea_input_ref[:] = [_copy.deepcopy(analyses)]" in src
    import re
    assert not re.search(r"\banalyses\s*=\s*_ea", src) and "_ea_sum[" not in src   # Rückgabe nie in Kandidatenlisten


LLM_ALLOWED = {"claim_extraction.py"}          # einziger (separater, SHADOW) LLM-Call: Claims strukturieren
NO_LLM_DECISION = ("expectation_gap.py", "regime_change.py", "cross_asset_confirmation.py", "timing.py",
                   "thesis.py", "ledger.py", "evaluation.py", "claims.py", "future_state.py", "data.py")


def test_ea_modules_never_write_production_files():
    from tests.test_promotion import _writes_to_protected
    for f in sorted((ROOT / "modules" / "expectation_alpha").glob("*.py")):
        assert _writes_to_protected(f) == [], f.name
        src = f.read_text(encoding="utf-8")
        for bad in ("openai", "apply_to_proposals", "history.json", "trade_score"):
            assert bad not in src, (f.name, bad)                 # kein Produktionspfad
        if f.name not in LLM_ALLOWED:
            assert "anthropic" not in src, f.name                 # kein LLM außer im Claim-Extraktor
    # Gap, Regime, Bestätigung, Status, Ledger, Outcomes, Evaluation: weder LLM noch Claim-Extraktor importiert
    for name in NO_LLM_DECISION:
        tree = ast.parse((ROOT / "modules" / "expectation_alpha" / name).read_text(encoding="utf-8"))
        imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in n.names} | {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        assert not {"claim_extraction", "anthropic"} & imported and "tracked_create" not in calls, name


# ── Promotion ───────────────────────────────────────────────────────────────
def _ea_contracts(*ids):
    cs = {c["hypothesis_id"]: c for c in hc.load() if c.get("eligible_stage") == "EA_NEWS_CANDIDATE"}
    return [cs[i] for i in ids]


def _row(oid, d: date, ticker, env, valid, regime, sector="Technology", sel="UNDERLYING", status="TRADE"):
    r = {"observation_id": oid, "stage": "EA_NEWS_CANDIDATE", "timestamp": f"{d.isoformat()}T15:00:00+00:00",
         "date": d.isoformat(), "ticker": ticker, "sector": sector, "regime": regime, "env": env, "status": status,
         "selected_expression": {"selected_expression": sel}, "expressions": [{"expression": "UNDERLYING"}]}
    r["contracts"] = eal.freeze_contracts(r, valid)
    return r


def _out(oid, ex, var, h, v):
    return {"observation_id": oid, "expression": ex, "variant": var, "horizon": h, "outcome": v, "outcome_net": v,
            "mfe": max(v, 0.0), "mae": min(v, 0.0), "outcome_method": eal.OUTCOME_METHOD}


def test_ea_promotion_capped_rejected_and_paired(tmp_path):
    cs = _ea_contracts("EA001_EXPECTATION_GAP", "EA003_ACCELERATION", "EA004_WAIT")
    reg, root = tmp_path / "reg.jsonl", tmp_path / "ea"
    assert all(s["status"] == "VALID" for s in hc.register(cs, reg, hc.load_policy()).values())
    valid = eal.registered_contracts(cs, reg)
    rng = random.Random(7)
    rows, outs = [], []
    days = [date(2026, 10, 10), date(2026, 10, 11)] + [date(2026, 10, 13) + timedelta(days=k) for k in range(200)]
    for i, d in enumerate(days):
        pre = d < date(2026, 10, 13)
        for j in range(2):
            oid = f"o{i:03d}{j}"
            aligned, accel, wait = (i + j) % 2, (i // 2 + j) % 2, int(j == 0)
            env = {"ea_gap_aligned": aligned, "ea_status_valid": 1, "ea_gap_accel_aligned": accel, "ea_is_wait": wait}
            rows.append(_row(oid, d, f"T{i}_{j}", env, valid, ("vix_low", "vix_high")[i % 2],
                             sector=("Technology", "Energy", "Industrials")[(i + 2 * j) % 3],
                             status="WAIT" if wait else "TRADE"))
            noise = rng.gauss(0, 0.005)
            # EA001: Gap-ausgerichtet besser; vor forward_start umgekehrt (darf nicht zählen)
            base = (0.04 if aligned else -0.02) * (-1 if pre else 1)
            # EA003 (nur aligned): Beschleunigung SCHLECHTER -> Gegenrichtung -> REJECT
            if aligned:
                base += -0.05 if accel else 0.03
            v = base + noise
            outs.append(_out(oid, "UNDERLYING", "immediate", 60, v))
            if wait:
                outs.append(_out(oid, "UNDERLYING", "triggered", 60, v + 0.01 + rng.gauss(0, 0.002)))
    (root / "candidates").mkdir(parents=True)
    eal.append_jsonl(root / "candidates" / "all.jsonl", rows, sort_keys=True, default=str)
    eal.append_jsonl(root / "outcomes.jsonl", outs, sort_keys=True)
    state = pc.run(contracts=cs, policy=hc.load_policy(), now=datetime(2027, 6, 1, tzinfo=UTC), registry=reg,
                   transitions=tmp_path / "tr.jsonl", ledger_dir=tmp_path / "champ", outcomes_path=tmp_path / "co.jsonl",
                   looks_path=tmp_path / "looks.jsonl", state_path=tmp_path / "state.json", history={}, approvals={},
                   safe_mode={"active": False}, final_mc_dir=tmp_path / "fm", final_mc_outcomes=tmp_path / "fmo.jsonl",
                   v2_dir=tmp_path / "v2", v2_outcomes=tmp_path / "v2o.jsonl", ea_root=root)
    h1 = state["hypotheses"]["EA001_EXPECTATION_GAP@v1"]
    h3 = state["hypotheses"]["EA003_ACCELERATION@v1"]
    h4 = state["hypotheses"]["EA004_WAIT@v1"]
    assert h1["eligible_stage"] == "EA_NEWS_CANDIDATE" and h1["evidence"]["n_observations"] == 400  # nichts vor forward_start
    assert h1["evidence"]["delta_expectancy"] > 0
    assert h1["state"] == "FORWARD_VALIDATED" and h1["influence_level"] == "NONE" and h1["recommendation"] is None
    assert h1["multiple_testing"]["hypothesis_family"] == "research_only@EA_NEWS_CANDIDATE"
    assert h3["state"] == "REJECTED" and h3["influence_level"] == "NONE"
    assert h4["evidence"]["n_observations"] == 200 and h4["evidence"]["delta_expectancy"] == pytest.approx(0.01, abs=2e-3)
    assert h4["state"] == "FORWARD_VALIDATED" and h4["influence_level"] == "NONE"
    for h in state["hypotheses"].values():
        assert h["influence_level"] == "NONE"


def test_adapter_never_applies_ea_contracts(monkeypatch, tmp_path):
    c = _ea_contracts("EA006_ABSTENTION")[0]
    forged = {hc.key(c): {"contract": c, "level": "ABSTENTION_ONLY", "state": "GUARDED_PRODUCTION",
                          "spec_hash": hc.spec_hash(c)}}
    monkeypatch.setattr(pia, "load_verified_state", lambda *a, **k: (forged, []))
    props = [{"ticker": "AAPL", "features": {"ea_is_abstain": 1, "ea_status_valid": 1},
              "trade_score": {"total": 80}}]
    kept, blocked, _recs = pia.apply_to_proposals(props, vix=18, context={"safe_mode_active": 0}, contracts=[c],
                                                  ledger_dir=tmp_path, policy=hc.load_policy())
    assert kept == props and blocked == []
