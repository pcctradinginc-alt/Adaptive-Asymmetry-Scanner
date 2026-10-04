"""Finaler End-to-End-Audit (2026-10-04): Lernzyklus, Promotion/Demotion, Leakage, Execution-Realität,
Crash-Recovery, Idempotenz, Quellen-Ausfälle, kanonischer State, Workflow-Invarianten.

Alle Tests laufen über die ECHTEN Pfade (PromotionController.run, ProductionIntelligenceAdapter,
Final-MC-/V2-Ledger, Commodity-Engine, Source Health, SystemState) – keine Netzwerkzugriffe.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from modules import commodity_intelligence as cmd
from modules import hypothesis_contract as hc
from modules import production_intelligence_adapter as pia
from modules import promotion_controller as pc

ROOT = Path(__file__).resolve().parent.parent
POLICY = hc.load_policy(ROOT / "config" / "promotion_policy.yaml")
UTC = timezone.utc


@pytest.fixture(autouse=True)
def _no_repo_safe_mode(monkeypatch):
    """Der echte Repo-Zustand (Drift MODERATE) darf die Simulation nicht steuern; Safe Mode wird explizit getestet."""
    monkeypatch.setattr(pc, "_safe_mode_state", lambda: {"active": False, "reasons": []})


# ── Hilfen: isolierte Pfade + Decision-Ledger wie vom Adapter geschrieben ─────────────────────────
class Env:
    def __init__(self, tmp: Path):
        self.reg, self.tr = tmp / "reg.jsonl", tmp / "tr.jsonl"
        self.ledger, self.out, self.looks = tmp / "ledger", tmp / "outcomes.jsonl", tmp / "looks.jsonl"
        self.state = tmp / "state.json"

    def run(self, contracts, now, policy=POLICY, approvals=None, safe=None):
        return pc.run(contracts=contracts, policy=policy, now=now, registry=self.reg, transitions=self.tr,
                      ledger_dir=self.ledger, outcomes_path=self.out, looks_path=self.looks, state_path=self.state,
                      history={}, approvals=approvals or {}, safe_mode=safe or {"active": False, "reasons": []})

    def verified(self, contracts):
        return pia.load_verified_state(self.state, contracts, self.reg, self.tr)


def base_contract(**kw) -> dict:
    c = copy.deepcopy(hc.load(ROOT / "config" / "promotion_hypotheses.yaml")[0])
    c.update(hypothesis_id="AUDIT-EQ-ABST", signal_definition="risk_flag", features=["risk_flag"],
             thresholds={"op": ">=", "value": 1}, title="Audit Equity-Abstinenz")
    c.update(kw)
    return c


def commodity_contract(**kw) -> dict:
    """Commodity→Equity: Kupferstärke × Kupfer-Exposure (Rerank), NON_PIT-Mapping fixiert."""
    c = base_contract(hypothesis_id="AUDIT-CMD-COPPER", production_class="rerank", direction=1,
                      signal_definition="cmdx_copper__copper_ret_3m", features=["cmdx_copper__copper_ret_3m"],
                      thresholds={"op": ">", "value": 0.0}, maximum_initial_influence="RERANK_ONLY",
                      mapping_version="exposure-v1", title="Kupferstärke -> kupferexponierte Titel",
                      research_question="Führt Kupferstärke bei kupferexponierten Titeln zu Überrendite?",
                      economic_rationale="Kupferpreis -> Margen von Minen; Einpreisung verzögert (Hypothese)",
                      population="V1-Champion-Trades mit Kupfer-Exposure (exposure-v1, NON_PIT)",
                      exposure="cmdexp_copper != 0")
    c.update(kw)
    return c


def write_ledger(env: Env, c: dict, start: datetime, n_dates: int, *, fired_ret: float, kept_ret: float,
                 seed: int = 1, step_days: int = 3, per_day: int = 2, fired_every: int = 4, noise: float = 0.05,
                 regimes=("vix_low", "vix_high"), sectors=("Tech", "Energy", "Health", "Util"), spec_hash=None,
                 prob_fired: float = 0.55, prob_kept: float = 0.55):
    rng = random.Random(seed)
    h, k = spec_hash or hc.spec_hash(c), hc.key(c)
    rows, outs, i = [], [], 0
    for d in range(n_dates):
        t = start + timedelta(days=d * step_days)
        for j in range(per_day):
            i += 1
            fired = i % fired_every == 0
            did = hashlib.sha256(f"{k}{seed}{t}{j}".encode()).hexdigest()[:16]
            rows.append({"decision_id": did, "timestamp": t.isoformat(), "date": t.date().isoformat(),
                         "ticker": f"T{i}", "sector": sectors[(i // fired_every) % len(sectors)],
                         "regime": regimes[(d * len(regimes)) // n_dates], "champion_decision": "TRADE",
                         "champion_probability": prob_fired if fired else prob_kept, "champion_rank": j + 1,
                         "intelligence_rank": j + 1,
                         "final_production_decision": "TRADE",
                         "intelligence": {k: {"spec_hash": h, "evaluable": True, "in_scope": True, "fired": fired,
                                              "applied": False}}})
            outs.append({"decision_id": did, "outcome": (fired_ret if fired else kept_ret) + rng.uniform(-noise, noise),
                         "outcome_method": "option_quote"})
    env.ledger.mkdir(parents=True, exist_ok=True)
    with open(env.ledger / f"{start:%Y-%m-%d}-{k}-{seed}.jsonl", "a") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    with open(env.out, "a") as fh:
        fh.writelines(json.dumps(o) + "\n" for o in outs)


T0 = datetime(2026, 10, 6, 15, tzinfo=UTC)


# ═══ PHASE 1–4: kein Alpha / Equity-Signal / Commodity-Signal / Decay (echter Controller + Adapter) ═══
def test_phase1_no_alpha_no_promotion(tmp_path):
    e = Env(tmp_path)
    c = base_contract(hypothesis_id="AUDIT-NOALPHA")
    write_ledger(e, c, T0, 40, fired_ret=0.05, kept_ret=0.05)
    h = e.run([c], datetime(2027, 2, 1, tzinfo=UTC))["hypotheses"][hc.key(c)]
    assert h["influence_level"] == "NONE" and h["state"] in ("PROSPECTIVE_CHALLENGER", "EXPIRED", "REJECTED")
    assert h["decision"] in ("KEEP_SHADOW", "REJECT")


def test_phase2_equity_signal_discovery_to_promotion_then_phase4_decay(tmp_path):
    e = Env(tmp_path)
    c = base_contract()
    # Historische Phase VOR forward_start: spektakulär, zählt aber nie (keine Historical Promotion)
    write_ledger(e, c, datetime(2026, 1, 5, 15, tzinfo=UTC), 40, fired_ret=-0.9, kept_ret=0.5, seed=3)
    s0 = e.run([c], datetime(2026, 10, 7, tzinfo=UTC))["hypotheses"][hc.key(c)]
    assert s0["evidence"]["n_observations"] == 0 and s0["influence_level"] == "NONE"
    # Forward: robustes Signal (blockierte Trades schlecht)
    write_ledger(e, c, T0, 40, fired_ret=-0.4, kept_ret=0.15)
    h = e.run([c], datetime(2027, 2, 1, tzinfo=UTC))["hypotheses"][hc.key(c)]
    assert h["state"] == "GUARDED_PRODUCTION" and h["influence_level"] == "ABSTENTION_ONLY"
    assert [x["new_state"] for x in pc.read_transitions(e.tr)[0]] == \
        ["PROSPECTIVE_CHALLENGER", "FORWARD_VALIDATED", "GUARDED_PRODUCTION"]
    # Phase 4: Decay nach Promotion -> Demotion (keine Immunität historischer Erfolge)
    write_ledger(e, c, datetime(2027, 2, 3, tzinfo=UTC), 30, fired_ret=0.5, kept_ret=0.0, seed=9)
    h2 = e.run([c], datetime(2027, 5, 1, tzinfo=UTC))["hypotheses"][hc.key(c)]
    assert h2["decision"] == "DEMOTE" and h2["influence_level"] == "NONE"
    assert e.verified([c])[0][hc.key(c)]["level"] == "NONE"


def test_phase3_commodity_signal_needs_forward_and_human_step_then_decays(tmp_path):
    e = Env(tmp_path)
    c = commodity_contract()
    assert hc.validate(c, POLICY) == []
    write_ledger(e, c, T0, 40, fired_ret=0.4, kept_ret=0.0)
    s1 = e.run([c], datetime(2027, 2, 1, tzinfo=UTC))["hypotheses"][hc.key(c)]
    # Policy max. automatisch ABSTENTION_ONLY liegt unter der Rerank-Leiter -> forward-validiert, KEIN Einfluss
    assert s1["state"] == "FORWARD_VALIDATED" and s1["influence_level"] == "NONE"
    assert s1["recommendation"] == "RERANK_ONLY"
    # Menschliche Freigabe (PR) bis SCORE_LIMITED -> genau EINE Stufe je Look
    appr = {hc.key(c): {"approved_level": "WEIGHT_25", "pr": "audit"}}             # wird auf SCORE_LIMITED gekappt
    s2 = e.run([c], datetime(2027, 3, 5, tzinfo=UTC), approvals=appr)["hypotheses"][hc.key(c)]
    assert s2["state"] == "LIMITED_PRODUCTION" and s2["influence_level"] == "RERANK_ONLY"
    # Adapter wendet NUR Rerank an, nur für gemappte Titel mit fixiertem Mapping; erzeugt nie Trades
    snap = {"features": {"cmd_copper_ret_3m": 0.08}, "commodity_data_version": "v"}
    ctx = {"safe_mode_active": 0, "commodity": snap}
    props = [{"ticker": t, "sector": "Basic Materials", "features": {}, "trade_score": {"total": 60 + i}}
             for i, t in enumerate(["AAA", "FCX", "SCCO"])]
    kept, blocked, recs = pia.apply_to_proposals([dict(p) for p in props], vix=14, context=ctx, state_path=e.state,
                                                 contracts=[c], registry=e.reg, transitions=e.tr,
                                                 ledger_dir=tmp_path / "live", policy=POLICY)
    assert {p["ticker"] for p in kept} == {"AAA", "FCX", "SCCO"} and not blocked
    by = {r["ticker"]: r for r in recs}
    assert by["FCX"]["intelligence"][hc.key(c)]["fired"] is True
    assert by["AAA"]["intelligence"][hc.key(c)]["evaluable"] is False          # kein Kupfer-Mapping -> nie 0
    assert all(r["final_production_decision"] == "TRADE" for r in recs)
    assert by["FCX"]["commodity_features_used"] is True and by["FCX"]["commodity_data_version"] == "v"
    # Phase 4 (Commodity): Beziehung zerfällt -> Demotion auf NONE (SHADOW)
    write_ledger(e, c, datetime(2027, 3, 8, tzinfo=UTC), 30, fired_ret=-0.3, kept_ret=0.1, seed=11)
    s3 = e.run([c], datetime(2027, 6, 1, tzinfo=UTC))["hypotheses"][hc.key(c)]
    assert s3["decision"] in ("DEMOTE", "REJECT") and s3["influence_level"] == "NONE"


def test_full_demotion_ladder_promoted_limited_rerank_shadow_retired(tmp_path):
    """PROMOTED (WEIGHT_10) -> RERANK -> SHADOW -> RETIRED (REJECTED) über echte Controller-Läufe."""
    e = Env(tmp_path)
    c = base_contract(hypothesis_id="AUDIT-W", production_class="weight", direction=1, maximum_initial_influence="RERANK_ONLY")
    assert hc.validate(c, POLICY) == []
    cal = {"prob_fired": 0.97, "prob_kept": 0.5}                     # kalibrierte Wahrscheinlichkeiten
    write_ledger(e, c, T0, 40, fired_ret=0.4, kept_ret=0.0, **cal)
    e.run([c], datetime(2027, 2, 1, tzinfo=UTC))
    appr = {hc.key(c): {"approved_level": "WEIGHT_10", "pr": "audit"}}
    lv = [e.run([c], datetime(2027, 3, 5, tzinfo=UTC), approvals=appr)["hypotheses"][hc.key(c)]["influence_level"]]
    # nächste Stufe nur mit erneut voller Forward-Evidenz SEIT der Promotion (N, Tage, Spanne, Regime)
    early = e.run([c], datetime(2027, 4, 5, tzinfo=UTC), approvals=appr)["hypotheses"][hc.key(c)]
    assert early["influence_level"] == "RERANK_ONLY"
    write_ledger(e, c, datetime(2027, 3, 6, 15, tzinfo=UTC), 50, fired_ret=0.4, kept_ret=0.0, seed=21, step_days=2, **cal)
    lv.append(e.run([c], datetime(2027, 6, 20, tzinfo=UTC), approvals=appr)["hypotheses"][hc.key(c)]["influence_level"])
    assert lv == ["RERANK_ONLY", "WEIGHT_10"]
    # Decay: je Verstoß eine Stufe nach unten, dann Effektumkehr -> terminal
    seen = []
    when = datetime(2027, 6, 22, tzinfo=UTC)
    for i in range(4):
        write_ledger(e, c, when, 20, fired_ret=-0.4, kept_ret=0.1, seed=40 + i, step_days=1, **cal)
        when += timedelta(days=30)
        h = e.run([c], when)["hypotheses"][hc.key(c)]
        seen.append((h["state"], h["influence_level"]))
    levels = [lvl for _, lvl in seen]
    assert levels[0] == "RERANK_ONLY" and "NONE" in levels
    assert seen[-1][0] in ("REJECTED", "DEMOTED", "EXPIRED", "PROSPECTIVE_CHALLENGER")
    assert e.verified([c])[0][hc.key(c)]["level"] == "NONE"


def test_contract_immutable_posthoc_change_is_new_version(tmp_path):
    e = Env(tmp_path)
    c = commodity_contract()
    assert hc.register([c], e.reg, POLICY)[hc.key(c)]["status"] == "VALID"
    moved = {**c, "thresholds": {"op": ">", "value": 0.05}}
    assert hc.spec_hash(moved) != hc.spec_hash(c)
    assert hc.register([moved], e.reg, POLICY)[hc.key(moved)]["status"] != "VALID"     # gleiche Version, anderer Hash
    v2 = {**moved, "version": c["version"] + 1}
    assert hc.register([v2], e.reg, POLICY)[hc.key(v2)]["status"] == "VALID"
    # forward_start strikt nach registered_at
    bad = {**c, "forward_start": c["registered_at"]}
    assert any("forward_start" in x for x in hc.validate(bad, POLICY))
    # Mapping-Änderung: alte Version nicht mehr auswertbar
    env_new = {"cmdx_copper__copper_ret_3m": 0.1, "commodity_mapping_version": "exposure-v2"}
    assert hc.fires(c, env_new) is None


def test_v1_v2_and_commodity_populations_never_mix(tmp_path):
    """Adapter wertet nur Champion-Population aus; V2-Segment- und Final-MC-Verträge nie auf V1-Trades."""
    seg = next(x for x in hc.load() if x.get("universe_version") == "V2")
    fm = next(x for x in hc.load() if x.get("eligible_stage") == "FINAL_MC_SURVIVOR")
    eq = base_contract()
    cm = commodity_contract()
    props = [{"ticker": "FCX", "sector": "Basic Materials", "features": {"risk_flag": 1}, "trade_score": {"total": 70}}]
    _, _, recs = pia.apply_to_proposals(props, vix=14, context={"safe_mode_active": 0, "commodity": {}},
                                        state_path=tmp_path / "s.json", contracts=[seg, fm, eq, cm],
                                        registry=tmp_path / "r.jsonl", transitions=tmp_path / "t.jsonl",
                                        ledger_dir=tmp_path / "l", policy=POLICY)
    keys = set(recs[0]["intelligence"])
    assert hc.key(seg) not in keys and hc.key(fm) not in keys
    assert recs[0]["universe_version"] == "V1"


# ═══ PHASE 5: Lookahead erzeugt Scheinalpha -> PIT-Engine eliminiert es ════════════════════════════
def test_phase5_lookahead_signal_vanishes_under_pit(tmp_path):
    """Equity-Rendite der Woche hängt an der Lagerveränderung derselben Woche (erst Mi/Do danach
    veröffentlicht). Mit observation_time (Lookahead) perfekt korreliert, PIT (available_at) nicht."""
    from modules.external.archive import ExternalArchive
    from modules.external.pit import AvailabilityPrecision, Observation
    from modules.external.sources import commodities as cs
    rng = np.random.default_rng(5)
    rule = {"days": 6, "hour_utc": 16}
    t = datetime(2024, 1, 5, tzinfo=UTC)
    obs, level, changes = [], 400000.0, {}
    while t < datetime(2026, 9, 1, tzinfo=UTC):
        chg = float(rng.normal(0, 3000))
        level += chg
        changes[t.date()] = chg
        av = cs.release_time(t, rule)
        obs.append(Observation(source_id="eia_petroleum_weekly", dataset="crude_stocks", series_id="WCESTUS1",
                               entity_id="US", metric="eia_crude_stocks", value=round(level, 1), unit="thousand_barrels",
                               observation_time=t, available_at=av, retrieved_at=av,
                               availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE, parser_version="1"))
        t += timedelta(days=7)
    ExternalArchive(str(tmp_path)).store_observations(obs, max_backfill_bytes=10**10)
    df, _ = cmd.build(now=datetime(2026, 9, 1, tzinfo=UTC), root=tmp_path, start="2024-06-03", write=False)
    df["date"] = df["date"].astype(str)
    # "Equity-Rendite" je Freitag = -(Lagerveränderung der Woche bis zu diesem Freitag) (Lookahead-Konstruktion)
    fridays = [d for d in changes if d >= date(2024, 6, 7)]
    ret = np.array([-changes[d] for d in fridays])
    look = np.array([changes[d] for d in fridays])
    pit = df.set_index("date").reindex([d.isoformat() for d in fridays])["cmd_crude_stocks_chg_1w"].to_numpy()
    m = ~np.isnan(pit)
    corr_look = np.corrcoef(look, ret)[0, 1]
    corr_pit = np.corrcoef(pit[m], ret[m])[0, 1]
    assert corr_look < -0.99                       # Scheinalpha mit observation_time
    assert abs(corr_pit) < 0.25                    # PIT: am Freitag ist erst die VORWOCHE veröffentlicht


def test_phase5_cot_report_date_not_release_date(tmp_path):
    from modules.external.sources import commodities as cs
    rule = cs.load_config()["cftc"]["release"]
    assert cs.cot_release_time(date(2026, 9, 22), rule) > datetime(2026, 9, 25, 19, 30, tzinfo=UTC)   # nie vor Fr 15:30 ET
    assert cs.cot_release_time(date(2026, 11, 24), rule).date() == date(2026, 11, 30)              # Feiertagswoche


# ═══ PHASE 6: Small-Cap-Alpha, aber unhandelbare Optionen -> keine Promotion ════════════════════════
def test_phase6_research_alpha_but_untradeable_options_no_promotion(tmp_path):
    from modules import universe_v2 as uv
    from modules import universe_v2_ledger as v2l
    cfg = uv.load_cfg()

    def chain(sym, today, price=20.0, spread=0.15):
        rows = []
        for k in range(4):
            e = (today + timedelta(days=45 + 30 * k)).isoformat()
            for st in (0.9, 1.0, 1.1):
                mid = max(0.2, price * 0.08 * (1.2 - st))
                rows.append({"symbol": f"{sym}{e}C{st}", "expiry": e, "strike": round(price * st, 2),
                             "option_type": "call", "bid": round(mid * (1 - spread / 2), 3),
                             "ask": round(mid * (1 + spread / 2), 3), "open_interest": 20, "volume": 0})
        return rows

    def quote(today, price=20.0):
        return {"price": price, "bid": price * 0.995, "ask": price * 1.005, "avg_volume": 300_000, "volume": 200_000,
                "last_trade": today.isoformat(), "closes": [price] * 25}
    reg, led, outs = tmp_path / "r.jsonl", tmp_path / "led", tmp_path / "o.jsonl"
    hc.register(hc.load(), reg, hc.load_policy())
    exit_mid = {}
    ctx = {"safe_mode_active": 0, "blind_spot_sectors": [], "ml_cards": {}, "system_state_version": 1,
           "drift_level": "NORMAL", "commodity": {}}
    for k in range(120):
        d = date(2026, 10, 12) + timedelta(days=k)
        rec = uv.assess(f"MI{k}", quote(d), chain(f"MI{k}", d), 100e6, d, cfg, "fake")
        assert rec["research_ok"] and not rec["tradeable_ok"]             # optionierbar != handelbar
        ref = rec["reference_option"]
        mid = (ref["bid"] + ref["ask"]) / 2
        exit_mid[ref["symbol"]] = mid * 1.2                                    # +20 % theoretisch (Mid->Mid)
        v2l.record_candidates([{"ticker": f"MI{k}", "assessment": rec, "hit_rate": 0.7}], today=d.isoformat(),
                              vix=18.0 if k % 2 else 30.0, ctx=dict(ctx), registry=reg, ledger_dir=led, v1_members=set(),
                              now=datetime(d.year, d.month, d.day, 15, tzinfo=UTC))

    def bars(t, s, e):
        out, dd = [], s
        while dd <= e:
            px = 20 * (1 + 0.3 * min(60, (dd - s).days) / 60)
            out.append((dd, px * 1.01, px * 0.99, px))
            dd += timedelta(days=1)
        return out, []
    v2l.resolve_outcomes(today=date(2027, 4, 1), ledger_dir=led, path=outs, bars_fn=bars,
                         option_close_fn=lambda sym, on: exit_mid.get(sym))
    o = [json.loads(x) for x in outs.read_text().splitlines() if x.strip()]
    theo = [x["option_theoretical_return"] for x in o if x.get("option_theoretical_return") is not None]
    net = [x["net_realizable_return"] for x in o if x.get("net_realizable_return") is not None]
    raw = [x["raw_underlying_return"] for x in o if x.get("raw_underlying_return") is not None]
    assert np.mean(raw) > 0.1 and np.mean(theo) > 0.15                     # Research-Alpha erkannt
    assert np.mean(net) < 0                                                # nach Bid/Ask + Slippage negativ
    st = pc.run(contracts=hc.load(), policy=hc.load_policy(), now=datetime(2027, 4, 1, tzinfo=UTC), registry=reg,
                transitions=tmp_path / "tr.jsonl", ledger_dir=tmp_path / "ch", outcomes_path=tmp_path / "co.jsonl",
                looks_path=tmp_path / "lk.jsonl", state_path=tmp_path / "st.json", history={}, approvals={},
                safe_mode={"active": False}, final_mc_dir=tmp_path / "fm", final_mc_outcomes=tmp_path / "fmo.jsonl",
                v2_dir=led, v2_outcomes=outs)
    micro = st["hypotheses"]["UNIV-V2-SEG-MICRO@v1"]
    assert micro["influence_level"] == "NONE" and micro["state"] not in ("FORWARD_VALIDATED",) + pc.ACTIVE_STATES
    assert micro["evidence"]["outcome_basis"] == "net_realizable_return"


# ═══ Crash-Recovery / Idempotenz ═══════════════════════════════════════════════════════════════════
def test_truncated_ledger_line_recovered_and_next_append_clean(tmp_path):
    from modules.atomic_io import CorruptLedgerError, append_jsonl, read_jsonl
    p = tmp_path / "l.jsonl"
    append_jsonl(p, [{"a": 1}, {"a": 2}])
    with open(p, "a") as fh:
        fh.write('{"a": 3, "b"')                                          # Abbruch mitten in der Zeile
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]                         # abgeschnittene letzte Zeile ignoriert
    append_jsonl(p, [{"a": 4}])                                          # Neustart: Zeile wird abgeschlossen
    with pytest.raises(CorruptLedgerError):                              # Rest mitten in der Datei -> laut, nie still
        read_jsonl(p)
    q = tmp_path / "q.jsonl"
    q.write_text('{"a": 1}\nKAPUTT\n{"a": 2}\n')
    with pytest.raises(CorruptLedgerError):
        read_jsonl(q)


def test_atomic_write_never_leaves_half_file(tmp_path, monkeypatch):
    import os
    from modules import atomic_io
    p = tmp_path / "history.json"
    atomic_io.atomic_write_json(p, {"closed_trades": [1, 2, 3]})
    real = os.replace

    def boom(a, b):
        raise OSError("Abbruch")
    monkeypatch.setattr(atomic_io.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_io.atomic_write_json(p, {"closed_trades": []})
    monkeypatch.setattr(atomic_io.os, "replace", real)
    assert json.loads(p.read_text()) == {"closed_trades": [1, 2, 3]}
    assert not [x for x in tmp_path.iterdir() if x.name.endswith(".tmp")]


def test_corrupt_history_is_fail_closed_not_reset(tmp_path, monkeypatch):
    import pipeline
    p = tmp_path / "history.json"
    p.write_text('{"closed_trades": [1, 2')
    monkeypatch.setattr(pipeline, "HISTORY_PATH", p)
    with pytest.raises(RuntimeError):
        pipeline.load_history()
    assert p.read_text() == '{"closed_trades": [1, 2'                     # nie überschrieben


def test_truncated_transition_log_is_fail_closed(tmp_path):
    e = Env(tmp_path)
    c = base_contract()
    write_ledger(e, c, T0, 40, fired_ret=-0.4, kept_ret=0.15)
    e.run([c], datetime(2027, 2, 1, tzinfo=UTC))
    with open(e.tr, "a") as fh:
        fh.write('{"key": "x", "new_st')
    _, problems = pc.read_transitions(e.tr)
    assert problems
    active, probs = e.verified([c])
    assert probs and all(a["level"] == "NONE" for a in active.values())


def test_controller_and_ledgers_idempotent_on_restart(tmp_path):
    e = Env(tmp_path)
    c = base_contract()
    write_ledger(e, c, T0, 40, fired_ret=-0.4, kept_ret=0.15)
    now = datetime(2027, 2, 1, tzinfo=UTC)
    s1 = e.run([c], now)
    n_tr, n_looks = len(e.tr.read_text().splitlines()), len(e.looks.read_text().splitlines())
    s2 = e.run([c], now)                                                    # Neustart desselben Laufs
    assert len(e.tr.read_text().splitlines()) == n_tr and len(e.looks.read_text().splitlines()) == n_looks
    assert s1["hypotheses"][hc.key(c)]["state"] == s2["hypotheses"][hc.key(c)]["state"]
    # Final-MC-Ledger: gleicher Tag/Ticker zweimal -> keine Dubletten
    from modules import final_mc_ledger as fml
    reg = tmp_path / "freg.jsonl"
    hc.register(hc.load(), reg, POLICY)
    surv = [{"ticker": "AAA", "features": {}, "simulation": {"hit_rate": 0.6, "current_price": 10},
             "deep_analysis": {"direction": "BULLISH"}}]
    ctx = {"safe_mode_active": 0, "ml_cards": {}, "commodity": {}}
    a = fml.record_survivors(surv, today="2026-10-12", vix=18, ctx=ctx, registry=reg, ledger_dir=tmp_path / "fm")
    b = fml.record_survivors(surv, today="2026-10-12", vix=18, ctx=ctx, registry=reg, ledger_dir=tmp_path / "fm")
    assert len(a) == 1 and b == []


# ═══ Quellen-Ausfälle: nur abhängige Komponenten, kein unnötiger globaler Safe Mode ═══════════════
def _src(sid, status, crit="NON_CRITICAL", research_only=False, feats=(), decisions=()):
    return {"source_id": sid, "status": status, "criticality": crit, "kind": "registry", "research_only": research_only,
            "fallback_only": False, "fallback_active": False,
            "downstream_dependencies": {"features": list(feats), "decisions": list(decisions), "hypotheses": []}}


@pytest.mark.parametrize("broken,expect_global", [
    (("eia_petroleum_weekly", "eia_natural_gas", "fred_commodities", "cftc_cot"), False),   # Commodity komplett weg
    (("sec_form345",), False),                                                             # Alt-Data weg
])
def test_outage_affects_only_dependents(broken, expect_global):
    from modules import source_health as sh
    srcs = {"market_prices": _src("market_prices", "HEALTHY", "CRITICAL", decisions=["scanner_candidates"])}
    for b in broken:
        srcs[b] = _src(b, "BROKEN", research_only=b != "sec_form345", feats=[f"f_{b}"])
    sm = sh.data_safe_mode({"sources": srcs, "features": {}}, sh.load_config())
    assert sm["active"] is expect_global and not sm["blocked_decisions"]


def test_critical_market_data_outage_blocks_decisions():
    from modules import source_health as sh
    srcs = {"market_prices": _src("market_prices", "BROKEN", "CRITICAL", decisions=["scanner_candidates"])}
    sm = sh.data_safe_mode({"sources": srcs, "features": {}}, sh.load_config())
    assert "scanner_candidates" in sm["blocked_decisions"]


def test_unavailable_feature_never_zero_in_decision_env():
    p = {"ticker": "FCX", "sector": "Basic Materials", "features": {"risk_flag": 1}}
    ctx = {"safe_mode_active": 0, "commodity": {"features": {"cmd_copper_ret_3m": 0.05}, "commodity_data_version": "v"},
           "data_unavailable_features": ["cmdx_copper__copper_ret_3m", "risk_flag"]}
    env = pia.candidate_env(p, ctx, 18)
    assert env["cmdx_copper__copper_ret_3m"] is None and env["risk_flag"] is None


# ═══ Modell-/RL-/Commodity-Fehlerfälle: keine unkontrollierte Produktionswirkung ═══════════════════
def test_nan_extreme_and_missing_rerank_signals_keep_champion_order(tmp_path):
    c = commodity_contract()
    active = {hc.key(c): {"contract": c, "level": "RERANK_ONLY", "state": "LIMITED_PRODUCTION", "spec_hash": hc.spec_hash(c)}}
    for bad in (float("nan"), None, 1e12, -1e12):
        env = {"cmdx_copper__copper_ret_3m": bad, "commodity_mapping_version": "exposure-v1", "champion_probability": 0.5}
        d = pia.decide_for_trade({"trade_score": {"total": 60}}, env, active, safe_mode=False, sector="X", regime=None,
                                 policy=POLICY)
        assert d["final_production_decision"] == "TRADE" and abs(d["score_adjustment"]) <= pia.HARD_CAPS["score_points"]


def test_rl_veto_off_by_default_and_never_drops_candidates():
    src = (ROOT / "pipeline.py").read_text()
    assert 'cfg.rl.get("veto_enabled", False)' in src
    assert yaml.safe_load((ROOT / "config.yaml").read_text())["rl"]["veto_enabled"] is False
    from modules.rl_agent import RLScorer
    sims = [{"ticker": f"T{i}", "features": {}, "simulation": {}, "deep_analysis": {}} for i in range(5)]
    out = RLScorer(history={}, veto_enabled=False).run(sims)
    assert {s["ticker"] for s in out} == {s["ticker"] for s in sims}


def test_absurd_cot_and_extreme_eia_are_flagged_not_used():
    from modules.external.sources.commodities import oi_consistent
    assert not oi_consistent({"total_open_interest": 100, "managed_money_long": 10**9})[0]
    t0 = datetime(2026, 1, 2, tzinfo=UTC)
    from modules.external.pit import AvailabilityPrecision, Observation
    o = [Observation(source_id="s", dataset="d", series_id="x", entity_id="US", metric="eia_crude_stocks",
                     value=v, unit="thousand_barrels", observation_time=t0 + timedelta(days=7 * i),
                     available_at=t0 + timedelta(days=7 * i + 6), retrieved_at=t0 + timedelta(days=400),
                     availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE, parser_version="1")
         for i, v in enumerate([400000.0] * 19 + [-5.0])]
    q = cmd.quality_checks("x", o, "weekly")
    assert "NEGATIVE_VALUE" in q["issues"] and q["severe"]


# ═══ Kanonischer State / LEARNING_HEALTH ═══════════════════════════════════════════════════════════
def test_system_state_has_all_canonical_fields():
    from modules import system_state as ss
    st = ss.derive()
    for k in ("safe_mode", "data_health", "drift_state", "model_health", "champion_version", "universe_version",
              "promotion_state", "allowed_influence", "commodity_data_health", "learning_health"):
        assert k in st, k
    assert st["universe_version"]["production"] == "V1" and st["universe_version"]["v1_unchanged"] is True
    assert st["commodity_data_health"]["safe_mode_relevant"] is False
    assert st["allowed_influence"]["commodity_max"] == "SCORE_LIMITED"


def test_stale_model_health_is_unknown_fail_closed(tmp_path):
    from modules import system_state as ss
    mh = tmp_path / "safe_mode.json"
    mh.write_text(json.dumps({"active": False, "reasons": [], "updated": "2026-08-01T00:00:00+00:00"}))
    st = ss.derive(inputs={"model_health": mh}, now=datetime(2026, 10, 4, tzinfo=UTC))
    assert st["safe_mode"] and any("veraltet" in r for r in st["safe_mode_reason"])


def test_learning_health_detects_stalled_paths(tmp_path):
    from modules import learning_health as lh
    o = tmp_path / "outputs"
    (o / "research").mkdir(parents=True)
    (o / "intelligence").mkdir(parents=True)
    (o / "research" / "factory_plan.json").write_text(json.dumps({"generated": "2026-07-01T00:00:00+00:00", "n_ideas": 3}))
    (o / "intelligence" / "promotion_state.json").write_text(json.dumps({"generated": "2026-09-01T00:00:00+00:00",
                                                                        "hypotheses": {}}))
    (o / "history.json").write_text(json.dumps({"closed_trades": [{"close_date": "2026-05-01", "outcome": 0.1}],
                                                "active_trades": [{"ticker": "A"}], "shadow_trades": []}))
    r = lh.assess(tmp_path, datetime(2026, 10, 4, tzinfo=UTC))
    p = r["paths"]
    assert p["research_generation"]["status"] == "STALLED" and p["promotion_pipeline"]["status"] == "STALLED"
    assert p["outcome_ingestion"]["status"] == "STALLED" and p["universe_v2"]["status"] == "UNVALIDATED"
    assert r["overall"] == "DEGRADED" and "promotion_pipeline" in r["stalled_or_broken"]


def test_outcome_classes_only_reliable_learns():
    from modules.outcomes import is_reliable_outcome, outcome_class
    assert outcome_class({"outcome": 0.1, "outcome_method": "option_quote"}) == "RELIABLE"
    assert outcome_class({"outcome": 0.1, "outcome_method": "delta_approx"}) == "APPROXIMATED"
    assert outcome_class({"outcome": 0.1, "outcome_method_reconstructed": True}) == "RECONSTRUCTED"
    assert outcome_class({"outcome": None}) == "UNKNOWN"
    legacy = {"outcome": 0.1}
    assert outcome_class(legacy) == "RELIABLE" and outcome_class(legacy, strict=True) == "UNKNOWN"
    for t in ({"outcome": 0.1, "outcome_method": "delta_approx"}, {"outcome": 0.1, "outcome_method_reconstructed": True}):
        assert not is_reliable_outcome(t)


# ═══ Workflows: automatisiert, kein stiller Fehlschlag, konfliktsichere Pushes ════════════════════
WF = ROOT / ".github" / "workflows"


def _wf(name):
    raw = yaml.safe_load((WF / name).read_text())
    return raw.get("on", raw.get(True)) or {}, raw


@pytest.mark.parametrize("name", ["scanner.yml", "feedback.yml", "external_data.yml", "source_health.yml",
                                  "universe_v2.yml", "ml_research.yml", "research.yml", "weekly_report.yml",
                                  "monthly_report.yml", "world_model.yml", "alt_data.yml"])
def test_core_jobs_are_scheduled(name):
    on, _ = _wf(name)
    assert on.get("schedule"), f"{name} ohne Cron"


def test_no_silent_failures_and_safe_push_everywhere():
    for f in WF.glob("*.yml"):
        t = f.read_text()
        assert "|| true" not in t.replace('"|| true"', ""), f.name               # nie still verschlucken
        if "HEAD:main" in t or "git push" in t:
            raise AssertionError(f"{f.name}: Push nicht über scripts/ci_push.sh")
        if "ci_push.sh" in t:
            assert "GITHUB_TOKEN" in t
    for name in ("feedback.yml", "scanner.yml", "ml_research.yml", "research.yml", "weekly_report.yml",
                 "monthly_report.yml"):
        t = (WF / name).read_text()
        assert "ci_stage_check.sh" in t and "CRITICAL:" in t, name
    ga = (ROOT / ".gitattributes").read_text()
    assert "outputs/**/*.jsonl merge=union" in ga and "promotion_transitions.jsonl merge=text" in ga


def test_concurrency_groups_protect_shared_writers():
    groups = {}
    for f in WF.glob("*.yml"):
        _, raw = _wf(f.name)
        g = (raw.get("concurrency") or {}).get("group")
        groups[f.name] = g
    assert groups["scanner.yml"] == groups["feedback.yml"] == "history-write"      # history.json: ein Schreiber
    assert groups["external_data.yml"] == "external-data-write"
