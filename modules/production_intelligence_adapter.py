"""modules/production_intelligence_adapter.py – EINZIGE Schnittstelle Research→Produktion.

pipeline.py ruft genau eine Funktion auf: apply_to_proposals(). Sie

  1. lädt promotion_state.json und verifiziert ALLES (state_hash, Vertrags-Hash ==
     Registry-Hash == State-Hash, Registry-/Transition-Ketten). Jeder Fehler ->
     fail-safe: Champion unverändert (Einfluss NONE), Entscheidung trotzdem geloggt;
  2. wertet jede registrierte Hypothese (Shadow UND aktiv) je Champion-Trade aus und
     friert das Ergebnis im Decision-Ledger ein (Grundlage der Forward-Evidenz);
  3. wendet NUR den vom PromotionController freigegebenen, gekappten Einfluss an:
       ABSTENTION_ONLY  Champion-Trade blockieren (nie einen erzeugen)
       RERANK_ONLY      Reihenfolge der vom Champion akzeptierten Trades
       SCORE_LIMITED    ±HARD_CAPS["score_points"] Trade-Score-Punkte
       WEIGHT_10/25     P_final = (1-w)·P_champion + w·P_intelligence, w <= 0.10/0.25
     Safe Mode: kein positiver Boost, kein Gewicht, keine neuen Trades – nur bereits
     validierte defensive Abstinenz;
  4. speichert je Trade Champion-, Intelligence- und finale Entscheidung parallel.

Kein anderer Research-Pfad darf Produktionsentscheidungen verändern
(tests/test_promotion.py prüft das statisch für pipeline.py und die Research-Module).
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from modules import hypothesis_contract as hc
from modules import promotion_controller as pc

log = logging.getLogger(__name__)

# Harte Decke – Policy kann nur senken, nie anheben.
HARD_CAPS = {"score_points": 3.0, "weight": {"WEIGHT_10": 0.10, "WEIGHT_25": 0.25}, "max_weight": 0.25}
LEVEL_ORDER = hc.INFLUENCE_LEVELS
NEXT_VALIDATION = Path("outputs/research/next_validation.json")
ABSTAIN, TRADE = "ABSTAIN", "TRADE"


# ── Zustand laden + verifizieren ────────────────────────────────────────────
def load_verified_state(state_path: Path | None = None, contracts: list[dict] | None = None,
                        registry: Path | None = None, transitions: Path | None = None) -> tuple[dict, list[str]]:
    """-> ({key: {contract, level, state, spec_hash}}, Probleme). Fehler -> leer (fail-safe)."""
    state_path = state_path or pc.STATE_PATH
    problems: list[str] = []
    if not state_path.exists():
        return {}, ["promotion_state.json fehlt – nur Shadow-Logging"]
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {}, [f"promotion_state.json unlesbar: {e}"]
    if pc.state_digest(state) != state.get("state_hash"):
        return {}, ["promotion_state.json manipuliert (state_hash)"]
    contracts = contracts if contracts is not None else hc.load()
    reg, reg_problems = hc.read_registry(registry)
    _, tr_problems = pc.read_transitions(transitions)
    if reg_problems or tr_problems:
        return {}, reg_problems + tr_problems
    reg_hash = {e["key"]: e["spec_hash"] for e in reg}
    cur = pc.current_states(transitions)
    out = {}
    for c in contracts:
        k = hc.key(c)
        h = hc.spec_hash(c)
        s = (state.get("hypotheses") or {}).get(k)
        if s is None or reg_hash.get(k) != h or s.get("spec_hash") != h or not s.get("integrity_ok"):
            problems.append(f"{k}: Hash/Registry-Abweichung – ignoriert")
            continue
        t = cur.get(k) or {}
        # Zustand aus der (verifizierten) Transition-Kette ist maßgeblich; State-Datei muss übereinstimmen
        if t.get("new_state") != s.get("state") or t.get("influence_level", "NONE") != s.get("influence_level"):
            problems.append(f"{k}: State-Datei weicht von Transition-Kette ab – ignoriert")
            continue
        level = t.get("influence_level", "NONE")
        cls_max = hc._CLASS_MAX_LEVEL.get(c["production_class"], "NONE")
        if LEVEL_ORDER.index(level) > LEVEL_ORDER.index(cls_max):
            problems.append(f"{k}: Stufe {level} > Klasse {c['production_class']} – gekappt")
            level = cls_max
        out[k] = {"contract": c, "level": level, "state": t.get("new_state"), "spec_hash": h}
    return out, problems


# ── Kontext ─────────────────────────────────────────────────────────────────
def research_context(today: str | None = None) -> dict:
    """Read-only Research-Kontext für die Regelauswertung (nie Schreibzugriff)."""
    ctx = {"safe_mode_active": None, "blind_spot_sectors": [], "ml_cards": {}}
    # Safe Mode = Modell/Drift (meta_cognition) ODER Daten (täglicher Source Health Check).
    # Unbekannt -> aktiv (fail-closed: keine positiven Boosts).
    from modules.source_health import effective_safe_mode
    sm = effective_safe_mode()                      # kanonischer SystemState (Standard-Eingaben)
    ctx["safe_mode_active"] = 1 if sm["active"] else 0
    ctx["safe_mode_reasons"] = sm["reasons"]
    ctx["data_disabled_signals"] = sm["disabled_signals"]
    ctx["data_unavailable_features"] = sm["unavailable_features"]
    ctx["drift_level"] = sm.get("drift_level")
    ctx["drift_consequences"] = sm.get("consequences") or {}
    ctx["system_state_version"] = sm.get("state_version")
    try:
        nv = json.loads(NEXT_VALIDATION.read_text(encoding="utf-8"))
        ctx["blind_spot_sectors"] = sorted({(c.get("common_properties") or {}).get("sector")
                                            for c in nv.get("blind_spot_clusters") or []} - {None})
    except (OSError, ValueError):
        pass
    try:
        from modules.ml_research import latest_cards
        ctx["ml_cards"] = latest_cards(today=today)
    except (OSError, ValueError, KeyError, ImportError) as e:
        log.warning(f"Adapter: ML-Karten nicht lesbar ({e})")
    try:                                            # Commodity-PIT-Merkmale (nur lesend, Einfluss nur per Vertrag)
        from modules import commodity_intelligence as _cmd
        ctx["commodity"] = _cmd.decision_snapshot()
    except Exception as e:  # noqa: BLE001 – optionaler Research-Kontext
        log.warning(f"Adapter: Commodity-Snapshot nicht lesbar ({e})")
        ctx["commodity"] = {}
    return ctx


def regime_label(vix) -> str | None:
    if vix is None:
        return None
    v = float(vix)
    return "vix_low" if v < 15 else "vix_normal" if v < 25 else "vix_high" if v < 35 else "vix_stress"


def candidate_env(p: dict, ctx: dict, vix=None) -> dict:
    """Merkmale je Trade: Champion-Features + Research-Kontext. Fehlend bleibt None (nie 0)."""
    env = {k: v for k, v in (p.get("features") or {}).items() if isinstance(v, (int, float, bool))}
    card = (ctx.get("ml_cards") or {}).get(p.get("ticker")) or {}
    sector = p.get("sector") or (p.get("info") or {}).get("sector") or (p.get("features") or {}).get("sector")
    env.update({"safe_mode_active": ctx.get("safe_mode_active"), "vix": vix,
                "trade_score": (p.get("trade_score") or {}).get("total"),
                "champion_probability": (p.get("simulation") or {}).get("hit_rate"),
                "ml_disagreement_sd": card.get("model_disagreement_sd"),
                "ml_exp_dd_60": card.get("expected_drawdown_60"),
                "ml_exp_ret_60": card.get("expected_return_60"),
                "ml_q10_ret_60": (card.get("interval_80") or [None])[0],
                "blind_spot_sector_match": (1 if sector in (ctx.get("blind_spot_sectors") or []) else 0)
                if sector else None})
    try:                                            # Commodity-Merkmale (RESEARCH; fehlend -> None, nie 0)
        from modules import commodity_intelligence as _cmd
        env.update({k: v for k, v in _cmd.decision_features(p.get("ticker"), ctx.get("commodity"),
                                                             ticker_known=bool(sector)).items()
                    if k != "commodity_data_version"})
    except Exception as e:  # noqa: BLE001
        log.warning(f"Adapter: Commodity-Merkmale nicht ableitbar ({e})")
    for f in ctx.get("data_unavailable_features") or []:   # Source Health: Quelle nicht nutzbar -> unbekannt, nie alt/0
        if f in env:
            env[f] = None
    return env


# ── Kern ────────────────────────────────────────────────────────────────────
def decide_for_trade(p: dict, env: dict, active: dict, *, safe_mode: bool, sector: str | None,
                     drift: dict | None = None,
                     regime: str | None, requested_weights: dict | None = None,
                     policy: dict | None = None) -> dict:
    """Reine Funktion: Champion-Trade -> Intelligence-/Final-Entscheidung (ohne I/O)."""
    policy = policy or {}
    infl = policy.get("influence") or {}
    score_cap = min(HARD_CAPS["score_points"], float(infl.get("score_adjustment_max_points", HARD_CAPS["score_points"])))
    per_h, abstain_by, abstain_applied, score_raw, weights = {}, [], [], 0.0, []
    p_champ = env.get("champion_probability")
    rerank_score, rerank_missing = 0.0, False
    for k, a in active.items():
        c, level = a["contract"], a["level"]
        cls, direction = c["production_class"], int(c["direction"])
        scope = hc.in_scope(c, sector, regime)
        f = hc.fires(c, env) if scope else None
        val = hc.evaluate_signal(c, env) if scope else None
        applied = False
        if cls == "abstention":
            if f:
                abstain_by.append(k)                       # was Intelligence täte (auch im Shadow)
                if level == "ABSTENTION_ONLY":              # angewendet nur bei Freigabe
                    abstain_applied.append(k)
                    applied = True
        elif cls in ("rerank", "score", "weight") and level in ("RERANK_ONLY", "SCORE_LIMITED", "WEIGHT_10",
                                                                 "WEIGHT_25"):
            if val is None:
                rerank_missing = True
            else:
                rerank_score += direction * val
                applied = True
            if cls in ("score", "weight") and level == "SCORE_LIMITED" and f:
                score_raw += float(c.get("score_points", score_cap)) * direction
            if cls == "weight" and level in ("WEIGHT_10", "WEIGHT_25") and val is not None:
                w_req = float((requested_weights or {}).get(k, HARD_CAPS["weight"][level]))
                w_pol = float(infl.get("weight_level_2" if level == "WEIGHT_25" else "weight_level_1",
                                       HARD_CAPS["weight"][level]))
                w_cap = min(w_req, HARD_CAPS["weight"][level], HARD_CAPS["max_weight"], w_pol)
                weights.append((k, max(0.0, w_cap), min(1.0, max(0.0, val))))
        per_h[k] = {"spec_hash": a["spec_hash"], "level": level, "state": a["state"], "in_scope": scope,
                    "evaluable": f is not None, "fired": bool(f), "signal_value": val, "applied": applied}
    intelligence_decision = ABSTAIN if abstain_by else TRADE
    score_adj = max(-score_cap, min(score_cap, score_raw))
    if safe_mode:
        score_adj = min(0.0, score_adj)          # kein positiver Boost
        weights = []                             # kein erhöhtes Ensemble-Gewicht
    elif drift:                                  # abgestufte Drift (SystemState): Boosts begrenzen
        if score_adj > 0:
            score_adj = score_adj * float(drift.get("positive_boost_cap", 1.0))
        if not drift.get("allow_weight_increase", True):
            weights = []
    prob_adj = 0.0
    if weights and p_champ is not None:
        w_tot = min(HARD_CAPS["max_weight"], sum(w for _, w, _ in weights))
        p_int = sum(w * v for _, w, v in weights) / max(1e-9, sum(w for _, w, _ in weights))
        prob_adj = round(((1 - w_tot) * p_champ + w_tot * p_int) - p_champ, 5)
    final = ABSTAIN if abstain_applied else TRADE
    reason = None
    if abstain_applied:
        reason = "validierte Abstinenz: " + ", ".join(abstain_applied)
    return {"intelligence_decision": intelligence_decision, "final_production_decision": final,
            "abstention_reason": reason, "hypotheses": per_h, "would_abstain_by": abstain_by,
            "abstain_applied_by": abstain_applied, "score_adjustment_raw": round(score_raw, 3),
            "score_adjustment": round(score_adj, 3), "probability_adjustment": prob_adj,
            "influence_level": max([a["level"] for a in active.values()] or ["NONE"], key=LEVEL_ORDER.index),
            "rerank_score": None if rerank_missing else round(rerank_score, 6)}


def _config_version() -> str | None:
    """Hash des Champion-Regelwerks – dieselbe Ableitung wie SystemState.champion_version (nur lesend)."""
    from modules.system_state import DEFAULT_INPUTS, _hash_file
    return _hash_file(DEFAULT_INPUTS["champion_config"])


def _prompt_version() -> str | None:
    try:                                    # Konstante aus deep_analysis (Hash aus System-Prompt + Template)
        from modules.deep_analysis import PROMPT_VERSION
        return PROMPT_VERSION
    except Exception:  # noqa: BLE001 – optional (Tests ohne LLM-Abhängigkeiten)
        return None


_CONFIG_VERSION = _config_version()
_PROMPT_VERSION = None


def apply_to_proposals(proposals: list[dict], *, vix=None, today: str | None = None, context: dict | None = None,
                       state_path: Path | None = None, contracts: list[dict] | None = None,
                       registry: Path | None = None, transitions: Path | None = None,
                       ledger_dir: Path | None = None, policy: dict | None = None,
                       requested_weights: dict | None = None, now: datetime | None = None,
                       meta_versions: dict | None = None,
                       trade_score_min: float | None = None,
                       mc_threshold: float | None = None) -> tuple[list[dict], list[tuple[dict, str]], list[dict]]:
    """-> (umzusetzende Trades in finaler Reihenfolge, [(blockierter Trade, Grund)], Entscheidungs-Records).
    Erzeugt NIE neue Trades: Ausgabe ⊆ Eingabe."""
    global _PROMPT_VERSION
    if _PROMPT_VERSION is None:
        _PROMPT_VERSION = _prompt_version()
    now = now or datetime.now(timezone.utc)
    today = today or now.strftime("%Y-%m-%d")
    policy = policy if policy is not None else hc.load_policy()
    ctx = context if context is not None else research_context(today)
    contracts = contracts if contracts is not None else hc.load()
    # Nur Verträge der Champion-Population wirken hier; andere Populationen (z. B. FINAL_MC_SURVIVOR)
    # werden ausschließlich in ihrem eigenen Ledger ausgewertet (modules/final_mc_ledger.py).
    from modules.final_mc_ledger import CHAMPION_STAGE, stage_of
    contracts = [c for c in contracts if stage_of(c) == CHAMPION_STAGE]
    active, problems = load_verified_state(state_path, contracts, registry, transitions)
    for pr in problems:
        log.warning(f"Adapter: {pr}")
    # Shadow-Auswertung auch ohne verifizierten State: alle gültig registrierten Verträge mit Einfluss NONE
    if not active:
        reg, reg_problems = hc.read_registry(registry)
        ok = {e["key"]: e["spec_hash"] for e in reg} if not reg_problems else {}
        active = {hc.key(c): {"contract": c, "level": "NONE", "state": "UNVERIFIED", "spec_hash": hc.spec_hash(c)}
                  for c in contracts if ok.get(hc.key(c)) == hc.spec_hash(c)}
    off = set(ctx.get("data_disabled_signals") or [])
    missing = set(ctx.get("data_unavailable_features") or [])
    if off or missing:                               # Pflichtdaten fehlen -> Signal gar nicht verwenden
        for k in [k for k, a in active.items() if k in off or a["contract"].get("hypothesis_id") in off
                  or set(a["contract"].get("features") or []) & missing]:
            log.warning(f"Adapter: {k} deaktiviert – Pflichtdaten laut Source Health nicht verfügbar")
            active.pop(k)
    safe_mode = bool(ctx.get("safe_mode_active"))
    regime = regime_label(vix)
    kept, blocked, records = [], [], []
    try:                                              # Abstention Intelligence (SHADOW, nur Messung)
        from modules import abstention_intelligence as _ai
        _ai_extras = _ai.load_context_extras(ctx)
    except Exception as e:  # noqa: BLE001 – Risikovektor ist optional, nie entscheidungsrelevant
        log.warning(f"Adapter: Risikovektor nicht verfügbar ({e})")
        _ai = _ai_extras = None
    for rank, p in enumerate(proposals, start=1):
        env = candidate_env(p, ctx, vix)
        risk = None
        if _ai is not None:
            try:
                risk = _ai.risk_vector(p, env, ctx, _ai_extras, trade_score_min=trade_score_min,
                                       mc_threshold=mc_threshold)
                rf = _ai.as_features(risk)
                env.update(rf)                        # für künftige, registrierte Verträge
                if isinstance(p.get("features"), dict):
                    p["features"].update(rf)          # -> Trade-Record -> abstention_proposals (Walk-Forward)
            except Exception as e:  # noqa: BLE001
                log.warning(f"Adapter: Risikovektor für {p.get('ticker')} nicht berechenbar ({e})")
        sector = p.get("sector") or (p.get("info") or {}).get("sector")
        d = decide_for_trade(p, env, active, safe_mode=safe_mode, sector=sector, drift=ctx.get("drift_consequences"),
                             regime=regime,
                             requested_weights=requested_weights, policy=policy)
        base = (p.get("trade_score") or {}).get("total")
        if (d["final_production_decision"] == TRADE and trade_score_min is not None and base is not None
                and d["score_adjustment"] < 0 and base + d["score_adjustment"] < trade_score_min):
            d["final_production_decision"] = ABSTAIN        # Score-Einfluss senkt unter die Champion-Schwelle
            d["abstention_reason"] = f"Score-Einfluss {d['score_adjustment']} -> unter trade_score_min"
            d["abstain_applied_by"] = [k for k, v in d["hypotheses"].items() if v["applied"]]
        rec = {"decision_id": hashlib.sha256(f"{today}|{p.get('ticker')}|{now.isoformat()}|{rank}".encode()
                                             ).hexdigest()[:16],
               "timestamp": now.isoformat(timespec="seconds"), "date": today, "ticker": p.get("ticker"),
               "sector": sector, "regime": regime, "champion_decision": TRADE, "champion_score": base,
               "champion_probability": env.get("champion_probability"), "champion_rank": rank,
               "intelligence_decision": d["intelligence_decision"],
               "intelligence_adjustment": {"score": d["score_adjustment"], "score_raw": d["score_adjustment_raw"],
                                           "probability": d["probability_adjustment"]},
               "final_production_decision": d["final_production_decision"],
               "production_score": (round(base + d["score_adjustment"], 2) if base is not None else None),
               "abstention_reason": d["abstention_reason"], "influence_level": d["influence_level"],
               "active_hypotheses": [k for k, v in d["hypotheses"].items() if v["level"] != "NONE"],
               "hypothesis_versions": {k: v["spec_hash"][:12] for k, v in d["hypotheses"].items()},
               "promotion_levels": {k: v["level"] for k, v in d["hypotheses"].items()},
               "intelligence": d["hypotheses"], "safe_mode_state": safe_mode,
               "system_state_version": ctx.get("system_state_version"), "drift_level": ctx.get("drift_level"),
               "meta_model_version": (meta_versions or {}).get("meta_model_version"),
               "world_model_version": (meta_versions or {}).get("world_model_version"),
               "data_snapshot": {"vix": vix, "env_hash": hashlib.sha256(json.dumps(env, sort_keys=True,
                                                                                    default=str).encode()).hexdigest()[:16]},
               "code_commit": pc.code_version(), "integrity_problems": problems, "risk_vector": risk,
               # Reproduzierbarkeit (Audit 2026-10-04): Versionen aller Eingaben; nicht exakt
               # reproduzierbare Live-Eingaben sind ausdrücklich markiert.
               "universe_version": p.get("universe_version") or "V1", "config_version": _CONFIG_VERSION,
               "contract_hashes": {k: v["spec_hash"] for k, v in d["hypotheses"].items()},
               "prompt_version": (p.get("deep_analysis") or {}).get("prompt_version") or _PROMPT_VERSION,
               "commodity_features_used": any(k.startswith(("cmd_", "cmdx_", "cmdexp_")) for c in
                                              (v["contract"] for v in active.values()) for k in c.get("features") or []),
               "commodity_data_version": (ctx.get("commodity") or {}).get("commodity_data_version"),
               "non_reproducible_inputs": ["live_option_chain", "llm_response", "live_quotes"],
               "confidence": None, "evidence_snapshot": {k: v["state"] for k, v in active.items()}}
        records.append(rec)
        if d["final_production_decision"] == ABSTAIN:
            blocked.append((p, "intelligence_abstention:" + ",".join(d["abstain_applied_by"])))
        else:
            kept.append((p, rec, d))
    # Rerank nur innerhalb der vom Champion akzeptierten Trades; fehlt einem Trade das
    # Rerank-Signal, bleibt die Champion-Reihenfolge (fehlend != 0).
    any_rerank = any(a["level"] in ("RERANK_ONLY", "SCORE_LIMITED", "WEIGHT_10", "WEIGHT_25")
                     for a in active.values())
    if any_rerank and not safe_mode and all(d["rerank_score"] is not None for _, _, d in kept):
        def sort_key(x):
            p, rec, d = x
            s = rec["production_score"] if rec["production_score"] is not None else float("-inf")
            pr = (rec["champion_probability"] or 0.0) + d["probability_adjustment"]
            return (-d["rerank_score"], -s, -pr, rec["champion_rank"])
        ordered = sorted(kept, key=sort_key)
    else:
        ordered = kept
    for i, (p, rec, _) in enumerate(ordered, start=1):
        rec["intelligence_rank"] = i
    for rec in records:
        rec.setdefault("intelligence_rank", None)
    write_ledger(records, ledger_dir)
    return [p for p, _, _ in ordered], blocked, records


def write_ledger(records: list[dict], ledger_dir: Path | None = None) -> None:
    ledger_dir = ledger_dir or pc.LEDGER_DIR
    if not records:
        return
    ledger_dir.mkdir(parents=True, exist_ok=True)
    month = records[0]["date"][:7]
    with open(ledger_dir / f"{month}.jsonl", "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def v2_segment_levels(state_path: Path | None = None, contracts: list[dict] | None = None,
                      registry: Path | None = None, transitions: Path | None = None) -> dict[str, str]:
    """Verifizierte Stufen der UNIVERSE_V2-Segment-Verträge (fail-safe: leer = alles SHADOW).
    Wirken ausschließlich auf V2-Kandidaten; V1-Champion-Trades werden davon nie berührt."""
    from modules.universe_v2_ledger import v2_contracts
    cs = v2_contracts(contracts)
    if not cs:
        return {}
    active, problems = load_verified_state(state_path, cs, registry, transitions)
    for pr in problems:
        log.warning(f"Adapter (V2): {pr}")
    return {k: a["level"] for k, a in active.items()}
