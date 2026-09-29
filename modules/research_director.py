"""
modules/research_director.py – Autonomer Research Director + Active Learning (NUR SHADOW)

    python -m modules.research_director     (CI: vor research_lab im monatlichen Lauf)

Entscheidet, welche Forschungsfragen als Nächstes getestet werden – aus
GEMESSENEN Befunden, nicht aus freier Ideengenerierung:
  * Blind-Spot-Cluster (next_validation.json)         -> "Segment meiden"-Hypothesen
  * Regime-Befunde (meta_learning.json)               -> regime-gegatete Varianten
  * knapp gescheiterte Hypothesen (INCONCLUSIVE)      -> regime-bedingte Retests
  * Modell-Drift / Failure-Profile                     -> Diagnose-Kandidaten
  * World-Model-Lücken/Unsicherheit, Kausal-Befunde   -> Active Learning (Datenlücken)
Jeder Kandidat: research_id, question, hypothesis, economic rationale,
required/available data, expected information gain, novelty, overfit risk,
research cost, priority. Priorität = EIG × Wert × Neuheit × Zuverlässigkeit
÷ (Beschaffungskosten × Forschungskosten).

Nur die besten K testbaren Kandidaten (Protokoll: max_new_per_run) werden als
Hypothesen an das Research-Lab übergeben (outputs/research/director_hypotheses.json)
– dort gelten dieselbe Prüfkette, BH über ALLE Tests und die Ähnlichkeits-
sperre gegen bereits Verworfenes. Neue Datenquellen werden nie automatisch als
vertrauenswürdig behandelt (Aufnahmeprüfung, config/data_catalog.yaml).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

OUT = Path("outputs/research")
CATALOG = Path("config/data_catalog.yaml")
MAX_NEW_PER_RUN = 5                  # begrenzt den Mehrfachtest-Nenner je Lauf

# Eigenschaft (blind_spots.properties) -> PIT-Ausdruck der Signal-Sprache (1 = im Segment)
PROPERTY_EXPR = {
    ("volatility", "high_vol"): "step(vol_60 - 0.17)", ("volatility", "low_vol"): "step(-0.17 - vol_60)",
    ("liquidity", "less_liquid"): "step(-log_dollar_vol)", ("liquidity", "liquid"): "step(log_dollar_vol)",
    ("momentum", "loser_12m"): "step(-0.17 - mom_12_1)", ("momentum", "winner_12m"): "step(mom_12_1 - 0.17)",
    ("recent_move", "extreme_5d_move"): "step(abs(ret_5d) - 0.4)",
    ("near_high", "near_52w_high"): "step(dist_52w_high - 0.3)",
    ("beta", "high_beta"): "step(beta_126 - 0.17)", ("beta", "low_beta"): "step(-0.17 - beta_126)",
    ("lottery", "lottery_profile"): "step(max_ret_21 - 0.4)",
    ("vix", "vix_ge_20"): "step(vix - 20)", ("vix", "vix_lt_20"): "step(20 - vix)",
    ("trend", "downtrend"): "step(-spy_trend_200)", ("trend", "uptrend"): "step(spy_trend_200)",
}


def _load(path: Path, default):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except (OSError, json.JSONDecodeError) as e:
        log.warning(f"research_director: {path} nicht lesbar ({e})")
        return default


def _rid(text: str) -> str:
    return "RD-" + hashlib.sha256(text.encode()).hexdigest()[:8]


def priority(eig: float, value: float, novelty: float, reliability: float, acq_cost: float, research_cost: float) -> float:
    return round(eig * value * novelty * reliability / max(acq_cost * research_cost, 1e-9), 5)


def _novelty(h: dict, tested: dict) -> float:
    from modules.research_lab import text_similarity
    sims = [text_similarity(h, r) for r in tested.values()]
    exact = any(str(r.get("signal", "")).replace(" ", "") == str(h.get("signal", "")).replace(" ", "") for r in tested.values())
    return 0.0 if exact else round(1.0 - max(sims, default=0.0), 3)


def candidates_from_blind_spots(clusters: list[dict]) -> list[dict]:
    out = []
    for c in clusters or []:
        props = c.get("common_properties") or {}
        exprs = [PROPERTY_EXPR.get((k, v)) for k, v in props.items()]
        if not exprs or any(e is None for e in exprs):
            out.append({"kind": "untestable", "question": f"Warum versagt das Ensemble im Segment {props}?",
                        "required_data": ["Sektor-/Ereignismerkmal in der Signal-Sprache"], "available_data": False,
                        "evidence_t": abs(c.get("lift", 1) - 1) * 3, "value": abs(c.get("typical_error") or 0),
                        "source_finding": c.get("id")})
            continue
        seg = " * ".join(exprs)
        out.append({"kind": "testable", "title": f"Segment meiden: {props}",
                    "question": f"Unterperformt das Segment {props} den Querschnitt systematisch?",
                    "statement": f"Titel im Segment {props} liefern unterdurchschnittliche 20-Tage-Relativrenditen (Blind Spot {c.get('id')})",
                    "signal": f"-({seg})", "direction": 1,
                    "rationale": f"Unknown-Unknown-Detektor: Lift {c.get('lift')}, typischer Fehler {c.get('typical_error')}",
                    "evidence_t": abs(c.get("lift", 1) - 1) * 3, "value": abs(c.get("typical_error") or 0),
                    "required_data": ["Feature-Store"], "available_data": True, "source_finding": c.get("id")})
    return out


def candidates_from_regimes(meta: dict, hyp_db: dict) -> list[dict]:
    """Meta-Befund: Ensemble wirkt nur in Stress-Regimen -> knapp gescheiterte
    Hypothesen regime-bedingt neu formulieren (neue id, eigener Test)."""
    out = []
    ref = (meta.get("approaches") or {}).get(meta.get("reference", "static_equal"), {})
    reg = ref.get("by_regime") or {}
    stress = (reg.get("vix_ge_20", {}).get("expectancy") or 0) - (reg.get("vix_lt_20", {}).get("expectancy") or 0)
    if stress <= 0:
        return out
    for hid, r in (hyp_db.get("hypotheses") or {}).items():
        if r.get("canonical_status") != "INCONCLUSIVE" or not r.get("signal") or "step(" in str(r.get("signal")):
            continue
        t = ((r.get("walk_forward") or {}).get("base") or {}).get("t_months") or 0
        for gate, name in (("step(vix - 20)", "VIX >= 20"), ("step(-spy_trend_200)", "SPY unter SMA200")):
            out.append({"kind": "testable", "title": f"{r.get('title')} – nur bei {name}",
                        "question": f"Wirkt '{r.get('title')}' nur im Stress-Regime ({name})?",
                        "statement": f"{r.get('statement') or r.get('title')} – eingeschränkt auf {name}",
                        "signal": f"({r['signal']}) * {gate}", "direction": int(r.get("direction", 1)),
                        "rationale": f"Meta-Validierung: Ensemble-Expectancy VIX>=20 minus VIX<20 = {round(stress, 4)}; "
                                     f"{hid} knapp gescheitert (t={t})",
                        "evidence_t": abs(t), "value": abs(stress), "required_data": ["Feature-Store"],
                        "available_data": True, "source_finding": hid})
    return out


def candidates_from_drift(meta: dict) -> list[dict]:
    out = []
    for mid, m in (meta.get("model_intelligence") or {}).items():
        if m.get("trend") == "deteriorating":
            out.append({"kind": "diagnostic", "question": f"Warum verliert {mid} an Prognosekraft (t={m.get('trend_t')})?",
                        "required_data": ["Basis-OOS je Segment"], "available_data": True,
                        "evidence_t": abs(m.get("trend_t") or 0), "value": abs((m.get("prior_ic") or 0) - (m.get("recent_ic") or 0)),
                        "source_finding": f"model_drift:{mid}"})
    return out


def onboarding_checklist(src: dict) -> dict:
    """Aufnahmeprüfung einer Quelle – Status je Schritt aus Katalog-Metadaten;
    'pending' heißt: muss vor Nutzung gemessen werden (nie automatisch vertraut)."""
    pit = src.get("pit")
    return {"provenance": "ok" if src.get("provider") else "missing",
            "license": "review_required" if "REVIEW_REQUIRED" in str(src.get("license")) else "ok",
            "historical_coverage": "ok" if (src.get("history_from") or 9999) <= 2016 else "insufficient",
            "timestamp_integrity": "ok" if pit in ("vintages", "filing_timestamp", "publication_date", "daily_close")
            else "forward_only",
            "revision_analysis": "ok" if pit == "vintages" else "pending",
            "missing_data_analysis": "pending", "leakage_audit": "pending",
            "incremental_oos_test": "done" if src.get("status") == "integrated" else "pending"}


def active_learning(world: dict, blind: list[dict], catalog: list[dict]) -> list[dict]:
    """Welche Information reduziert die Unsicherheit am stärksten?"""
    cur = (world or {}).get("current") or {}
    dims = {}
    for k, v in cur.items():
        if k.endswith("_state"):
            d = k[:-6]
            unc = cur.get(f"{d}_uncertainty")
            dims[d] = 1.0 if v in ("unavailable", "no_data") else (unc if unc is not None else 0.5)
    blind_need = {"event_risk": sum(1 for c in blind or [] if "extreme_5d_move" in str(c.get("common_properties")))}
    out = []
    for src in catalog:
        if src.get("status") == "integrated":
            continue
        eig = max([dims.get(d, 0.0) for d in src.get("dimension", [])] + [min(1.0, blind_need.get(d, 0) / 3)
                                                                         for d in src.get("dimension", [])])
        chk = onboarding_checklist(src)
        reliability = 1.0 if chk["timestamp_integrity"] == "ok" else 0.3
        feasible = chk["historical_coverage"] == "ok" and chk["timestamp_integrity"] == "ok"
        out.append({"source": src["id"], "dimensions": src.get("dimension"), "current_uncertainty": round(eig, 3),
                    "expected_information_gain": round(eig * (1.0 if feasible else 0.2), 3),
                    "available": src.get("status") != "candidate" or feasible,
                    "reliability": reliability, "acquisition_cost": src.get("acquisition_cost", 3),
                    "priority": priority(eig, 1.0, 1.0, reliability, src.get("acquisition_cost", 3), 1.0),
                    "onboarding_checklist": chk, "license": src.get("license"),
                    "note": "nicht vertrauenswürdig bis alle Schritte 'ok/done'"})
    return sorted(out, key=lambda x: -x["priority"])


def plan(meta: dict, nextv: dict, world: dict, hyp_db: dict, catalog: list[dict], max_new: int = MAX_NEW_PER_RUN) -> dict:
    tested = hyp_db.get("hypotheses") or {}
    blind = nextv.get("blind_spot_clusters") or []
    raw = candidates_from_blind_spots(blind) + candidates_from_regimes(meta, hyp_db) + candidates_from_drift(meta)
    cands = []
    n_family = len(tested)
    for c in raw:
        c["research_id"] = _rid(c.get("signal") or c["question"])
        c["hypothesis"] = c.get("statement") or c["question"]
        c["economic_rationale"] = c.get("rationale") or "gemessener Befund, siehe source_finding"
        c["novelty"] = _novelty(c, tested) if c["kind"] == "testable" else 1.0
        c["expected_information_gain"] = round(min(1.0, c.get("evidence_t", 0) / 3.0), 3)
        c["risk_of_overfitting"] = round(1 - 1 / (1 + 0.05 * n_family + (0.3 if "step(" in str(c.get("signal")) else 0)), 3)
        c["research_cost"] = 1.0 if c["kind"] == "testable" else 3.0
        c["priority"] = priority(c["expected_information_gain"], max(c.get("value", 0) * 100, 0.1), c["novelty"],
                                 1 - c["risk_of_overfitting"], 1.0 if c.get("available_data") else 3.0, c["research_cost"])
        cands.append(c)
    cands.sort(key=lambda c: -c["priority"])
    chosen = [c for c in cands if c["kind"] == "testable" and c["novelty"] > 0.2 and c["research_id"] not in tested][:max_new]
    hyps = [{"id": c["research_id"], "title": c["title"], "statement": c["statement"], "signal": c["signal"],
             "direction": c["direction"], "rationale": c["economic_rationale"],
             "created_at": datetime.now(timezone.utc).date().isoformat()} for c in chosen]
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_candidates": len(cands), "n_selected_for_testing": len(hyps), "max_new_per_run": max_new,
            "candidates": cands, "selected": [c["research_id"] for c in chosen], "hypotheses": hyps,
            "active_learning": active_learning(world, blind, catalog)}


def run() -> dict:
    meta = _load(OUT / "meta_learning.json", {})
    nextv = _load(OUT / "next_validation.json", {})
    world = _load(OUT / "world_model.json", {})
    hyp_db = _load(OUT / "hypothesis_db.json", {})
    catalog = (yaml.safe_load(CATALOG.read_text(encoding="utf-8")) or {}).get("sources", []) if CATALOG.exists() else []
    rep = plan(meta, nextv, world, hyp_db, catalog)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "research_candidates.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    (OUT / "director_hypotheses.json").write_text(json.dumps({"generated": rep["generated"],
                                                             "hypotheses": rep["hypotheses"]}, indent=1, ensure_ascii=False))
    (OUT / "active_learning.json").write_text(json.dumps({"generated": rep["generated"],
                                                         "data_gaps": rep["active_learning"]}, indent=1, ensure_ascii=False))
    return rep


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    rep = run()
    print(json.dumps({k: rep[k] for k in ("n_candidates", "n_selected_for_testing", "selected")}, indent=1))
    for c in rep["candidates"][:10]:
        print(f"- [{c['priority']}] {c['kind']}: {c['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
