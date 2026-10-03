"""
modules/inquiry.py – Kernfrage des Systems als nachvollziehbare Kette je offener Unklarheit.

  1. Welche Information verstehe ich noch nicht?        Blind-Spot-Cluster, Modell-Drift, Fehlerursachen
  2. Welche falsifizierbare Erklärung?                  verknüpfte Hypothesen (Director, Fabrik, Verträge)
  3. Welche Daten würden sie bestätigen/widerlegen?      benötigte Daten, Verfügbarkeit, DATA_GAP + freie Quellen
  4. Funktioniert sie auf wirklich neuen Daten?           Status historisch vs. prospektive Forward-Evidenz
  5. Verbessert sie Entscheidungen genug für Änderung?   Produktionseinfluss (PromotionController)

Status je Kette (fortschreitend):
  OPEN_QUESTION          keine Hypothese formuliert
  NEEDS_DATA             Hypothese(n) nur mit Datenlücke
  HISTORICALLY_REJECTED  alle getesteten Erklärungen historisch verworfen -> neue Erklärung nötig
  UNDER_TEST             Erklärung in Test / historisch offen
  FORWARD_TEST           prospektiver Challenger sammelt neue Daten
  BEHAVIOUR_CHANGED      validierte Erklärung mit Produktionseinfluss

Rein lesend, keine Produktionswirkung. Ausgabe: outputs/research/inquiry_chains.json.
    python -m modules.inquiry
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

RS = Path("outputs/research")
INTEL = Path("outputs/intelligence")
OUT = RS / "inquiry_chains.json"
FORWARD_STATES = ("PROSPECTIVE_CHALLENGER", "FORWARD_VALIDATED")
PRODUCTION_STATES = ("GUARDED_PRODUCTION", "LIMITED_PRODUCTION", "FULL_PRODUCTION")


def _load(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError) as e:
        log.warning(f"inquiry: {p} nicht lesbar ({e})")
        return None


def anomalies(nextv: dict | None, mstate: dict | None, fail: dict | None, sysstate: dict | None) -> list[dict]:
    """Was versteht das System nicht? Nur gemessene Befunde."""
    out = []
    for c in (nextv or {}).get("blind_spot_clusters") or []:
        out.append({"id": c.get("id"), "kind": "blind_spot",
                    "finding": f"systematischer Fehler {c.get('typical_error')} bei {c.get('common_properties')} "
                               f"(n={c.get('n')}, Lift {c.get('lift')})",
                    "keys": [c.get("id")] + (["blind_spot_sector_match"]
                                             if (c.get("common_properties") or {}).get("sector") else []),
                    "sector": (c.get("common_properties") or {}).get("sector")})
    md = ((((sysstate or {}).get("drift_state") or {}).get("components") or {}).get("model") or {})
    for m in md.get("deteriorating") or []:
        out.append({"id": f"model_drift:{m}", "kind": "model_drift", "finding": f"Modell {m} verliert Prognosekraft",
                    "keys": [f"model_drift:{m}"]})
    for f in ((((sysstate or {}).get("drift_state") or {}).get("components") or {}).get("feature") or {}) \
            .get("features_out_of_range") or []:
        out.append({"id": f"feature_drift:{f}", "kind": "feature_drift",
                    "finding": f"Merkmal {f} außerhalb des Trainingsbereichs", "keys": [f"drift_{f}", f]})
    prim = (((fail or {}).get("reliable") or {}).get("primary") or {})
    for cause, v in sorted(prim.items(), key=lambda kv: -((kv[1] or {}).get("share") or 0))[:3]:
        out.append({"id": f"failure:{cause}", "kind": "prediction_error",
                    "finding": f"Verlust-Ursache {cause}: {round(100 * (v.get('share') or 0))} % der Verlierer "
                               f"(Ø {v.get('mean_outcome')})", "keys": ["champion_trades", cause]})
    return out


def explanations(a: dict, director: dict | None, hyp_db: dict | None, factory_plan: dict | None,
                 factory_results: dict | None, promo: dict | None, contracts: list[dict],
                 proposals: dict | None = None) -> list[dict]:
    """Verknüpfte, falsifizierbare Erklärungen mit Datenlage und Status."""
    db = (hyp_db or {}).get("hypotheses") or {}
    keys = {str(k) for k in a.get("keys") or [] if k}
    out = []
    for c in (director or {}).get("candidates") or []:
        if c.get("source_finding") in keys or (a["kind"] == "blind_spot" and c.get("source_finding") == a["id"]):
            h = db.get(c.get("research_id")) or {}
            out.append({"id": c.get("research_id"), "source": "director", "explanation": c.get("question"),
                        "data_needed": c.get("required_data"), "data_available": c.get("available_data"),
                        "status": h.get("canonical_status") or "NOT_YET_TESTED", "evidence": "historisch",
                        "reasons": (h.get("reasons") or [])[:2]})
    res = (factory_results or {}).get("results") or {}
    for h in (factory_plan or {}).get("ideas") or []:
        if (h.get("family") in keys or any(k in str(h.get("signal")) for k in keys if k.startswith(("drift_",)))
                or (a["kind"] == "feature_drift" and set(h.get("domain_features") or []) & keys)):
            r = res.get(h.get("id")) or {}
            rd = h.get("readiness") or {}
            out.append({"id": h.get("id"), "source": "factory", "explanation": h.get("statement") or h.get("title"),
                        "H0": h.get("H0"), "data_needed": h.get("domain_features"),
                        "data_available": rd.get("status") == "READY", "free_sources": rd.get("free_sources"),
                        "status": "DATA_GAP" if h.get("plan_status") == "DATA_GAP" else
                        (r.get("status") or h.get("plan_status")), "evidence": "historisch",
                        "reasons": (r.get("reasons") or [])[:2]})
    if a["kind"] == "prediction_error":            # Erklärungen aus echten Champion-Fehlern (abstention_proposals)
        for p in (proposals or {}).get("proposals") or []:
            wf = p.get("walk_forward") or {}
            out.append({"id": wf.get("rule"), "source": "abstention_proposals",
                        "explanation": f"Champion-Trades mit {wf.get('rule')} laufen schlechter",
                        "data_needed": [wf.get("rule")], "data_available": True,
                        "status": "HISTORICALLY_VALIDATED (Entwurf, Registrierung per PR)", "evidence": "historisch"})
        for r in (proposals or {}).get("rejected") or []:
            out.append({"id": r.get("rule"), "source": "abstention_proposals",
                        "explanation": f"Champion-Trades mit {r.get('rule')} laufen schlechter",
                        "data_needed": [r.get("rule")], "data_available": True, "status": "REJECTED",
                        "evidence": "historisch", "reasons": [r.get("reason")]})
    states = (promo or {}).get("hypotheses") or {}
    for c in contracts:
        feats = set(c.get("features") or [])
        if feats & keys:
            k = f"{c.get('hypothesis_id')}@v{c.get('version', 1)}"
            st = states.get(k) or {}
            ev = st.get("evidence") or {}
            out.append({"id": k, "source": "contract", "explanation": c.get("research_question") or c.get("title"),
                        "H0": c.get("H0"), "data_needed": sorted(feats), "data_available": True,
                        "status": st.get("state") or "REGISTERED", "evidence": "prospektiv (Forward)",
                        "forward_n": ev.get("n_observations", 0), "delta_expectancy": ev.get("delta_expectancy"),
                        "influence_level": st.get("influence_level") or "NONE"})
    return out


def chain_status(expl: list[dict]) -> str:
    if not expl:
        return "OPEN_QUESTION"
    sts = [e.get("status") for e in expl]
    if any(e.get("influence_level") not in (None, "NONE") or e.get("status") in PRODUCTION_STATES for e in expl):
        return "BEHAVIOUR_CHANGED"
    if any(s in FORWARD_STATES for s in sts):
        return "FORWARD_TEST"
    if any(str(s).startswith("HISTORICALLY_VALIDATED") for s in sts):
        return "UNDER_TEST"
    if all(s == "DATA_GAP" for s in sts):
        return "NEEDS_DATA"
    if all(s in ("REJECTED", "DEMOTED", "EXPIRED") for s in sts if s != "DATA_GAP"):
        return "HISTORICALLY_REJECTED"
    return "UNDER_TEST"


NEXT_STEP = {
    "OPEN_QUESTION": "falsifizierbare Erklärung formulieren (Director/Fabrik)",
    "NEEDS_DATA": "Datenlücke schließen (freie Quelle anbinden, PIT prüfen)",
    "HISTORICALLY_REJECTED": "neue, nicht ähnliche Erklärung formulieren; verworfene nicht kosmetisch wiederholen",
    "UNDER_TEST": "historischen Walk-Forward abschließen",
    "FORWARD_TEST": "Forward-Daten abwarten (keine Entscheidung vor Mindest-N/Spanne)",
    "BEHAVIOUR_CHANGED": "Wirkung überwachen; Demotion bei Verschlechterung",
}


def build(root: Path = Path("."), now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    rs, intel = root / RS, root / INTEL
    try:
        from modules import hypothesis_contract as hc
        contracts = hc.load()
    except Exception as e:  # noqa: BLE001 – ohne Verträge bleiben die übrigen Verknüpfungen
        log.warning(f"inquiry: Verträge nicht ladbar ({e})")
        contracts = []
    an = anomalies(_load(rs / "next_validation.json"), _load(rs / "machine_state.json"),
                   _load(rs / "failure_analysis.json"), _load(root / "outputs/state/system_state.json"))
    chains = []
    for a in an:
        ex = explanations(a, _load(rs / "research_candidates.json"), _load(rs / "hypothesis_db.json"),
                          _load(rs / "factory_plan.json"), _load(rs / "factory_results.json"),
                          _load(intel / "promotion_state.json"), contracts, _load(intel / "contract_proposals.json"))
        st = chain_status(ex)
        chains.append({"question_1_not_understood": a, "question_2_explanations": ex,
                       "question_3_data": [{"id": e["id"], "needed": e.get("data_needed"),
                                            "available": e.get("data_available"),
                                            "free_sources": e.get("free_sources")} for e in ex],
                       "question_4_new_data": [{"id": e["id"], "status": e.get("status"), "evidence": e.get("evidence"),
                                                "forward_n": e.get("forward_n")} for e in ex],
                       "question_5_behaviour": "Produktion geändert" if st == "BEHAVIOUR_CHANGED" else "keine Änderung",
                       "status": st, "next_step": NEXT_STEP[st]})
    counts: dict[str, int] = {}
    for c in chains:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    try:
        from modules import research_memory as rm
        learning = rm.learning_curve(rm.load())
    except Exception as e:  # noqa: BLE001
        log.warning(f"inquiry: Lernkurve nicht berechenbar ({e})")
        learning = None
    return {"generated": now.isoformat(timespec="seconds"), "n": len(chains), "status_counts": counts,
            "chains": chains, "research_learning": learning}


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    res = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(f"{res['n']} Ketten: {res['status_counts']}")
    for c in res["chains"]:
        print(f"- [{c['status']}] {c['question_1_not_understood']['finding']} -> {c['next_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
