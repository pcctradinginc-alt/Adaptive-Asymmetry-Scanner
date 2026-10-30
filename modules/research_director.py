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

Options-Lernmodell (python -m modules.research_director --options, wöchentlich nach option_learner evaluate):
  * option_gaps(): messbare Lücken aus status_<h>d.json – dünn befüllte Merkmale, nicht bewertbare gewählte
    Formen, einseitige Evidenz, Überanpassungsrisiko, fehlende Modellalternativen, fehlende Evidenz.
    Ergebnis: outputs/research/option_learner/improvement.json (auch in meta_cognition sichtbar).
  * option_proposals(): je Lücke strukturierte Vorschläge mit Änderung, Hypothese, erwartetem Nutzen, Kosten,
    Datenverfügbarkeit, vorab festgelegter Falsifikation und Freigabebedarf. Automatisch testbar sind nur
    Varianten aus config/option_learner_variants.yaml (eine Änderung, freigegebener Wert, ohne Kosten);
    Datenquellen, Erfassung und Stichprobe bleiben Vorschläge zur menschlichen Freigabe.
  * Automatisch testbare Vorschläge registriert option_learner.register_variants() als isolierte Varianten
    (begrenzt, je Methode eine, nie eine falsifizierte Änderung erneut); sie laufen ab dem Folgetag im Scanner.

Nur die besten K testbaren Kandidaten (Protokoll: max_new_per_run) werden als
Hypothesen an das Research-Lab übergeben (outputs/research/director_hypotheses.json)
– dort gelten dieselbe Prüfkette, BH über ALLE Tests und die Ähnlichkeits-
sperre gegen bereits Verworfenes. Neue Datenquellen werden nie automatisch als
vertrauenswürdig behandelt (Aufnahmeprüfung, config/data_catalog.yaml).
"""

from __future__ import annotations

import argparse
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
OPTION_OUT = OUT / "option_learner"
OPTION_HORIZONS = (20, 5)
MIN_LABELS_FOR_COVERAGE = 50         # Merkmalsabdeckung erst ab so vielen geprüften Labels beurteilen
LABELS_PER_FEATURE = 20              # Faustregel gegen Überanpassung: geprüfte Labels je Merkmal
MIN_TRADES_FOR_COVERAGE = 10

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


# ── Options-Lernmodell: messbare Lücken ──────────────────────────────────────

def _gap(gid: str, kind: str, horizon: int, finding: str, metric: str, value, threshold, severity: float,
         **kw) -> dict:
    return {"gap_id": f"{kind}:{gid}:{horizon}d", "kind": kind, "horizon": horizon, "finding": finding,
            "metric": metric, "value": value, "threshold": threshold,
            "severity": round(min(1.0, max(0.0, severity)), 3), **kw}


def option_gaps(status: dict, variants: list[dict] | None = None) -> list[dict]:
    """Lücken des Options-Lernmodells je Horizont aus status_<h>d.json – jede mit Befund, Metrik, Wert, Schwelle
    und Schweregrad 0–1. Ohne Messung keine Lücke; ein fehlender Status ist selbst eine Lücke."""
    from modules import option_learner as ol
    out = []
    for h in sorted(status):
        st = status[h] or {}
        if not st:
            out.append(_gap("status_missing", "evidence", h, f"kein Status für {h} T (evaluate nicht gelaufen)",
                            "status", None, "vorhanden", 1.0))
            continue
        n = st.get("verified_labels") or 0
        wf, fw, model = st.get("walk_forward") or {}, st.get("forward") or {}, st.get("model") or {}
        if n >= MIN_LABELS_FOR_COVERAGE:
            for f, c in sorted((st.get("feature_coverage") or {}).items()):
                if c < ol.MIN_FEATURE_COVERAGE:
                    out.append(_gap(f"sparse_feature:{f}", "data", h,
                                    f"Merkmal {f} nur in {c:.0%} der {n} geprüften Labels befüllt – vom Modell verworfen",
                                    "feature_coverage", c, ol.MIN_FEATURE_COVERAGE, 1 - c, feature=f))
        for name, s in (("walk_forward", wf), ("forward", fw)):
            cov = s.get("coverage")
            if (s.get("trades") or 0) >= MIN_TRADES_FOR_COVERAGE and cov is not None and cov < ol.MIN_COVERAGE:
                out.append(_gap(f"outcome_coverage:{name}", "data", h,
                                f"{name}: nur {cov:.0%} der gewählten Formen mit geprüftem Ergebnis "
                                f"({s.get('unevaluable')})", "coverage", cov, ol.MIN_COVERAGE, 1 - cov,
                                reasons=s.get("unevaluable")))
        for w in (wf.get("representativeness") or {}).get("warnings") or []:
            out.append(_gap("representativeness:" + hashlib.sha256(w.encode()).hexdigest()[:6], "representativeness",
                            h, f"Walk-Forward-Evidenz: {w}", "evidence_mix", None, "nicht einseitig, >= 90 Tage", 0.5))
        k = model.get("features") or 0
        if n and k and n / k < LABELS_PER_FEATURE:
            out.append(_gap("overfit_risk", "model", h,
                            f"{k} Merkmale auf {n} geprüfte Labels ({n / k:.0f} je Merkmal, Faustregel >= "
                            f"{LABELS_PER_FEATURE})", "labels_per_feature", round(n / k, 1), LABELS_PER_FEATURE,
                            1 - n / k / LABELS_PER_FEATURE))
        if wf.get("verdict") == "NOT_VALIDATED":
            out.append(_gap("no_skill", "model", h, f"Walk-Forward NOT_VALIDATED (Rang-IC t {wf.get('ic_t')}, "
                            f"KI {wf.get('selected_ci95')})", "walk_forward_verdict", "NOT_VALIDATED", "VALIDATED", 0.8))
        ms, ma = wf.get("mean_selected"), wf.get("mean_all")
        if ms is not None and ma is not None and ms <= ma:
            out.append(_gap("selection_no_value", "model", h, f"gewählte Fälle Ø {ms} nicht besser als alle Formen "
                            f"Ø {ma}", "mean_selected_minus_all", round(ms - ma, 6), "> 0", 0.6))
        vd = wf.get("vs_default") or {}
        if vd.get("ci95") and vd["ci95"][1] < 0:
            out.append(_gap("choice_worse_than_default", "model", h, f"Wahl der Form schlechter als die Standardform "
                            f"(Δ {vd.get('mean_diff')}, KI {vd['ci95']})", "vs_default", vd.get("mean_diff"), "> 0", 0.7))
        if k and not any(v.get("horizon") == h for v in variants or []):
            out.append(_gap("untested_alternatives", "model", h, f"für {h} T wurde noch keine Modell-, Merkmals- oder "
                            "Auswahlvariante prospektiv gegen die Basis getestet", "variants_tested", 0, ">= 1", 0.4))
        missing = [f"{name}: {s.get('dates', 0)}/{ol.MIN_OOS_DATES} Testtage, {s.get('independent_periods', 0)}/"
                   f"{ol.MIN_INDEPENDENT_PERIODS} Perioden, {s.get('n_selected', 0)}/{ol.MIN_SELECTED} Fälle"
                   for name, s in (("walk_forward", wf), ("forward", fw)) if s.get("verdict") == "NEED_MORE_DATA"]
        if missing:
            out.append(_gap("insufficient", "evidence", h, "NEED_MORE_DATA – " + "; ".join(missing), "verdict",
                            "NEED_MORE_DATA", "VALIDATED / FORWARD_POSITIVE", 0.3, missing=missing))
    return sorted(out, key=lambda g: (-g["severity"], g["gap_id"]))


# Lückenart -> freigegebene Methoden ("up"/"down": Regularisierung stärker/schwächer als die Basis)
GAP_METHODS = {
    "overfit_risk": (("ridge_alpha", "up"), ("drop_feature_group", None)),
    "no_skill": (("drop_feature_group", None), ("ridge_alpha", "up"), ("ridge_alpha", "down")),
    "selection_no_value": (("min_pred", None),),
    "choice_worse_than_default": (("min_pred", None),),
    "untested_alternatives": (("ridge_alpha", "up"), ("min_pred", None), ("drop_feature_group", None)),
}
MECHANISM = {
    "ridge_alpha": "Stärkere/schwächere Regularisierung verschiebt den Kompromiss zwischen Varianz und Verzerrung",
    "min_pred": "Erst ab einer Netto-Prognose über der Kostenunsicherheit handeln: weniger, dafür bessere Fälle",
    "drop_feature_group": "Eine Merkmalsgruppe trägt eventuell nur Rauschen bei; ohne sie generalisiert das Modell besser",
}
PROPOSAL_COST = {"variant": 1.0, "data_pipeline": 2.0, "data_capture": 3.0, "sampling": 3.0}


def _pid(kind: str, change: dict, horizon: int) -> str:
    return "OLP-" + hashlib.sha256(json.dumps([kind, change, horizon], sort_keys=True).encode()).hexdigest()[:8]


def _falsification(cfg: dict, horizon: int) -> dict:
    from modules import option_learner as ol
    f = (cfg or {}).get("falsification") or {}
    return {"primary_metric": "mittlere Tagesdifferenz netto (Variante − Basis), gleiche Kandidaten, gleiches Budget",
            "comparators": ["Basis-Lernversion (gleiche Labels, gleicher Tag)", "eingefrorene Referenz der Basis"],
            "accept_if": f"Bonferroni-KI (family_alpha {f.get('family_alpha', 0.05)}) ganz über 0 bei >= "
                         f"{ol.MIN_OOS_DATES} Tagen, >= {ol.MIN_INDEPENDENT_PERIODS} unabhängigen Perioden, >= "
                         f"{ol.MIN_SELECTED} Kandidaten",
            "reject_if": "KI ganz unter 0 (FALSIFIED_WORSE) oder nach max_days kein signifikanter Vorteil "
                         "(FALSIFIED_NO_EFFECT)",
            "max_days": int(((f.get("max_days") or {}).get(horizon)) or 4 * horizon + 60),
            "family_alpha": float(f.get("family_alpha", 0.05))}


def _variant_overrides(method: str, direction: str | None, cfg: dict) -> list[dict]:
    from modules import option_learner as ol
    allowed = ((cfg or {}).get("approved_methods") or {}).get(method)
    if not allowed:
        return []
    if method == "drop_feature_group":
        return [{method: g} for g in allowed]
    vals = [float(v) for v in allowed]
    if method == "ridge_alpha":
        vals = [v for v in vals if (v > ol.RIDGE_ALPHA if direction == "up" else v < ol.RIDGE_ALPHA)]
        vals = sorted(vals, reverse=direction != "up")          # nächstliegender Schritt zuerst
    return [{method: v} for v in vals]


def option_proposals(gaps: list[dict], cfg: dict | None = None, catalog: list[dict] | None = None) -> list[dict]:
    """Strukturierte Verbesserungsvorschläge je gemessener Lücke. Erwarteter Nutzen = Schweregrad der Lücke als
    Priorität (keine erfundene Renditeprognose); die Falsifikation ist vorab festgelegt. auto_experiment nur für
    freigegebene, kostenlose Varianten; alles mit Daten-, Erfassungs- oder Stichprobenänderung braucht Freigabe."""
    from modules import option_learner as ol
    cfg = cfg if cfg is not None else ol.variant_config()
    props = {}

    def add(kind, change, gap, title, hypothesis, cost, availability, falsification, auto, weight=1.0, **kw):
        pid = _pid(kind, change, gap["horizon"])
        if pid in props:
            props[pid]["gap_ids"].append(gap["gap_id"])
            props[pid]["priority"] = max(props[pid]["priority"], round(gap["severity"] * weight, 4))
            return
        props[pid] = {"proposal_id": pid, "kind": kind, "horizon": gap["horizon"], "gap_ids": [gap["gap_id"]],
                      "title": title, "change": change, "hypothesis": hypothesis,
                      "expected_benefit": {"target": falsification.get("primary_metric")
                                           or "Lücke geschlossen (siehe Falsifikation)",
                                           "priority_basis": f"Schweregrad {gap['severity']} der Lücke {gap['gap_id']}",
                                           "finding": gap["finding"]},
                      "cost": cost, "data_availability": availability, "falsification": falsification,
                      "auto_experiment": auto, "approval_required": not auto,
                      "priority": round(gap["severity"] * weight / PROPOSAL_COST.get(kind, 4.0), 4), **kw}

    variant_cost = {"api_calls": 0, "llm_calls": 0, "paid_data": False,
                    "compute": "ein zusätzliches Ridge-Training je Scanner-Lauf"}
    for g in gaps:
        name = g["gap_id"].split(":")[1]
        for method, direction in GAP_METHODS.get(name, ()):
            for change in _variant_overrides(method, direction, cfg):
                try:
                    ol.validate_overrides(change, cfg)
                except ValueError:
                    continue
                (m, v), = change.items()
                add("variant", change, g, f"Variante {m}={v} ({g['horizon']} T)",
                    f"Mit {m}={v} trifft das Modell auf denselben neuen Kandidaten netto bessere Entscheidungen als "
                    f"die Basis. Mechanismus: {MECHANISM[m]}.", variant_cost,
                    "vorhanden: dieselben geprüften Labels und Kandidaten wie die Basis",
                    _falsification(cfg, g["horizon"]), True)
        if name == "sparse_feature":
            add("data_pipeline", {"feature": g["feature"]}, g, f"Merkmal {g['feature']} befüllen oder streichen",
                f"Das Ledger-Feld {g['feature']} fehlt in fast allen Kandidaten; befüllt könnte das Modell es nutzen, "
                "sonst gehört es nicht in die Spezifikation.",
                {"api_calls": "keine zusätzlichen erwartet", "llm_calls": 0, "paid_data": False,
                 "human_effort": "Code-Prüfung der Erfassung"}, "Feld im Ledger leer – Erfassung prüfen",
                {"accept_if": f"Abdeckung >= {ol.MIN_FEATURE_COVERAGE:.0%} binnen 20 Handelstagen nach Korrektur",
                 "reject_if": "sonst Merkmal aus der Spezifikation entfernen (neue Experimentversion)"}, False)
        elif name == "outcome_coverage":
            add("data_capture", {"reasons": g.get("reasons")}, g, "Quote-Erfassung für gewählte Kontrakte schließen",
                "Fehlende Entry-/Exit-Quotes der gewählten Formen verhindern die Bewertung und können sie verzerren.",
                {"api_calls": "zusätzliche Quote-Abrufe im bestehenden Erfassungs-Workflow", "llm_calls": 0,
                 "paid_data": False, "human_effort": "Erfassungslogik ändern"},
                "Quotes nur bei rechtzeitiger Erfassung verfügbar",
                {"accept_if": f"Abdeckung >= {ol.MIN_COVERAGE:.0%} in den nächsten 20 fälligen Handelstagen",
                 "reject_if": "Abdeckung bleibt darunter: Ursache (z. B. Liquidität) statt Erfassung prüfen"}, False)
        elif g["kind"] == "representativeness":
            add("sampling", {"warning": g["finding"]}, g, "Evidenzbasis verbreitern (Richtungssignale, bearishe Fälle)",
                "Eine einseitige Stichprobe belegt nicht alle Marktsituationen; zusätzliche Snapshots würden das zeigen.",
                {"api_calls": "zusätzliche Kontrakt-Snapshots je Scanner-Lauf", "llm_calls": 0, "paid_data": False,
                 "human_effort": "Stichprobenregel im Kandidaten-Ledger ändern"}, "erfordert neue Snapshots",
                {"accept_if": "keine Repräsentativitätswarnung binnen 90 Tagen",
                 "reject_if": "Warnung bleibt bestehen: Einschränkung im Leistungszeugnis dauerhaft ausweisen"}, False)
        elif name == "no_skill":
            for src in catalog or []:
                if src.get("status") == "integrated":
                    continue
                cost = float(src.get("acquisition_cost") or 3)
                paid = cost >= 5 or "kommerziell" in str(src.get("license", "")).lower()
                add("data_source", {"source": src.get("id")}, g, f"Datenquelle {src.get('id')} prüfen",
                    f"Zusätzliche Information ({', '.join(src.get('dimension') or [])}) könnte die fehlende "
                    "Prognosekraft liefern.",
                    {"api_calls": "Abrufe der neuen Quelle", "llm_calls": 0, "paid_data": paid,
                     "acquisition_cost": cost, "license": src.get("license")},
                    {"onboarding_checklist": onboarding_checklist(src), "history_from": src.get("history_from")},
                    {"accept_if": "nach Aufnahmeprüfung als Merkmalsvariante: Bonferroni-KI der Tagesdifferenz > 0",
                     "reject_if": "kein signifikanter Vorteil nach max_days oder Aufnahmeprüfung nicht bestanden"},
                    False, weight=1.0 / (2 * cost))
    # Gleichstand: mehr stützende Lücken zuerst, dann die Reihenfolge der Methoden in GAP_METHODS
    order = {pid: i for i, pid in enumerate(props)}
    return sorted(props.values(), key=lambda p: (-p["priority"], -len(p["gap_ids"]), order[p["proposal_id"]]))


def _option_status(root: Path = OPTION_OUT) -> dict:
    return {h: _load(root / f"status_{h}d.json", {}) for h in OPTION_HORIZONS}


def run_options(root: Path = OPTION_OUT, now: datetime | None = None) -> dict:
    """Wöchentlich nach option_learner evaluate: Lücken und Vorschläge des Options-Lernmodells; freigegebene,
    kostenlose Varianten werden als isolierte Experimente registriert (Start am Folgetag) -> improvement.json."""
    from modules import option_learner as ol
    now = now or datetime.now(timezone.utc)
    cfg = ol.variant_config()
    gaps = option_gaps(_option_status(root), list(ol.variant_registry(root).values()))
    catalog = (yaml.safe_load(CATALOG.read_text(encoding="utf-8")) or {}).get("sources", []) if CATALOG.exists() else []
    proposals = option_proposals(gaps, cfg, catalog)
    registered = ol.register_variants(root, proposals, now, cfg)
    rep = {"generated": now.isoformat(timespec="seconds"), "gaps": gaps, "proposals": proposals,
           "registered_variants": [v["variant_id"] for v in registered]}
    root.mkdir(parents=True, exist_ok=True)
    (root / "improvement.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    ol.report_card(root)
    return rep


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


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--options", action="store_true", help="nur Options-Lernmodell: Lücken und Vorschläge")
    if ap.parse_args(argv).options:
        from modules import research_memory as rm
        rep = run_options()
        print(f"Research Memory: {rm.sync()} neue Einträge")
        for g in rep["gaps"][:10]:
            print(f"- [{g['severity']}] {g['gap_id']}: {g['finding']}")
        for p in rep["proposals"][:10]:
            print(f"  -> [{p['priority']}] {p['proposal_id']} {p['kind']} {p['change']} auto={p['auto_experiment']}")
        return 0
    rep = run()
    print(json.dumps({k: rep[k] for k in ("n_candidates", "n_selected_for_testing", "selected")}, indent=1))
    for c in rep["candidates"][:10]:
        print(f"- [{c['priority']}] {c['kind']}: {c['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
