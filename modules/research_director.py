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

Self-Improvement Notifications (python -m modules.research_director --notify [--dry-run], wöchentlich danach):
  * improvement_tasks(): technische Verbesserungsaufgaben NUR aus belegten Befunden (leere Merkmale, fehlende
    Erfassung bewerteter Entscheidungen, einseitige Evidenz, belegte/widerrufene Varianten-Ergebnisse, nicht testbare
    Varianten, nachgewiesene Modellschwächen ohne automatische Abhilfe, Freigabeprüfung, dauerhaft gestörte
    Datenquellen, regelmäßig nachgestartete Workflows) – nie aus NEED_MORE_DATA, Wartezeit oder Heuristik.
  * notify_tasks(): täglich (--critical) nur P0 sofort, samstags Digest mit höchstens 3 neuen/wesentlich geänderten
    P1–P3-Aufgaben; Dedupe über Fingerabdruck; OPEN/RESOLVED/UNKNOWN (behoben nur mit aktuellem Nachweis);
    append-only Verlauf outputs/research/improvement_tasks.jsonl; feste Vorlagen mit kopierfertiger
    Claude-Code-Aufgabe, kein LLM, kein automatischer Code.

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
import html
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
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
    tb = (cfg or {}).get("test_budget") or {}
    budget = f"total_alpha {tb.get('total_alpha', 0.05)} / max_tests {tb.get('max_tests', 20)}"
    max_days = int(((f.get("max_days") or {}).get(horizon)) or 4 * horizon + 60)
    return {"primary_metric": "mittlere Tagesdifferenz netto (Variante − Basis), gleiche Kandidaten, gleiches Budget",
            "economic_metric": "Tagesrendite netto der Variante gegenüber Nicht-Handeln (gleiches Budget je Kandidat)",
            "comparators": ["Basis-Lernversion (gleiche Labels, gleicher Tag)", "eingefrorene Referenz der Basis",
                            "Nicht-Handeln"],
            "looks_days": ol.look_days(max_days),
            "accept_if": (f"an einem vorab festgelegten Prüfzeitpunkt BEIDE KI ganz über 0 (relativ zur Basis UND "
                          f"netto gegenüber Nicht-Handeln; Teil-Alpha nach O'Brien-Fleming aus dem bei der "
                          f"Registrierung zugeteilten Slot des dauerhaften Fehlerbudgets {budget}), gemeinsame "
                          f"Ergebnisabdeckung >= {ol.MIN_COVERAGE:.0%}, >= {ol.MIN_OOS_DATES} Tage, >= "
                          f"{ol.MIN_INDEPENDENT_PERIODS} unabhängige Perioden, >= {ol.MIN_SELECTED} Kandidaten"),
            "reject_if": ("relativ KI ganz unter 0 an einem Prüfzeitpunkt (FALSIFIED_WORSE); an der letzten Prüfung "
                          "kein relativer Vorteil (FALSIFIED_NO_EFFECT) bzw. nur relativ, nicht netto "
                          "(BETTER_THAN_BASE_NOT_PROFITABLE)"),
            "max_days": max_days, "test_budget": budget}


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
           "registered_variants": [v["variant_id"] for v in registered],
           "test_budget": ol.test_budget(root, cfg)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "improvement.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    ol.report_card(root)
    return rep


# ── Self-Improvement Notifications: technische Verbesserungsaufgaben per Mail ─────────────────────────────────────
# Aus BELEGTEN Befunden der vorhandenen Research-Ausgaben werden technische Verbesserungsaufgaben abgeleitet,
# append-only gespeichert und als kopierfertige Claude-Code-Aufgabe gemeldet. Deterministische Vorlagen: kein LLM,
# keine Claude-API, keine Markt-API. Es wird nie Code geändert, nichts automatisch umgesetzt, nichts am Handel.
# Keine Aufgabe (und keine Mail) aus: NEED_MORE_DATA / normaler Wartezeit, laufenden Prüfungen, Heuristiken ohne
# Test (Überanpassungs-Faustregel, nur Punktschätzung) und Lücken, für die gerade automatische Varianten laufen.
TASKS = OUT / "improvement_tasks.jsonl"          # append-only: detected/changed/reopened/resolved/notified/send_failed
TASKS_STATE = OUT / "improvement_tasks.json"     # abgeleiteter Zustand je Aufgabe (Research Memory, meta_cognition)
TASK_DECISIONS = Path("config/improvement_task_decisions.yaml")   # menschliche Entscheidungen je Task-ID
TASK_INPUTS = {"option_root": OPTION_OUT, "source_history": Path("outputs/health/source_health_history.jsonl"),
               "source_snapshot": Path("outputs/health/source_health_snapshot.json"),
               "workflow_status": Path("outputs/state/workflow_status.json"),
               "ledger": Path("outputs/candidate_ledger")}
INPUT_LABEL = {"improvement": "option_learner/improvement.json", "source_snapshot": "source_health_snapshot.json",
               "workflow_status": "workflow_status.json"}
DIGEST_MAX = 3                                   # höchstens so viele neue/geänderte Aufgaben je Kalenderwoche
CRITICAL_KINDS = ("withdrawn_success", "release_integrity")   # P0: täglicher Lauf (--critical), sonst Wochen-Digest
FRESH_DAYS = {"option": 10, "seq": 10, "health": 2, "workflow": 2}   # ältere Daten sind nicht beurteilbar
MODES = ("weekly", "critical")
SUBJECT = "[Scanner] Technischer Verbesserungsvorschlag – {}"
PRIORITY_TEXT = {"P0": "kritisch", "P1": "hoch", "P2": "mittel", "P3": "niedrig"}
CATEGORY_TEXT = {"data": "Daten", "capture": "Datenerfassung", "model": "Modell", "testing": "Testbarkeit",
                 "infrastructure": "Pipeline/Infrastruktur", "research_result": "Forschungsergebnis",
                 "critical": "Kritischer Evidenzwiderspruch"}
HUMAN_CLOSED = ("REJECTED", "DONE", "IN_PROGRESS")       # menschlich entschieden: keine weitere Mail
SOURCE_BAD = ("BROKEN", "FAILED", "STALE")
SOURCE_PERSIST_DAYS = 7
RECOVERY_WINDOW_DAYS, RECOVERY_MIN_DAYS = 7, 5
NO_COST = {"api_calls": 0, "llm_calls": 0, "paid_data": False, "compute": "vernachlässigbar"}
BOUNDARIES = ("Keine Änderungen an Trading-Secrets, Orders, Positionsgrößen, Risikolimits, Champion oder "
              "Produktionsregeln. Keine zusätzlichen kostenpflichtigen APIs oder Datenquellen. Kein automatischer "
              "Merge – menschliche Freigabe bleibt zwingend.")


def _num_or_none(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _code_refs(token: str, limit: int = 6) -> list[str]:
    """Fundstellen eines Feld- oder Workflow-Namens im Produktionscode (deterministische Textsuche, ohne Tests)."""
    refs = []
    for path in sorted([*Path("modules").rglob("*.py"), Path("pipeline.py"), Path("feedback.py")]):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        refs += [f"{path.as_posix()}:{i}" for i, line in enumerate(lines, 1) if token in line][:2]
        if len(refs) >= limit:
            break
    return refs[:limit]


def _field_locations(rows: list[dict] | None, field: str) -> dict | None:
    """Wo und in welchem Format ein Merkmal im Kandidaten-Ledger steht (Beleg, keine Annahme)."""
    if not rows:
        return None
    n, numeric, other, top, keys = 0, 0, {}, 0, {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        n += 1
        top += field in r
        v = (r.get("features") or {}).get(field) if isinstance(r.get("features"), dict) else None
        if v is None:
            continue
        if _num_or_none(v) is not None:
            numeric += 1
            continue
        other[type(v).__name__] = other.get(type(v).__name__, 0) + 1
        if isinstance(v, dict):
            for k, x in v.items():
                keys[k] = keys.get(k, 0) + (_num_or_none(x) is not None)
    return {"rows": n, "in_features": numeric + sum(other.values()), "numeric": numeric,
            "non_numeric_types": dict(sorted(other.items())),
            "numeric_by_key": dict(sorted(keys.items(), key=lambda kv: (-kv[1], kv[0]))[:8]), "top_level": top}


def _task(kind: str, subject: str, priority: str, category: str, short: str, **f) -> dict:
    """Aufgabe mit stabiler ID (Erkennungsart + Gegenstand) und Fingerabdruck der wesentlichen Belege: nur eine
    wesentliche Änderung (andere Größenordnung, Status, Priorität) ergibt eine neue Benachrichtigung."""
    tid = "IMP-" + hashlib.sha256(f"{kind}|{subject}".encode()).hexdigest()[:8]
    material = {"priority": priority, **(f.pop("material", None) or {})}
    return {"task_id": tid, "kind": kind, "subject": subject, "priority": priority, "category": category,
            "short": short[:80], "fingerprint": hashlib.sha256(json.dumps(material, sort_keys=True, default=str)
                                                               .encode()).hexdigest()[:12],
            "material": material, **f}


def _bucket(cov: float) -> str:
    return "leer" if cov == 0 else "<25%" if cov < 0.25 else "<50%" if cov < 0.5 else "<80%"


def _tasks_empty_features(gaps: list[dict], status: dict, ledger: list[dict] | None, issues: list[str]) -> list[dict]:
    """Leere oder kaum befüllte Merkmale (Lücke sparse_feature: < 50 % befüllt bei >= 50 geprüften Labels)."""
    from modules import option_learner as ol
    by_feature: dict = {}
    for g in gaps:
        if g.get("gap_id", "").split(":")[1:2] != ["sparse_feature"]:
            continue
        cov, h, feat = _num_or_none(g.get("value")), g.get("horizon"), g.get("feature")
        if cov is None or not 0 <= cov <= 1 or not isinstance(h, int) or not feat:
            issues.append(f"Lücke {g.get('gap_id')}: unvollständig – ignoriert")
            continue
        by_feature.setdefault(feat, {})[h] = cov
    out = []
    for feat, covs in sorted(by_feature.items()):
        loc = _field_locations(ledger, feat)
        labels = {h: (status.get(h) or {}).get("verified_labels") or "?" for h in covs}
        lines = [f"{h} T: {feat} in {covs[h]:.0%} der {labels[h]} geprüften Options-Labels befüllt (Schwelle "
                 f"{ol.MIN_FEATURE_COVERAGE:.0%}) – das Modell verwirft das Merkmal" for h in sorted(covs)]
        cause = "Das Merkmal wird zum Entscheidungszeitpunkt nicht oder unter anderem Namen ins Ledger geschrieben."
        if loc:
            lines.append(f"Kandidaten-Ledger: {feat} in {loc['in_features']} von {loc['rows']} Zeilen unter "
                         f"'features', davon {loc['numeric']} als Zahl; als eigenes Feld außerhalb von 'features': "
                         f"{loc['top_level']}")
            if loc["non_numeric_types"]:
                lines.append(f"Format der übrigen Werte: {loc['non_numeric_types']}"
                             + (f"; Zahlen je Schlüssel des Objekts: {loc['numeric_by_key']}"
                                if loc["numeric_by_key"] else ""))
                cause = (f"Das Ledger speichert {feat} als {', '.join(loc['non_numeric_types'])} statt als Zahl; "
                         "option_learner.features() liest nur Zahlen (_num) und verwirft den Wert vollständig.")
            elif loc["top_level"] and not loc["in_features"]:
                cause = (f"{feat} steht als eigenes Ledger-Feld außerhalb von 'features'; option_learner.features() "
                         "liest nur row['features'].")
        refs, empty = _code_refs(feat), all(c == 0 for c in covs.values())
        out.append(_task(
            "empty_feature", feat, "P1" if empty else "P2", "data",
            f"Merkmal {feat} im Options-Lernmodell {'leer' if empty else 'kaum befüllt'}",
            scope={"feature": feat, "horizons": sorted(covs)},
            finding=(f"Das Merkmal {feat} ist in den geprüften Options-Labels {'leer' if empty else 'kaum befüllt'}: "
                     + ", ".join(f"{h} T {c:.0%} von {labels[h]} Labels" for h, c in sorted(covs.items())) + "."),
            problem=("Ein Merkmal, das das Modell mangels Werten verwirft, steht nur scheinbar in der Spezifikation: "
                     "Information, die erfasst wird, geht verloren, und Varianten mit dieser Merkmalsgruppe testen "
                     "etwas anderes als beabsichtigt."),
            severity_reason=("Merkmal vollständig leer" if empty else "Merkmal überwiegend leer")
            + " – betrifft jede Prognose des Options-Lernmodells.",
            evidence=lines, sources=["outputs/research/option_learner/improvement.json",
                                     "outputs/research/option_learner/status_<h>d.json", "outputs/candidate_ledger/"],
            files=["modules/option_learner.py (LEDGER_FEATURES, features)", *refs],
            cause=cause,
            verify=(f"In outputs/candidate_ledger/ Format und Speicherort von {feat} auszählen und mit "
                    "option_learner.features() vergleichen."),
            change=(f"Entweder einen numerischen Wert von {feat} dort ablegen bzw. auslesen, wo das Lernmodell ihn "
                    "erwartet (eine klar dokumentierte Größe), oder das Merkmal bewusst aus LEDGER_FEATURES "
                    "streichen."),
            benefit=("Das Merkmal wird tatsächlich nutzbar oder die Spezifikation ehrlich. Ob es die Prognose "
                     "verbessert, ist offen und wird erst prospektiv gemessen – kein Renditeversprechen."),
            risks=("Eine Änderung der Merkmale ergibt eine neue Experimentversion: Forward-Evidenz, Referenz und "
                   "Varianten beginnen für diese Version neu."),
            effort="klein (eine Lese- bzw. Erfassungsstelle, Tests)", costs=NO_COST,
            acceptance=[(f"Abdeckung von {feat} in neuen geprüften Labels >= {ol.MIN_FEATURE_COVERAGE:.0%} binnen "
                         "20 Handelstagen nach Umsetzung (oder Merkmal entfernt)"),
                        "keine Änderung an Prognosen anderer Merkmale"],
            tests=[(f"Test: option_learner.features() liefert für eine Ledger-Zeile im tatsächlichen Format von "
                    f"{feat} den erwarteten Zahlenwert"),
                   "Test: fehlender oder ungültiger Wert bleibt fehlend (keine erfundenen Werte)"],
            material={"coverage": {h: _bucket(c) for h, c in covs.items()},
                      "format": sorted((loc or {}).get("non_numeric_types") or {}),
                      "top_level": bool((loc or {}).get("top_level"))}))
    return out


def _tasks_capture(gaps: list[dict], issues: list[str]) -> list[dict]:
    """Fehlende geprüfte Ergebnisse der gewählten Formen (Lücke outcome_coverage). Bewusst nicht aus den
    Gesamtzählern der Label-Herkunft: dort steckt Altbestand vor dem Start des Quote-Archivs (nie prüfbar, kein
    behebbarer Defekt) – maßgeblich sind die tatsächlich bewerteten Entscheidungen."""
    from modules import option_learner as ol
    out = []
    for g in gaps:
        if g.get("gap_id", "").split(":")[1:2] != ["outcome_coverage"]:
            continue
        cov, reasons = _num_or_none(g.get("value")), g.get("reasons")
        if cov is None or not isinstance(reasons, dict):
            issues.append(f"Lücke {g.get('gap_id')}: unvollständig – ignoriert")
            continue
        name = g["gap_id"].split(":")[2]
        top = max(reasons, key=reasons.get) if reasons else "unbekannt"
        out.append(_task(
            "outcome_capture", f"{name}:{g.get('horizon')}", "P1", "capture",
            f"Ergebnisse gewählter Optionsformen fehlen ({cov:.0%} geprüft, {g.get('horizon')} T)",
            scope={"name": name, "horizon": g.get("horizon")},
            finding=f"{name} ({g.get('horizon')} T): nur {cov:.0%} der gewählten Formen haben ein geprüftes "
                    f"Netto-Ergebnis; Ausfälle: {reasons}.",
            problem=("Fehlende Entry-/Exit-Quotes machen Entscheidungen unbewertbar. Fehlen sie nicht zufällig "
                     "(z. B. bei illiquiden Kontrakten), ist jede Auswertung verzerrt; unter "
                     f"{ol.MIN_COVERAGE:.0%} gibt es deshalb kein Urteil."),
            severity_reason="blockiert die Bewertung des Modells (INSUFFICIENT_COVERAGE).",
            evidence=[g.get("finding", ""), f"häufigster Grund: {top} ({reasons.get(top)})"],
            sources=["outputs/research/option_learner/status_<h>d.json"],
            files=["modules/research_evidence_discovery.py (provenance)", *_code_refs(top)],
            cause=f"Die Quote-Erfassung erreicht die gewählten Kontrakte nicht rechtzeitig (häufigster Grund {top}).",
            verify=f"Für die betroffenen Kandidaten prüfen, warum {top} gesetzt wird (Archiv-Zeitpunkte, Kontrakte).",
            change="Erfassungslücke im bestehenden Quote-Archiv-Workflow schließen; keine neue Datenquelle.",
            benefit="Bewertbare Entscheidungen statt INSUFFICIENT_COVERAGE; kein Renditeversprechen.",
            risks="Mehr Quote-Abrufe im bestehenden Workflow (Rate-Limits beachten).",
            effort="mittel", costs={**NO_COST, "api_calls": "zusätzliche Abrufe im bestehenden, kostenlosen Archiv"},
            acceptance=[f"Abdeckung >= {ol.MIN_COVERAGE:.0%} in den nächsten 20 fälligen Handelstagen"],
            tests=[f"Regressionstest für den Fall {top} (Kontrakt wird archiviert bzw. Grund korrekt gesetzt)"],
            material={"coverage": _bucket(cov), "top_reason": top}))
    return out


def _tasks_sample_bias(gaps: list[dict]) -> list[dict]:
    """Einseitige Evidenz (Stichprobe überwiegend Forschung bzw. bullisch) – nicht die reine Zeitraum-Warnung,
    die nur mit Warten verschwindet."""
    out = []
    for g in gaps:
        if g.get("kind") != "representativeness" or "überwiegend" not in str(g.get("finding")):
            continue
        text = str(g["finding"]).split("Walk-Forward-Evidenz: ", 1)[-1]
        out.append(_task(
            "sample_bias", text.split(":")[0], "P3", "data", f"Evidenz einseitig: {text.split(':')[0]}",
            scope={"warning": text.split(":")[0], "horizon": g.get("horizon")},
            finding=f"{g.get('horizon')} T: {text}", problem="Die Evidenz deckt nur einen Teil der Marktsituationen "
            "ab; ein Ergebnis gilt dann nicht automatisch für produktive Richtungssignale.",
            severity_reason="schränkt die Übertragbarkeit ein, kein Fehler im engeren Sinn.",
            evidence=[g["finding"]], sources=["outputs/research/option_learner/improvement.json"],
            files=["modules/candidate_ledger.py (Stichprobenregel)", "modules/option_learner.py (_representativeness)"],
            cause="Die Options-Snapshots werden überwiegend für eine Gruppe von Kandidaten erfasst.",
            verify="Verteilung der Snapshots nach Stichprobe und Richtung im Ledger auszählen.",
            change="Stichprobenregel so erweitern, dass die unterrepräsentierte Gruppe Snapshots erhält.",
            benefit="breitere, übertragbare Evidenz; kein Renditeversprechen.",
            risks="mehr Snapshots je Scanner-Lauf (bestehende, kostenlose Quelle)", effort="mittel",
            costs={**NO_COST, "api_calls": "zusätzliche Kontrakt-Snapshots je Lauf (bestehende Quelle)"},
            acceptance=["keine Repräsentativitätswarnung dieser Art binnen 90 Tagen"],
            tests=["Test der erweiterten Stichprobenregel"], material={"warning": text.split(":")[0]}))
    return out


def _variants_of(status: dict) -> list[dict]:
    return [{**v, "horizon": h} for h, st in sorted(status.items())
            for v in (((st or {}).get("variants") or {}).get("variants") or {}).values() if isinstance(v, dict)]


def _tasks_variants(status: dict, gaps: list[dict]) -> list[dict]:
    """Forschungsergebnisse der Varianten: belegte Übernahme, Widerruf (kritisch), nicht testbare Varianten,
    erschöpftes Fehlerbudget und nachgewiesene Modellschwächen, deren automatische Abhilfen ausgeschöpft sind."""
    out, variants = [], _variants_of(status)
    for v in variants:
        vid, ov = v.get("variant_id"), v.get("overrides") or {}
        if not vid or not isinstance(ov, dict) or len(ov) != 1:
            continue
        (m, val), = ov.items()
        ev, prop = v.get("current_evidence") or {}, v.get("change_proposal") or {}
        lr = (v.get("vs_base") or {}).get("learning_minus_reference") or {}
        nt = (v.get("vs_base") or {}).get("learning_minus_no_trade") or {}
        facts = [(f"Variante {vid} ({m}={val}, {v.get('horizon')} T, Status {v.get('status')}, Prüfung "
                  f"{v.get('look')}/{v.get('looks')}, Bericht {(v.get('look_report') or {}).get('report_id')})"),
                 f"relativ zur Basis: {v.get('relative_verdict')}, Δ {lr.get('mean')}/Tag, KI {lr.get('ci95')}",
                 f"netto gegenüber Nicht-Handeln: {v.get('economic_verdict')}, KI {nt.get('ci95')}",
                 f"Abdeckung {(v.get('comparison') or {}).get('coverage')}"]
        if ev.get("contradicts_decision") or prop.get("state") in ("WITHDRAWN", "REVIEW_REQUIRED"):
            out.append(_task(
                "withdrawn_success", vid, "P0", "critical", f"WARNUNG: belegte Verbesserung {m}={val} widerrufen",
                scope={"variant_id": vid, "horizon": v.get("horizon")},
                finding=(f"Für die Variante {m}={val} lag ein Übernahmevorschlag vor; nach korrigierten oder neuen "
                         f"Daten ist der Vorteil nicht mehr belegt (Vorschlag {prop.get('state')}, aktuelle Evidenz "
                         f"{ev.get('status')}, relativ {ev.get('relative_verdict')}, netto "
                         f"{ev.get('economic_verdict')})."),
                problem="Eine bereits gemeldete oder umgesetzte Verbesserung stützt sich auf nicht mehr gültige "
                        "Evidenz. Wurde sie übernommen, läuft das Modell mit einer unbelegten Änderung.",
                severity_reason="kritischer Evidenzwiderspruch – sofortige Prüfung.",
                evidence=facts + [f"Änderungsvorschlag: outputs/research/option_learner/change_proposals/{vid}.md"],
                sources=["outputs/research/option_learner/status_<h>d.json",
                         f"outputs/research/option_learner/change_proposals/{vid}.json",
                         "outputs/research/option_learner/look_reports/"],
                files=["modules/option_learner.py (Parameter der Variante)", f"change_proposals/{vid}.md"],
                cause="Nachgelieferte oder korrigierte Labels bzw. Quotes haben den Prüfbericht verändert.",
                verify="Eingefrorene Prüfberichte (alte und neue Fassung) vergleichen; prüfen, ob die Änderung "
                       "bereits in main übernommen wurde.",
                change="Falls übernommen: Rücknahme per eigenem PR vorbereiten; sonst den Vorschlag verwerfen.",
                benefit="verhindert, dass eine unbelegte Änderung im Modell bleibt.",
                risks="Rücknahme ergibt eine neue Experimentversion.", effort="klein", costs=NO_COST,
                acceptance=["Entscheidung dokumentiert (Rücknahme-PR oder verworfener Vorschlag)"],
                tests=["Regressionstest, dass die Basisparameter dem dokumentierten Stand entsprechen"],
                material={"state": prop.get("state"), "evidence": ev.get("status")}))
        elif v.get("status") in ("PROPOSE_ADOPTION", "CONFIRMED_ADOPTION") and prop.get("state") == "VALID":
            out.append(_task(
                "adoption", vid, "P1", "research_result", f"Belegte Verbesserung {m}={val} zur Übernahme prüfen",
                scope={"variant_id": vid, "horizon": v.get("horizon")},
                finding=(f"Die Variante {m}={val} ist an einem vorab festgelegten Prüfzeitpunkt relativ besser als "
                         "die Basis UND netto besser als Nicht-Handeln (Status " + str(v.get("status")) + ")."),
                problem="Ohne Umsetzung bleibt ein belegter Vorteil ungenutzt; umgesetzt wird nur per menschlichem PR.",
                severity_reason="statistisch belegtes Forschungsergebnis (korrigiertes Teil-Alpha).",
                evidence=facts, sources=[f"outputs/research/option_learner/change_proposals/{vid}.md",
                                         "outputs/research/option_learner/look_reports/"],
                files=["modules/option_learner.py (Parameter der Basis)", f"change_proposals/{vid}.md"],
                cause="kein Defekt – Forschungsergebnis legt eine Parameteränderung nahe.",
                verify="Prüfbericht und aktuellen Evidenzstatus (current_status VALID) im Änderungsvorschlag prüfen.",
                change=f"{m} der Basis auf {val} setzen (eine Änderung, neue Experimentversion).",
                benefit=f"gemessen: Δ {lr.get('mean')} je Tag gegenüber der Basis (KI {lr.get('ci95')}); keine "
                        "Zusage für die Zukunft.",
                risks="neue Experimentversion: Forward-Evidenz und Freigabe beginnen neu.", effort="klein",
                costs=NO_COST, acceptance=["Basis nutzt den neuen Wert; neue Experimentversion dokumentiert"],
                tests=["experiment_id-Test an die neue Version anpassen", "Regressionstest des Parameters"],
                material={"status": v.get("status"), "look": v.get("look")}))
        elif v.get("look") and v.get("relative_verdict") == "INSUFFICIENT_COVERAGE":
            excl = (v.get("comparison") or {}).get("excluded") or {}
            top = max(excl, key=excl.get) if excl else "unbekannt"
            out.append(_task(
                "variant_untestable", vid, "P2", "testing", f"Variante {m}={val} mangels Daten nicht testbar",
                scope={"variant_id": vid, "horizon": v.get("horizon")},
                finding=f"An Prüfung {v.get('look')} der Variante {m}={val} fehlen zu viele Ergebnisse "
                        f"(Abdeckung {(v.get('comparison') or {}).get('coverage')}, Ausfälle {excl}).",
                problem="Der vorab festgelegte Test kann kein Urteil fällen; das Fehlerbudget wird verbraucht, ohne "
                        "dass etwas gelernt wird.",
                severity_reason="Prüfung ohne Urteil (INSUFFICIENT_COVERAGE).", evidence=facts,
                sources=["outputs/research/option_learner/status_<h>d.json"],
                files=["modules/option_learner.py (_run_variant, evaluate_variants)", *_code_refs(top)],
                cause=f"Häufigster Ausfallgrund {top}: Entscheidungen oder Ergebnisse fehlen.",
                verify=f"Für die betroffenen Kandidaten nachvollziehen, warum {top} auftritt.",
                change="Erfassung der fehlenden Entscheidungen bzw. Ergebnisse beheben.",
                benefit="Varianten werden wieder entscheidbar; kein Renditeversprechen.", risks="gering",
                effort="mittel", costs=NO_COST, acceptance=["Abdeckung der nächsten Prüfung >= 80 %"],
                tests=[f"Regressionstest für den Ausfallgrund {top}"], material={"top_reason": top}))
    budgets = [((st or {}).get("variants") or {}).get("test_budget") for st in status.values()]
    if any(isinstance(b, dict) and b.get("exhausted") for b in budgets):
        b = next(b for b in budgets if isinstance(b, dict) and b.get("exhausted"))
        out.append(_task(
            "budget_exhausted", "test_budget", "P3", "testing", "Fehlerbudget der Varianten-Tests erschöpft",
            scope={},
            finding=f"Alle {b.get('max_tests')} Test-Slots des dauerhaften Fehlerbudgets (total_alpha "
                    f"{b.get('total_alpha')}) sind verbraucht; neue Varianten werden nicht mehr registriert.",
            problem="Weitere Modellverbesserungen können nicht mehr statistisch korrekt getestet werden.",
            severity_reason="stoppt die automatische Forschung, nicht den Betrieb.",
            evidence=[f"test_budget: {b}"], sources=["outputs/research/option_learner/variants.jsonl"],
            files=["config/option_learner_variants.yaml (test_budget)", "modules/option_learner.py (test_budget)"],
            cause="kein Defekt – das vorab festgelegte Budget ist planmäßig aufgebraucht.",
            verify="Prüfen, welche Varianten welche Slots verbraucht haben (variants.jsonl).",
            change="Menschliche Entscheidung: bewusster, dokumentierter Neustart des Registers mit neuem Budget "
                   "oder Ende der Variantenforschung.", benefit="klare Grundlage für weitere Tests.",
            risks="ein Neustart darf frühere Ergebnisse nicht umdeuten", effort="klein", costs=NO_COST,
            acceptance=["Entscheidung dokumentiert"], tests=["Test, dass der Neustart das alte Register erhält"],
            material={"used": b.get("used")}))
    proven = {g["gap_id"]: g for g in gaps if g.get("gap_id", "").split(":")[1:2] in (["no_skill"],
                                                                                      ["choice_worse_than_default"])}
    budget_left = not any(isinstance(b, dict) and b.get("exhausted") for b in budgets)
    for gid, g in sorted(proven.items()):
        related = [v for v in variants if gid in (v.get("gap_ids") or [])]
        if any(not v.get("closed_at") and v.get("status") not in (
                "FALSIFIED_WORSE", "FALSIFIED_NO_EFFECT", "BETTER_THAN_BASE_NOT_PROFITABLE") for v in related):
            continue                                  # automatische Abhilfe läuft noch: normales Warten
        if budget_left and not related:
            continue                                  # Varianten werden erst registriert: normales Warten
        out.append(_task(
            "proven_weakness", gid, "P2", "model", f"Nachgewiesene Modellschwäche ({g['gap_id'].split(':')[1]}, "
            f"{g.get('horizon')} T) ohne automatische Abhilfe",
            scope={"gap_id": gid, "horizon": g.get("horizon")},
            finding=f"{g.get('finding')}; automatisch getestete Varianten: "
                    + (", ".join(f"{v.get('variant_id')} {v.get('status')}" for v in related) or "keine möglich"),
            problem="Die freigegebenen automatischen Varianten haben die Schwäche nicht behoben; ohne technische "
                    "Änderung bleibt sie bestehen.",
            severity_reason="durch einen abgeschlossenen Test belegt (nicht nur vermutet).",
            evidence=[g.get("finding", "")] + [f"{v.get('variant_id')}: {v.get('status')}" for v in related],
            sources=["outputs/research/option_learner/improvement.json",
                     "outputs/research/option_learner/status_<h>d.json"],
            files=["modules/option_learner.py (decide, fit)", "config/option_learner_variants.yaml (approved_methods)"],
            cause="Die zulässigen Variantenmethoden decken die Ursache nicht ab.",
            verify="Fehleranalyse der gewählten Formen gegen die Standardform bzw. der Prognosen nach Merkmalsgruppe.",
            change="Eine neue, begründete Variantenmethode in config/option_learner_variants.yaml vorschlagen "
                   "(menschlicher PR) – kein direkter Eingriff in die Basis.",
            benefit="ermöglicht einen sauberen, vorab festgelegten Test einer neuen Abhilfe; kein Renditeversprechen.",
            risks="jede neue Variante verbraucht einen Slot des Fehlerbudgets", effort="mittel", costs=NO_COST,
            acceptance=["neue Methode in der Allowlist mit Begründung und Falsifikation"],
            tests=["Test, dass validate_overrides die neue Methode akzeptiert und Unbekanntes ablehnt"],
            material={"related": sorted(f"{v.get('variant_id')}:{v.get('status')}" for v in related)}))
    return out


def _tasks_release(seqs: dict) -> list[dict]:
    """Sequenzielle Forward-Prüfung der Basis ({Horizont: forward_sequential}, täglich aus dem Scanner bzw.
    wöchentlich aus dem Status): beschädigter Prüfplan/Fehler kritisch; fehlende Entscheidungen; letzte Prüfung
    ohne Bestätigung."""
    out = []
    for h, seq in sorted(seqs.items()):
        if not isinstance(seq, dict) or not seq.get("status"):
            continue
        s, b = seq["status"], seq.get("binding") or {}
        common = {"sources": [f"outputs/research/option_learner/status_{h}d.json",
                              "outputs/research/option_learner/forward_plans/",
                              "outputs/research/option_learner/forward_looks/"], "costs": NO_COST}
        if s in ("PLAN_INVALID", "ERROR"):
            out.append(_task(
                "release_integrity", f"{h}:{s}", "P0", "critical", f"WARNUNG: Freigabeprüfung {h} T defekt ({s})",
                scope={"horizon": h},
                finding=f"Die sequenzielle Forward-Prüfung ({h} T) meldet {s}"
                        + (f": {seq.get('error')}" if seq.get("error") else "") + ".",
                problem="Die Freigabe ist dadurch dauerhaft gesperrt; bei PLAN_INVALID wurde ein vorab festgelegter "
                        "Prüfplan verändert oder beschädigt.", severity_reason="Integrität der Freigabe betroffen.",
                evidence=["forward_sequential: " + json.dumps({k: seq.get(k) for k in ("status", "plan_id", "error")},
                                                              default=str)],
                files=["modules/option_learner.py (forward_plan, forward_sequential)"],
                cause="Prüfplan-Datei verändert/beschädigt oder Laufzeitfehler in der Prüfung.",
                verify="git-Historie der Plandatei und das Scanner-Log prüfen.",
                change="Ursache beheben; einen Plan nie still ersetzen – Neustart nur dokumentiert per PR.",
                benefit="funktionierende, nachvollziehbare Freigabe.", risks="gering", effort="klein",
                acceptance=["Status wieder AWAITING_LOOK/RUNNING/CONFIRMED"],
                tests=["Regressionstest für die Fehlerursache"], material={"status": s}, **common))
        elif b.get("verdict") == "INSUFFICIENT_COVERAGE" and _num_or_none(b.get("decision_coverage")) is not None \
                and b["decision_coverage"] < 0.8:
            out.append(_task(
                "decision_capture", f"{h}", "P1", "capture",
                f"Tagesentscheidungen des Hauptmodells fehlen ({b['decision_coverage']:.0%}, {h} T)",
                scope={"horizon": h},
                finding=f"An Prüfung {b.get('look')} ({h} T) haben nur {b['decision_coverage']:.0%} der Ledger-"
                        f"Kandidaten eine am Signaltag festgehaltene Entscheidung; Ausfälle {b.get('excluded')}.",
                problem="Ohne vollständige Tagesentscheidungen kann die Freigabe nie bestätigt werden.",
                severity_reason="blockiert die statistische Freigabe.", evidence=[f"Prüfung: {b}"],
                files=["modules/option_learner.py (recommend)", ".github/workflows/scanner.yml"],
                cause="Kandidaten kommen nach dem Scanner-Lauf ins Ledger oder der Lauf fällt aus.",
                verify="Ledger-Zeitstempel der fehlenden Kandidaten mit den Journal-Zeitpunkten vergleichen.",
                change="Entscheidung für jeden Ledger-Kandidaten am Signaltag sicherstellen.",
                benefit="die vorab festgelegte Prüfung wird entscheidbar.", risks="gering", effort="mittel",
                acceptance=["Entscheidungsabdeckung >= 80 % an der nächsten Prüfung"],
                tests=["Regressionstest: jeder Kandidat des Tages erhält eine Entscheidung"],
                material={"coverage": _bucket(b["decision_coverage"])}, **common))
        elif s in ("NOT_CONFIRMED", "REJECTED_NEGATIVE"):
            out.append(_task(
                "base_not_confirmed", f"{h}:{seq.get('plan_id')}", "P2", "research_result",
                f"Hauptmodell {h} T im Forward-Test {'negativ' if s == 'REJECTED_NEGATIVE' else 'nicht bestätigt'}",
                scope={"horizon": h, "plan_id": seq.get("plan_id")},
                finding=f"Die vorab festgelegte Forward-Prüfung ({h} T, Plan {seq.get('plan_id')}) endet mit {s} "
                        f"(Prüfung {b.get('look')}: KI {b.get('selected_ci95')} zum Niveau {b.get('ci_level')}).",
                problem="Für diese Experimentversion bleibt die Freigabe dauerhaft gesperrt; nur eine neue, "
                        "begründete Version kann erneut geprüft werden.",
                severity_reason="abgeschlossener, statistisch korrekter Test.", evidence=[f"Prüfung: {b}"],
                files=["modules/option_learner.py (experiment)", "config/option_learner_variants.yaml"],
                cause="Die Merkmale oder die Auswahlregel tragen prospektiv keinen nachweisbaren Netto-Vorteil.",
                verify="Leistungszeugnis, Varianten-Ergebnisse und Lücken auswerten.",
                change=("Eine begründete neue Experimentversion vorschlagen (menschlicher PR) oder das Modell ruhen "
                        "lassen."),
                benefit="ehrliche Grundlage statt dauerhaft gesperrter Version.", risks="neue Version startet neu",
                effort="mittel", acceptance=["Entscheidung dokumentiert"],
                tests=["experiment_id-Test der neuen Version"],
                material={"status": s}, **common))
    return out


def _tasks_infrastructure(history: list[dict], snapshot: dict, wf_status: dict, now: datetime) -> list[dict]:
    """Dauerhaft ausgefallene wichtige Datenquellen und Workflows, die regelmäßig nachgestartet werden müssen."""
    out, sources = [], (snapshot or {}).get("sources") or {}
    by_source: dict = {}
    for e in history or []:
        if isinstance(e, dict) and e.get("source_id") and e.get("checked_at") and e.get("status"):
            by_source.setdefault(e["source_id"], []).append(e)
    for sid, meta in sorted(sources.items()):
        if not isinstance(meta, dict) or meta.get("status") not in SOURCE_BAD \
                or meta.get("criticality") not in ("CRITICAL", "IMPORTANT"):
            continue
        events = sorted(by_source.get(sid, []), key=lambda e: e["checked_at"])
        streak = []
        for e in events:
            streak = [*streak, e] if e["status"] in SOURCE_BAD else []
        if not streak:
            continue
        try:
            since = datetime.fromisoformat(streak[0]["checked_at"])
        except ValueError:
            continue
        days = (now - since).days
        if days < SOURCE_PERSIST_DAYS:
            continue                                   # kurzfristige Störung: meldet bereits das Ops-Monitoring
        last = streak[-1]
        out.append(_task(
            "source_failure", sid, "P1" if meta["criticality"] == "CRITICAL" else "P2", "infrastructure",
            f"Datenquelle {sid} seit {days} Tagen {meta['status']}",
            scope={"source_id": sid},
            finding=f"Die {meta['criticality']}-Quelle {sid} ist seit {streak[0]['checked_at'][:10]} ({days} Tage, "
                    f"{len(streak)} Prüfungen) durchgehend {', '.join(sorted({e['status'] for e in streak}))}.",
            problem="Merkmale aus dieser Quelle fehlen oder veralten; nachgelagerte Modelle arbeiten mit Fallbacks.",
            severity_reason=f"{meta['criticality']}-Quelle, länger als {SOURCE_PERSIST_DAYS} Tage gestört.",
            evidence=[(f"letzter Erfolg {meta.get('last_success')}, letzte Beobachtung "
                       f"{meta.get('latest_observation')}, Datenalter {meta.get('data_age_days')} Tage, erwartet "
                       f"{meta.get('expected_freshness')}"),
                      f"letzte Prüfung {last.get('checked_at')}: {last.get('status')}, Fehler {last.get('error')}"],
            sources=["outputs/health/source_health_snapshot.json", "outputs/health/source_health_history.jsonl"],
            files=_code_refs(sid) or ["modules/external/"], cause="Format- oder Endpunktänderung der Quelle bzw. "
            "ausbleibende Veröffentlichung.",
            verify="Ingest der Quelle lokal ausführen und Fehlermeldung bzw. Datenstand prüfen.",
            change="Ingest an die Quelle anpassen oder die Quelle bewusst deaktivieren (status_override).",
            benefit="aktuelle Merkmale statt Fallback; kein Renditeversprechen.", risks="gering", effort="mittel",
            costs=NO_COST, acceptance=["Quelle wieder HEALTHY oder bewusst deaktiviert"],
            tests=["Regressionstest des Parsers mit dem aktuellen Antwortformat"],
            material={"status": meta["status"], "weeks": days // 7}))
    recov = (wf_status or {}).get("recoveries") or {}
    if isinstance(recov, dict):
        window = {(now.date() - timedelta(days=i)).isoformat() for i in range(RECOVERY_WINDOW_DAYS)}
        counts: dict = {}
        for day, wfs in recov.items():
            if day in window and isinstance(wfs, dict):
                for wf in wfs:
                    counts.setdefault(wf, []).append(day)
        latest = ((wf_status or {}).get("days") or {})
        for wf, days in sorted(counts.items()):
            if len(days) < RECOVERY_MIN_DAYS:
                continue
            all_days = sorted(d for d, w in recov.items() if isinstance(w, dict) and wf in w)
            info = (latest.get(max(days)) or {}).get(wf) or {}         # letzter Tag mit Nachstart
            out.append(_task(
                "workflow_recovery", wf, "P2", "infrastructure",
                f"Workflow {wf} muss regelmäßig nachgestartet werden",
                scope={"workflow": wf},
                finding=f"Der Workflow {wf} wurde an {len(days)} der letzten {RECOVERY_WINDOW_DAYS} Tage vom Watchdog "
                        f"nachgestartet (insgesamt {len(all_days)} Tage seit {all_days[0]}).",
                problem="Der geplante Lauf startet nicht zuverlässig; nachgelagerte Schritte warten oder nutzen ältere "
                        "Daten, der Watchdog verbraucht täglich zusätzliche Läufe.",
                severity_reason="dauerhaftes Muster, kein Einzelfall.",
                evidence=[f"Nachstarts: {sorted(days)}",
                          (f"{max(days)}: Datei {info.get('file')}, erwartet {info.get('expected')}, Frist "
                           f"{info.get('deadline')}, letzter Lauf {info.get('actual_last_run')}, Zustand "
                           f"{info.get('state')}, Aktion {info.get('action')}, Grund {info.get('reason')}")],
                sources=["outputs/state/workflow_status.json"],
                files=[f".github/workflows/{info.get('file') or wf + '.yml'}", "scripts/workflow_watchdog.py",
                       "modules/workflow_health.py"],
                cause="Der Cron-Zeitpunkt kollidiert mit GitHub-Verzögerungen oder einer Abhängigkeit.",
                verify="Startzeiten der geplanten Läufe der letzten zwei Wochen mit expected/deadline vergleichen.",
                change="Zeitplan oder Abhängigkeit des Workflows anpassen, sodass er ohne Nachstart läuft.",
                benefit="weniger Läufe, pünktliche Daten.", risks="gering", effort="klein", costs=NO_COST,
                acceptance=["keine Nachstarts an 7 aufeinanderfolgenden Tagen"],
                tests=["Test der Watchdog-Erwartung für den neuen Zeitplan"],
                material={"chronic": True}))
    return out


def _ts(value) -> datetime | None:
    """Zeitstempel oder Datum (ISO) -> datetime in UTC; ungültig -> None."""
    try:
        d = datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None
    return None if d is None else d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _fresh(stamp, days: int, now: datetime, problem: str | None = None) -> tuple[str, str]:
    """("ok", "") oder ("unknown", Grund): fehlt, beschädigt, ohne Zeitstempel, veraltet, aus der Zukunft."""
    if problem:
        return "unknown", problem
    d = _ts(stamp)
    if d is None:
        return "unknown", "ohne gültigen Zeitstempel"
    age = (now - d).total_seconds() / 86400
    if age < -1:
        return "unknown", f"Zeitstempel {str(stamp)[:10]} liegt in der Zukunft"
    if age > days:
        return "unknown", f"veraltet (Stand {str(stamp)[:10]}, älter als {days} Tage)"
    return "ok", ""


def input_quality(inputs: dict, now: datetime) -> dict[str, tuple[str, str]]:
    """Beurteilbarkeit jeder Quelle: nur "ok" (vorhanden, gültig, aktuell) darf Aufgaben erzeugen oder als behoben
    bestätigen; alles andere ergibt UNKNOWN statt eines Statuswechsels."""
    problems = inputs.get("problems") or {}
    imp = inputs.get("improvement")
    q = {"improvement": _fresh(imp.get("generated") if isinstance(imp, dict) else None, FRESH_DAYS["option"], now,
                               problems.get("improvement") or (None if isinstance(imp, dict) and imp else "fehlt"))}
    status = inputs.get("status") or {}
    for h in sorted({*OPTION_HORIZONS, *(k for k in status if str(k).isdigit())}, key=int):
        st = status.get(h) if isinstance(status, dict) else None
        q[f"status:{int(h)}"] = _fresh(st.get("as_of") if isinstance(st, dict) else None, FRESH_DAYS["option"], now,
                                       problems.get(f"status:{h}") or (None if isinstance(st, dict) and st
                                                                       else "fehlt"))
        seq, seq_as_of = _seq_of(inputs, int(h))
        q[f"seq:{int(h)}"] = _fresh(seq_as_of, FRESH_DAYS["seq"], now, None if seq else "fehlt")
    snap = inputs.get("source_snapshot")
    q["source_snapshot"] = _fresh(snap.get("generated") if isinstance(snap, dict) else None, FRESH_DAYS["health"], now,
                                  problems.get("source_snapshot") or (None if isinstance(snap, dict) and snap
                                                                      else "fehlt"))
    wf = inputs.get("workflow_status")
    days = [d for d in ((wf or {}).get("days") or {}) if _ts(d)] if isinstance(wf, dict) else []
    q["workflow_status"] = _fresh(max(days) if days else None, FRESH_DAYS["workflow"], now,
                                  problems.get("workflow_status") or (None if days else "fehlt"))
    return q


def _seq_of(inputs: dict, h: int) -> tuple[dict | None, str | None]:
    """forward_sequential eines Horizonts: der neuere Stand aus der täglichen Empfehlung oder dem Wochenstatus."""
    st = (inputs.get("status") or {}).get(h) if isinstance(inputs.get("status"), dict) else None
    cand = [(str(st.get("as_of")), st.get("forward_sequential"))] if isinstance(st, dict) else []
    daily = (inputs.get("daily_seq") or {}).get(h)
    if isinstance(daily, dict):
        cand.append((str(daily.get("as_of")), daily.get("forward_sequential")))
    cand = [(a, sq) for a, sq in cand if isinstance(sq, dict) and sq.get("status") and _ts(a)]
    return (max(cand, key=lambda c: c[0])[1], max(cand, key=lambda c: c[0])[0]) if cand else (None, None)


def improvement_tasks(inputs: dict, now: datetime | None = None) -> tuple[list[dict], list[str]]:
    """Alle belegten technischen Verbesserungsaufgaben aus den vorhandenen Research-Ausgaben -> (Aufgaben,
    ignorierte Eingaben). Ausgewertet wird nur, was vorhanden, gültig und aktuell ist (input_quality); fehlende,
    beschädigte oder veraltete Quellen erzeugen keine Aufgabe."""
    now = now or datetime.now(timezone.utc)
    issues: list[str] = []
    q = input_quality(inputs, now)
    for key, (state, why) in sorted(q.items()):
        if state != "ok" and why != "fehlt":
            issues.append(f"{INPUT_LABEL.get(key, key)}: {why} – nicht ausgewertet (ignoriert)")
    imp = inputs.get("improvement") if q["improvement"][0] == "ok" else {}
    gaps = [g for g in (imp or {}).get("gaps") or [] if isinstance(g, dict) and isinstance(g.get("gap_id"), str)]
    if len(gaps) != len((imp or {}).get("gaps") or []):
        issues.append("improvement.json: Lücken ohne gültige gap_id ignoriert")
    raw = inputs.get("status") if isinstance(inputs.get("status"), dict) else {}
    status = {int(h): st for h, st in raw.items() if str(h).isdigit() and isinstance(st, dict) and st
              and q.get(f"status:{int(h)}", ("unknown",))[0] == "ok"}
    gaps = [g for g in gaps if not isinstance(g.get("horizon"), int) or g["horizon"] in status]
    seqs = {h: _seq_of(inputs, h)[0] for h in {*OPTION_HORIZONS, *status} if q.get(f"seq:{h}", ("",))[0] == "ok"}
    tasks = (_tasks_empty_features(gaps, status, inputs.get("ledger"), issues) + _tasks_capture(gaps, issues)
             + _tasks_sample_bias(gaps) + _tasks_variants(status, gaps) + _tasks_release(seqs)
             + _tasks_infrastructure(inputs.get("source_history") if q["source_snapshot"][0] == "ok" else [],
                                     inputs.get("source_snapshot") if q["source_snapshot"][0] == "ok" else {},
                                     inputs.get("workflow_status") if q["workflow_status"][0] == "ok" else {}, now))
    as_of = max([str(st.get("as_of")) for st in status.values() if st.get("as_of")]
                + [str((imp or {}).get("generated", ""))[:10]])
    unique = {}
    for t in tasks:
        unique.setdefault(t["task_id"], {**t, "data_as_of": as_of or now.date().isoformat()})
    return sorted(unique.values(), key=lambda t: (t["priority"], t["task_id"])), issues


def resolution(task: dict, inputs: dict, now: datetime | None = None) -> tuple[str, str]:
    """Zustand einer bekannten Aufgabe, die gerade NICHT erkannt wurde: ("RESOLVED", Beleg) nur, wenn aktuelle,
    gültige Daten die Behebung positiv zeigen; ("UNKNOWN", Grund), wenn die nötigen Daten fehlen, beschädigt,
    veraltet oder nicht aussagekräftig sind; ("OPEN", Grund), wenn die Daten den Mangel weiter zeigen."""
    from modules import option_learner as ol
    now = now or datetime.now(timezone.utc)
    q, kind, sc = input_quality(inputs, now), task.get("kind"), task.get("scope")
    if not isinstance(sc, dict):
        return "UNKNOWN", "Aufgabe ohne Geltungsbereich – nicht beurteilbar"

    def missing(*keys) -> str | None:
        bad = [f"{INPUT_LABEL.get(k, k)} {q.get(k, ('unknown', 'fehlt'))[1]}" for k in keys
               if q.get(k, ("unknown",))[0] != "ok"]
        return "; ".join(bad) or None
    h = sc.get("horizon")
    status = inputs.get("status") or {}
    st = status.get(h) if isinstance(h, int) else None
    imp = inputs.get("improvement") or {}
    if kind == "empty_feature":
        hs = [x for x in sc.get("horizons") or [] if isinstance(x, int)]
        if not hs or (why := missing("improvement", *[f"status:{x}" for x in hs])):
            return "UNKNOWN", why or "ohne Horizont"
        for x in hs:
            sx = status[x]
            feats = ((sx.get("experiment") or {}).get("features") or {}).get("ledger")
            if isinstance(feats, list) and sc.get("feature") not in feats:
                continue                               # Merkmal bewusst entfernt
            if (sx.get("verified_labels") or 0) < MIN_LABELS_FOR_COVERAGE:
                return "UNKNOWN", f"{x} T: weniger als {MIN_LABELS_FOR_COVERAGE} geprüfte Labels – nicht beurteilbar"
            cov = _num_or_none((sx.get("feature_coverage") or {}).get(sc.get("feature")))
            if cov is None:
                return "UNKNOWN", f"{x} T: Abdeckung von {sc.get('feature')} nicht gemessen"
            if cov < ol.MIN_FEATURE_COVERAGE:
                return "OPEN", f"{x} T: Abdeckung {cov:.0%}"
        return "RESOLVED", "aktuelle Daten: Abdeckung >= " + f"{ol.MIN_FEATURE_COVERAGE:.0%} bzw. Merkmal entfernt"
    needs_status = kind in ("outcome_capture", "sample_bias", "proven_weakness", "adoption", "withdrawn_success",
                            "variant_untestable")
    if needs_status and (not isinstance(h, int) or (why := missing(f"status:{h}"))):
        return "UNKNOWN", why or "ohne Horizont"
    if kind == "outcome_capture":
        s = st.get(sc.get("name")) or {}
        cov, trades = _num_or_none(s.get("coverage")), s.get("trades") or 0
        if cov is None or trades < MIN_TRADES_FOR_COVERAGE:
            return "UNKNOWN", f"zu wenige bewertbare Trades ({trades})"
        return ("RESOLVED", f"Abdeckung {cov:.0%} bei {trades} Trades") if cov >= ol.MIN_COVERAGE else \
            ("OPEN", f"Abdeckung {cov:.0%}")
    if kind == "sample_bias":
        rep = (st.get("walk_forward") or {}).get("representativeness") or {}
        if not rep.get("by_sample"):
            return "UNKNOWN", "keine bewerteten Entscheidungen"
        hit = any(str(w).startswith(str(sc.get("warning"))) for w in rep.get("warnings") or [])
        return ("OPEN", "Warnung besteht") if hit else ("RESOLVED", "aktuelle Evidenz ohne diese Warnung")
    if kind == "proven_weakness":
        if why := missing("improvement"):
            return "UNKNOWN", why
        if any(g.get("gap_id") == sc.get("gap_id") for g in imp.get("gaps") or [] if isinstance(g, dict)):
            return "OPEN", "Lücke besteht (automatische Abhilfe läuft)"
        if (st.get("walk_forward") or {}).get("verdict") in (None, "NEED_MORE_DATA"):
            return "UNKNOWN", "Walk-Forward ohne Urteil"
        return "RESOLVED", "aktuelle Lückenanalyse ohne diese Schwäche"
    if kind in ("adoption", "withdrawn_success", "variant_untestable"):
        v = (((st.get("variants") or {}).get("variants")) or {}).get(sc.get("variant_id"))
        if not isinstance(v, dict):
            return "UNKNOWN", "Variante im aktuellen Status nicht enthalten"
        prop, ev = v.get("change_proposal") or {}, v.get("current_evidence") or {}
        if kind == "adoption":
            on = v.get("status") in ("PROPOSE_ADOPTION", "CONFIRMED_ADOPTION") and prop.get("state") == "VALID"
            return ("OPEN", "Übernahmevorschlag gültig") if on else \
                ("RESOLVED", f"Status {v.get('status')}, Vorschlag {prop.get('state')}")
        if kind == "withdrawn_success":
            on = ev.get("contradicts_decision") or prop.get("state") in ("WITHDRAWN", "REVIEW_REQUIRED")
            return ("OPEN", "Widerspruch besteht") if on else \
                ("RESOLVED", f"Vorschlag {prop.get('state')}, kein Widerspruch")
        if not v.get("look"):
            return "UNKNOWN", "noch keine Prüfung"
        return ("OPEN", "INSUFFICIENT_COVERAGE") if v.get("relative_verdict") == "INSUFFICIENT_COVERAGE" else \
            ("RESOLVED", f"Prüfung {v.get('look')}: {v.get('relative_verdict')}")
    if kind == "budget_exhausted":
        budgets = [((status.get(x) or {}).get("variants") or {}).get("test_budget")
                   for x in status if q.get(f"status:{x}", ("",))[0] == "ok"]
        budgets = [b for b in budgets if isinstance(b, dict) and "exhausted" in b]
        if not budgets:
            return "UNKNOWN", "kein aktueller Varianten-Status"
        return ("OPEN", "erschöpft") if any(b["exhausted"] for b in budgets) else ("RESOLVED", "Budget verfügbar")
    if kind in ("release_integrity", "decision_capture", "base_not_confirmed"):
        if not isinstance(h, int) or (why := missing(f"seq:{h}")):
            return "UNKNOWN", why or "ohne Horizont"
        seq = _seq_of(inputs, h)[0]
        if kind == "release_integrity":
            return ("OPEN", seq["status"]) if seq["status"] in ("PLAN_INVALID", "ERROR") else \
                ("RESOLVED", f"Freigabeprüfung {seq['status']}")
        if kind == "decision_capture":
            b = seq.get("binding") or {}
            if not b:
                return "UNKNOWN", "noch keine Prüfung"
            low = b.get("verdict") == "INSUFFICIENT_COVERAGE" and (_num_or_none(b.get("decision_coverage")) or 0) < 0.8
            return ("OPEN", "Entscheidungsabdeckung < 80 %") if low else \
                ("RESOLVED", f"Prüfung {b.get('look')}: {b.get('verdict')}")
        same = seq.get("plan_id") == sc.get("plan_id")
        return ("OPEN", seq["status"]) if same and seq["status"] in ("NOT_CONFIRMED", "REJECTED_NEGATIVE") else \
            ("RESOLVED", f"neuer Prüfplan {seq.get('plan_id')}" if not same else f"Status {seq['status']}")
    if kind == "source_failure":
        if why := missing("source_snapshot"):
            return "UNKNOWN", why
        meta = ((inputs.get("source_snapshot") or {}).get("sources") or {}).get(sc.get("source_id"))
        if not isinstance(meta, dict) or not meta.get("status"):
            return "UNKNOWN", "Quelle im aktuellen Snapshot nicht enthalten"
        return ("OPEN", meta["status"]) if meta["status"] in SOURCE_BAD else ("RESOLVED", f"Status {meta['status']}")
    if kind == "workflow_recovery":
        if why := missing("workflow_status"):
            return "UNKNOWN", why
        window = {(now.date() - timedelta(days=i)).isoformat() for i in range(RECOVERY_WINDOW_DAYS)}
        recov = (inputs.get("workflow_status") or {}).get("recoveries") or {}
        hits = sorted(d for d, w in recov.items() if d in window and isinstance(w, dict) and sc.get("workflow") in w)
        return ("OPEN", f"Nachstarts {hits}") if hits else \
            ("RESOLVED", f"kein Nachstart in den letzten {RECOVERY_WINDOW_DAYS} Tagen")
    return "UNKNOWN", f"unbekannte Aufgabenart {kind}"


def _load_checked(path: Path):
    """(Inhalt, Problem): fehlende Datei -> "fehlt", kein gültiges JSON-Objekt -> "beschädigt"."""
    if not path.exists():
        return {}, "fehlt"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning(f"research_director: {path} nicht lesbar ({e})")
        return {}, "beschädigt"
    return (data, None) if isinstance(data, dict) else ({}, "beschädigt")


def load_task_inputs(paths: dict | None = None, now: datetime | None = None) -> dict:
    """Eingaben der Aufgaben-Erkennung aus den vorhandenen Dateien – fehlende oder beschädigte Quellen werden als
    Problem vermerkt (nie durch Annahmen ersetzt). Dazu der jüngste Stand der Freigabeprüfung aus den täglichen
    Empfehlungen des Scanners (für die tägliche P0-Prüfung)."""
    from modules.atomic_io import read_jsonl
    p, now = {**TASK_INPUTS, **(paths or {})}, now or datetime.now(timezone.utc)
    root, problems = Path(p["option_root"]), {}
    imp, problems["improvement"] = _load_checked(root / "improvement.json")
    status, daily = {}, {}
    for h in OPTION_HORIZONS:
        status[h], problems[f"status:{h}"] = _load_checked(root / f"status_{h}d.json")
        recs = sorted((root / "recommendations").glob(f"????-??-??_{h}d.json")) if (root / "recommendations").exists() \
            else []
        if recs:
            rec, bad = _load_checked(recs[-1])
            if not bad and isinstance(rec.get("forward_sequential"), dict):
                daily[h] = {"as_of": rec.get("date") or recs[-1].name[:10],
                            "forward_sequential": rec["forward_sequential"]}
    snapshot, problems["source_snapshot"] = _load_checked(Path(p["source_snapshot"]))
    wf, problems["workflow_status"] = _load_checked(Path(p["workflow_status"]))
    ledger = None
    try:
        from modules import research_evidence_discovery as red
        ledger = red.load(p["ledger"], now.date()) if Path(p["ledger"]).exists() else None
    except (OSError, ValueError) as e:
        log.warning(f"research_director: Ledger nicht lesbar ({e}) – Formatbelege fehlen")
    return {"improvement": imp, "status": status, "daily_seq": daily,
            "source_history": read_jsonl(p["source_history"]) if Path(p["source_history"]).exists() else [],
            "source_snapshot": snapshot, "workflow_status": wf, "ledger": ledger,
            "problems": {k: v for k, v in problems.items() if v}}


# ── Vorlagen (deterministisch) ──

def _repo() -> str:
    return os.environ.get("GITHUB_REPOSITORY") or "dieses Repository"


def claude_task(t: dict) -> str:
    """Kopierfertige Aufgabe für Claude Code: zuerst Ursache prüfen, dann gezielt ändern, Tests, PR ohne Merge."""
    lines = [f"Repository: {_repo()} (neuer Branch von main)",
             f"Aufgabe {t['task_id']}: {t['short']}", "",
             f"Befund (Research-Daten vom {t['data_as_of']}):", t["finding"], "", "Belege:",
             *[f"- {e}" for e in t["evidence"]], "",
             "Vermutete Ursache – HYPOTHESE, noch nicht bestätigt:", t["cause"], "",
             "Vorgehen:",
             (f"1. Ursache zuerst verifizieren: {t['verify']} Bestätigt sich die Hypothese nicht, nichts ändern "
              "und den tatsächlichen Befund berichten."),
             "2. Nur gezielte Änderungen, voraussichtlich in: " + "; ".join(t["files"]) + ".",
             f"3. Vorgeschlagene Änderung: {t['change']}",
             "4. Regressionstests ergänzen: " + "; ".join(t["tests"]) + ".",
             "5. Volle Testsuite (pytest) und Ruff ausführen.",
             "6. Pull Request erstellen – KEIN automatischer Merge.", "",
             "Akzeptanzkriterien:", *[f"- {a}" for a in t["acceptance"]], "",
             f"Grenzen: {BOUNDARIES}"]
    return "\n".join(lines)


def _cost_text(c: dict) -> str:
    return (f"API-Aufrufe: {c.get('api_calls')}; LLM-/Claude-Aufrufe: {c.get('llm_calls')}; kostenpflichtige Daten: "
            f"{'ja' if c.get('paid_data') else 'nein'}; Rechenaufwand: {c.get('compute', 'vernachlässigbar')}. Keine "
            "zusätzlichen Infrastrukturkosten durch diese Meldung selbst (feste Vorlage, kein KI-Aufruf).")


def task_text(t: dict, reason: str) -> str:
    return "\n".join([
        (f"Task-ID {t['task_id']} · Priorität {t['priority']} ({PRIORITY_TEXT[t['priority']]}) · Kategorie "
         f"{CATEGORY_TEXT.get(t['category'], t['category'])} · Status {t.get('status', 'OPEN')} ({reason})"),
        "", "1. Was wurde erkannt?", t["finding"],
        "", "2. Warum ist das problematisch?", t["problem"],
        "", "3. Wie schwerwiegend ist das Problem?", f"{PRIORITY_TEXT[t['priority']]}: {t['severity_reason']}",
        "", "4. Welche Verbesserung wird vorgeschlagen?", t["change"], f"Erwarteter Nutzen: {t['benefit']}",
        f"Risiken: {t['risks']}", f"Aufwand: {t['effort']}",
        f"Vermutete Ursache (Hypothese, nicht bestätigt): {t['cause']}",
        "", "5. Welche Daten belegen das?", *[f"- {e}" for e in t["evidence"]],
        "Quellen: " + ", ".join(t["sources"]) + f" (Stand {t['data_as_of']})",
        "", "6. Entstehen zusätzliche Infrastrukturkosten?", _cost_text(t["costs"]),
        "", "7. Aufgabe für Claude Code (zum Kopieren)", "----- BEGIN CLAUDE CODE TASK -----", claude_task(t),
        "----- END CLAUDE CODE TASK -----"])


REASON_TEXT = {"NEW": "neu erkannt", "CHANGED": "Befund wesentlich geändert", "REOPENED": "erneut aufgetreten",
               "PENDING": "Versand stand noch aus"}


def render_mail(tasks: list[tuple[dict, str]]) -> tuple[str, str, str]:
    """(Betreff, HTML, Text) für eine oder mehrere Aufgaben – feste Vorlage, kein KI-Aufruf."""
    first = tasks[0][0]
    subject = SUBJECT.format(first["short"] + (f" (+{len(tasks) - 1} weitere)" if len(tasks) > 1 else ""))
    head = ("Automatisch aus belegten Research-Befunden erzeugt (feste Vorlage, kein KI-Aufruf, keine API-Kosten). "
            "Nichts wird automatisch geändert; jede Umsetzung braucht deine Freigabe (PR ohne Auto-Merge).")
    parts = [head, ""]
    for i, (t, reason) in enumerate(tasks, 1):
        parts += [f"=== {i}/{len(tasks)}: {t['short']} ===", task_text(t, REASON_TEXT.get(reason, reason)), ""]
    parts += [(f"Entscheidung festhalten (keine weitere Mail zu dieser Aufgabe): {TASK_DECISIONS} – "
               "tasks: {<Task-ID>: {status: REJECTED|IN_PROGRESS|DONE, note: ...}}")]
    text = "\n".join(parts)
    body = html.escape(text)
    return subject, f"<pre style='font-family:monospace;white-space:pre-wrap'>{body}</pre>", text


# ── Zustand, Dedupe und Versand ──

def task_state(events: list[dict]) -> dict:
    """Zustand je Aufgabe aus dem append-only Ereignisprotokoll (zeitlich sortiert: parallele Läufe werden per
    Union-Merge zusammengeführt, ihre Zeilen können verschränkt sein)."""
    st: dict = {}
    for e in sorted((e for e in events if isinstance(e, dict) and e.get("task_id")), key=lambda e: str(e.get("at"))):
        s = st.setdefault(e["task_id"], {"status": "OPEN", "notified_fingerprint": None, "notifications": [],
                                         "send_failures": 0})
        ev = e.get("event")
        if ev in ("detected", "changed", "reopened"):
            s.update(task=e.get("task"), fingerprint=e.get("fingerprint"), status="OPEN", unknown_reason=None,
                     last_changed=e.get("at"), first_detected=s.get("first_detected") or e.get("at"))
        elif ev == "confirmed":                        # nach UNKNOWN wieder nachgewiesen, Befund unverändert
            s.update(status="NOTIFIED" if s["notified_fingerprint"] == s.get("fingerprint") else "OPEN",
                     unknown_reason=None)
        elif ev == "unknown":
            s.update(status="UNKNOWN", unknown_reason=e.get("reason"), unknown_since=e.get("at"))
        elif ev == "resolved":
            s.update(status="RESOLVED", resolved_at=e.get("at"), resolved_reason=e.get("reason"))
        elif ev == "notified":
            s["notified_fingerprint"] = e.get("fingerprint")
            s["notifications"].append({k: e.get(k) for k in ("at", "channel", "subject", "fingerprint", "reason")})
            if s["status"] == "OPEN" and e.get("fingerprint") == s.get("fingerprint"):
                s["status"] = "NOTIFIED"
        elif ev == "send_failed":
            s["send_failures"] += 1
            s["last_send_error"] = {k: e.get(k) for k in ("at", "mail_status", "error")}
    return st


def _decisions() -> dict:
    try:
        d = yaml.safe_load(TASK_DECISIONS.read_text(encoding="utf-8")) if TASK_DECISIONS.exists() else {}
    except (OSError, yaml.YAMLError) as e:
        log.warning(f"research_director: {TASK_DECISIONS} nicht lesbar ({e})")
        return {}
    return {k: v for k, v in ((d or {}).get("tasks") or {}).items() if isinstance(v, dict)}


def _is_critical(t: dict) -> bool:
    return (t or {}).get("kind") in CRITICAL_KINDS


def notify_tasks(tasks: list[dict], now: datetime | None = None, *, mode: str = "weekly", dry_run: bool = False,
                 path: Path | None = None, state_path: Path | None = None, send=None, resolve=None,
                 decisions: dict | None = None, issues: list[str] | None = None) -> dict:
    """Aufgaben verwalten und melden – je Lauf genau eine Zuständigkeit (gemeinsame Historie, getrennte Aufgaben):
    * mode="critical" (täglich): nur P0 (CRITICAL_KINDS), jede neue/geänderte P0-Aufgabe sofort als eigene Mail.
    * mode="weekly" (Samstag): nur P1–P3, ein Digest mit höchstens DIGEST_MAX Aufgaben je Kalenderwoche.
    Da jede Aufgabe nur von einer Laufart gemeldet wird, können parallele Läufe dieselbe Aufgabe nie doppelt melden.
    Unveränderte, bereits gemeldete Aufgaben nie erneut; keine Mail ohne Handlungsbedarf. Nicht mehr erkannte
    Aufgaben: RESOLVED nur, wenn `resolve(task)` die Behebung mit aktuellen, gültigen Daten belegt; fehlen die Daten
    oder sind sie beschädigt/veraltet -> UNKNOWN (Befund und Historie bleiben, keine Erfolgsmeldung). Nur ein
    Versand mit Status "sent" zählt; Fehler (auch fehlende Konfiguration) werden protokolliert und beim nächsten
    Lauf derselben Art erneut versucht. dry_run: nichts senden, nichts speichern."""
    from modules.atomic_io import append_jsonl, read_jsonl
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} nicht in {MODES}")
    now, path, state_path = now or datetime.now(timezone.utc), path or TASKS, state_path or TASKS_STATE
    decisions = _decisions() if decisions is None else decisions
    resolve = resolve or (lambda _t: ("UNKNOWN", "keine Prüfdaten für eine Behebung"))
    if send is None:
        from modules import mailer
        send = mailer.send_mail
    at = now.isoformat(timespec="seconds")
    events = read_jsonl(path) if path.exists() else []
    state, new_events, pending = task_state(events), [], []
    mine = (lambda t: _is_critical(t)) if mode == "critical" else (lambda t: not _is_critical(t))
    current = {t["task_id"]: t for t in tasks if mine(t)}
    for tid, t in current.items():
        prev = state.get(tid)
        ev = ("detected" if prev is None else "reopened" if prev["status"] == "RESOLVED"
              else "changed" if prev.get("fingerprint") != t["fingerprint"]
              else "confirmed" if prev["status"] == "UNKNOWN" else None)
        if ev:
            new_events.append({"event": ev, "task_id": tid, "at": at, "fingerprint": t["fingerprint"],
                               **({"task": t} if ev != "confirmed" else {})})
        decided = str((decisions.get(tid) or {}).get("status", "")).upper()
        already = (prev or {}).get("notified_fingerprint") == t["fingerprint"] and ev in (None, "confirmed")
        if decided in HUMAN_CLOSED or already:
            continue
        reason = {"detected": "NEW", "changed": "CHANGED", "reopened": "REOPENED"}.get(ev, "PENDING")
        pending.append((t, reason))
    for tid, s in sorted(state.items()):
        if tid in current or s["status"] == "RESOLVED" or not mine(s.get("task")):
            continue
        verdict, why = resolve(s.get("task") or {})
        if verdict == "RESOLVED":
            new_events.append({"event": "resolved", "task_id": tid, "at": at, "reason": why})
        elif verdict == "UNKNOWN" and s["status"] != "UNKNOWN":
            new_events.append({"event": "unknown", "task_id": tid, "at": at, "reason": why})
    week = now.isocalendar()[:2]
    sent_this_week = sum(1 for s in state.values() for n in s["notifications"] if n.get("channel") == "digest"
                         and _ts(n.get("at")) and _ts(n["at"]).isocalendar()[:2] == week)
    pending.sort(key=lambda p: (p[0]["priority"], p[0]["task_id"]))
    quota = len(pending) if mode == "critical" else max(0, DIGEST_MAX - sent_this_week)
    mails = ([("critical", [p]) for p in pending] if mode == "critical"
             else ([("digest", pending[:quota])] if pending[:quota] else []))
    report = {"generated": at, "mode": mode, "dry_run": dry_run, "tasks_detected": len(current),
              "input_issues": issues or [], "pending": [p[0]["task_id"] for p in pending],
              "deferred_by_weekly_limit": [p[0]["task_id"] for p in pending[quota:]],
              "status_changes": [{k: e.get(k) for k in ("event", "task_id", "reason")} for e in new_events
                                 if e["event"] in ("resolved", "unknown", "confirmed", "reopened")],
              "mails": [], "mail_status": "nothing_to_send" if not mails else None}
    for channel, items in mails:
        subject, body_html, text = render_mail(items)
        entry = {"channel": channel, "subject": subject, "tasks": [t["task_id"] for t, _r in items]}
        if dry_run:
            report["mails"].append({**entry, "status": "dry_run", "text": text})
            continue
        try:
            res = send(subject, body_html, text) or {}
        except Exception as exc:  # noqa: BLE001 – ein Versandfehler zählt nie als gesendet, der Lauf geht weiter
            res = {"status": "failed", "error": type(exc).__name__}
        status = res.get("status") or "failed"
        report["mails"].append({**entry, "status": status, "error": res.get("error")})
        for t, r in items:
            if status == "sent":
                new_events.append({"event": "notified", "task_id": t["task_id"], "at": at, "channel": channel,
                                   "subject": subject, "fingerprint": t["fingerprint"], "reason": r})
            else:
                new_events.append({"event": "send_failed", "task_id": t["task_id"], "at": at, "channel": channel,
                                   "mail_status": status, "error": res.get("error")})
    if mails:
        st = [m["status"] for m in report["mails"]]
        report["mail_status"] = ("dry_run" if dry_run else "sent" if all(x == "sent" for x in st)
                                 else "not_configured" if "not_configured" in st else "failed")
    if report["mail_status"] in ("not_configured", "failed"):
        log.error("research_director: Verbesserungs-Mail nicht versendet (%s) – bleibt offen und wird beim "
                  "nächsten Lauf erneut versucht", report["mail_status"])
    if dry_run:
        return report
    if new_events:
        append_jsonl(path, new_events, ensure_ascii=False, sort_keys=True, default=str)
    previous = _load(state_path, {}) if state_path.exists() else {}
    last = {k: report[k] for k in ("generated", "mail_status", "pending", "deferred_by_weekly_limit",
                                   "input_issues", "status_changes")} | {
        "mails": [{k: m.get(k) for k in ("channel", "subject", "tasks", "status", "error")} for m in report["mails"]]}
    final = task_state(events + new_events)
    snapshot = {"generated": at, "last_runs": {**(previous.get("last_runs") or {}), mode: last},
                "tasks": {tid: {**(s.get("task") or {}), "status": str((decisions.get(tid) or {}).get("status")
                                                                         or s["status"]).upper(),
                                "decision_note": (decisions.get(tid) or {}).get("note"),
                                "first_detected": s.get("first_detected"), "last_changed": s.get("last_changed"),
                                "resolved_at": s.get("resolved_at") if s["status"] == "RESOLVED" else None,
                                "resolved_reason": s.get("resolved_reason") if s["status"] == "RESOLVED" else None,
                                "unknown_reason": s.get("unknown_reason") if s["status"] == "UNKNOWN" else None,
                                "notifications": s["notifications"], "send_failures": s["send_failures"],
                                "last_send_error": s.get("last_send_error")}
                          for tid, s in sorted(final.items())}}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return report


def run_notify(now: datetime | None = None, dry_run: bool = False, paths: dict | None = None,
               mode: str = "weekly") -> dict:
    """Aufgaben erkennen und melden: mode="weekly" (ml_research.yml, Samstag: P1–P3-Digest, Research Memory) bzw.
    mode="critical" (improvement_notify.yml, täglich: nur P0). Beide nutzen dieselbe Historie."""
    now = now or datetime.now(timezone.utc)
    inputs = load_task_inputs(paths, now)
    tasks, issues = improvement_tasks(inputs, now)
    report = notify_tasks(tasks, now, mode=mode, dry_run=dry_run, issues=issues,
                          resolve=lambda t: resolution(t, inputs, now))
    if not dry_run and mode == "weekly":
        from modules import research_memory as rm
        report["research_memory_new"] = rm.sync()
    return report


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
    ap.add_argument("--notify", action="store_true",
                    help="technische Verbesserungsaufgaben erkennen und melden (Wochen-Digest, kritisch sofort)")
    ap.add_argument("--dry-run", action="store_true", help="mit --notify: nur Vorschau, nichts senden oder speichern")
    ap.add_argument("--critical", action="store_true",
                    help="mit --notify: täglicher Lauf, nur kritische P0-Aufgaben (sonst Wochen-Digest P1–P3)")
    args = ap.parse_args(argv)
    if args.notify:
        rep = run_notify(dry_run=args.dry_run, mode="critical" if args.critical else "weekly")
        print(json.dumps({k: rep.get(k) for k in ("mode", "mail_status", "tasks_detected", "pending",
                                                    "deferred_by_weekly_limit", "status_changes", "input_issues")},
                         ensure_ascii=False))
        for m in rep["mails"]:
            print(f"--- {m['status']}: {m['subject']} ({', '.join(m['tasks'])})")
            if args.dry_run:
                print(m["text"])
        return 2 if rep["mail_status"] in ("not_configured", "failed") else 0
    if args.options:
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
