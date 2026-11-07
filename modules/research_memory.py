"""modules/research_memory.py – Research Memory, Ähnlichkeitssuche, Meta-Learning über
Forschungsrichtungen und der (optionale) Research-Memory-Reviewer.

Keine neue Datenhaltung neben den bestehenden: das Gedächtnis SYNCHRONISIERT append-only
aus den vorhandenen Speichern
  * outputs/research/hypothesis_db.json            (Research-Lab: Director, Fabrik, Config, Alt-Data)
  * outputs/research/alt_data_validation.json      (Quellen-Bewertungen KEEP/MODIFY/REJECT)
  * outputs/intelligence/contract_proposals.json   (Abstinenz-Vorschläge Walk-Forward)
  * outputs/intelligence/promotion_state.json      (Produktions-Hypothesen inkl. Forward-Zustand)
  * outputs/research/factory_results.json          (Robustheits-Batterie der Fabrik)
  * outputs/research/factory_plan.json             (nur DATA_GAP-Ideen: fehlende Daten + freie Quellen)
  * outputs/research/option_learner/status_<h>d.json (Varianten-Experimente des Options-Lernmodells, prospektiv)
  * outputs/research/option_learner/improvement.json (Vorschläge, die eine menschliche Freigabe brauchen)
  * outputs/research/improvement_tasks.json         (technische Verbesserungsaufgaben mit Status und Versand)
  * outputs/research/error_research.json           (fehlergetriebene Forschung: Hypothesen aus Forward-
                                                    Prognosefehlern, durch bekannte Faktoren erklärte Muster,
                                                    Datenlücken – research_director --errors)
nach outputs/research/research_memory.jsonl. Jeder Eintrag: Spezifikation, Daten, Tests,
Resultat, Regime, Gründe – und ob die Evidenz PROSPEKTIV (Forward) ist.

Reviewer: sieht AUSSCHLIESSLICH prospektiv validierte Einträge (FORWARD_VALIDATED oder
Produktionszustände). Er wird nach der unabhängigen Kandidatenanalyse aufgerufen und
schreibt nur Beobachtungen (Ledger) – er fließt nicht in die Analyse ein und hat keinen
Produktionseinfluss (der läuft ausschließlich über den ProductionIntelligenceAdapter).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from modules.atomic_io import append_jsonl

MEMORY = Path("outputs/research/research_memory.jsonl")
DIRECTIONS = Path("outputs/research/research_directions.json")
SOURCES = {
    "hypothesis_db": Path("outputs/research/hypothesis_db.json"),
    "alt_validation": Path("outputs/research/alt_data_validation.json"),
    "abstention_proposals": Path("outputs/intelligence/contract_proposals.json"),
    "promotion_state": Path("outputs/intelligence/promotion_state.json"),
    "factory_results": Path("outputs/research/factory_results.json"),
    "factory_plan": Path("outputs/research/factory_plan.json"),
    "surprise_study": Path("outputs/research/surprise_study.json"),
    "commodity_evaluation": Path("outputs/research/commodity_evaluation.json"),
    "option_status_20d": Path("outputs/research/option_learner/status_20d.json"),
    "option_status_5d": Path("outputs/research/option_learner/status_5d.json"),
    "option_improvement": Path("outputs/research/option_learner/improvement.json"),
    "improvement_tasks": Path("outputs/research/improvement_tasks.json"),
    "alpha_option_economics": Path("outputs/research/autonomous_alpha/option_economics.json"),
    "alpha_option_validation": Path("outputs/research/autonomous_alpha/option_validation.json"),
    "error_research": Path("outputs/research/error_research.json"),
}
# Fehlergetriebene Forschung (research_director.error_research) -> Memory-Status. PENDING_FORWARD ist weder Erfolg
# noch abgeschlossener Test; Erfolg nur nach prospektiver Bestätigung (FORWARD_VALIDATED).
ERROR_RESEARCH_STATUS = {"PENDING_FORWARD": "PENDING_FORWARD", "REJECTED_WF": "REJECTED",
                         "FORWARD_REJECTED": "REJECTED", "FORWARD_NOT_CONFIRMED": "REJECTED",
                         "FORWARD_CONFIRMED": "FORWARD_VALIDATED", "INCONCLUSIVE": "INCONCLUSIVE",
                         "NO_BUDGET": "INCONCLUSIVE", "INVALID_MODIFIED": "REJECTED"}
# Alpha-to-Options: Phase-B-Labels sind explorativ (kein abgeschlossener Test), Phase-C-Ergebnisse sind vorab
# registrierte prospektive Tests. "Qualifiziert" ist ein Forschungserfolg, nie ein Produktionsstatus.
ALPHA_ECON_STATUS = {"CANDIDATE_FOR_PROSPECTIVE_VALIDATION": "PENDING_FORWARD", "NEED_MORE_DATA": "DATA_GAP",
                     "COMPARISON_DATA_MISSING": "DATA_GAP", "NO_ECONOMIC_EDGE_AFTER_COSTS": "INCONCLUSIVE",
                     "NOT_BETTER_THAN_RANDOM_CONTROL": "INCONCLUSIVE", "ADVANTAGE_NOT_ESTABLISHED": "INCONCLUSIVE"}
ALPHA_PLAN_STATUS = {"FALSIFIED": "REJECTED", "NOT_CONFIRMED": "REJECTED", "NO_BUDGET": "INCONCLUSIVE",
                     "PLAN_INVALID": "INCONCLUSIVE"}
# Varianten-Status des Options-Lernmodells -> Memory-Status (laufend = weder Erfolg noch Test)
# Erfolg nur bei relativem UND wirtschaftlichem Vorteil (PROPOSE_/CONFIRMED_ADOPTION); "nur relativ besser" ist
# kein Erfolg. Laufende Varianten sind weder Erfolg noch abgeschlossener Test.
OPTION_VARIANT_STATUS = {"PROPOSE_ADOPTION": "FORWARD_VALIDATED", "CONFIRMED_ADOPTION": "FORWARD_VALIDATED",
                         "BETTER_THAN_BASE_NOT_PROFITABLE": "INCONCLUSIVE", "FALSIFIED_WORSE": "REJECTED",
                         "FALSIFIED_NO_EFFECT": "REJECTED", "BASE_CHANGED": "EXPIRED",
                         "BETTER_THAN_BASE_ONLY": "PENDING_FORWARD", "RUNNING": "PENDING_FORWARD",
                         "RESEARCH_ONLY": "PENDING_FORWARD"}
PROSPECTIVE_STATES = ("FORWARD_VALIDATED", "GUARDED_PRODUCTION", "LIMITED_PRODUCTION", "FULL_PRODUCTION")
# Kanonischer Status eines research_lab-Datensatzes – EINZIGE Implementierung (research_lab.canonical_status delegiert
# hierher). Audit 2026-11-07: Leser (research_memory, meta_cognition, hypothesis_factory, inquiry, hypothesis_contract,
# research_director) lasen das GESPEICHERTE Feld canonical_status. In outputs/research/hypothesis_db.json (Stand 03.11.,
# vor #165) stehen 15 nur per Textähnlichkeit gesperrte, nie getestete Hypothesen noch als REJECTED; research.yml
# schreibt die Datei erst am 02.12. neu. Bis dahin zählten sie überall als widerlegt.
HYPOTHESIS_CANONICAL = {"accepted": "EXPLORATORY", "historical_exploratory_pending_forward": "EXPLORATORY", "rejected": "REJECTED", "rejected_leakage_or_invalid": "REJECTED",
                        "rejected_locked": "REJECTED", "duplicate": "REJECTED",
                        "blocked_similar_to_rejected": "NOT_TESTED", "invalid_modified": "REJECTED",
                        "not_significant_after_fdr": "INCONCLUSIVE", "passed_pre_fdr": "INCONCLUSIVE",
                        "passed_pending_locked": "RETEST_LATER", "insufficient_coverage": "RETEST_LATER",
                        "blocked_data": "RETEST_LATER"}


def hypothesis_status(rec: dict | None) -> str:
    """Nur per Textähnlichkeit gesperrt = nie getestet (NOT_TESTED), auch wenn ältere Läufe REJECTED gespeichert haben;
    sonst der gespeicherte kanonische Status bzw. die Zuordnung des Roh-Status."""
    rec = rec or {}
    if rec.get("status") == "blocked_similar_to_rejected":
        return "NOT_TESTED"
    # A legacy cached ACCEPTED/ROBUST/VALIDATED is NOT independent proof.
    # The historical locked interval was reused; pre-existing JSON files may
    # still carry old canonical labels until the next monthly lab run.
    if rec.get("status") in ("accepted", "historical_exploratory_pending_forward"):
        return "EXPLORATORY"
    historical = rec.get("locked_holdout_status") == "CONTAMINATED" or rec.get("evidence_class") in (
        "EXPLORATORY_HISTORICAL", "HISTORICAL_RESEARCH", "HISTORICAL_ROBUSTNESS_BATTERY")
    if historical and str(rec.get("canonical_status") or rec.get("status") or "").upper() in (
        "ACCEPTED", "ROBUST", "VALIDATED", "CONFIRMED"):
        return "EXPLORATORY"
    if rec.get("canonical_status"):
        return rec["canonical_status"]
    if rec.get("status") == "prior_result":
        return "REJECTED"
    return HYPOTHESIS_CANONICAL.get(rec.get("status"), "INCONCLUSIVE")


TESTED = ("EXPLORATORY", "REJECTED", "INCONCLUSIVE", "RETEST_LATER", "NOT_ROBUST",
          "PENDING_FORWARD", *PROSPECTIVE_STATES, "DEMOTED", "EXPIRED")
# Research success has one evidence meaning: an independently validated forward result.
SUCCESS = PROSPECTIVE_STATES
_NAME = re.compile(r"[a-z_][a-z0-9_]*")


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def tokens_of(spec: dict) -> set[str]:
    """Inhalts-Tokens: Signal-Merkmale, Exposure, Richtung, Domäne/Familie."""
    sig = str(spec.get("signal") or spec.get("signal_definition") or "").lower()
    names = {n for n in _NAME.findall(sig) if n not in ("rank", "sign", "abs", "step", "min", "max")}
    out = {f"f:{n}" for n in names}
    for k in ("family", "domain", "exposure_sector", "population_sector", "commodity"):
        if spec.get(k):
            out.add(f"{k}:{str(spec[k]).lower()}")
    if spec.get("direction") is not None:
        out.add(f"dir:{int(spec['direction'])}")
    return out


def similarity(a: dict, b: dict) -> float:
    ta, tb = tokens_of(a), tokens_of(b)
    if not ta or not tb:
        return 0.0
    j = len(ta & tb) / len(ta | tb)
    sa = re.sub(r"\s+", "", str(a.get("signal") or a.get("signal_definition") or ""))
    sb = re.sub(r"\s+", "", str(b.get("signal") or b.get("signal_definition") or ""))
    if sa and sa == sb:
        j = max(j, 0.95)
    return round(j, 3)


def _entry(kind: str, hid: str, status: str, spec: dict, **kw) -> dict:
    e = {"kind": kind, "hypothesis_id": hid, "status": status, "spec": spec,
         "prospective": status in PROSPECTIVE_STATES, **kw}
    e["entry_key"] = hashlib.sha256(json.dumps({k: e[k] for k in ("kind", "hypothesis_id", "status")} |
                                               {"ev": kw.get("evidence")}, sort_keys=True, default=str)
                                    .encode()).hexdigest()[:16]
    return e


def collect(sources: dict | None = None) -> list[dict]:
    """Einträge aus allen vorhandenen Speichern (ohne Schreiben)."""
    src = {k: _load(v) for k, v in (sources or SOURCES).items()}
    out = []
    db = src.get("hypothesis_db") or {}
    for hid, r in (db.get("hypotheses") or {}).items():
        wf = (r.get("walk_forward") or {}).get("base") or {}
        out.append(_entry("research", hid, hypothesis_status(r),
                          {"signal": r.get("signal"), "direction": r.get("direction"), "title": r.get("title"),
                           "family": r.get("family"), "domain": r.get("domain"),
                           "exposure_sector": r.get("exposure_sector")},
                          source=r.get("source"), evidence={"mean": wf.get("mean"), "t": wf.get("t_months"),
                                                           "years_positive": wf.get("years_positive_share")},
                          regimes=r.get("regimes"), reasons=r.get("reasons"), data_kind="historical_walk_forward"))
    for sid, r in ((src.get("alt_validation") or {}).get("sources") or {}).items():
        out.append(_entry("alt_source", sid, {"KEEP": "EXPLORATORY", "MODIFY": "INCONCLUSIVE"}.get(r.get("verdict"), "REJECTED"),
                          {"signal": " ".join(r.get("selected_features") or []), "domain": sid},
                          source="alt_data", evidence={"verdict": r.get("verdict")}, reasons=[r.get("verdict_reason")],
                          data_kind="historical_walk_forward_ablation"))
    ap = src.get("abstention_proposals") or {}
    for p in ap.get("rejected") or []:
        out.append(_entry("abstention", p.get("rule"), "REJECTED", {"signal": p.get("rule"), "direction": -1,
                                                                    "domain": "champion_trades"},
                          source="abstention_proposals", evidence={"test_delta": p.get("test_delta")},
                          reasons=[p.get("reason")], data_kind="historical_walk_forward"))
    for p in ap.get("proposals") or []:
        wf = p.get("walk_forward") or {}
        out.append(_entry("abstention", wf.get("rule"), "EXPLORATORY", {"signal": wf.get("rule"), "direction": -1,
                                                                     "domain": "champion_trades"},
                          source="abstention_proposals", evidence=wf, data_kind="historical_walk_forward"))
    for k, h in ((src.get("promotion_state") or {}).get("hypotheses") or {}).items():
        ev = h.get("evidence") or {}
        st = h.get("state") or "IDEA"
        if st not in PROSPECTIVE_STATES and st not in ("DEMOTED", "EXPIRED", "REJECTED"):
            st = "PENDING_FORWARD"           # registriert, aber (noch) nicht forward-validiert: weder Erfolg noch Test
        out.append(_entry("promotion", k, st, {"signal": h.get("description"),
                                                                    "domain": "champion_trades"},
                          source=h.get("source_type"), evidence={"n": ev.get("n_observations"),
                                                                 "delta": ev.get("delta_expectancy"), "ci": ev.get("ci")},
                          reasons=h.get("reasons"), data_kind="prospective_forward", state=h.get("state"),
                          sector_scope=h.get("sector_scope"), regime_scope=h.get("regime_scope")))
    for h in (src.get("factory_plan") or {}).get("ideas") or []:
        if h.get("plan_status") == "DATA_GAP":
            rd = h.get("readiness") or {}
            out.append(_entry("factory", h.get("id"), "DATA_GAP",
                              {k: h.get(k) for k in ("signal", "direction", "family", "domain", "exposure_sector",
                                                     "commodity")},
                              source="factory", evidence=None, reasons=[rd.get("reason")],
                              free_sources=rd.get("free_sources") or [], data_kind="none"))
    for name, r in ((src.get("surprise_study") or {}).get("hypotheses") or {}).items():
        d = r.get("decision") or {}
        b = ((r.get("h20") or {}).get("base_cost") or {})
        out.append(_entry("research", f"SURPRISE-{name}",
                          {"KEEP": "ACCEPTED", "MODIFY": "INCONCLUSIVE"}.get(d.get("verdict"), "REJECTED"),
                          {"signal": name, "title": r.get("description"), "family": "surprise_engine",
                           "domain": "earnings_expectations"},
                          source="surprise_engine", evidence={"mean": b.get("mean"), "t": b.get("t_months"),
                                                              "years_positive": b.get("years_positive_share"),
                                                              "placebo_p": (r.get("h20") or {}).get("placebo_p")},
                          reasons=d.get("reasons"), data_kind="historical_walk_forward"))
    for cid, c in ((src.get("surprise_study") or {}).get("counter_hypotheses") or {}).items():
        if c.get("status") == "LOCKED" or not c.get("decision"):
            continue                       # prospektiver Holdout noch gesperrt -> kein Memory-Eintrag
        d = c.get("decision") or {}
        b = ((c.get("h20") or {}).get("base_cost") or {})
        out.append(_entry("research", f"SURPRISE-{cid}",
                          {"KEEP": "ACCEPTED", "MODIFY": "INCONCLUSIVE"}.get(d.get("verdict"), "REJECTED"),
                          {"signal": cid, "title": c.get("description"), "family": "surprise_engine",
                           "domain": "earnings_expectations", "idea_source": "counter_hypothesis"},
                          source="surprise_engine", evidence={"mean": b.get("mean"), "t": b.get("t_months"),
                                                              "spec_hash": c.get("spec_hash")},
                          reasons=d.get("reasons"), data_kind="prospective_locked_holdout"))
    for hid, r in ((src.get("factory_results") or {}).get("results") or {}).items():
        out.append(_entry("factory", hid, r.get("status"), r.get("spec") or {}, source="factory",
                          evidence=r.get("tests"), reasons=r.get("reasons"), data_kind=r.get("data_kind")))
    out += commodity_entries(src.get("commodity_evaluation"), src.get("factory_results"))
    out += option_entries(*(src.get(k) for k in ("option_status_20d", "option_status_5d", "option_improvement")))
    out += task_entries(src.get("improvement_tasks"))
    out += alpha_option_entries(src.get("alpha_option_economics"), src.get("alpha_option_validation"))
    out += error_entries(src.get("error_research"))
    return out


def error_entries(rep: dict | None) -> list[dict]:
    """Fehlergetriebene Forschung: jede registrierte Hypothese (Familie error:<Quellenfamilie>, eigener Schlüssel
    error_key -> nie erneut registriert), jedes durch Größe erklärte Muster (NOT_ROBUST: senkt den Familien-Prior),
    jede Datenlücke und jede nachträgliche Datenänderung (Audit). Evidenz enthält nur EINGEFRORENE Stufen (finale
    Forward-Prüfung aus dem Journal), keinen Zwischenstand – ein Eintrag je abgeschlossenem Ergebnis."""
    rep = rep or {}
    out = []
    for h in rep.get("hypotheses") or []:
        sp, wf = h.get("spec") or {}, h.get("walk_forward") or {}
        final = next((v for v in (h.get("forward") or {}).get("looks") or [] if v.get("final")), None)
        status = ERROR_RESEARCH_STATUS.get(h.get("status"))
        if not status:
            continue
        out.append(_entry("error_research", h["id"], status,
                          {"signal": sp.get("signal"), "direction": sp.get("direction"), "family": sp.get("family"),
                           "domain": "forecast_error", "idea_source": "forecast_error", "error_key": h.get("key"),
                           "horizon": sp.get("horizon"), "priority": (h.get("priority") or {}).get("priority"),
                           "spec_hash": h.get("spec_hash")},
                          source="error_research",
                          evidence={"walk_forward": wf.get("status"), "wf_mean": wf.get("mean"),
                                    "forward_outcome": (final or {}).get("outcome"),
                                    "forward_look": (final or {}).get("look"),
                                    "forward_mean": ((final or {}).get("stats") or {}).get("mean"),
                                    "forward_p": ((final or {}).get("stats") or {}).get("p_one_sided"),
                                    "alpha": (final or {}).get("alpha"),
                                    "data_sha256": ((final or {}).get("data_ref") or {}).get("inputs_sha256"),
                                    "slot": (h.get("forward_plan") or {}).get("slot"),
                                    "option_check": (h.get("option_check") or {}).get("status")},
                          reasons=[h.get("reason")] if h.get("reason") and final else [],
                          data_kind="prospective_forward_frozen" if final else "ledger_walk_forward_seen_data"))
        for a in h.get("audits") or []:
            out.append(_entry("error_audit", f"{h['id']}:look{a['look']}:{a['current_sha256'][:12]}", "DATA_GAP",
                              {"signal": sp.get("signal"), "domain": "forecast_error"}, source="error_research",
                              evidence={"frozen_outcome": a.get("frozen_outcome"),
                                        "reassessment": a.get("reassessment")},
                              reasons=["Daten nach eingefrorener Prüfung geändert – Entscheidung bleibt, "
                                       "Neubewertung getrennt"], data_kind="audit"))
    for hz, v in sorted((rep.get("horizons") or {}).items()):
        for c in v.get("candidates") or []:
            if c.get("verdict") != "EXPLAINED_BY_KNOWN_FACTOR":
                continue
            d = c.get("discovery") or {}
            out.append(_entry("error_pattern", f"ERRPAT-{c['id']}-{hz}d", "NOT_ROBUST",
                              {"signal": c["id"], "family": "error:" + "+".join(c.get("families") or []),
                               "domain": "forecast_error", "idea_source": "forecast_error", "horizon": int(hz)},
                              source="error_research",
                              evidence={"residual_ic": d.get("mean"), "t": d.get("t"),
                                        "control_size_t": (c.get("control_size") or {}).get("t")},
                              reasons=c.get("reasons"), data_kind="ledger_discovery_seen_data"))
    for g in rep.get("data_gaps") or []:
        out.append(_entry("error_gap", g["gap_id"], "DATA_GAP",
                          {"signal": g.get("kind"), "domain": "forecast_error", "dimensions": g.get("dimensions")},
                          source="error_research", evidence={"finding": g.get("finding")},
                          reasons=[g.get("decision")],
                          free_sources=[s["source"] if isinstance(s, dict) else s for s in g.get("sources") or []],
                          data_kind="none"))
    return out


def alpha_option_entries(econ: dict | None, validation: dict | None) -> list[dict]:
    """Alpha-to-Options: Phase-B-Forschungsentscheidung je Hypothese und Struktur sowie Phase-C-Prüfpläne."""
    out = []
    for hid, per in sorted(((econ or {}).get("hypotheses") or {}).items()):
        for st, e in sorted((per or {}).items()):
            label = (e.get("research_decision") or {}).get("label")
            if label not in ALPHA_ECON_STATUS:
                continue
            m = (e.get("metrics") or {}).get("ADVERSE") or {}
            out.append(_entry("alpha_option_research", f"{hid}:{st}", ALPHA_ECON_STATUS[label],
                              {"signal": hid, "family": f"alpha_option_{st.lower()}", "domain": "options_net"},
                              source="option_economics", evidence={"label": label, "n": m.get("n"),
                                                                   "net_ev": m.get("net_ev_return")},
                              reasons=(e.get("research_decision") or {}).get("reasons"),
                              data_kind="prospective_quote_verified_paper"))
    for p in (validation or {}).get("plans") or []:
        if p.get("classification") == "QUALIFIED_FOR_HUMAN_REVIEW" and p.get("status") == "CONFIRMED":
            status = "FORWARD_VALIDATED"
        else:
            status = ALPHA_PLAN_STATUS.get(p.get("status"), "PENDING_FORWARD")
        b = p.get("binding") or {}
        out.append(_entry("alpha_option_validation", str(p.get("plan_id")), status,
                          {"signal": p.get("hypothesis_id"), "family": f"alpha_option_{str(p.get('structure')).lower()}",
                           "domain": "options_net"},
                          source="option_validation", evidence={"status": p.get("status"), "look": b.get("look"),
                                                                "report": b.get("report_id"),
                                                                "classification": p.get("classification")},
                          reasons=(p.get("qualification") or {}).get("failed"),
                          data_kind="prospective_preregistered_paper"))
    return out


def task_entries(state: dict | None) -> list[dict]:
    """Technische Verbesserungsaufgaben (research_director --notify): Befund, Priorität, Status und Versand.
    Ein neuer Eintrag entsteht bei neuem Status oder wesentlich geändertem Befund (Fingerabdruck)."""
    out = []
    for tid, t in sorted(((state or {}).get("tasks") or {}).items()):
        if not isinstance(t, dict) or not t.get("kind"):
            continue
        out.append(_entry("dev_task", tid, str(t.get("status") or "OPEN"),
                          {"signal": t.get("subject"), "family": f"dev_task_{t.get('kind')}",
                           "domain": t.get("category")},
                          source="research_director", evidence={"fingerprint": t.get("fingerprint"),
                                                                "priority": t.get("priority")},
                          reasons=[t.get("finding")], title=t.get("short"),
                          notifications=len(t.get("notifications") or []), data_kind="research_outputs"))
    return out


def option_entries(*sources) -> list[dict]:
    """Options-Lernmodell: Varianten-Experimente (prospektiv auf identischen Kandidaten, Bonferroni) mit Status,
    Evidenz und Grund; dazu Vorschläge, die eine menschliche Freigabe brauchen (Daten, Erfassung, Stichprobe)."""
    out = []
    for src in sources:
        for vid, v in (((src or {}).get("variants") or {}).get("variants") or {}).items():
            ((method, value),) = (v.get("overrides") or {"?": None}).items()
            lr = (v.get("vs_base") or {}).get("learning_minus_reference") or {}
            ev = v.get("current_evidence") or {}
            status = ("REVIEW_REQUIRED" if ev.get("contradicts_decision")      # korrigierte Daten widersprechen
                      else OPTION_VARIANT_STATUS.get(v.get("status"), "INCONCLUSIVE"))
            out.append(_entry("option_experiment", vid, status,
                              {"signal": f"{method}={value}", "family": "option_learner_variant",
                               "domain": "options_net", "experiment_id": v.get("experiment_id")},
                              source="option_learner",
                              evidence={"relative": v.get("relative_verdict"), "economic": v.get("economic_verdict"),
                                        "look": v.get("look"), "mean": lr.get("mean"), "ci": lr.get("ci95"),
                                        "coverage": (v.get("comparison") or {}).get("coverage"),
                                        "days": (v.get("vs_base") or {}).get("days"),
                                        "alpha_slot": v.get("alpha_slot"), "alpha_look": v.get("alpha_look"),
                                        "look_report": (v.get("look_report") or {}).get("report_id"),
                                        "proposal": (v.get("change_proposal") or {}).get("state"),
                                        "evidence_status": ev.get("status")},
                              reasons=[v.get("relative_verdict"), v.get("economic_verdict")],
                              gap_ids=v.get("gap_ids"), proposal_id=v.get("proposal_id"),
                              data_kind="prospective_forward_options"))
        for p in (src or {}).get("proposals") or []:
            if p.get("approval_required"):
                out.append(_entry("option_proposal", p["proposal_id"], "APPROVAL_REQUIRED",
                                  {"signal": json.dumps(p.get("change"), sort_keys=True, default=str),
                                   "family": f"option_{p.get('kind')}", "domain": "options_net"},
                                  source="research_director", evidence={"priority": p.get("priority")},
                                  reasons=p.get("gap_ids"), cost=p.get("cost"), data_kind="none"))
    return out


def commodity_entries(evaluation: dict | None, factory_results: dict | None = None) -> list[dict]:
    """Commodity-Gedächtnis: domain, commodity, equity_population, mechanism, features, result, n, OOS,
    forward, failure_reason, fingerprint – je getesteter Commodity-Hypothese und je Ablation (BASE vs.
    BASE+COMMODITY). Fingerprint = Inhalts-Tokens -> Beinahe-Duplikate werden über similarity blockiert."""
    out = []
    for hid, r in ((factory_results or {}).get("results") or {}).items():
        sp = r.get("spec") or {}
        if not sp.get("commodity"):
            continue
        t = r.get("tests") or {}
        out.append(_entry("commodity", hid, r.get("status"), sp, source="factory",
                          domain=sp.get("domain"), commodity=sp.get("commodity"),
                          equity_population=sp.get("equity_population"), mechanism=sp.get("mechanism"),
                          features=sp.get("domain_features"), result=r.get("status"), n=t.get("n_cohorts"),
                          oos=t.get("walk_forward") or t.get("oos"), forward=None,
                          failure_reason="; ".join(map(str, r.get("reasons") or [])) or None,
                          fingerprint=sorted(tokens_of(sp)), data_kind=r.get("data_kind")))
    for g, r in ((evaluation or {}).get("groups") or {}).items():
        st = {"KEEP": "ACCEPTED", "MODIFY": "INCONCLUSIVE"}.get(r.get("verdict"), "REJECTED")
        sp = {"signal": " ".join(r.get("selected_features") or []), "domain": f"commodity_{g}",
              "family": f"commodity_ablation_{g}"}
        out.append(_entry("commodity", f"CMD-ABLATION-{g}", st, sp, source="commodity_evaluation",
                          domain=f"commodity_{g}", commodity=r.get("commodities"),
                          equity_population=r.get("population"), mechanism="BASE vs. BASE+COMMODITY (Ablation)",
                          features=r.get("selected_features"), result=r.get("verdict"), n=r.get("n_rows"),
                          oos=r.get("oos_summary"), forward=r.get("forward"),
                          failure_reason=r.get("verdict_reason") if st != "ACCEPTED" else None,
                          fingerprint=sorted(tokens_of(sp)), data_kind="historical_walk_forward_ablation"))
    return out


def sync(path: Path | None = None, sources: dict | None = None, now: str | None = None) -> int:
    """Neue/geänderte Einträge append-only anhängen. -> Anzahl neuer Einträge."""
    path = path or MEMORY
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen = {e["entry_key"] for e in load(path)}
    new = [e for e in collect(sources) if e["entry_key"] not in seen]
    if new:
        append_jsonl(path, [{**e, "recorded_at": now} for e in new], ensure_ascii=False, sort_keys=True, default=str)
    return len(new)


def load(path: Path | None = None) -> list[dict]:
    path = path or MEMORY
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def latest(entries: list[dict]) -> dict[str, dict]:
    """Letzter Stand je (kind, hypothesis_id)."""
    out = {}
    for e in entries:
        out[f"{e['kind']}:{e['hypothesis_id']}"] = e
    return out


def search(spec: dict, entries: list[dict], threshold: float) -> list[tuple[str, str, float]]:
    """Ähnliche, bereits behandelte Hypothesen (jeder Status) -> [(key, status, sim)] absteigend."""
    hits = []
    for k, e in latest(entries).items():
        s = similarity(spec, e.get("spec") or {})
        if s >= threshold:
            hits.append((k, e.get("status"), s))
    return sorted(hits, key=lambda x: -x[2])


# ── Meta-Learning über Forschungsrichtungen ─────────────────────────────────
def direction_of(e: dict) -> str:
    sp = e.get("spec") or {}
    if sp.get("family"):
        return f"family:{sp['family']}"
    if e.get("kind") == "research":
        return f"research:{e.get('source') or 'config'}"
    return f"{e.get('kind')}:{sp.get('domain') or e.get('source') or 'n/a'}"


def directions(entries: list[dict], prior: tuple[float, float] = (1.0, 4.0)) -> dict:
    """Je Richtung: getestet, Erfolge (robust/akzeptiert), prospektiv bestätigt, verschwendete
    Kapazität; Beta-Posterior der Erfolgsrate + Unsicherheit (für EIG)."""
    a0, b0 = prior
    agg: dict[str, dict] = {}
    for e in latest(entries).values():
        d = agg.setdefault(direction_of(e), {"tested": 0, "success": 0, "prospective": 0, "data_gap": 0,
                                             "rejected": 0})
        st = e.get("status")
        if st == "DATA_GAP":
            d["data_gap"] += 1
            continue
        if st in TESTED:
            d["tested"] += 1
            d["success"] += st in SUCCESS
            d["prospective"] += st in PROSPECTIVE_STATES
            d["rejected"] += st in ("REJECTED", "NOT_ROBUST", "DEMOTED", "EXPIRED")
    # Zweite Ebene: Ideentyp (cross_domain, drift, source_condition, …) – Prior für neue Familien
    for e in latest(entries).values():
        src = (e.get("spec") or {}).get("idea_source")
        st = e.get("status")
        if not src or st == "DATA_GAP" or st not in TESTED:
            continue
        d = agg.setdefault(f"idea_source:{src}", {"tested": 0, "success": 0, "prospective": 0, "data_gap": 0,
                                                  "rejected": 0})
        d["tested"] += 1
        d["success"] += st in SUCCESS
        d["prospective"] += st in PROSPECTIVE_STATES
        d["rejected"] += st in ("REJECTED", "NOT_ROBUST", "DEMOTED", "EXPIRED")
    out = {}
    for k, d in agg.items():
        a, b = a0 + d["success"] + d["prospective"], b0 + d["tested"] - d["success"]
        mean = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        out[k] = {**d, "posterior_success": round(mean, 4), "posterior_sd": round(sd, 4),
                  "wasted_capacity": d["rejected"], "assessment": (
                      "liefert Forward-Mehrwert" if d["prospective"] else
                      "historisch vielversprechend" if d["success"] else
                      "verschwendet Kapazität" if d["tested"] >= 3 else "zu wenig getestet")}
    return out


def write_directions(entries: list[dict], path: Path | None = None, prior=(1.0, 4.0)) -> dict:
    path = path or DIRECTIONS
    d = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prior": list(prior),
         "directions": directions(entries, prior)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    return d


# ── Research-Memory-Reviewer (nach der unabhängigen Analyse) ────────────────
def review_candidate(candidate: dict, entries: list[dict] | None = None) -> list[dict]:
    """Nur PROSPEKTIV validierte Erkenntnisse; prüft, ob Scope (Sektor/Regime) auf den Kandidaten
    zutrifft. Gibt Beobachtungen zurück – keine Entscheidung, kein Score."""
    entries = entries if entries is not None else load()
    sector, regime = candidate.get("sector"), candidate.get("regime")
    out = []
    for e in latest(entries).values():
        if not e.get("prospective") or e.get("status") not in PROSPECTIVE_STATES:
            continue
        ss, rs = e.get("sector_scope") or ["all"], e.get("regime_scope") or ["all"]
        if ("all" in ss or sector in ss) and ("all" in rs or regime in rs):
            out.append({"hypothesis": e["hypothesis_id"], "status": e["status"], "evidence": e.get("evidence"),
                        "applies_because": f"Scope Sektor {ss} / Regime {rs}"})
    return out


# ── Lernt die Forschung selbst? ─────────────────────────────────────────────
def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def learning_curve(entries: list[dict], min_n: int = 10) -> dict:
    """Wird die Forschung besser? (1) Erfolgsquote getesteter Hypothesen je Quartal (Zeitpunkt der
    Erfassung), (2) Kalibrierung der Priorisierung: Rangkorrelation zwischen vergebener Priorität und
    Erfolg (> 0 = das System wählt die aussichtsreicheren Ideen zuerst). Nur gemessene Werte."""
    tested = [e for e in latest(entries).values() if e.get("status") in TESTED]
    by_q: dict[str, list[int]] = {}
    for e in tested:
        ts = str(e.get("recorded_at") or "")[:7]
        if len(ts) == 7:
            q = f"{ts[:4]}-Q{(int(ts[5:7]) - 1) // 3 + 1}"
            by_q.setdefault(q, []).append(int(e.get("status") in SUCCESS))
    curve = {q: {"tested": len(v), "success_rate": round(sum(v) / len(v), 3)} for q, v in sorted(by_q.items())}
    pr = [(float((e.get("spec") or {})["priority"]), int(e.get("status") in SUCCESS)) for e in tested
          if isinstance((e.get("spec") or {}).get("priority"), (int, float))]
    calib = {"n": len(pr), "spearman": None}
    if len(pr) >= min_n and len({y for _, y in pr}) == 2:
        rx, ry = _rank([x for x, _ in pr]), _rank([float(y) for _, y in pr])
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        vx = sum((a - mx) ** 2 for a in rx) ** 0.5
        vy = sum((b - my) ** 2 for b in ry) ** 0.5
        calib["spearman"] = round(cov / (vx * vy), 3) if vx and vy else None
    return {"by_quarter": curve, "priority_calibration": calib,
            "by_idea_source": {k[12:]: v for k, v in directions(entries).items() if k.startswith("idea_source:")}}
