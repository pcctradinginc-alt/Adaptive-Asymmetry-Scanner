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

MEMORY = Path("outputs/research/research_memory.jsonl")
DIRECTIONS = Path("outputs/research/research_directions.json")
SOURCES = {
    "hypothesis_db": Path("outputs/research/hypothesis_db.json"),
    "alt_validation": Path("outputs/research/alt_data_validation.json"),
    "abstention_proposals": Path("outputs/intelligence/contract_proposals.json"),
    "promotion_state": Path("outputs/intelligence/promotion_state.json"),
    "factory_results": Path("outputs/research/factory_results.json"),
    "factory_plan": Path("outputs/research/factory_plan.json"),
}
PROSPECTIVE_STATES = ("FORWARD_VALIDATED", "GUARDED_PRODUCTION", "LIMITED_PRODUCTION", "FULL_PRODUCTION")
TESTED = ("ACCEPTED", "REJECTED", "INCONCLUSIVE", "RETEST_LATER", "ROBUST", "NOT_ROBUST", "PROSPECTIVE_CHALLENGER",
          *PROSPECTIVE_STATES, "DEMOTED", "EXPIRED")
SUCCESS = ("ACCEPTED", "ROBUST", "PROSPECTIVE_CHALLENGER", *PROSPECTIVE_STATES)
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
    for k in ("family", "domain", "exposure_sector", "population_sector"):
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
        out.append(_entry("research", hid, r.get("canonical_status") or "INCONCLUSIVE",
                          {"signal": r.get("signal"), "direction": r.get("direction"), "title": r.get("title"),
                           "family": r.get("family"), "domain": r.get("domain"),
                           "exposure_sector": r.get("exposure_sector")},
                          source=r.get("source"), evidence={"mean": wf.get("mean"), "t": wf.get("t_months"),
                                                           "years_positive": wf.get("years_positive_share")},
                          regimes=r.get("regimes"), reasons=r.get("reasons"), data_kind="historical_walk_forward"))
    for sid, r in ((src.get("alt_validation") or {}).get("sources") or {}).items():
        out.append(_entry("alt_source", sid, {"KEEP": "ACCEPTED", "MODIFY": "INCONCLUSIVE"}.get(r.get("verdict"), "REJECTED"),
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
        out.append(_entry("abstention", wf.get("rule"), "ACCEPTED", {"signal": wf.get("rule"), "direction": -1,
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
                              {k: h.get(k) for k in ("signal", "direction", "family", "domain", "exposure_sector")},
                              source="factory", evidence=None, reasons=[rd.get("reason")],
                              free_sources=rd.get("free_sources") or [], data_kind="none"))
    for hid, r in ((src.get("factory_results") or {}).get("results") or {}).items():
        out.append(_entry("factory", hid, r.get("status"), r.get("spec") or {}, source="factory",
                          evidence=r.get("tests"), reasons=r.get("reasons"), data_kind=r.get("data_kind")))
    return out


def sync(path: Path | None = None, sources: dict | None = None, now: str | None = None) -> int:
    """Neue/geänderte Einträge append-only anhängen. -> Anzahl neuer Einträge."""
    path = path or MEMORY
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen = {e["entry_key"] for e in load(path)}
    new = [e for e in collect(sources) if e["entry_key"] not in seen]
    if new:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            for e in new:
                fh.write(json.dumps({**e, "recorded_at": now}, ensure_ascii=False, sort_keys=True, default=str) + "\n")
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
