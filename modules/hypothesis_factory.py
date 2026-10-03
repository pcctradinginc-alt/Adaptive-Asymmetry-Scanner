"""modules/hypothesis_factory.py – Scientific Hypothesis Factory (NUR SHADOW/RESEARCH).

    python -m modules.hypothesis_factory plan [--llm]   (CI ml_research.yml: nach Director, vor Research-Lab)
    python -m modules.hypothesis_factory evaluate        (CI: nach Research-Lab, ML_PANEL_CACHE)

Erweitert die bestehende Kette, statt sie zu duplizieren:

  IDEE      Cross-Domain-Familien (config/research_domains.yaml), Cross-Source-Divergenzen,
            Drift-/Regime-Befunde (machine_state.json). Ein LLM darf optional NUR den
            ökonomischen Mechanismus formulieren – nie Zahlen, Evidenz oder Ergebnisse.
  HYPOTHESE falsifizierbar: Population, Exposure, Signal, Richtung, Lag/Horizont,
            Kontrollgruppe, Primärmetrik, Mechanismus, Failure Condition, H0, H_alt; spec_hash.
  PRÜFUNG   Datenbereitschaft (PIT-Feature-Store? Mapping belastbar? Historie/Abdeckung?)
            -> sonst DATA_GAP mit kostenlosen Quellen (nie simuliert). Ähnlichkeitssuche im
            Research Memory (jeder Status), Varianten-Deckel je Familie, Budget mit
            Explorationsanteil (config/factory_protocol.yaml, gepinnt).
  TEST      Research-Lab (bestehende Kette: Leakage-Whitelist, Duplikat-Fingerprint,
            Walk-Forward, Kosten, Hälften, Regime, BH über ALLE, Locked) – dann hier die
            Robustheits-Batterie: Placebo (Permutation je Stichtag), Lag-Profil, Sektor- und
            Größen-Replikation, Ablation (Residual-IC nach allen bestehenden Merkmalen).
  FORWARD   Nur ROBUST -> PROSPECTIVE_CHALLENGER (append-only, eingefroren); ab dann zählen
            ausschließlich neue Forward-Kohorten (factory_forward_ledger.jsonl).
  PRODUKTION nie direkt. Einziger Weg: Vertrag mit Champion-Population per menschlichem PR in
            config/promotion_hypotheses.yaml -> PromotionController -> Adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modules import research_memory as rm

log = logging.getLogger(__name__)

PROTOCOL = Path("config/factory_protocol.yaml")
DOMAINS = Path("config/research_domains.yaml")
OUT = Path("outputs/research")
HYP_OUT = OUT / "factory_hypotheses.json"
PLAN_OUT = OUT / "factory_plan.json"
RESULTS_OUT = OUT / "factory_results.json"
CHALLENGERS = OUT / "factory_challengers.jsonl"
FORWARD_LEDGER = OUT / "factory_forward_ledger.jsonl"
LLM_MODEL = "claude-opus-5-5"
DIVERGENCES = [  # Cross-Source-Divergenz: alternatives Signal widerspricht dem Preis
    {"id": "div_procurement_vs_price", "domain": "procurement", "feature": "ted_awards_z", "price": "mom_3m",
     "direction": 1, "relevance": 0.5, "exploratory": True,
     "mechanism": "Steigende öffentliche Aufträge bei fallendem Kurs: Markt preist neue Umsatzsichtbarkeit noch nicht ein."},
    {"id": "div_insider_vs_price", "domain": "insider", "feature": "sec_insider_net_value_90d", "price": "mom_3m",
     "direction": 1, "relevance": 0.5, "exploratory": True,
     "mechanism": "Insider kaufen netto, während der Kurs fällt: private Information widerspricht dem Preis."},
]


def load_protocol(path: Path | None = None) -> dict:
    return yaml.safe_load((path or PROTOCOL).read_text(encoding="utf-8"))


def load_domains(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DOMAINS).read_text(encoding="utf-8"))


def _hid(*parts) -> str:
    return "FAC-" + hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8].upper()


def spec_hash(h: dict) -> str:
    keys = ("signal", "direction", "population", "exposure", "control_group", "horizon", "lag", "primary_metric",
            "failure_condition")
    return hashlib.sha256(json.dumps({k: h.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()[:16]


# ── Ideen -> falsifizierbare Hypothesen ─────────────────────────────────────
def _exp_name(sector: str) -> str | None:
    from modules.research_lab import SECTOR_EXPOSURES
    return next((k for k, v in SECTOR_EXPOSURES.items() if v == sector), None)


def _finish(h: dict) -> dict:
    rule = f"[{h['signal']}] × Richtung {h['direction']:+d}"
    h["H0"] = (f"H0: {rule} hat keinen Effekt auf {h['primary_metric']} über {h['horizon']} Handelstage "
               f"gegenüber {h['control_group']}.")
    h["H_alt"] = (f"H_alt: Effekt in Gegenrichtung. Gewinnt H_alt, wird die Hypothese REJECTED – kein "
                  f"Vorzeichenwechsel; eine Gegenhypothese ist neu zu registrieren und erneut zu prüfen.")
    h["statement"] = f"{h['title']}: {h['mechanism']}"
    h["rationale"] = h["mechanism"]
    h["spec_hash"] = spec_hash(h)
    return h


def ideas_cross_domain(domains: dict, now: str) -> list[dict]:
    out = []
    for f in domains.get("families") or []:
        d = (domains.get("domains") or {}).get(f["domain"], {})
        exp = _exp_name(f["sector"]) if f.get("sector") else None
        feature = f.get("feature")
        signal = (f"{exp} * sign({feature} - {f.get('center', 0)})" if feature and exp else None)
        out.append(_finish({
            "id": _hid(f["id"], signal), "family": f["id"], "domain": f["domain"], "idea_source": "cross_domain",
            "exploratory": bool(f.get("exploratory")), "relevance": float(f.get("relevance", 0.5)),
            "title": f"{d.get('label', f['domain'])} × {f.get('sector')}",
            "population": f"PIT-S&P-500, Sektor {f.get('sector')} vs. übrige Titel", "exposure_sector": f.get("sector"),
            "exposure": f"Sektor {f.get('sector')} (heutige Zuordnung, nicht PIT)",
            "signal": signal, "domain_kind": d.get("kind", "missing"), "domain_features": d.get("features") or [],
            "free_sources": d.get("free_sources") or [], "direction": int(f.get("direction", 1)),
            "lag": "Merkmal zum Stichtag (PIT), Rendite ab Folgetag", "horizon": 20,
            "control_group": "Querschnittsmittel desselben Stichtags (alle übrigen Titel)",
            "primary_metric": "netto Top-Dezil-Überrendite je 20 Handelstage (10 bp/Seite)",
            "mechanism": f["mechanism"], "mechanism_source": "template",
            "failure_condition": "Walk-Forward netto <= 0 oder nicht signifikant nach BH, Placebo nicht übertroffen, "
                                 "Effekt verschwindet in Replikation oder nach Herausrechnen bestehender Merkmale",
            "created_at": now}))
    return out


def ideas_divergence(now: str) -> list[dict]:
    out = []
    for dv in DIVERGENCES:
        signal = f"rank({dv['feature']}) * step(-{dv['price']})"
        out.append(_finish({
            "id": _hid(dv["id"], signal), "family": dv["id"], "domain": dv["domain"], "idea_source": "cross_source_divergence",
            "exploratory": dv["exploratory"], "relevance": dv["relevance"], "title": f"Divergenz {dv['feature']} vs. {dv['price']}",
            "population": "PIT-S&P-500 mit verfügbarem Alt-Feature", "exposure": "firmenspezifisch",
            "signal": signal, "domain_kind": "alt_feature", "domain_features": [dv["feature"]], "free_sources": [],
            "direction": dv["direction"], "lag": "PIT (available_at <= Stichtag)", "horizon": 20,
            "control_group": "Querschnittsmittel desselben Stichtags", "primary_metric": "netto Top-Dezil-Überrendite je 20 T",
            "mechanism": dv["mechanism"], "mechanism_source": "template",
            "failure_condition": "wie Cross-Domain (Walk-Forward, BH, Placebo, Replikation, Ablation)", "created_at": now}))
    return out


def ideas_drift(machine_state: dict | None, now: str) -> list[dict]:
    """Abschwächende Merkmale -> regime-bedingte Hypothese (wirkt das Merkmal nur bei niedriger Vol?)."""
    from modules import ml_research as ml
    out = []
    for item in (machine_state or {}).get("which_features_are_decaying") or []:
        f = next((x for x in ml.ALL_FEATURES if x in str(item)), None)
        if not f or f in ("vix", "vix_chg_21", "spy_trend_200", "spy_mom_63", "tnx", "curve_10y_3m", "cpi_yoy",
                          "fed_assets_13w_chg", "usd_63d_chg", "wti_63d_chg"):
            continue
        signal = f"rank({f}) * step(20 - vix)"
        out.append(_finish({
            "id": _hid("drift", signal), "family": f"drift_{f}", "domain": "drift", "idea_source": "drift",
            "exploratory": False, "relevance": 0.4, "title": f"{f} nur im Niedrig-Vol-Regime",
            "population": "PIT-S&P-500", "exposure": "firmenspezifisch", "signal": signal, "domain_kind": "pit_panel",
            "domain_features": [f, "vix"], "free_sources": [], "direction": 1, "lag": "PIT", "horizon": 20,
            "control_group": "Querschnittsmittel desselben Stichtags", "primary_metric": "netto Top-Dezil-Überrendite je 20 T",
            "mechanism": f"Gemessener Decay von {f} (Meta-Cognition): Effekt könnte regimeabhängig sein statt verschwunden.",
            "mechanism_source": "measured_drift", "failure_condition": "wie Cross-Domain", "created_at": now}))
    return out[:2]


# ── Datenbereitschaft ───────────────────────────────────────────────────────
def readiness(h: dict, panel: pd.DataFrame | None, protocol: dict) -> dict:
    from modules import ml_research as ml
    from modules.alt_data.registry import ALT_FEATURES
    from modules.research_lab import SECTOR_EXPOSURES
    dq = protocol["priority"]["data_quality"]
    if h.get("domain_kind") == "missing" or not h.get("signal"):
        return {"status": "DATA_GAP", "reason": "Datendomäne nicht im PIT-Feature-Store",
                "free_sources": h.get("free_sources") or [], "data_quality": 0.0, "cost": None}
    names = set(rm.tokens_of({"signal": h["signal"]}))
    feats = {n[2:] for n in names if n.startswith("f:")}
    unknown = [f for f in feats if f not in ml.ALL_FEATURES and f not in ALT_FEATURES and f not in SECTOR_EXPOSURES]
    if unknown:
        return {"status": "DATA_GAP", "reason": f"Merkmale unbekannt: {unknown}", "free_sources": h.get("free_sources") or [],
                "data_quality": 0.0, "cost": None}
    quality, cost, flags = dq["pit_panel"], protocol["priority"]["cost"]["pit_panel"], []
    if any(f in ALT_FEATURES for f in feats):
        quality, cost = dq["alt_feature"], protocol["priority"]["cost"]["alt_feature"]
    if any(f in SECTOR_EXPOSURES for f in feats):
        quality = min(quality, dq["non_pit_mapping"])
        flags.append("non_pit_mapping: Sektorzuordnung von heute (Mapping MEDIUM)")
    cov = None
    if panel is not None:
        dev = panel[panel["date"] >= pd.Timestamp("2016-01-01")]
        present = [f for f in feats if f in dev.columns]
        missing = [f for f in feats if f not in dev.columns and f not in SECTOR_EXPOSURES]
        if missing:
            return {"status": "DATA_GAP", "reason": f"nicht im Panel: {missing}", "free_sources": h.get("free_sources") or [],
                    "data_quality": 0.0, "cost": None}
        cov = float(dev[present].notna().all(axis=1).mean()) if present else 1.0
        if cov < 0.3:
            return {"status": "DATA_GAP", "reason": f"Abdeckung ab 2016 nur {cov:.0%} (Historie/Mapping zu dünn)",
                    "free_sources": h.get("free_sources") or [], "data_quality": round(quality * cov, 3), "cost": cost}
    return {"status": "READY", "flags": flags, "coverage_2016plus": None if cov is None else round(cov, 3),
            "data_quality": quality, "cost": cost}


# ── Priorität und Budget ────────────────────────────────────────────────────
def priority(h: dict, dirs: dict, memory_max_sim: float, protocol: dict) -> float:
    a0, b0 = protocol["priority"]["prior_success"]
    d = dirs.get(f"family:{h['family']}") or {}
    mean = d.get("posterior_success", a0 / (a0 + b0))
    sd = d.get("posterior_sd", math.sqrt(a0 * b0 / ((a0 + b0) ** 2 * (a0 + b0 + 1))))
    eig = mean + sd                                       # optimistisch unter Unsicherheit (UCB)
    novelty = max(0.0, 1.0 - memory_max_sim)
    r = h["readiness"]
    if r["status"] != "READY" or not r.get("cost"):
        return 0.0
    return round(eig * h["relevance"] * novelty * r["data_quality"] / r["cost"], 5)


def plan(panel: pd.DataFrame | None = None, protocol: dict | None = None, domains: dict | None = None,
         memory: list[dict] | None = None, machine_state: dict | None = None, now: datetime | None = None) -> dict:
    protocol = protocol or load_protocol()
    domains = domains or load_domains()
    now = now or datetime.now(timezone.utc)
    now_s = now.isoformat(timespec="seconds")
    memory = memory if memory is not None else rm.load()
    dirs = rm.directions(memory, tuple(protocol["priority"]["prior_success"]))
    b = protocol["budget"]
    ideas = ideas_cross_domain(domains, now_s) + ideas_divergence(now_s) + ideas_drift(machine_state, now_s)
    fam_tested: dict[str, set] = {}
    own: dict[str, str] = {}
    for e in rm.latest(memory).values():
        fam = (e.get("spec") or {}).get("family")
        if fam and e.get("status") in rm.TESTED:
            fam_tested.setdefault(fam, set()).add(e["hypothesis_id"])
            own[e["hypothesis_id"]] = e["status"]
    for h in ideas:
        h["readiness"] = readiness(h, panel, protocol)
        hits = rm.search(h, memory, 0.0)[:1]
        sim = hits[0][2] if hits else 0.0
        blocked = [x for x in rm.search(h, memory, b["similarity_block"]) if not x[0].endswith(":" + h["id"])
                   and x[1] in rm.TESTED]
        h["novelty"] = round(1 - sim, 3)
        h["priority"] = priority(h, dirs, sim, protocol)
        if h["readiness"]["status"] == "DATA_GAP":
            h["plan_status"] = "DATA_GAP"
        elif h["id"] in own and own[h["id"]] != "RETEST_LATER":
            h["plan_status"], h["memory_status"] = "ALREADY_TESTED", own[h["id"]]   # nie still wiederholen
        elif blocked:
            h["plan_status"], h["similar_to"] = "SIMILAR_TO_TESTED", blocked[:3]
        elif len(fam_tested.get(h["family"], ())) >= b["max_variants_per_family"]:
            h["plan_status"] = "FAMILY_EXHAUSTED"
        else:
            h["plan_status"] = "CANDIDATE"
    cands = sorted((h for h in ideas if h["plan_status"] == "CANDIDATE"), key=lambda h: -h["priority"])
    slots = int(b["tests_per_run"])
    n_expl = math.ceil(b["exploratory_share"] * slots)
    chosen = [h for h in cands if h["exploratory"]][:n_expl]
    chosen += [h for h in cands if h not in chosen][: max(0, slots - len(chosen))]
    for h in cands:
        h["plan_status"] = "SELECTED" if h in chosen else "NOT_SELECTED_BUDGET"
    return {"generated": now_s, "protocol": protocol["version"], "budget": {"slots": slots, "exploratory_slots": n_expl},
            "n_ideas": len(ideas), "ideas": ideas, "selected": [h["id"] for h in chosen],
            "directions_used": {k: v for k, v in dirs.items() if k.startswith("family:")}}


def lab_hypotheses(plan_res: dict) -> list[dict]:
    keep = ("id", "title", "statement", "signal", "direction", "rationale", "created_at", "family", "domain",
            "exposure_sector", "spec_hash", "H0", "H_alt", "population", "control_group", "failure_condition")
    return [{k: h.get(k) for k in keep} for h in plan_res["ideas"] if h["plan_status"] == "SELECTED"]


# ── Optional: LLM formuliert NUR den Mechanismus ────────────────────────────
def llm_mechanism(h: dict, client=None) -> str | None:
    """Mechanismus-Text (max. 3 Sätze) – ohne Zahlen/Evidenz. Fehler/Zahlen -> None (Vorlage bleibt)."""
    try:
        import anthropic
        client = client or anthropic.Anthropic()
        resp = client.messages.create(
            model=LLM_MODEL, max_tokens=1024, output_config={"effort": "low", "format": {
                "type": "json_schema", "schema": {"type": "object", "additionalProperties": False,
                                                  "required": ["mechanism"],
                                                  "properties": {"mechanism": {"type": "string"}}}}},
            system=("Du formulierst ausschließlich einen plausiblen ökonomischen Wirkmechanismus für eine noch "
                    "UNGETESTETE Hypothese. Nenne keine Zahlen, Studien, Renditen, Wahrscheinlichkeiten oder "
                    "Belege und behaupte nicht, dass der Zusammenhang existiert. Höchstens drei Sätze, Deutsch."),
            messages=[{"role": "user", "content": json.dumps({k: h.get(k) for k in (
                "title", "population", "exposure", "signal", "direction", "horizon")}, ensure_ascii=False)}])
        if getattr(resp, "stop_reason", None) == "refusal":
            return None
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        mech = str(json.loads(text).get("mechanism", "")).strip()
    except Exception as e:  # noqa: BLE001 – optional; Vorlage bleibt
        log.warning(f"Fabrik: LLM-Mechanismus nicht verfügbar ({type(e).__name__})")
        return None
    if not mech or any(ch.isdigit() for ch in mech) or len(mech) > 600:
        return None                                   # Zahlen = mögliche erfundene Evidenz -> verwerfen
    return mech


# ── Robustheits-Batterie (nach dem Research-Lab) ────────────────────────────
def _top_minus_univ(d: pd.DataFrame) -> float | None:
    from modules import ml_research as ml
    coh = ml.cohort_returns(d, "score")
    return None if coh.empty else float(coh["top_vs_univ"].mean() - 2 * ml.COST_BASE)


def battery(panel: pd.DataFrame, h: dict, protocol: dict, seed: int = 7) -> dict:
    from modules import ml_research as ml
    from modules import research_lab as rl
    rb = protocol["robustness"]
    per = ml.PROTOCOL["periods"]
    panel = rl.add_exposures(panel)
    sig = rl.eval_signal(panel, h["signal"]) * int(h["direction"])
    test = rl.window(panel, f"{per['first_test_year']}-01-01", per["locked_from"], per["locked_from"])
    d = test.assign(score=sig.loc[test.index])
    out, fails = {}, []
    obs = _top_minus_univ(d)
    out["observed"] = obs
    # Placebo: Signal innerhalb jedes Stichtags permutieren
    rng = np.random.default_rng(seed)
    plac = []
    for _ in range(int(rb["placebo_permutations"])):
        perm = d.groupby("date")["score"].transform(lambda s: s.to_numpy()[rng.permutation(len(s))])
        v = _top_minus_univ(d.assign(score=perm))
        if v is not None:
            plac.append(v)
    p = (1 + sum(v >= (obs or -9) for v in plac)) / (1 + len(plac)) if plac else 1.0
    out["placebo"] = {"n": len(plac), "p": round(p, 4), "q95": round(float(np.quantile(plac, 0.95)), 5) if plac else None}
    if obs is None or p > rb["placebo_max_p"]:
        fails.append(f"Placebo nicht übertroffen (p={p:.3f})")
    # Lag-Profil (Wochen = Zeilen je Ticker)
    srt = panel.sort_values(["ticker", "date"])
    lags = {}
    for k in rb["lag_weeks"]:
        lagged = sig.loc[srt.index].groupby(srt["ticker"]).shift(int(k))
        lags[str(k)] = _top_minus_univ(test.assign(score=lagged.reindex(test.index)))
    out["lag_profile"] = lags
    if (lags.get("0") or -1) <= 0:
        fails.append(f"Lag 0 nicht positiv ({lags.get('0')})")
    # Größen-Replikation (verwandtes Universum)
    med = d.groupby("date")[rb["replication_split"]].transform("median")
    big, small = _top_minus_univ(d[d[rb["replication_split"]] >= med]), _top_minus_univ(d[d[rb["replication_split"]] < med])
    out["replication_size"] = {"large": big, "small": small}
    if not ((big or -1) > 0 and (small or -1) > 0):
        fails.append(f"Replikation groß/klein uneinheitlich ({big}, {small})")
    # Sektor-Replikation (nur für nicht sektor-gescopte Hypothesen)
    if not h.get("exposure_sector") and "sector" in d:
        res = {s: _top_minus_univ(g) for s, g in d.groupby("sector") if g["date"].nunique() >= 10}
        res = {s: v for s, v in res.items() if v is not None}
        share = sum(v > 0 for v in res.values()) / len(res) if res else 0.0
        out["replication_sector"] = {"share_positive": round(share, 3), "by_sector": res}
        if share < rb["sector_min_share_same_sign"]:
            fails.append(f"Sektor-Replikation {share:.0%} < {rb['sector_min_share_same_sign']:.0%}")
    # Ablation: Residual-IC nach allen bestehenden Querschnitts-Merkmalen
    xs = [f for f in ml.ALL_FEATURES if f in d and d.groupby("date")[f].nunique().median() > 1]
    resid = pd.Series(np.nan, index=d.index)
    for _, g in d.groupby("date"):
        g = g[["score"] + xs].dropna()
        if len(g) < 30 or g["score"].nunique() < 2:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].rank().to_numpy() for c in xs])
        y = g["score"].rank().to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        if r.std() > 1e-6 * y.std():                  # sonst vollständig erklärt: kein eigener Gehalt (NaN)
            resid.loc[g.index] = r
    ic = ml.ic_stats(ml.daily_ic(d.assign(resid=resid), "resid"))
    out["residual_ic"] = ic
    if (ic.get("t_months") or 0) < rb["residual_ic_min_t"]:
        fails.append(f"kein inkrementeller Gehalt nach bestehenden Merkmalen (Residual-IC t={ic.get('t_months')})")
    return {"tests": out, "robust": not fails, "reasons": fails}


def _next_monday(now: datetime) -> str:
    d = (now + timedelta(days=(7 - now.weekday()) % 7 or 7)).date()
    return d.isoformat()                              # Datum wie alt_hypotheses.yaml (Panel-Daten tz-naiv)


def evaluate(panel: pd.DataFrame, protocol: dict | None = None, now: datetime | None = None,
             db_path: Path | None = None, hyp_path: Path | None = None) -> dict:
    """Nach dem Research-Lab: Batterie nur für Lab-Bestandene; ROBUST -> PROSPECTIVE_CHALLENGER."""
    from modules import research_lab as rl
    protocol = protocol or load_protocol()
    now = now or datetime.now(timezone.utc)
    now_s = now.isoformat(timespec="seconds")
    hyps = json.loads((hyp_path or HYP_OUT).read_text()).get("hypotheses", []) if (hyp_path or HYP_OUT).exists() else []
    db = rl.load_db(db_path or rl.DB_PATH).get("hypotheses", {})
    known = {json.loads(x)["hypothesis_id"] for x in CHALLENGERS.read_text().splitlines() if x.strip()} \
        if CHALLENGERS.exists() else set()
    results = {}
    for h in hyps:
        rec = db.get(h["id"]) or {}
        lab = (rec.get("canonical_status") or rl.canonical_status(rec)) if rec else "NOT_TESTED"
        spec = {k: h.get(k) for k in ("signal", "direction", "family", "domain", "exposure_sector", "spec_hash")}
        if lab != "ACCEPTED":
            results[h["id"]] = {"status": "REJECTED" if lab == "REJECTED" else lab, "spec": spec,
                                "reasons": rec.get("reasons") or [f"Research-Lab: {lab}"],
                                "data_kind": "historical_walk_forward"}
            continue
        bt = battery(panel, h, protocol)
        status = "PROSPECTIVE_CHALLENGER" if bt["robust"] else "NOT_ROBUST"
        results[h["id"]] = {"status": status, "spec": spec, "tests": bt["tests"], "reasons": bt["reasons"],
                            "data_kind": "historical_walk_forward+robustness"}
        if bt["robust"] and h["id"] not in known:
            c = {"hypothesis_id": h["id"], "spec_hash": h["spec_hash"], "signal": h["signal"],
                 "direction": int(h["direction"]), "registered_at": now_s, "forward_start": _next_monday(now),
                 "spec": h, "note": "eingefroren; Promotion nur über PromotionController nach Champion-Vertrag (PR)"}
            CHALLENGERS.parent.mkdir(parents=True, exist_ok=True)
            with open(CHALLENGERS, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(c, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    out = {"generated": now_s, "results": results}
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    record_forward(panel)
    return out


def record_forward(panel: pd.DataFrame) -> int:
    """Prospective Challenger: Top-Dezile je Stichtag >= forward_start (append-only)."""
    if not CHALLENGERS.exists():
        return 0
    from modules.alt_data import contracts as ac
    from modules import research_lab as rl
    cs = [json.loads(x) for x in CHALLENGERS.read_text().splitlines() if x.strip()]
    st = {c["hypothesis_id"]: {"status": "VALID", "spec_hash": c["spec_hash"]} for c in cs}
    return ac.record_forward(rl.add_exposures(panel), cs, st, FORWARD_LEDGER)


def _panel_from_cache() -> pd.DataFrame | None:
    import pickle
    cache = os.environ.get("ML_PANEL_CACHE")
    if cache and Path(cache).exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)  # noqa: S301 – eigene, im selben Job erzeugte Datei
    return None


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "evaluate"])
    ap.add_argument("--llm", action="store_true", help="Mechanismus-Text optional per LLM (nie Evidenz)")
    args = ap.parse_args(argv)
    rm.sync()
    panel = _panel_from_cache()
    if args.cmd == "plan":
        ms = json.loads(Path("outputs/research/machine_state.json").read_text()) \
            if Path("outputs/research/machine_state.json").exists() else None
        res = plan(panel, machine_state=ms)
        if args.llm:
            for h in res["ideas"]:
                if h["plan_status"] == "SELECTED":
                    m = llm_mechanism(h)
                    if m:
                        h.update(mechanism_template=h["mechanism"], mechanism=m, mechanism_source=f"llm:{LLM_MODEL}")
        PLAN_OUT.parent.mkdir(parents=True, exist_ok=True)
        PLAN_OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        HYP_OUT.write_text(json.dumps({"generated": res["generated"], "hypotheses": lab_hypotheses(res)},
                                      indent=1, ensure_ascii=False), encoding="utf-8")
        counts = {}
        for h in res["ideas"]:
            counts[h["plan_status"]] = counts.get(h["plan_status"], 0) + 1
        print(f"Fabrik: {res['n_ideas']} Ideen {counts}; ausgewählt {res['selected']}")
    else:
        if panel is None:
            log.error("Fabrik evaluate: ML_PANEL_CACHE fehlt")
            return 1
        res = evaluate(panel)
        print({k: v["status"] for k, v in res["results"].items()})
    rm.sync()
    rm.write_directions(rm.load())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
