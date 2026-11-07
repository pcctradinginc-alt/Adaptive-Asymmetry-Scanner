"""modules/hypothesis_factory.py – Scientific Hypothesis Factory (NUR SHADOW/RESEARCH).

    python -m modules.hypothesis_factory plan [--llm]   (CI ml_research.yml: nach Director, vor Research-Lab)
    python -m modules.hypothesis_factory evaluate        (CI: nach Research-Lab, ML_PANEL_CACHE)
    python -m modules.hypothesis_factory forward         (CI wöchentlich: Forward-Ledger der Challenger, ML_PANEL_CACHE)

Erweitert die bestehende Kette, statt sie zu duplizieren:

  IDEE      Cross-Domain-Familien (config/research_domains.yaml), Cross-Source-Divergenzen,
            Drift-/Regime-Befunde (machine_state.json) und GEMESSENE Anomalien (scan_anomalies: fester
            Suchraum aus Divergenz, verzögerter Makro-Reaktion, Widerspruch, Peer-Reaktion, Regime-Beziehung;
            nur im Discovery-Fenster, BH über alle gescannten Ausdrücke). Ein LLM darf optional NUR den
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

from modules import cost_telemetry
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
    {"id": "div_fundamentals_vs_price", "domain": "fundamentals", "feature": "xbrl_sue", "price": "mom_3m",
     "direction": 1, "relevance": 0.6, "exploratory": False,
     "mechanism": "Berichtete Gewinnüberraschung bei fallendem Kurs: belegte Fundamentaldaten widersprechen dem Preis."},
    # Commodity (RESEARCH): Rohstoff-Signal für die Exposure-Richtung günstig, Aktie fiel trotzdem
    {"id": "div_oil_vs_equity", "domain": "commodity_oil_price", "feature": "cmdx_oil__wti_ret_20d", "price": "mom_3m",
     "signal": "rank(cmdx_oil__wti_ret_20d) * step(-mom_3m)", "direction": 1, "relevance": 0.4, "exploratory": True,
     "mechanism": "Ölpreis bewegte sich zugunsten der Exposure-Richtung, Aktie fiel dennoch: Markt preist die "
                  "Rohstoffbewegung verzögert ein (oder die Beziehung ist zerfallen – Test entscheidet)."},
    {"id": "div_gas_vs_equity", "domain": "commodity_gas_price", "feature": "cmdx_natural_gas__henry_hub_ret_20d",
     "price": "mom_3m", "signal": "rank(cmdx_natural_gas__henry_hub_ret_20d) * step(-mom_3m)", "direction": 1,
     "relevance": 0.3, "exploratory": True,
     "mechanism": "Gaspreisbewegung zugunsten der Exposure-Richtung bei fallender Aktie: verzögerte Einpreisung."},
]
DIVERGENCE_SOURCE = {"procurement": "ted_procurement", "insider": "sec_deep_events",
                     "fundamentals": "sec_xbrl_fundamentals", "commodity_oil_price": "commodity_price",
                     "commodity_gas_price": "commodity_price"}
SOURCE_CONDITIONS = OUT / "source_conditions.json"
MAX_CONDITION_IDEAS = 4
COND_MIN_T = 2.0


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
    h["H_alt"] = ("H_alt: Effekt in Gegenrichtung. Gewinnt H_alt, wird die Hypothese REJECTED – kein "
                  "Vorzeichenwechsel; eine Gegenhypothese ist neu zu registrieren und erneut zu prüfen.")
    h["statement"] = f"{h['title']}: {h['mechanism']}"
    h["rationale"] = h["mechanism"]
    h["spec_hash"] = spec_hash(h)
    return h


def ideas_cross_domain(domains: dict, now: str) -> list[dict]:
    out = []
    for f in domains.get("families") or []:
        d = (domains.get("domains") or {}).get(f["domain"], {})
        if f.get("commodity"):
            out.append(_commodity_idea(f, d, now))
            continue
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


def _commodity_idea(f: dict, d: dict, now: str) -> dict:
    """Commodity-Domäne: Signal = cmdexp_<rohstoff> (erwartete Exposure-Richtung, NON_PIT-Mapping)
    × sign(Datums-Feature − center). Die LLM/Vorlage liefert nur den Mechanismus – Population, Horizont,
    Baseline, OOS (Walk-Forward + Locked) und Mehrfachtest-Kontrolle (BH) sind fest."""
    from modules.commodity_intelligence import EXPOSURE_COLUMNS, mapping_status
    exp = f"cmdexp_{f['commodity']}"
    feature = f.get("feature")
    signal = f"{exp} * sign({feature} - {f.get('center', 0)})" if exp in EXPOSURE_COLUMNS and feature else None
    ms = mapping_status()
    return _finish({
        "id": _hid(f["id"], signal), "family": f["id"], "domain": f["domain"], "idea_source": "cross_domain",
        "commodity": f["commodity"], "exploratory": bool(f.get("exploratory")),
        "relevance": float(f.get("relevance", 0.5)), "title": f"{d.get('label', f['domain'])} × {f['commodity']}-Exposure",
        "population": f"PIT-S&P-500; Titel mit {f['commodity']}-Exposure-Mapping vs. übrige Titel mit bekanntem Sektor",
        "equity_population": f"cmdexp_{f['commodity']} != 0 (Mapping {ms.get('version')}, {ms.get('point_in_time_status')})",
        "exposure": f"Commodity-Exposure {f['commodity']} (heutige Zuordnung, nicht PIT)", "non_pit_mapping": True,
        "signal": signal, "domain_kind": d.get("kind", "missing"), "domain_features": d.get("features") or [],
        "free_sources": d.get("free_sources") or [], "source_id": d.get("source"), "direction": int(f.get("direction", 1)),
        "lag": "Datums-Feature zum Stichtag 21:00 UTC (available_at <= Stichtag), Rendite ab Folgetag", "horizon": 20,
        "baseline": "Querschnittsmittel desselben Stichtags; zusätzlich Ablation gegen bestehende Merkmale",
        "control_group": "Querschnittsmittel desselben Stichtags (alle übrigen Titel)",
        "primary_metric": "netto Top-Dezil-Überrendite je 20 Handelstage (10 bp/Seite)",
        "oos": "Walk-Forward-Testjahre + gesperrtes Locked-Fenster; danach Forward ab Registrierung",
        "multiple_testing": "Benjamini-Hochberg über alle Fabrik-Hypothesen des Laufs",
        "mechanism": f["mechanism"], "mechanism_source": "template",
        "failure_condition": "Walk-Forward netto <= 0 oder nicht signifikant nach BH, Placebo nicht übertroffen, "
                             "Effekt verschwindet in Replikation, nach Herausrechnen bestehender Merkmale oder "
                             "zerfällt über die Zeit (RELATIONSHIP DECAYED)",
        "created_at": now})


def ideas_divergence(now: str) -> list[dict]:
    out = []
    for dv in DIVERGENCES:
        signal = dv.get("signal") or f"rank({dv['feature']}) * step(-{dv['price']})"
        out.append(_finish({
            "id": _hid(dv["id"], signal), "family": dv["id"], "domain": dv["domain"], "idea_source": "cross_source_divergence",
            "source_id": DIVERGENCE_SOURCE.get(dv["domain"]),
            "exploratory": dv["exploratory"], "relevance": dv["relevance"], "title": f"Divergenz {dv['feature']} vs. {dv['price']}",
            "population": "PIT-S&P-500 mit verfügbarem Alt-Feature", "exposure": "firmenspezifisch",
            "signal": signal, "domain_kind": "alt_feature", "domain_features": [dv["feature"]], "free_sources": [],
            "direction": dv["direction"], "lag": "PIT (available_at <= Stichtag)", "horizon": 20,
            "control_group": "Querschnittsmittel desselben Stichtags", "primary_metric": "netto Top-Dezil-Überrendite je 20 T",
            "mechanism": dv["mechanism"], "mechanism_source": "template",
            "failure_condition": "wie Cross-Domain (Walk-Forward, BH, Placebo, Replikation, Ablation)", "created_at": now}))
    return out


def ideas_source_conditions(conditions: dict | None, now: str) -> list[dict]:
    """Aus Daten generiert: Quelle × Sektor/Regime-Zellen mit |t| >= COND_MIN_T – NUR aus den
    Auswahljahren (selection), getestet wird danach im Lab-Walk-Forward der späteren Jahre."""
    from modules.research_lab import SECTOR_EXPOSURES
    by_sector = {v: k for k, v in SECTOR_EXPOSURES.items()}
    cells = []
    for sid, src in ((conditions or {}).get("sources") or {}).items():
        for f, c in (src.get("selection") or {}).items():
            for sec, hz in (c.get("sector") or {}).items():
                st = hz.get("20") or {}
                if by_sector.get(sec) and abs(st.get("t_months") or 0) >= COND_MIN_T:
                    cells.append((abs(st["t_months"]), sid, f, "sector", sec, int(np.sign(st.get("mean_ic") or 0))))
            for reg, hz in (c.get("regime") or {}).items():
                st = hz.get("20") or {}
                if reg.startswith("vol:") and abs(st.get("t_months") or 0) >= COND_MIN_T:
                    cells.append((abs(st["t_months"]), sid, f, "regime", reg, int(np.sign(st.get("mean_ic") or 0))))
    out = []
    for t, sid, f, kind, key, sign in sorted(cells, reverse=True)[:MAX_CONDITION_IDEAS]:
        if sign == 0:
            continue
        if kind == "sector":
            signal, scope = f"{by_sector[key]} * rank({f})", f"Sektor {key}"
        else:
            signal = f"rank({f}) * step(vix - 20)" if key.endswith("vix_ge_20") else f"rank({f}) * step(20 - vix)"
            scope = f"Regime {key}"
        out.append(_finish({
            "id": _hid("cond", signal, sign), "family": f"cond_{f}_{key.replace(':', '_').replace(' ', '_').lower()}",
            "domain": sid, "source_id": sid, "idea_source": "source_condition", "exploratory": False, "relevance": 0.5,
            "title": f"{f} wirkt bedingt ({scope})", "population": f"PIT-S&P-500, {scope}",
            "exposure_sector": key if kind == "sector" else None,
            "exposure": scope + (" (heutige Zuordnung, nicht PIT)" if kind == "sector" else ""),
            "signal": signal, "domain_kind": "alt_feature", "domain_features": [f], "free_sources": [],
            "direction": sign, "lag": "PIT (available_at <= Stichtag)", "horizon": 20,
            "control_group": "Querschnittsmittel desselben Stichtags", "primary_metric": "netto Top-Dezil-Überrendite je 20 T",
            "mechanism": f"Gemessen in den Auswahljahren ({conditions.get('selection_years')}): Quelle {sid} zeigt "
                         f"nur bedingt Information. Ökonomischer Mechanismus offen – Test entscheidet.",
            "mechanism_source": "measured_condition", "selection_t": round(t, 2),
            "failure_condition": "wie Cross-Domain (Walk-Forward der späteren Jahre, BH, Placebo, Replikation, Ablation)",
            "created_at": now}))
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


def ideas_open_questions(system_state: dict | None, now: str) -> list[dict]:
    """Kernfrage-Schleife: Merkmal außerhalb des Trainingsbereichs (gemessen im SystemState) ->
    falsifizierbare Frage, ob der stärkste Basis-Effekt (12-1-Momentum) in diesem Bereich anders wirkt.
    Schwelle = p99 des Trainingsbereichs aus dem Drift-Befund (keine Wahl aus Testdaten)."""
    from modules import ml_research as ml
    feat = (((system_state or {}).get("drift_state") or {}).get("components") or {}).get("feature") or {}
    out = []
    for f in feat.get("features_out_of_range") or []:
        info = (feat.get("per_feature") or {}).get(f) or {}
        p99 = info.get("p99")
        if f not in ml.ALL_FEATURES or not isinstance(p99, (int, float)):
            continue
        signal = f"rank(mom_12_1) * step({f} - {round(float(p99), 4)})"
        out.append(_finish({
            "id": _hid("openq", signal), "family": f"openq_{f}", "domain": "regime", "idea_source": "open_question",
            "exploratory": True, "relevance": 0.6, "title": f"Momentum außerhalb des {f}-Trainingsbereichs",
            "population": "PIT-S&P-500", "exposure": f"Regime {f} > p99 des Trainingsbereichs",
            "signal": signal, "domain_kind": "pit_panel", "domain_features": ["mom_12_1", f], "free_sources": [],
            "direction": 1, "lag": "PIT", "horizon": 20, "control_group": "Querschnittsmittel desselben Stichtags",
            "primary_metric": "netto Top-Dezil-Überrendite je 20 T",
            "mechanism": f"{f} liegt außerhalb des bisher gesehenen Bereichs (SystemState-Drift). Offene Frage: "
                         f"tragen die Basis-Effekte dort weiter, oder kehrt sich ihre Wirkung um?",
            "mechanism_source": "measured_drift", "failure_condition": "wie Cross-Domain; zu wenige Beobachtungen "
                                                                       "im Bereich -> DATA_GAP statt Schätzung",
            "created_at": now}))
    return out[:2]


# ── Anomalie-getriebene Ideen: gemessen, nur im Auswahlfenster ──────────────
# Statt nur fester Vorlagen leitet die Fabrik Forschungsfragen aus GEMESSENEN Auffälligkeiten ab. Suchraum: feste,
# vorab erklärte Beziehungstypen über vorhandene PIT-Merkmale (keine neuen Daten, kein generierter Code: jede Idee
# ist ein Ausdruck der Research-Lab-Sprache). Gemessen wird NUR im Discovery-Fenster des geschützten Protokolls
# (train_start..discovery_end, Labels enden davor) – geprüft wird danach in der unveränderten Lab-Kette auf den
# späteren Jahren (Walk-Forward, BH über alle, Locked nur informativ) und erst nach Registrierung prospektiv.
# Mehrfachtest: BH über ALLE gescannten Ausdrücke dieses Laufs (auch nicht auswertbare, p = 1); höchstens
# MAX_ANOMALY_IDEAS Ideen, danach gelten Ähnlichkeitssperre, Familien-Deckel und Budget der Fabrik unverändert.
ANOMALY_VERSION = "anomaly-v1"
MAX_ANOMALY_IDEAS = 4
ANOMALY_MIN_COVERAGE = 0.5                # wie hypothesis_acceptance.min_coverage im geschützten Protokoll
ANOMALY_PRICE = ("mom_3m", "rev_1m")
ANOMALY_MACRO = ("vix_chg_21", "spy_mom_63", "fed_assets_13w_chg", "usd_63d_chg", "wti_63d_chg")
ANOMALY_PEER = ("mom_12_1", "mom_3m", "rev_1m", "vol_20", "dist_52w_high", "ret_5d")
ANOMALY_REGIMES = {"VIX >= 20": "step(vix - 20)", "VIX < 20": "step(20 - vix)",
                   "SPY-Aufwärtstrend": "step(spy_trend_200)", "SPY-Abwärtstrend": "step(-spy_trend_200)"}
ANOMALY_CONTRADICTIONS = (
    ("xbrl_sue", "step(sec_8k_negative_90d)", "negative 8-K-Meldungen", ("sec_8k_negative_90d",)),
    ("xbrl_sue", "step(-ret_5d)", "fallender Kurs der letzten Woche", ("ret_5d",)),
    ("sec_insider_net_value_90d", "step(sec_8k_negative_90d)", "negative 8-K-Meldungen", ("sec_8k_negative_90d",)),
)
ANOMALY_TYPES = {
    "divergence": {"relevance": 0.5, "label": "Unerwartete Divergenz zweier Datenreihen",
                   "mechanism": "{x} ist hoch, während {p} gefallen ist: belegte Unternehmensinformation widerspricht "
                                "dem Preis. Hypothese: der Markt preist sie verzögert ein (oder die Divergenz ist "
                                "berechtigt – der Test entscheidet)."},
    "macro_reaction": {"relevance": 0.5, "label": "Verzögerte Reaktion auf einen Wirtschaftsindikator",
                       "mechanism": "Veränderungen von {m} wirken über Kosten, Nachfrage oder Finanzierung auf "
                                    "Unternehmen des Sektors {s}. Hypothese: die Aktien reagieren verzögert – die "
                                    "Rendite wird erst ab dem Folgetag über 20 Handelstage gemessen."},
    "contradiction": {"relevance": 0.5, "label": "Widerspruch zwischen Fundamentaldaten, Nachrichten und Kurs",
                      "mechanism": "{x} signalisiert Stärke, gleichzeitig {n}: Fundamentaldaten und Nachrichten/Kurs "
                                   "widersprechen sich. Hypothese: eine der beiden Informationen ist fehlbewertet."},
    "peer_reaction": {"relevance": 0.4, "label": "Unterschiedliche Reaktion vergleichbarer Unternehmen",
                      "mechanism": "Innerhalb des Sektors {s} wirkt {f} anders als im Gesamtmarkt. Hypothese: ein "
                                   "sektorspezifischer Mechanismus (Zyklik, Kapitalintensität, Regulierung) verändert "
                                   "die Bepreisung dieses Merkmals."},
    "regime_relationship": {"relevance": 0.4, "label": "Veränderte Beziehung eines bekannten Faktors je Marktregime",
                            "mechanism": "{f} wirkt im Regime {r} anders als im Durchschnitt. Hypothese: "
                                         "Risikoappetit bzw. Volatilität verändert die Bepreisung dieses Faktors."},
    "option_vs_stock": {"relevance": 0.6, "label": "Abweichung zwischen Aktienbewegung und Optionsbewertung",
                        "mechanism": "Implizite Volatilität bzw. Skew weichen von der späteren Aktienbewegung ab. "
                                     "Hypothese: der Optionsmarkt nimmt Information vorweg oder überschätzt Risiken."},
}


def anomaly_templates() -> list[dict]:
    """Fester, vorab erklärter Suchraum (Anzahl fließt vollständig in die BH-Korrektur ein)."""
    from modules import ml_research as ml
    from modules.alt_data.registry import ALT_FEATURES
    from modules.research_lab import SECTOR_EXPOSURES
    firm_alt = [f for f in ALT_FEATURES if not f.startswith("cmdx_")]
    out = []
    for x in firm_alt:
        for p in ANOMALY_PRICE:
            out.append({"type": "divergence", "signal": f"rank({x}) * step(-{p})", "features": [x, p],
                        "fmt": {"x": x, "p": p}, "key": f"{x}_vs_{p}"})
    for exp, sector in SECTOR_EXPOSURES.items():
        for m in ANOMALY_MACRO:
            out.append({"type": "macro_reaction", "signal": f"{exp} * sign({m} - 0)", "features": [exp, m],
                        "fmt": {"m": m, "s": sector}, "key": f"{m}_x_{exp}", "exposure_sector": sector})
    for x, cond, text, extra in ANOMALY_CONTRADICTIONS:
        out.append({"type": "contradiction", "signal": f"rank({x}) * {cond}", "features": [x, *extra],
                    "fmt": {"x": x, "n": text}, "key": f"{x}_{extra[0]}"})
    for exp, sector in SECTOR_EXPOSURES.items():
        for f in ANOMALY_PEER:
            out.append({"type": "peer_reaction", "signal": f"{exp} * rank({f})", "features": [exp, f],
                        "fmt": {"f": f, "s": sector}, "key": f"{f}_in_{exp}", "exposure_sector": sector})
    for f in ml.STOCK_FEATURES:
        for name, cond in ANOMALY_REGIMES.items():
            feats = [f] + [n for n in ("vix", "spy_trend_200") if n in cond]
            out.append({"type": "regime_relationship", "signal": f"rank({f}) * {cond}", "features": feats,
                        "fmt": {"f": f, "r": name}, "key": f"{f}_{name}"})
    return out


def scan_anomalies(panel: pd.DataFrame | None, known_keys: set | frozenset = frozenset()) -> dict:
    """Misst jeden Suchraum-Ausdruck im Discovery-Fenster (Rang-IC gegen die Querschnitts-Überrendite, t über
    Monate, zweiseitig) und gibt die BH-Überlebenden (q des geschützten Protokolls) zurück. Bereits bekannte
    Ausdrücke (gleicher AST-Schlüssel wie getestete, Fabrik- oder Lab-Discovery-Signale) werden nicht erneut
    gescannt. Fehlende Merkmale -> DATA_GAP, nie geschätzt. Deterministisch: gleiche Daten, gleiche Liste."""
    from modules import ml_research as ml
    from modules import research_lab as rl
    per, q = ml.PROTOCOL["periods"], float(ml.PROTOCOL["multiple_testing"]["fdr_q"])
    window_txt = f"{per['train_start']}..{per['discovery_end']}"
    if panel is None or panel.empty:
        return {"version": ANOMALY_VERSION, "status": "NO_PANEL", "n_scanned": 0, "survivors": [], "data_gaps": {},
                "window": window_txt}
    panel = rl.add_exposures(panel)
    lo, hi = pd.Timestamp(per["train_start"]), pd.Timestamp(per["discovery_end"])
    sub = panel[(panel["date"] >= lo) & (panel["date"] < hi)]
    win = rl.window(sub, lo, hi, hi)
    stats, pv, gaps, seen, by_key = {}, {}, {}, set(known_keys), {}
    for t in anomaly_templates():
        try:
            key = rl.signal_key(t["signal"])
        except rl.SignalError:
            continue
        if key in seen:
            continue
        seen.add(key)
        missing = [f for f in t["features"] if f not in sub.columns or sub[f].notna().sum() == 0]
        if missing:
            gaps.setdefault(t["type"], set()).update(missing)
            continue
        s = rl.eval_signal(sub, t["signal"]).loc[win.index]
        cov = float(s.notna().mean()) if len(s) else 0.0
        if cov < ANOMALY_MIN_COVERAGE:                     # zählt als Test (p = 1, konservatives BH), Lücke benannt
            pv[key], stats[key], by_key[key] = 1.0, {"coverage": round(cov, 3), "status": "LOW_COVERAGE"}, t
            thin = [f for f in t["features"] if win.empty or win[f].notna().mean() < ANOMALY_MIN_COVERAGE]
            gaps.setdefault(t["type"], set()).update(thin or t["features"])
            continue
        st = ml.ic_stats(ml.daily_ic(win.assign(_s=s), "_s"))
        tm = st.get("t_months")
        pv[key] = 1.0 if tm is None else min(1.0, 2 * rl.p_one_sided(abs(tm)))
        stats[key], by_key[key] = {**st, "coverage": round(cov, 3)}, t
    sig = rl.benjamini_hochberg(pv, q) if pv else set()
    surv = []
    for key in sorted(sig, key=lambda k: (pv[k], k))[:MAX_ANOMALY_IDEAS]:
        st = stats[key]
        if not st.get("mean_ic"):
            continue
        surv.append({**by_key[key], "signal_key": key, "direction": 1 if st["mean_ic"] > 0 else -1,
                     "selection": {"mean_ic": st["mean_ic"], "t_months": st.get("t_months"),
                                   "p_two_sided": round(pv[key], 6), "n_dates": st.get("n_dates"),
                                   "coverage": st["coverage"], "fdr_q": q, "n_scanned": len(pv), "window": window_txt}})
    by_type: dict[str, int] = {}
    for k in pv:
        by_type[by_key[k]["type"]] = by_type.get(by_key[k]["type"], 0) + 1
    return {"version": ANOMALY_VERSION, "status": "OK", "window": window_txt, "n_scanned": len(pv),
            "n_significant": len(sig), "survivors": surv, "by_type": dict(sorted(by_type.items())),
            "data_gaps": {k: sorted(v) for k, v in sorted(gaps.items())}}


def ideas_anomalies(scan: dict, now: str) -> list[dict]:
    """BH-Überlebende des Anomalie-Scans -> falsifizierbare Fabrik-Ideen; Typen ohne Daten -> DATA_GAP-Ideen.
    Die Richtung stammt aus dem Auswahlfenster und ist danach fest (kein Vorzeichenwechsel im Test)."""
    out = []
    for a in scan.get("survivors") or []:
        tp = ANOMALY_TYPES[a["type"]]
        sel = a["selection"]
        out.append(_finish({
            "id": _hid("anomaly", a["signal"], a["direction"]), "family": f"anom_{a['type']}_{a['key']}".lower(),
            "domain": a["type"], "idea_source": "anomaly", "anomaly_type": a["type"], "exploratory": True,
            "relevance": tp["relevance"], "title": f"{tp['label']}: {a['signal']}",
            "population": "PIT-S&P-500" + (f", Sektor {a['exposure_sector']} vs. übrige Titel"
                                           if a.get("exposure_sector") else ""),
            "exposure_sector": a.get("exposure_sector"),
            "exposure": (f"Sektor {a['exposure_sector']} (heutige Zuordnung, nicht PIT)" if a.get("exposure_sector")
                         else "firmenspezifisch"),
            "signal": a["signal"], "domain_kind": "pit_panel", "domain_features": a["features"], "free_sources": [],
            "direction": a["direction"], "lag": "PIT (available_at <= Stichtag), Rendite ab Folgetag", "horizon": 20,
            "control_group": "Querschnittsmittel desselben Stichtags (alle übrigen Titel)",
            "primary_metric": "netto Top-Dezil-Überrendite je 20 Handelstage (10 bp/Seite)",
            "mechanism": tp["mechanism"].format(**a["fmt"]), "mechanism_source": "measured_anomaly",
            "anomaly": {"version": ANOMALY_VERSION, **sel},
            "failure_condition": "Walk-Forward der späteren Jahre netto <= 0 oder nicht signifikant nach BH über "
                                 "alle Hypothesen, Placebo nicht übertroffen, Effekt verschwindet in Replikation oder "
                                 "nach Herausrechnen bestehender Merkmale; Auswahlfenster zählt nie als Beleg",
            "created_at": now}))
    gap_types = set(scan.get("data_gaps") or {})
    if scan.get("status") == "OK":
        gap_types.add("option_vs_stock")      # keine PIT-Optionshistorie im Research-Panel (Archiv erst ab 2026-10-12)
    for tp in sorted(gap_types):
        if tp != "option_vs_stock" and any(a["type"] == tp for a in scan.get("survivors") or []):
            continue
        feats = (scan.get("data_gaps") or {}).get(tp) or ["options_iv_history", "options_skew_history"]
        out.append(_finish({
            "id": _hid("anomaly_gap", tp), "family": f"anom_gap_{tp}", "domain": tp, "idea_source": "anomaly",
            "anomaly_type": tp, "exploratory": True, "relevance": ANOMALY_TYPES[tp]["relevance"],
            "title": f"{ANOMALY_TYPES[tp]['label']} – Daten fehlen", "population": "PIT-S&P-500",
            "exposure": "firmenspezifisch", "signal": None, "domain_kind": "missing", "domain_features": feats,
            "free_sources": (["outputs/options_archive (eigenes Quote-Archiv, ab 2026-10-12; Historie zu kurz)"]
                             if tp == "option_vs_stock" else []),
            "direction": 1, "lag": "PIT", "horizon": 20, "control_group": "Querschnittsmittel desselben Stichtags",
            "primary_metric": "netto Top-Dezil-Überrendite je 20 T",
            "mechanism": ANOMALY_TYPES[tp]["mechanism"].format(x="Merkmal", p="Preis", m="Indikator", s="Sektor",
                                                               n="Gegeninformation", f="Merkmal", r="Regime"),
            "mechanism_source": "measured_anomaly",
            "failure_condition": "nicht testbar ohne Point-in-Time-Historie – DATA_GAP, nie simuliert",
            "created_at": now}))
    return out


# ── Forschungsdossier je Idee (Dokumentationspflicht, rein beschreibend) ───────────────────────────
def dossier(h: dict) -> dict:
    """Was, warum, womit, wann verfügbar, gegen was, Falsifikation, Datenlage, Aufwand und Neuheit – aus den
    Feldern der Idee abgeleitet (keine neuen Annahmen, keine Zahlen außer gemessenen)."""
    r = h.get("readiness") or {}
    feats = list(h.get("domain_features") or [])
    gaps = [r["reason"]] if r.get("status") == "DATA_GAP" and r.get("reason") else []
    gaps += [f for f in r.get("flags") or []]
    sig = h.get("signal")
    return {
        "research_question": (f"Sagt {sig} (Richtung {int(h.get('direction', 1)):+d}) die {h.get('primary_metric')} "
                              f"gegenüber {h.get('control_group')} voraus?" if sig else
                              f"Offene Frage ({h.get('title')}): ohne Point-in-Time-Daten nicht testbar."),
        "mechanism": h.get("mechanism"), "mechanism_source": h.get("mechanism_source"),
        "data": {"features": feats, "available_at": h.get("lag"),
                 "point_in_time": not h.get("non_pit_mapping") and not any("non_pit" in f for f in gaps)},
        "population": h.get("population"), "direction": h.get("direction"), "lag": h.get("lag"),
        "horizon_trading_days": h.get("horizon"),
        "comparison_model": h.get("baseline") or f"{h.get('control_group')}; zusätzlich Ablation gegen alle "
                                                 "bestehenden Merkmale (Robustheits-Batterie)",
        "primary_metric": h.get("primary_metric"), "falsification": h.get("failure_condition"),
        "data_quality": r.get("data_quality"), "coverage_2016plus": r.get("coverage_2016plus"), "data_gaps": gaps,
        "effort_cost": {"relative_cost": r.get("cost"), "api_calls": 0, "llm_calls": 0, "paid_data": False,
                        "compute": "eine Walk-Forward-Auswertung im bestehenden Research-Lab"},
        "novelty": h.get("novelty"),
        "selection_evidence": ("Auswahlfenster " + str((h.get("anomaly") or {}).get("window"))
                               + " – nur Priorisierung, kein Beleg") if h.get("anomaly") else None,
    }


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
    from modules.commodity_intelligence import EXPOSURE_COLUMNS, SIGNAL_DATE_FEATURES
    known_cmd = set(EXPOSURE_COLUMNS) | set(SIGNAL_DATE_FEATURES)
    unknown = [f for f in feats if f not in ml.ALL_FEATURES and f not in ALT_FEATURES and f not in SECTOR_EXPOSURES
               and f not in known_cmd]
    if unknown:
        return {"status": "DATA_GAP", "reason": f"Merkmale unbekannt: {unknown}", "free_sources": h.get("free_sources") or [],
                "data_quality": 0.0, "cost": None}
    quality, cost, flags = dq["pit_panel"], protocol["priority"]["cost"]["pit_panel"], []
    if any(f in ALT_FEATURES for f in feats):
        quality, cost = dq["alt_feature"], protocol["priority"]["cost"]["alt_feature"]
    if any(f in SECTOR_EXPOSURES for f in feats):
        quality = min(quality, dq["non_pit_mapping"])
        flags.append("non_pit_mapping: Sektorzuordnung von heute (Mapping MEDIUM)")
    if any(f in known_cmd or f.startswith("cmdx_") for f in feats):
        quality, cost = min(quality, dq["non_pit_mapping"]), protocol["priority"]["cost"]["alt_feature"]
        flags.append("non_pit_mapping: Commodity-Exposure-Mapping von heute (NON_PIT)")
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
PLAUSIBILITY = {"measured_condition": 0.75}      # gemessen, aber ökonomischer Mechanismus ausdrücklich offen
PROSPECTIVE_NEGATIVE = ("DEMOTED", "EXPIRED")    # prospektiv widerlegt (Promotion-Kette)


def independent_evidence(h: dict, memory: list[dict], threshold: float = 0.5) -> dict:
    """Vorhandene UNABHÄNGIGE (prospektive) Evidenz zu ähnlichen Ideen im Research Memory: Anzahl bestätigter und
    widerlegter Einträge. Widerlegte ohne bestätigte senken die Priorität (keine sinnlose Wiederholung)."""
    hits = rm.search(h, memory, threshold) if h.get("signal") else []
    pos = sorted(k for k, st, _s in hits if st in rm.PROSPECTIVE_STATES)
    neg = sorted(k for k, st, _s in hits if st in PROSPECTIVE_NEGATIVE)
    return {"prospective_positive": len(pos), "prospective_negative": len(neg), "related": (pos + neg)[:5]}


def priority(h: dict, dirs: dict, memory_max_sim: float, protocol: dict) -> float:
    a0, b0 = protocol["priority"]["prior_success"]
    d = dirs.get(f"family:{h['family']}") or {}
    if not d.get("tested") and h.get("source_id"):       # neue Familie: Erfahrung mit der Quelle (Meta-Learning)
        d = dirs.get(f"research:alt_data:{h['source_id']}") or d
    if not d.get("tested") and h.get("idea_source"):     # sonst: Erfahrung mit dem Ideentyp (hierarchisch)
        d = dirs.get(f"idea_source:{h['idea_source']}") or d
    mean = d.get("posterior_success", a0 / (a0 + b0))
    sd = d.get("posterior_sd", math.sqrt(a0 * b0 / ((a0 + b0) ** 2 * (a0 + b0 + 1))))
    eig = mean + sd                                       # optimistisch unter Unsicherheit (UCB)
    novelty = max(0.0, 1.0 - memory_max_sim)
    r = h["readiness"]
    if r["status"] != "READY" or not r.get("cost"):
        h["priority_components"] = {"status": r["status"]}
        return 0.0
    # Ergänzung 2026-11-01 (nur Reihenfolge, nie Budget, Prüfkette oder BH): Faktoren <= 1, ohne Angabe 1.
    plaus = PLAUSIBILITY.get(h.get("mechanism_source"), 1.0)          # Mechanismus offen -> nachrangig
    cov = r.get("coverage_2016plus")
    testab = 1.0 if cov is None else round(min(1.0, cov / ANOMALY_MIN_COVERAGE), 3)
    ev = h.get("independent_evidence") or {}
    contra = 0.5 if ev.get("prospective_negative") and not ev.get("prospective_positive") else 1.0
    h["priority_components"] = {"eig": round(eig, 4), "relevance": h["relevance"], "novelty": round(novelty, 3),
                                "data_quality": r["data_quality"], "cost": r["cost"], "plausibility": plaus,
                                "testability": testab, "independent_evidence": contra,
                                "role": "nur Priorisierung – historischer Erfolg ist kein unabhängiger Alpha-Nachweis"}
    return round(eig * h["relevance"] * novelty * r["data_quality"] / r["cost"] * plaus * testab * contra, 5)


def plan(panel: pd.DataFrame | None = None, protocol: dict | None = None, domains: dict | None = None,
         memory: list[dict] | None = None, machine_state: dict | None = None, now: datetime | None = None,
         conditions: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    domains = domains or load_domains()
    now = now or datetime.now(timezone.utc)
    now_s = now.isoformat(timespec="seconds")
    memory = memory if memory is not None else rm.load()
    dirs = rm.directions(memory, tuple(protocol["priority"]["prior_success"]))
    b = protocol["budget"]
    if conditions is None and SOURCE_CONDITIONS.exists():
        try:
            conditions = json.loads(SOURCE_CONDITIONS.read_text())
        except (OSError, ValueError):
            conditions = None
    sys_state = None
    try:
        from modules import system_state as _ss
        sys_state = json.loads(Path(_ss.STATE).read_text())
    except (OSError, ValueError, ImportError):
        sys_state = None
    ideas = (ideas_cross_domain(domains, now_s) + ideas_divergence(now_s) + ideas_drift(machine_state, now_s)
             + ideas_open_questions(sys_state, now_s)
             + ideas_source_conditions(conditions, now_s))
    try:                         # ein Fehler im Scan stoppt nicht die übrigen Ideen, wird aber sichtbar (main -> rot)
        scan = scan_anomalies(panel, known_signal_keys(ideas, memory)) if panel is not None else \
            {"version": ANOMALY_VERSION, "status": "NO_PANEL", "n_scanned": 0, "survivors": [], "data_gaps": {}}
    except Exception as exc:  # noqa: BLE001
        log.error("Fabrik: Anomalie-Scan fehlgeschlagen (%s: %s)", type(exc).__name__, exc)
        scan = {"version": ANOMALY_VERSION, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                "n_scanned": 0, "survivors": [], "data_gaps": {}}
    ideas += ideas_anomalies(scan, now_s)
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
        h["independent_evidence"] = independent_evidence(h, memory)
        h["priority"] = priority(h, dirs, sim, protocol)
        h["dossier"] = dossier(h)
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
            "anomaly_scan": {k: v for k, v in scan.items() if k != "survivors"} | {
                "survivor_ids": [h["id"] for h in ideas if h.get("idea_source") == "anomaly" and h.get("signal")]},
            "directions_used": {k: v for k, v in dirs.items() if k.startswith("family:")}}


def known_signal_keys(ideas: list[dict], memory: list[dict]) -> set[str]:
    """AST-Schlüssel bereits bekannter Signale: Ideen dieses Laufs, getestete Einträge im Research Memory und der
    Discovery-Suchraum des Research-Labs – ein bekanntes Signal wird nicht unter neuem Namen erneut gescannt."""
    from modules import research_lab as rl
    exprs = [h.get("signal") for h in ideas] + rl.discovery_candidates()
    exprs += [(e.get("spec") or {}).get("signal") for e in rm.latest(memory).values() if e.get("status") in rm.TESTED]
    keys = set()
    for x in filter(None, exprs):
        try:
            keys.add(rl.signal_key(x))
        except (rl.SignalError, SyntaxError, TypeError):
            continue
    return keys


def lab_hypotheses(plan_res: dict) -> list[dict]:
    keep = ("id", "title", "statement", "signal", "direction", "rationale", "created_at", "family", "domain",
            "exposure_sector", "spec_hash", "H0", "H_alt", "population", "control_group", "failure_condition",
            "idea_source", "anomaly")
    return [{k: h.get(k) for k in keep} for h in plan_res["ideas"] if h["plan_status"] == "SELECTED"]


# ── Optional: LLM formuliert NUR den Mechanismus ────────────────────────────
def llm_mechanism(h: dict, client=None) -> str | None:
    """Mechanismus-Text (max. 3 Sätze) – ohne Zahlen/Evidenz. Fehler/Zahlen -> None (Vorlage bleibt)."""
    if client is None and not cost_telemetry.allow("hypothesis_factory"):
        return None                 # Kosten-Guard: Research zuerst gedrosselt; Vorlage bleibt
    try:
        import anthropic
        client = client or anthropic.Anthropic()
        resp = cost_telemetry.tracked_create(
            client, workflow="hypothesis_factory", stage="llm_mechanism",
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
    # Ablation: Residual-IC nach allen bestehenden Querschnitts-Merkmalen; Signale ohne Sektor-Bezug zusätzlich
    # nach Sektor (2026-11-01, nur strenger: ein reiner Sektor-Effekt bekommt keinen Zusatznutzen). Sektor-Timing-
    # Hypothesen (exp_*) würden durch die Sektor-Dummies je Stichtag vollständig "erklärt" – dort wäre das falsch.
    xs = [f for f in ml.ALL_FEATURES if f in d and d.groupby("date")[f].nunique().median() > 1]
    sector_scoped = bool(h.get("exposure_sector")) or "exp_" in str(h.get("signal"))
    sec = [] if sector_scoped else [c for c in rl.SECTOR_EXPOSURES
                                    if c in d and d.groupby("date")[c].nunique().median() > 1]
    ic = _residual_ic(d, xs + sec)
    out["residual_ic"] = ic
    out["ablation_controls"] = {"features": len(xs), "sector_dummies": len(sec)}
    if (ic.get("t_months") or 0) < rb["residual_ic_min_t"]:
        fails.append(f"kein inkrementeller Gehalt nach bestehenden Merkmalen (Residual-IC t={ic.get('t_months')})")
    # Welche bekannte Faktorgruppe erklärt den Effekt allein? (beschreibend; Regime: Lab-Regime-Verdikte)
    raw_t = ml.ic_stats(ml.daily_ic(d, "score")).get("t_months")
    groups = {k: [c for c in v if c in d and (k == "sector" or c in xs)] for k, v in KNOWN_FACTOR_GROUPS.items()}
    groups["sector"] = sec
    expl = {}
    for k, cols in groups.items():
        if cols and (raw_t or 0) >= rb["residual_ic_min_t"]:
            t_k = _residual_ic(d, cols).get("t_months")
            expl[k] = {"residual_t": t_k, "explains": (t_k or 0) < rb["residual_ic_min_t"]}
    out["explained_by"] = {"raw_ic_t": raw_t, "groups": expl,
                           "regime": "Research-Lab: Regime-Verdikte (mindestens 3 von 4 Regimen gleiches Vorzeichen)"}
    return {"tests": out, "robust": not fails, "reasons": fails}


KNOWN_FACTOR_GROUPS = {
    "momentum": ("mom_12_1", "mom_3m", "rev_1m", "ret_5d", "rs_63", "dist_52w_high"),
    "volatility": ("vol_20", "vol_60", "vol_ratio", "max_ret_21", "beta_126"),
    "size_liquidity": ("log_dollar_vol", "relvol_5_60"),
    "sector": (),                                   # Sektor-Dummies (nur für Signale ohne Sektor-Bezug)
}


def _residual_ic(d: pd.DataFrame, cols: list[str]) -> dict:
    """Rang-IC des Signal-Residuums nach Querschnitts-Regression auf `cols` je Stichtag."""
    from modules import ml_research as ml
    resid = pd.Series(np.nan, index=d.index)
    for _, g in d.groupby("date"):
        g = g[["score"] + cols].dropna()
        if len(g) < 30 or g["score"].nunique() < 2:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].rank().to_numpy() for c in cols])
        y = g["score"].rank().to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        if r.std() > 1e-6 * y.std():                  # sonst vollständig erklärt: kein eigener Gehalt (NaN)
            resid.loc[g.index] = r
    return ml.ic_stats(ml.daily_ic(d.assign(resid=resid), "resid"))


def _next_monday(now: datetime) -> str:
    d = (now + timedelta(days=(7 - now.weekday()) % 7 or 7)).date()
    return d.isoformat()                              # Datum wie alt_hypotheses.yaml (Panel-Daten tz-naiv)


# Historische Prüfung nach adaptiver Auswahl ist explorativ; das Locked-Holdout ist kontaminiert (research_protocol
# locked_status) und kann nur ablehnen, nie bestätigen. Unabhängig bestätigt wird ausschließlich prospektiv.
EVIDENCE_CLASS = {"evidence_class": "EXPLORATORY_HISTORICAL",
                  "locked_holdout": "CONTAMINATED – nur informativ, keine unabhängige Bestätigung",
                  "independent_confirmation": "nur prospektiv (Forward-Ledger ab Registrierung, Alpha→Options)"}


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
        lab = rl.canonical_status(rec) if rec else "NOT_TESTED"     # nie das gespeicherte Feld allein (Textsperre)
        spec = {k: h.get(k) for k in ("signal", "direction", "family", "domain", "exposure_sector", "spec_hash",
                                      "idea_source", "exploratory", "priority", "novelty", "commodity",
                                      "equity_population", "mechanism", "domain_features", "non_pit_mapping",
                                      "anomaly_type")}
        # A contaminated historic screen may still *select* a prospectively
        # registered challenger; it cannot independently CONFIRM one.
        historical_screen = (rec.get("lab_screen_passed") is True
                             or rec.get("status") == "accepted")
        if not (historical_screen and lab == "EXPLORATORY"):
            results[h["id"]] = {"status": "REJECTED" if lab == "REJECTED" else lab, "spec": spec,
                                "reasons": rec.get("reasons") or [f"Research-Lab: {lab}"],
                                "data_kind": "historical_walk_forward", **EVIDENCE_CLASS}
            continue
        bt = battery(panel, h, protocol)
        status = "PROSPECTIVE_CHALLENGER" if bt["robust"] else "NOT_ROBUST"
        # Robustness is a historical *selection* result, not a forward success.
        # The actual confirmation is made by the frozen forward ledger only.
        results[h["id"]] = {"status": status, "spec": spec, "tests": bt["tests"], "reasons": bt["reasons"],
                            "data_kind": "historical_walk_forward+robustness", **EVIDENCE_CLASS}
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
    ap.add_argument("cmd", choices=["plan", "evaluate", "forward"])
    ap.add_argument("--llm", action="store_true", help="Mechanismus-Text optional per LLM (nie Evidenz)")
    args = ap.parse_args(argv)
    if args.cmd == "forward":                   # wöchentlich: nur Forward-Ledger der eingefrorenen Challenger
        panel = _panel_from_cache()
        if panel is None:
            log.error("Fabrik forward: ML_PANEL_CACHE fehlt")
            return 1
        print(f"Fabrik forward: {record_forward(panel)} neue Forward-Stichtage")
        return 0
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
        print(f"Fabrik: {res['n_ideas']} Ideen {counts}; ausgewählt {res['selected']}; Anomalie-Scan "
              f"{res['anomaly_scan'].get('status')} ({res['anomaly_scan'].get('n_scanned')} Ausdrücke)")
        if res["anomaly_scan"].get("status") == "ERROR":       # übrige Ideen sind geschrieben; Stufe trotzdem rot
            rm.sync()
            rm.write_directions(rm.load())
            return 1
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
