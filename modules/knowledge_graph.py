"""
modules/knowledge_graph.py – versionierter Knowledge Graph (NUR SHADOW, Phase 4)

    python -m modules.knowledge_graph        (CI: Validierung braucht Netzwerk über world_model)

Evidenz-/Widerspruchs-Quelle für den High-Confidence-Scanner: welche Firmen,
Sektoren, Industrien und Makro-Indikatoren hängen wie zusammen – und wie gut ist
jede Aussage belegt?

Regeln
  * KEINE erfundenen Kanten. Jede Kante trägt source (Datei/Modul), timestamp,
    confidence (HIGH/MEDIUM/LOW), evidence_type (curated_config | measured_oos |
    measured_insample | reference_data), version, uncertain. Ohne Quelle/Evidenz
    wirft der Builder einen Fehler.
  * Quellen: outputs/research/sector_map.json (Firma -> Sektor, reference_data),
    config/industry_exposure.yaml + config/weather_exposures.yaml (Relevanz-Kategorien
    -> exposed_to, curated_config; LOW => uncertain), config/port_universe.yaml
    (Ports/Routen, nur Knoten + Zugehörigkeit), outputs/research/causal_research.json
    (Indikator -> Sektor-ETF, measured_oos, Konfidenz aus causal_confidence; nur
    Beziehungen ab predictive_relationship). Keine LLM-Kanten.
  * Relevanz-Kategorien tragen keine Richtung (sign=None); nur gemessene Kanten haben ein Vorzeichen.
  * Version = Hash des Inhalts (ohne Zeitstempel). Jede neue Version wird append-only in
    outputs/research/knowledge_graph_versions.jsonl abgelegt.

Validierung (config/intelligence_protocol.yaml, knowledge_graph): "Historical evidence
propagation" – hilft der KG-Indikator-Schock über causal_research.sector_tilts hinaus?
Walk-Forward-Rang-IC-Differenz mit Monats-Block-Bootstrap; KEEP / MODIFY / REJECT nach
vorab festgelegter Regel. Da die Messkanten aus causal_research stammen, ist der
erwartete Zusatznutzen gering.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from modules import causal_research as cr

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "config" / "intelligence_protocol.yaml"
KP = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))["knowledge_graph"]
OUT_DIR = Path("outputs/research")
KG_JSON = OUT_DIR / "knowledge_graph.json"
KG_VERSIONS = OUT_DIR / "knowledge_graph_versions.jsonl"
VALIDATION_JSON = OUT_DIR / "knowledge_graph_validation.json"

EVIDENCE_TYPES = ("curated_config", "measured_oos", "measured_insample", "reference_data")
CONFIDENCES = ("HIGH", "MEDIUM", "LOW")
NODE_TYPES = ("Company", "Sector", "Industry", "Macro variable", "Economic indicator", "Commodity",
              "Currency", "Index", "Port", "Route", "Theme")
# Kantentyp -> Richtung der Wirkungsausbreitung relativ zur gespeicherten Richtung (from -> to)
PROPAGATION = {
    "historically_leads": "forward", "predictive_of": "forward",          # Indikator -> Index
    "represents_sector": "forward",                                      # Index -> Sektor
    "belongs_to_sector": "reverse",                                      # Sektor wirkt auf seine Firmen
    "maps_to_industry": "reverse",                                       # Industrie wirkt auf Sektor
    "exposed_to": "reverse",                                             # Thema wirkt auf Exponierte
}
MEASURED = ("measured_oos", "measured_insample")
INDICATOR_TYPES = {"wti_63d": "Commodity", "copper_63d": "Commodity", "copper_gold_63d": "Commodity",
                   "usd_63d": "Currency"}


# ── Aufbau ───────────────────────────────────────────────────────────────────

class _Builder:
    def __init__(self, now: str):
        self.now = now
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict] = {}

    def node(self, nid: str, ntype: str, label: str | None = None, **attrs) -> str:
        if ntype not in NODE_TYPES:
            raise ValueError(f"unbekannter Knotentyp {ntype}")
        self.nodes.setdefault(nid, {"id": nid, "type": ntype, "label": label or nid.split(":", 1)[-1], "attrs": attrs})
        return nid

    def edge(self, etype: str, frm: str, to: str, *, source: str, confidence: str, evidence_type: str,
             uncertain: bool, sign: int | None = None, timestamp: str | None = None, key: str = "", **attrs) -> None:
        if not source:
            raise ValueError("Kante ohne Quelle")
        if evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"Kante ohne gültigen evidence_type: {evidence_type}")
        if confidence not in CONFIDENCES:
            raise ValueError(f"Kante ohne gültige confidence: {confidence}")
        if frm not in self.nodes or to not in self.nodes:
            raise ValueError(f"Kante {etype} {frm}->{to}: Knoten fehlt")
        eid = f"{etype}|{frm}|{to}|{key}"
        self.edges[eid] = {"id": eid, "type": etype, "from": frm, "to": to, "source": source,
                           "timestamp": timestamp or self.now, "confidence": confidence,
                           "evidence_type": evidence_type, "version": None, "uncertain": bool(uncertain),
                           "sign": sign, "attrs": attrs,
                           # Audit P2-3: nur gemessene Kanten haben einen belegten Gültigkeitsbeginn;
                           # Referenz-/Config-Kanten sind HEUTIGER Stand und nicht point-in-time.
                           "point_in_time": evidence_type == "measured_oos",
                           "valid_from": (timestamp or self.now) if evidence_type == "measured_oos" else None}


def content_hash(nodes: dict, edges: dict) -> str:
    """Hash des Inhalts – ohne timestamp/version (sonst würde jeder Neuaufbau eine neue Version erzeugen)."""
    core = {"nodes": {k: nodes[k] for k in sorted(nodes)},
            "edges": {k: {f: v for f, v in edges[k].items() if f not in ("timestamp", "version")} for k in sorted(edges)}}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_graph(sector_map: dict | None = None, industry_exposure: dict | None = None,
                weather_exposures: dict | None = None, causal: dict | None = None,
                port_universe: dict | None = None, now: str | None = None) -> dict:
    """Baut den Graphen nur aus den übergebenen Quellen. -> {"version","generated","schema","nodes","edges"}."""
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    b = _Builder(now)
    r2c = KP["relevance_to_confidence"]

    # Sektor-ETF (Index) <-> Sektor: Referenz aus causal_research.SECTOR_ETF_MAP
    for etf, sec in cr.SECTOR_ETF_MAP.items():
        b.node(f"Index:{etf}", "Index", etf)
        b.node(f"Sector:{sec}", "Sector", sec)
        b.edge("represents_sector", f"Index:{etf}", f"Sector:{sec}", source="modules/causal_research.py:SECTOR_ETF_MAP",
               confidence="HIGH", evidence_type="reference_data", uncertain=False)

    for tk, sec in (sector_map or {}).items():
        if not sec:
            continue
        b.node(f"Sector:{sec}", "Sector", sec)
        b.node(f"Company:{tk}", "Company", tk)
        b.edge("belongs_to_sector", f"Company:{tk}", f"Sector:{sec}", source="outputs/research/sector_map.json",
               confidence="HIGH", evidence_type="reference_data", uncertain=False)

    if industry_exposure:
        src = "config/industry_exposure.yaml"
        for ind_name, spec in (industry_exposure.get("industries") or {}).items():
            b.node(f"Industry:{ind_name}", "Industry", ind_name)
            for k, rel in (spec or {}).items():
                if not k.endswith("_relevance") or rel not in r2c:          # NONE/unbekannt -> keine Kante
                    continue
                theme = k[: -len("_relevance")]
                b.node(f"Theme:{theme}", "Theme", theme)
                b.edge("exposed_to", f"Industry:{ind_name}", f"Theme:{theme}", source=src, confidence=r2c[rel],
                       evidence_type="curated_config", uncertain=(rel == "LOW"), relevance=rel)
        for sec, ind_name in (industry_exposure.get("sector_fallback") or {}).items():
            if f"Industry:{ind_name}" not in b.nodes:
                log.warning(f"knowledge_graph: sector_fallback {sec}->{ind_name}: Industrie unbekannt, Kante übersprungen")
                continue
            b.node(f"Sector:{sec}", "Sector", sec)
            b.edge("maps_to_industry", f"Sector:{sec}", f"Industry:{ind_name}", source=src, confidence="LOW",
                   evidence_type="curated_config", uncertain=True, note="grober Sektor-Fallback")
        for ind_name, groups in (industry_exposure.get("portwatch_industry_map") or {}).items():
            if f"Industry:{ind_name}" not in b.nodes:
                continue
            for g in groups or []:
                b.node(f"Theme:maritime:{g}", "Theme", f"maritime:{g}")
                b.edge("exposed_to", f"Industry:{ind_name}", f"Theme:maritime:{g}", source=src, confidence="LOW",
                       evidence_type="curated_config", uncertain=True, note="PortWatch-Industriezuordnung, ungeprüft")

    for tk, spec in ((weather_exposures or {}).get("ticker_overrides") or {}).items():
        if not (spec or {}).get("source"):                                  # Regel der Config: ohne Beleg keine Aufnahme
            continue
        b.node("Theme:weather", "Theme", "weather")
        b.node(f"Company:{tk}", "Company", tk)
        b.edge("exposed_to", f"Company:{tk}", "Theme:weather", source="config/weather_exposures.yaml",
               confidence="MEDIUM", evidence_type="curated_config", uncertain=False,
               exposure_type=spec.get("exposure_type"), hub_codes=spec.get("hub_codes"))

    if port_universe:
        src = "config/port_universe.yaml"
        for grp, ports in (port_universe.get("groups") or {}).items():
            b.node(f"Route:{grp}", "Route", grp)
            for p in ports or []:
                b.node(f"Port:{p['name']}", "Port", p["name"], country=p.get("country"))
                b.edge("part_of_route", f"Port:{p['name']}", f"Route:{grp}", source=src, confidence="HIGH",
                       evidence_type="reference_data", uncertain=False)
        for c in port_universe.get("chokepoints") or []:
            b.node(f"Route:{c['slug']}", "Route", c["name"], chokepoint=True)

    if causal:
        stamp = causal.get("generated")
        for r in causal.get("relations") or []:
            etf, drv = r.get("target"), r.get("driver")
            if r.get("level") not in KP["causal_edge_levels"] or etf not in cr.SECTOR_ETF_MAP or not r.get("observed_sign"):
                continue
            conf = KP["causal_confidence_to_confidence"].get(r["evidence"]["causal_confidence"], "LOW")
            etype = "predictive_of" if r["level"] == "predictive_relationship" else "historically_leads"
            b.node(f"Indicator:{drv}", INDICATOR_TYPES.get(drv, "Economic indicator"), drv)
            b.edge(etype, f"Indicator:{drv}", f"Index:{etf}", source="outputs/research/causal_research.json",
                   confidence=conf, evidence_type="measured_oos", uncertain=(conf == "LOW"),
                   sign=int(r["observed_sign"]), timestamp=stamp, key=str(r["horizon_weeks"]),
                   horizon_weeks=r["horizon_weeks"], level=r["level"], relation_id=r.get("id"),
                   q_bh=(r.get("stats") or {}).get("q_bh"), causal_evidence=r.get("causal_evidence", "none"))

    version = content_hash(b.nodes, b.edges)
    for e in b.edges.values():
        e["version"] = version
    return {"version": version, "generated": now, "schema": KP["version"], "nodes": b.nodes, "edges": list(b.edges.values())}


def validate_graph(graph: dict) -> list[str]:
    """Liste von Verstößen (leer = ok): jede Kante mit Quelle, gültigem evidence_type/confidence, bekannten Knoten."""
    bad = []
    for e in graph["edges"]:
        if not e.get("source") or e.get("evidence_type") not in EVIDENCE_TYPES or e.get("confidence") not in CONFIDENCES:
            bad.append(f"{e.get('id')}: Quelle/Evidenz/Konfidenz fehlt")
        if e["from"] not in graph["nodes"] or e["to"] not in graph["nodes"]:
            bad.append(f"{e.get('id')}: Knoten fehlt")
        if e.get("evidence_type") in MEASURED and e.get("sign") not in (-1, 1):
            bad.append(f"{e.get('id')}: gemessene Kante ohne Vorzeichen")
    return bad


# ── Speichern / Versionen ────────────────────────────────────────────────────

def save(graph: dict, out_dir: Path | str = OUT_DIR) -> dict:
    """Schreibt knowledge_graph.json; hängt die Version (voller Graph) append-only an
    knowledge_graph_versions.jsonl an, falls dieser Hash dort noch nicht steht."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "knowledge_graph.json").write_text(json.dumps(graph, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    vf = out / "knowledge_graph_versions.jsonl"
    known = {v["version"] for v in _read_versions(vf)}
    new = graph["version"] not in known
    if new:
        rec = {"version": graph["version"], "generated": graph["generated"], "n_nodes": len(graph["nodes"]),
               "n_edges": len(graph["edges"]), "graph": graph}
        with open(vf, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return {"version": graph["version"], "new_version": new, "path": str(out / "knowledge_graph.json")}


def _read_versions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("knowledge_graph: defekte Zeile in der Versionsliste übersprungen")
    return out


def load(path: Path | str = KG_JSON) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_version(version: str, out_dir: Path | str = OUT_DIR) -> dict | None:
    for v in _read_versions(Path(out_dir) / "knowledge_graph_versions.jsonl"):
        if v["version"] == version:
            return v["graph"]
    return None


# ── Abfragen ─────────────────────────────────────────────────────────────────

def _label(score: float) -> str:
    m = KP["confidence_label_min"]
    return "HIGH" if score >= m["HIGH"] else "MEDIUM" if score >= m["MEDIUM"] else "LOW"


def _mul(a, b):
    return None if a is None or b is None else a * b


def propagate(graph: dict, node_id: str, max_depth: int = 3) -> list[dict]:
    """Von node_id betroffene Knoten (Tiefe <= max_depth) mit bestem Pfad. Kombinierte Konfidenz =
    Produkt der Kanten-Gewichte (KP.confidence_weight); sign = Produkt der Vorzeichen (None, wenn ein
    Pfadstück keine Richtung hat). Einfache Pfade, kein Zyklus."""
    if node_id not in graph["nodes"]:
        return []
    w = KP["confidence_weight"]
    adj: dict[str, list] = {}
    for e in graph["edges"]:
        d = PROPAGATION.get(e["type"])
        if d == "forward":
            adj.setdefault(e["from"], []).append((e["to"], e))
        elif d == "reverse":
            adj.setdefault(e["to"], []).append((e["from"], e))
    best: dict[str, dict] = {}

    def dfs(cur, path, edges, score, sign, depth):
        for nxt, e in adj.get(cur, []):
            if nxt in path:
                continue
            sc = score * w[e["confidence"]]
            sg = _mul(sign, e["sign"] if e.get("sign") is not None else (1 if e["type"] in MEASURED_OR_STRUCT else None))
            p, es = path + [nxt], edges + [e]
            cand = {"node": nxt, "type": graph["nodes"][nxt]["type"], "label": graph["nodes"][nxt]["label"],
                    "depth": depth + 1, "path": p, "edge_types": [x["type"] for x in es],
                    "confidence_score": round(sc, 4), "confidence": _label(sc), "sign": sg,
                    "uncertain": any(x["uncertain"] for x in es), "evidence_types": sorted({x["evidence_type"] for x in es}),
                    "sources": sorted({x["source"] for x in es})}
            cur_best = best.get(nxt)
            if cur_best is None or (sc, -cand["depth"]) > (cur_best["confidence_score_raw"], -cur_best["depth"]):
                cand["confidence_score_raw"] = sc
                best[nxt] = cand
            if depth + 1 < max_depth:
                dfs(nxt, p, es, sc, sg, depth + 1)

    dfs(node_id, [node_id], [], 1.0, 1, 0)
    res = []
    for c in best.values():
        c.pop("confidence_score_raw", None)
        res.append(c)
    return sorted(res, key=lambda c: (-c["confidence_score"], c["depth"], c["node"]))


# Kantentypen, die ohne explizites Vorzeichen strukturell "+1" (gleiche Richtung) bedeuten
MEASURED_OR_STRUCT = ("represents_sector", "belongs_to_sector")


def _edges_into(graph: dict, node_id: str, types: tuple) -> list[dict]:
    return [e for e in graph["edges"] if e["to"] == node_id and e["type"] in types]


def find_contradictions(graph: dict, target: str) -> list[dict]:
    """Widerspruch = zwei gemessene Kanten mit entgegengesetztem Vorzeichen auf dieselbe Ziel-Entität."""
    meas = [e for e in _edges_into(graph, target, ("historically_leads", "predictive_of")) if e["evidence_type"] in MEASURED]
    out = []
    for i, a in enumerate(meas):
        for b in meas[i + 1:]:
            if a["sign"] * b["sign"] < 0:
                out.append({"target": target, "edges": [_edge_brief(a), _edge_brief(b)],
                            "note": "entgegengesetzte gemessene Vorzeichen auf dieselbe Ziel-Entität"})
    return out


def _edge_brief(e: dict) -> dict:
    return {"id": e["id"], "indicator": e["from"].split(":", 1)[-1], "type": e["type"], "sign": e["sign"],
            "horizon_weeks": e["attrs"].get("horizon_weeks"), "level": e["attrs"].get("level"),
            "confidence": e["confidence"], "evidence_type": e["evidence_type"], "uncertain": e["uncertain"],
            "source": e["source"]}


def ticker_evidence(graph: dict, ticker: str) -> dict:
    """{"exposures", "leading_indicators", "contradictions"} für eine Firma (leer, wenn unbekannt)."""
    cid = f"Company:{ticker}"
    res = {"exposures": [], "leading_indicators": [], "contradictions": []}
    if cid not in graph["nodes"]:
        return res
    w = KP["confidence_weight"]
    out_edges: dict[str, list] = {}
    for e in graph["edges"]:
        if e["type"] in ("belongs_to_sector", "maps_to_industry", "exposed_to"):
            out_edges.setdefault(e["from"], []).append(e)

    def walk(cur, path, es, score):
        for e in out_edges.get(cur, []):
            if e["to"] in path:
                continue
            sc, p, x = score * w[e["confidence"]], path + [e["to"]], es + [e]
            if graph["nodes"][e["to"]]["type"] == "Theme":
                res["exposures"].append({"theme": graph["nodes"][e["to"]]["label"], "path": p,
                                         "confidence_score": round(sc, 4), "confidence": _label(sc),
                                         "uncertain": any(y["uncertain"] for y in x),
                                         "evidence_types": sorted({y["evidence_type"] for y in x}),
                                         "sources": sorted({y["source"] for y in x})})
            else:
                walk(e["to"], p, x, sc)
    walk(cid, [cid], [], 1.0)
    res["exposures"].sort(key=lambda r: (-r["confidence_score"], r["theme"]))
    sectors = [e["to"] for e in graph["edges"] if e["from"] == cid and e["type"] == "belongs_to_sector"]
    for s in sectors:
        for rep in _edges_into(graph, s, ("represents_sector",)):
            res["leading_indicators"] += [_edge_brief(e) | {"target": rep["from"]}
                                          for e in _edges_into(graph, rep["from"], ("historically_leads", "predictive_of"))]
            res["contradictions"] += find_contradictions(graph, rep["from"])
    return res


def event_impact(graph: dict, indicator: str, direction) -> list[dict]:
    """Indikator-Schock ("up"/"down" bzw. +1/-1) -> betroffene Sektoren/Titel mit erwarteter RELATIVER Richtung
    (gegenüber SPY: outperform/underperform/unknown) und Evidenz."""
    d = {"up": 1, "down": -1, 1: 1, -1: -1}.get(direction)
    if d is None:
        raise ValueError("direction muss up/down (oder +1/-1) sein")
    nid = indicator if indicator.startswith("Indicator:") else f"Indicator:{indicator}"
    out = []
    for c in propagate(graph, nid, max_depth=3):
        if c["type"] not in ("Sector", "Index", "Company"):
            continue
        s = None if c["sign"] is None else c["sign"] * d
        out.append({"node": c["node"], "type": c["type"], "ticker": c["label"] if c["type"] == "Company" else None,
                    "expected_direction": "unknown" if s is None else ("outperform" if s > 0 else "underperform"),
                    "confidence": c["confidence"], "confidence_score": c["confidence_score"], "path": c["path"],
                    "evidence_types": c["evidence_types"], "uncertain": c["uncertain"], "sources": c["sources"]})
    return out


# ── Validierung: Historical evidence propagation ─────────────────────────────

def kg_scores(graph: dict, ind: pd.DataFrame, dates, vcfg: dict | None = None) -> pd.DataFrame:
    """KG-Score je (Stichtag, ETF) = Σ z_Indikator(t) × (Vorzeichen × Konfidenzgewicht der Propagation
    Indikator -> Index). z expandierend nur mit Vergangenheit."""
    v = {**KP["validation"], **(vcfg or {})}
    W: dict[str, dict[str, float]] = {}
    for nid, n in graph["nodes"].items():
        if n["type"] in ("Economic indicator", "Macro variable", "Commodity", "Currency") and nid.startswith("Indicator:"):
            for c in propagate(graph, nid, max_depth=1):
                if c["type"] == "Index" and c["sign"]:
                    W.setdefault(n["label"], {})[c["label"]] = c["sign"] * c["confidence_score"]
    cols = [c for c in W if c in ind.columns]
    mu = ind[cols].expanding(min_periods=v["indicator_min_history_weeks"]).mean().shift(1)
    sd = ind[cols].expanding(min_periods=v["indicator_min_history_weeks"]).std().shift(1)
    z = ((ind[cols] - mu) / sd).clip(-v["z_clip"], v["z_clip"]).fillna(0.0)
    rows = []
    for t in [pd.Timestamp(d) for d in dates if pd.Timestamp(d) in ind.index]:
        acc: dict[str, float] = {}
        for c in cols:
            for etf, wt in W[c].items():
                acc[etf] = acc.get(etf, 0.0) + float(z.at[t, c]) * wt
        rows += [(t, etf, s) for etf, s in acc.items()]
    return pd.DataFrame(rows, columns=["date", "sector_etf", "tilt"])


def _xs_z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def validate_propagation(graph: dict, ind: pd.DataFrame, px: pd.DataFrame, tilts: pd.DataFrame,
                         locked_from: pd.Timestamp, cfg: dict | None = None) -> dict:
    """Rang-IC (Sektor-Querschnitt) von Tilts, KG-Score und Kombination; Differenz kombiniert − Tilts mit
    Monats-Block-Bootstrap. Nur Ziele mit Ende vor dem Locked-Holdout."""
    v = {**KP["validation"], **(cfg or {})}
    if tilts is None or tilts.empty:
        return {"verdict": "REJECT", "verdict_reason": "keine Tilts (causal_research liefert keine Beziehungen)", "n_dates": 0}
    dates = sorted(tilts["date"].unique())
    kg = kg_scores(graph, ind, dates, v)
    fwd, _, end = cr._grid_returns(px, list(ind.index), v["eval_horizon_weeks"])
    T = tilts.pivot(index="date", columns="sector_etf", values="tilt")
    K = kg.pivot(index="date", columns="sector_etf", values="tilt") if not kg.empty else pd.DataFrame()
    ic = {"tilt": {}, "kg": {}, "combined": {}}
    for t in T.index:
        e_t = end.get(t)
        if t not in fwd.index or t not in K.index or e_t is None or pd.isna(e_t) or e_t >= locked_from:
            continue
        d = pd.concat([T.loc[t], K.loc[t], fwd.loc[t]], axis=1, keys=["tilt", "kg", "y"]).dropna()
        if len(d) < v["min_sectors"]:
            continue
        d["combined"] = _xs_z(d["tilt"]) + _xs_z(d["kg"])
        ry = d["y"].rank()
        for k in ic:
            ic[k][t] = d[k].rank().corr(ry)
    S = pd.DataFrame(ic).dropna()
    if S.empty:
        return {"verdict": "REJECT", "verdict_reason": "keine auswertbaren Stichtage", "n_dates": 0}
    seed, n, a = v["bootstrap_seed"], v["bootstrap_n"], v["bootstrap_alpha"]
    diff = month_diff = cr.month_block_bootstrap(S["combined"] - S["tilt"], n, seed, a)
    kgb = cr.month_block_bootstrap(S["kg"], n, seed, a)
    keep = diff.get("lo") is not None and diff["lo"] > 0
    modify = kgb.get("lo") is not None and kgb["lo"] > 0
    verdict = "KEEP" if keep else "MODIFY" if modify else "REJECT"
    why = ("kombinierte IC − Tilt-IC mit Bootstrap-Untergrenze > 0" if keep else
           "KG-Score allein informativ (Untergrenze > 0), aber ohne Zusatznutzen über die Tilts -> nur Evidenz-/Widerspruchsquelle"
           if modify else "kein Zusatznutzen und keine eigenständige Vorhersage -> KG bleibt Dokumentation")
    return {"verdict": verdict, "verdict_reason": why, "n_dates": int(len(S)),
            "ic_mean": {k: cr._f(float(S[k].mean())) for k in S}, "ic_diff_combined_minus_tilt": {
                "mean": cr._f(month_diff.get("mean")), "ci_lo": cr._f(month_diff.get("lo")), "ci_hi": cr._f(month_diff.get("hi"))},
            "kg_ic": {"mean": cr._f(kgb.get("mean")), "ci_lo": cr._f(kgb.get("lo")), "ci_hi": cr._f(kgb.get("hi"))},
            "caveat": "Kanten stammen aus causal_research (Auswahl nutzt Daten bis zum Locked-Holdout) -> optimistisch"}


# ── Lauf ─────────────────────────────────────────────────────────────────────

def load_inputs(root: Path | str = ".") -> dict:
    root = Path(root)
    spec = {"sector_map": ("outputs/research/sector_map.json", json.loads),
            "causal": ("outputs/research/causal_research.json", json.loads),
            "industry_exposure": ("config/industry_exposure.yaml", yaml.safe_load),
            "weather_exposures": ("config/weather_exposures.yaml", yaml.safe_load),
            "port_universe": ("config/port_universe.yaml", yaml.safe_load)}
    out = {}
    for k, (rel, parse) in spec.items():
        p = root / rel
        if not p.exists():
            log.warning(f"knowledge_graph: Quelle {rel} fehlt -> keine Kanten daraus")
            out[k] = None
            continue
        try:
            out[k] = parse(p.read_text(encoding="utf-8"))
        except (ValueError, yaml.YAMLError) as e:
            log.warning(f"knowledge_graph: Quelle {rel} nicht lesbar ({e}) -> keine Kanten daraus")
            out[k] = None
    return out


def run(validate_now: bool = True) -> dict:
    graph = build_graph(**load_inputs())
    bad = validate_graph(graph)
    if bad:
        raise ValueError(f"Knowledge Graph verletzt Kantenregeln: {bad[:5]}")
    info = save(graph)
    rep = {"generated": graph["generated"], "schema": KP["version"], "graph_version": graph["version"],
           "n_nodes": len(graph["nodes"]), "n_edges": len(graph["edges"]), "save": info,
           "edges_by_type": {t: sum(1 for e in graph["edges"] if e["type"] == t) for t in sorted({e["type"] for e in graph["edges"]})}}
    if validate_now:
        from modules import ml_research as ml
        from modules import world_model as wm
        causal = load_inputs()["causal"] or {}
        sel = [r for r in causal.get("relations", []) if r.get("level") in cr.CP["tilt"]["tilt_levels"]]
        px = wm.fetch_market()
        ind, _ = wm.build_world(px, wm.load_archive(), None)
        first = pd.Timestamp(year=KP["validation"]["first_test_year"], month=1, day=1)
        tilts = cr.sector_tilts(ind, px, sel, [d for d in ind.index if d >= first])
        rep["validation"] = validate_propagation(graph, ind, px, tilts, ml.LOCKED_FROM)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return rep


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
