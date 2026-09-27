"""
modules/external/reporting.py – Kompakte Darstellung des externen Kontexts
für die TÄGLICHEN Reports (Status-/Trade-Mail, Daily-Markdown).

Liest AUSSCHLIESSLICH bereits vorhandene, persistierte Daten:
  - outputs/external_data/snapshots/<YYYY-MM>/<id>.json  (neuester Snapshot)
  - outputs/external_data/health/source_health.json       (Source-Health)

Kein Netzwerk, kein Schreibzugriff, keine Gate-/Score-Logik — reine
Observability-Darstellung. Toleriert fehlende Snapshot/Health-Dateien
(z.B. weil die Ingestion-Integration in pipeline.py noch nicht auf diesem
Stand ist) und liefert dann einen expliziten "keine externen Daten"-Zustand,
statt zu werfen.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SNAPSHOT_ROOT = Path("outputs/external_data/snapshots")
DEFAULT_HEALTH_PATH   = Path("outputs/external_data/health/source_health.json")

CHOKEPOINT_PRIMITIVES = ("suez_z", "panama_z", "hormuz_z", "malacca_z", "bab_el_mandeb_z")
FREIGHT_STATE_KEYS = (
    ("us_freight_state", "us_freight_confidence", "US"),
    ("eu_freight_state", "eu_freight_confidence", "EU"),
    ("asia_freight_state", "asia_freight_confidence", "Asien"),
)


def _mode() -> str:
    try:
        from modules.config import cfg
        return str(getattr(getattr(cfg, "external_context", None), "mode", "shadow"))
    except Exception:
        return "shadow"


def load_latest_snapshot(root: Path | str = DEFAULT_SNAPSHOT_ROOT) -> dict | None:
    """Neuester Snapshot (nach Dateiname/mtime) unter snapshots/<YYYY-MM>/<id>.json.
    None wenn Ordner fehlt/leer/kaputt — nie ein Raise."""
    try:
        root = Path(root)
        if not root.exists():
            return None
        files = sorted(root.glob("*/*.json"), key=lambda p: (p.parent.name, p.name))
        if not files:
            return None
        data = json.loads(files[-1].read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def load_source_health(path: Path | str = DEFAULT_HEALTH_PATH) -> dict:
    try:
        path = Path(path)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def source_health_summary(health: dict) -> dict:
    """PASS/FAIL/STALE-Zähler + Liste der fehlschlagenden Quellen."""
    counts = {"PASS": 0, "FAIL": 0, "STALE": 0, "OTHER": 0}
    failing = []
    for source_id, h in (health or {}).items():
        if not isinstance(h, dict):
            continue
        status = h.get("status", "")
        staleness = h.get("staleness", "")
        if staleness == "STALE":
            counts["STALE"] += 1
        if status == "FAIL":
            counts["FAIL"] += 1
            failing.append(source_id)
        elif status == "PASS":
            counts["PASS"] += 1
        else:
            counts["OTHER"] += 1
    return {"counts": counts, "failing": sorted(failing)}


def chokepoint_anomalies(primitives: dict, threshold: float = 2.0) -> list[dict]:
    """Chokepoints mit |z| >= threshold."""
    out = []
    for key in CHOKEPOINT_PRIMITIVES:
        z = primitives.get(key)
        if isinstance(z, (int, float)) and abs(z) >= threshold:
            out.append({"key": key, "z": z})
    return out


def regional_freight_states(states: dict) -> list[dict]:
    out = []
    for state_key, conf_key, label in FREIGHT_STATE_KEYS:
        state = states.get(state_key)
        if state is None:
            continue
        out.append({"label": label, "state": state, "confidence": states.get(conf_key)})
    return out


def weather_exposure_summary(primitives: dict) -> dict:
    return {
        "active_tropical_system": bool(primitives.get("active_tropical_system")),
        "weather_disruption_index": primitives.get("weather_disruption_index"),
        "hdd_anomaly": primitives.get("hdd_anomaly"),
        "cdd_anomaly": primitives.get("cdd_anomaly"),
    }


def source_attributions(sources_dir: str = "config/external_sources") -> list[str]:
    """Pflicht-Quellenangaben aller aktivierten Quellen mit `attribution`-Feld
    in der Registry (z.B. IMF PortWatch). Fehler -> leere Liste."""
    try:
        from modules.external.registry import load_source_configs
        out = []
        for s in load_source_configs(sources_dir).values():
            if s.get("enabled") and s.get("attribution") and s["attribution"] not in out:
                out.append(s["attribution"])
        return out
    except Exception:
        return []


def build_compact_context(snapshot: dict | None = None, health: dict | None = None) -> dict | None:
    """
    Baut den kompakten Kontext für die tägliche Darstellung aus einem
    Snapshot-Dict + der Source-Health. None wenn kein Snapshot vorliegt
    (→ Aufrufer zeigt "keine externen Daten").
    """
    if snapshot is None:
        snapshot = load_latest_snapshot()
    if snapshot is None:
        return None
    health = health if health is not None else load_source_health()

    primitives = snapshot.get("primitives") or {}
    states = snapshot.get("states") or {}

    return {
        "mode": _mode(),
        "snapshot_id": snapshot.get("snapshot_id"),
        "available_at": snapshot.get("available_at") or snapshot.get("generated_at"),
        "source_health": source_health_summary(health),
        "regional_freight_states": regional_freight_states(states),
        "global_maritime_state": states.get("global_maritime_state"),
        "global_maritime_confidence": states.get("global_maritime_confidence"),
        "chokepoint_anomalies": chokepoint_anomalies(primitives),
        "weather": weather_exposure_summary(primitives),
        "attributions": source_attributions(),
        "candidates": [
            {
                "ticker": c.get("ticker"),
                "relation": (c.get("relation") or {}).get("relation") if isinstance(c.get("relation"), dict) else c.get("relation"),
                "materiality": (c.get("relation") or {}).get("materiality") if isinstance(c.get("relation"), dict) else c.get("materiality"),
                "exposure_source": (c.get("ticker_exposure") or {}).get("exposure_source") if isinstance(c.get("ticker_exposure"), dict) else c.get("exposure_source"),
            }
            for c in (snapshot.get("candidates") or [])
            if isinstance(c, dict) and c.get("ticker")
        ],
    }


def _shadow_header(mode: str) -> str:
    return ("SHADOW — NICHT in der Produktionsentscheidung verwendet"
            if mode == "shadow" else f"mode={mode}")


def render_html_block(context: dict | None) -> str:
    """Kompaktes HTML-Fragment für Status-/Trade-Mail. context=None → einzeilig
    'keine externen Daten'."""
    if not context:
        return (
            "<div style='margin-top:10px;padding:8px 14px;background:#f8fafc;"
            "border-radius:6px;font-size:11px;color:#94a3b8;'>🌍 Externer Kontext: "
            "keine externen Daten</div>"
        )
    header = _shadow_header(context["mode"])
    hs = context["source_health"]["counts"]
    failing = context["source_health"]["failing"]
    fail_str = f" ({', '.join(failing[:5])})" if failing else ""

    regional_sep = " · ".join(
        f"{r['label']}: {r['state']}"
        + (f" (Konf. {r['confidence']:.0%})" if isinstance(r.get("confidence"), (int, float)) else "")
        for r in context["regional_freight_states"]
    ) or "–"

    maritime = context.get("global_maritime_state") or "–"
    maritime_conf = context.get("global_maritime_confidence")
    maritime_str = f"{maritime}" + (f" (Konf. {maritime_conf:.0%})" if isinstance(maritime_conf, (int, float)) else "")

    chokepoints = context.get("chokepoint_anomalies") or []
    chokepoint_str = (
        ", ".join(f"{c['key']}={c['z']:+.1f}" for c in chokepoints) if chokepoints else "keine"
    )

    weather = context.get("weather") or {}
    weather_str = (
        f"Tropensystem aktiv: {'ja' if weather.get('active_tropical_system') else 'nein'}"
        + (f" · Disruption-Index {weather['weather_disruption_index']:.2f}"
           if isinstance(weather.get("weather_disruption_index"), (int, float)) else "")
    )

    cand_rows = ""
    for c in (context.get("candidates") or [])[:8]:
        mat = c.get("materiality")
        mat_str = f"{mat:.0%}" if isinstance(mat, (int, float)) else "–"
        cand_rows += (
            f"<li>{c.get('ticker', '?')}: {c.get('relation', '–')} "
            f"(Materialität {mat_str}, Quelle {c.get('exposure_source', '–')})</li>"
        )
    cand_html = f"<ul style='margin:4px 0 0;padding-left:18px;'>{cand_rows}</ul>" if cand_rows else ""

    return f"""
    <div style="margin-top:10px;padding:10px 14px;background:#f8fafc;border:1px solid #e2e8f0;
                border-radius:6px;font-size:11px;color:#334155;">
      <b style="color:#0369a1;">🌍 Externer Kontext</b> — <i>{header}</i>
      <div style="margin-top:4px;">
        Source-Health: {hs.get('PASS', 0)} PASS / {hs.get('FAIL', 0)} FAIL / {hs.get('STALE', 0)} STALE{fail_str}<br>
        Freight (regional): {regional_sep}<br>
        Maritime (global): {maritime_str}<br>
        Chokepoint-Anomalien (|z|&ge;2): {chokepoint_str}<br>
        Wetter: {weather_str}
      </div>
      {cand_html}
      {"<div style='margin-top:6px;color:#64748b;'>Quellen: " + " · ".join(context.get("attributions") or []) + "</div>" if context.get("attributions") else ""}
    </div>"""


def render_markdown_lines(context: dict | None) -> list[str]:
    """Kompakter Markdown-Block für die Daily-Markdown. context=None → eine
    Zeile 'keine externen Daten'."""
    if not context:
        return ["", "### 🌍 Externer Kontext", "", "_keine externen Daten_", ""]

    header = _shadow_header(context["mode"])
    hs = context["source_health"]["counts"]
    failing = context["source_health"]["failing"]
    lines = [
        "", "### 🌍 Externer Kontext", "",
        f"**{header}**", "",
        f"- Source-Health: {hs.get('PASS', 0)} PASS / {hs.get('FAIL', 0)} FAIL / "
        f"{hs.get('STALE', 0)} STALE" + (f" ({', '.join(failing[:5])})" if failing else ""),
    ]
    regional = context.get("regional_freight_states") or []
    if regional:
        parts = [
            f"{r['label']}: {r['state']}"
            + (f" (Konf. {r['confidence']:.0%})" if isinstance(r.get("confidence"), (int, float)) else "")
            for r in regional
        ]
        lines.append(f"- Freight (regional): {' · '.join(parts)}")
    maritime = context.get("global_maritime_state")
    if maritime:
        conf = context.get("global_maritime_confidence")
        conf_str = f" (Konf. {conf:.0%})" if isinstance(conf, (int, float)) else ""
        lines.append(f"- Maritime (global): {maritime}{conf_str}")
    chokepoints = context.get("chokepoint_anomalies") or []
    lines.append(
        "- Chokepoint-Anomalien (|z|>=2): "
        + (", ".join(f"{c['key']}={c['z']:+.1f}" for c in chokepoints) if chokepoints else "keine")
    )
    weather = context.get("weather") or {}
    w_str = f"Tropensystem aktiv: {'ja' if weather.get('active_tropical_system') else 'nein'}"
    if isinstance(weather.get("weather_disruption_index"), (int, float)):
        w_str += f" · Disruption-Index {weather['weather_disruption_index']:.2f}"
    lines.append(f"- Wetter: {w_str}")
    if context.get("attributions"):
        lines.append("- Quellen: " + " · ".join(context["attributions"]))
    candidates = context.get("candidates") or []
    if candidates:
        lines.append("- Kandidaten (Relation/Materialität/Quelle):")
        for c in candidates[:8]:
            mat = c.get("materiality")
            mat_str = f"{mat:.0%}" if isinstance(mat, (int, float)) else "–"
            lines.append(f"  - {c.get('ticker', '?')}: {c.get('relation', '–')} "
                         f"(Materialität {mat_str}, Quelle {c.get('exposure_source', '–')})")
    lines.append("")
    return lines
