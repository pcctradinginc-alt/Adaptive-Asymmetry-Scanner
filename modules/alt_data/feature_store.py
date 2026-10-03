"""Anbindung der Alternative-Data-Features an das ml_research-Panel (SHADOW).

Join über (date, ticker). Fehlt eine Quelle/Datei -> Spalten NaN und
Verfügbarkeit 0 (nie 0 als Ersatzwert). Werte wurden PIT berechnet
(available_at <= Stichtag 21:00 UTC, siehe sec_features)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from modules.alt_data.registry import SOURCES

log = logging.getLogger(__name__)


def attach(panel: pd.DataFrame, sources: dict | None = None, health: dict | None = None) -> pd.DataFrame:
    """health: Source-Health-Snapshot (Default: outputs/health, falls vorhanden)."""
    out = panel
    if health is None:
        try:
            from modules.source_health import SNAPSHOT
            import json
            health = json.loads(SNAPSHOT.read_text(encoding="utf-8")) if SNAPSHOT.exists() else {}
        except (OSError, ValueError):
            health = {}
    for sid, s in (sources or SOURCES).items():
        try:
            out = _attach_one(out, s)
            out = apply_health(out, s, health)
        except Exception as e:  # noqa: BLE001 – optionale Quelle darf die Research-Pipeline nie brechen
            log.error(f"alt_data: Feature-Store {sid} nicht anbindbar ({type(e).__name__}: {e}) -> NaN/verfügbar 0")
            out = out.assign(**{c: np.nan for c in s["features"]}, **{s["availability_col"]: 0.0})
    return out


def apply_health(df: pd.DataFrame, s: dict, health: dict | None) -> pd.DataFrame:
    """Quelle laut täglichem Health Check nicht nutzbar -> ab Ausfallzeitpunkt (failed_since)
    Features NaN und Verfügbarkeit 0 (nie 0 als Ersatzwert). Cache nur mit ausdrücklich
    zulässiger Datenalterung: Werte bleiben, Spalte <availability>_stale = 1.
    Zeilen VOR dem Ausfall bleiben unverändert (damals gültige PIT-Daten)."""
    srcs = (health or {}).get("sources") or {}
    feats = (health or {}).get("features") or {}
    av = s["availability_col"]
    df[f"{av}_stale"] = 0.0
    for cid in s.get("contracts") or []:
        h = srcs.get(cid)
        if not h or h.get("status") in ("HEALTHY", "DEGRADED") or h.get("fallback_active"):
            continue
        since = pd.Timestamp(h.get("failed_since") or h.get("checked_at") or health.get("generated"))
        since = since.tz_convert(None) if since.tzinfo else since
        rows = pd.to_datetime(df["date"]).dt.tz_localize(None) >= since.normalize()
        if not rows.any():
            continue
        cached = all((feats.get(f) or {}).get("stale") for f in s["features"])
        if cached:
            df.loc[rows, f"{av}_stale"] = 1.0
            log.warning(f"alt_data: {cid} {h['status']} – Cache innerhalb zulässiger Datenalterung (stale=1)")
        else:
            df.loc[rows, s["features"]] = np.nan
            df.loc[rows, av] = 0.0
            log.warning(f"alt_data: {cid} {h['status']} seit {since.date()} – {int(rows.sum())} Zeilen unavailable")
    return df


def _attach_one(out: pd.DataFrame, s: dict) -> pd.DataFrame:
    cols = s["features"] + [s["availability_col"]]
    p = Path(s["path"])
    if not p.exists():
        log.warning(f"alt_data: Feature-Store fehlt ({p}) -> Spalten NaN/verfügbar 0")
        return out.assign(**{c: np.nan for c in s["features"]}, **{s["availability_col"]: 0.0})
    f = pd.read_csv(p, usecols=["date", "ticker"] + cols, parse_dates=["date"])
    f = f.drop_duplicates(["date", "ticker"]).rename(columns={"date": "_fdate"})
    # gleiche Zeit-Einheit/-zone erzwingen (CI 2026-10-03: M8[s] vs. M8[us] -> MergeError)
    f["_fdate"] = pd.to_datetime(f["_fdate"]).dt.tz_localize(None).astype("datetime64[ns]")
    f["ticker"] = f["ticker"].astype(str)
    f = f.sort_values("_fdate")
    # Feature-Stichtag <= Panel-Datum (Feiertagswochen: Panel-Donnerstag nimmt den
    # Vorwochen-Freitag, nie einen späteren Stichtag), höchstens 7 Tage alt.
    base = out.drop(columns=[c for c in cols if c in out.columns]).reset_index(drop=True)
    base["_row"] = np.arange(len(base))
    left = base.assign(_pdate=pd.to_datetime(base["date"]).dt.tz_localize(None).astype("datetime64[ns]"),
                       _tk=base["ticker"].astype(str))
    m = pd.merge_asof(left.sort_values("_pdate"), f.rename(columns={"ticker": "_tk"}), left_on="_pdate",
                      right_on="_fdate", by="_tk", direction="backward", tolerance=pd.Timedelta(days=7))
    out = m.sort_values("_row").drop(columns=["_row", "_fdate", "_pdate", "_tk"]).reset_index(drop=True)
    out[s["availability_col"]] = out[s["availability_col"]].fillna(0.0)
    return out
