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


def attach(panel: pd.DataFrame, sources: dict | None = None) -> pd.DataFrame:
    out = panel
    for sid, s in (sources or SOURCES).items():
        cols = s["features"] + [s["availability_col"]]
        p = Path(s["path"])
        if not p.exists():
            log.warning(f"alt_data: Feature-Store {sid} fehlt ({p}) -> Spalten NaN/verfügbar 0")
            out = out.assign(**{c: np.nan for c in s["features"]}, **{s["availability_col"]: 0.0})
            continue
        f = pd.read_csv(p, usecols=["date", "ticker"] + cols, parse_dates=["date"])
        f = f.drop_duplicates(["date", "ticker"]).rename(columns={"date": "_fdate"}).sort_values("_fdate")
        # Feature-Stichtag <= Panel-Datum (Feiertagswochen: Panel-Donnerstag nimmt den
        # Vorwochen-Freitag, nie einen späteren Stichtag), höchstens 7 Tage alt.
        base = out.drop(columns=[c for c in cols if c in out.columns]).reset_index(drop=True)
        base["_row"] = np.arange(len(base))
        m = pd.merge_asof(base.sort_values("date"), f, left_on="date", right_on="_fdate", by="ticker",
                          direction="backward", tolerance=pd.Timedelta(days=7))
        out = m.sort_values("_row").drop(columns=["_row", "_fdate"]).reset_index(drop=True)
        out[s["availability_col"]] = out[s["availability_col"]].fillna(0.0)
    return out
