"""modules/expectation_alpha/expectation_gap.py – gap = model_future_state − market_implied_expectation.

Domänen (config/expectation_alpha.yaml, `domains`):
  level_gap  gleiche Einheit (Prozentpunkte). Beispiel Inflation: CPI-3M-Rate annualisiert minus
             5J-Breakeven. Der Rohwert ist direkt interpretierbar.
  z_gap      Modell- und Marktseite liegen in unterschiedlichen Einheiten. Jede Komponente wird mit
             expandierendem z nur aus der Vergangenheit normiert; je Seite wird gemittelt; gap = Modell-z − Markt-z.
  unavailable  keine PIT-Reihe (Credit-Fundamental, Earnings-Konsens) -> UNAVAILABLE. Nie simuliert.

Je Domäne:
* Rohwert, z und Perzentil des Gaps gegen seine eigene Historie (nur Vergangenheit);
* Historienfenster, RoC-Satz des Gaps und des Modell-Zustands;
* Komponenten mit Provenienz.

Fehlt eine Seite zum Stichtag, ist der Gap UNAVAILABLE. Er wird nie mit 0 gefüllt.
"""
from __future__ import annotations

import math
from datetime import datetime, time, timezone

import numpy as np
import pandas as pd

from modules.expectation_alpha import data as eadata
from modules.expectation_alpha.future_state import expanding_percentile, expanding_z, roc_set
from modules.expectation_alpha.schemas import (FeatureValue, INSUFFICIENT_DATA, OK, STALE, UNAVAILABLE, rnd)

# Komponente -> (Art, Quelle/Metrik, Transformation, Frische-Klasse, Einheit)
#   macro: world_model.MACRO-Indikator (PIT-Vintages)   market: aus Tagesschlüssen
#   archive: eigene PIT-Reihe (FRED)                    commodity: commodity_intelligence-Merkmal
COMPONENTS = {
    "cpi_3m_ann": ("archive", ("fred_regime_macro", "us_cpi"), "ann_3m", "monthly", "pct"),
    "breakeven_5y": ("archive", ("fred_market_expectations", "us_breakeven_5y"), "level", "daily", "pct"),
    "policy_path_2y": ("archive_diff", (("fred_market_expectations", "us_treasury_2y"),
                                        ("fred_market_expectations", "us_fed_funds_effective")),
                       "DGS2 - DFF", "daily", "pct_points"),
    "cpi_trend": ("macro", "cpi_trend", "3M-Rate ann. - Jahresrate", "monthly", "ratio"),
    "payems_3m": ("macro", "payems_3m", "91T-Veränderung", "monthly", "ratio"),
    "claims_13w_neg": ("macro", "claims_13w_neg", "-(91T-Veränderung)", "weekly", "ratio"),
    "indpro_3m": ("macro", "indpro_3m", "91T-Veränderung", "monthly", "ratio"),
    "retail_yoy": ("macro", "retail_yoy", "365T-Veränderung", "monthly", "ratio"),
    "cyc_def_63d": ("market", "cyc_def_63d", "63T-Rendite Zykliker - Defensive", "daily", "ratio"),
    "copper_gold_63d": ("market", "copper_gold_63d", "63T-Veränderung Kupfer/Gold", "daily", "ratio"),
    "xli_spy_63d": ("market", "xli_spy_63d", "63T-Rendite XLI - SPY", "daily", "ratio"),
    "iwm_spy_63d": ("market", "iwm_spy_63d", "63T-Rendite IWM - SPY", "daily", "ratio"),
    "neg_crude_stocks_vs_5y": ("commodity", "cmd_crude_stocks_vs_5y", "-(Bestand vs. 5J-Mittel)", "weekly", "ratio"),
    "wti_ret_60d": ("commodity", "cmd_wti_ret_60d", "60T-Rendite WTI", "daily", "ratio"),
}
COMMODITY_SIGN = {"neg_crude_stocks_vs_5y": -1.0}
MARKET_SOURCES = {"cyc_def_63d": "XLI,XLY,XLB,XLF/XLP,XLU,XLV", "copper_gold_63d": "HG=F/GC=F",
                  "xli_spy_63d": "XLI,SPY", "iwm_spy_63d": "IWM,SPY"}


def extra_market(px: pd.DataFrame) -> pd.DataFrame:
    """Marktmerkmale, die world_model.market_frame nicht enthält (nur Renditen, bis Close t)."""
    m = pd.DataFrame(index=px.index)

    def chg(sym, n=63):
        return px[sym] / px[sym].shift(n) - 1.0 if sym in px else pd.Series(np.nan, index=px.index)
    m["iwm_spy_63d"] = chg("IWM") - chg("SPY")
    return m.replace([np.inf, -np.inf], np.nan)


def component_frame(inp: eadata.Inputs, dates: list[pd.Timestamp], cfg: dict) -> pd.DataFrame:
    """Wöchentliche PIT-Reihen aller Gap-Komponenten (Spalten) auf dem Raster `dates`."""
    from modules import world_model as wm
    ages = (cfg.get("data") or {}).get("max_age_days") or {}
    cols: dict[str, pd.Series] = {}
    if not dates:
        return pd.DataFrame()
    mac = wm.macro_frame(inp.archive_obs, dates) if inp.archive_obs else pd.DataFrame(index=dates)
    mk = pd.DataFrame(index=dates)
    if not inp.px.empty and "SPY" in inp.px:
        mk = pd.concat([wm.market_frame(inp.px), extra_market(inp.px)], axis=1).reindex(dates)
    cm = inp.commodity.reindex(dates) if not inp.commodity.empty else pd.DataFrame(index=dates)
    for name, (kind, ref, how, freq, _unit) in COMPONENTS.items():
        age = int(ages.get(freq, 75))
        if kind == "macro":
            cols[name] = mac[ref] if ref in mac else pd.Series(np.nan, index=dates)
        elif kind == "market":
            cols[name] = mk[ref] if ref in mk else pd.Series(np.nan, index=dates)
        elif kind == "commodity":
            s = cm[ref] if ref in cm else pd.Series(np.nan, index=dates)
            cols[name] = s.astype(float) * COMMODITY_SIGN.get(name, 1.0)
        elif kind == "archive":
            src, metric = ref
            cols[name] = eadata.pit_weekly(inp.archive_obs.get(src, []), metric, dates, age, how=how)
        elif kind == "archive_diff":
            (s1, m1), (s2, m2) = ref
            a = eadata.pit_weekly(inp.archive_obs.get(s1, []), m1, dates, age)
            b = eadata.pit_weekly(inp.archive_obs.get(s2, []), m2, dates, age)
            cols[name] = a - b
    return pd.DataFrame(cols, index=pd.DatetimeIndex(dates)).astype(float)


def component_provenance(name: str, inp: eadata.Inputs, frame: pd.DataFrame, cfg: dict) -> dict:
    """FeatureValue des Komponentenwerts am letzten Stichtag (Provenienz je Quelle)."""
    kind, ref, how, freq, unit = COMPONENTS[name]
    d = frame.index[-1]
    val = frame[name].iloc[-1] if name in frame else np.nan
    ages = (cfg.get("data") or {}).get("max_age_days") or {}
    cut = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if kind in ("archive", "archive_diff"):
        src, metric = ref if kind == "archive" else ref[0]
        o = eadata.latest_vintage(inp.archive_obs.get(src, []), metric, cut)
        fv = eadata.feature_from_obs(name, o, unit=unit, source=f"{src}:{metric}" if kind == "archive"
                                     else f"{src}:{ref[0][1]}-{ref[1][1]}", transformation=how,
                                     as_of=cut, max_age_days=int(ages.get(freq, 75)),
                                     value=None if pd.isna(val) else float(val))
        if pd.isna(val) and fv.status == OK:
            fv.value, fv.status = None, UNAVAILABLE
        return fv.to_dict()
    if kind == "macro":
        from modules import world_model as wm
        src, metric, _h, mx = wm.MACRO[ref]
        o = eadata.latest_vintage(inp.archive_obs.get(src, []), metric, cut)
        fv = eadata.feature_from_obs(name, o, unit=unit, source=f"{src}:{metric}", transformation=how,
                                     as_of=cut, max_age_days=int(mx), value=None if pd.isna(val) else float(val))
        if pd.isna(val):
            fv.value, fv.status = None, (STALE if fv.status == STALE else UNAVAILABLE)
        return fv.to_dict()
    close_h = int((cfg.get("data") or {}).get("close_complete_hour_utc", 21))
    avail = datetime.combine(d.date(), time(close_h), timezone.utc).isoformat()
    src = (f"yfinance:{MARKET_SOURCES.get(name, name)}" if kind == "market"
           else f"commodity_intelligence:{ref}")
    return FeatureValue(name, None if pd.isna(val) else float(val), unit, src, observed_at=d.date().isoformat(),
                        available_at=avail, retrieved_at=inp.decision_time.isoformat(timespec="seconds"),
                        transformation=how, freshness_days=(inp.decision_time - datetime.fromisoformat(avail))
                        .total_seconds() / 86400.0, confidence=None if pd.isna(val) else 1.0).to_dict()


def _side(frame: pd.DataFrame, comps: list[str], min_hist: int) -> pd.Series:
    """Mittel der vergangenheitsnormierten Komponenten; mindestens die Hälfte muss vorliegen."""
    zs = pd.concat([expanding_z(frame[c], min_hist) for c in comps], axis=1) if comps else pd.DataFrame()
    if zs.empty:
        return pd.Series(np.nan, index=frame.index)
    need = math.ceil(len(comps) / 2)
    return zs.mean(axis=1).where(zs.notna().sum(axis=1) >= need)


def domain_gaps(inp: eadata.Inputs, dates: list[pd.Timestamp], cfg: dict) -> dict:
    """Alle Domänen-Gaps am letzten Stichtag (+ Wochenreihe des Gap-Rohwerts für Tests/Report)."""
    min_hist = int((cfg.get("data") or {}).get("min_history_weeks", 52))
    out: dict = {}
    frame = component_frame(inp, dates, cfg) if dates else pd.DataFrame()
    for dom, spec in (cfg.get("domains") or {}).items():
        kind = spec.get("kind")
        res: dict = {"domain": dom, "kind": kind, "unit": spec.get("unit"), "meaning": spec.get("meaning")}
        if kind == "unavailable":
            out[dom] = {**res, "status": UNAVAILABLE, "reason": spec.get("reason")}
            continue
        if frame.empty:
            out[dom] = {**res, "status": UNAVAILABLE, "reason": "keine Daten (Raster leer)"}
            continue
        if kind == "level_gap":
            mc, kc = [spec["model"]["indicator"]], [spec["market"]["indicator"]]
            model, market = frame[mc[0]], frame[kc[0]]
        elif kind == "z_gap":
            mc, kc = list(spec["model"]["components"]), list(spec["market"]["components"])
            model, market = _side(frame, mc, min_hist), _side(frame, kc, min_hist)
        else:
            raise ValueError(f"unbekannte Gap-Art {kind}")
        gap = model - market
        gz, gp = expanding_z(gap, min_hist), expanding_percentile(gap, min_hist)
        res["model"] = {"description": spec["model"].get("description"), "value": rnd(model.iloc[-1], 4),
                        "components": {c: component_provenance(c, inp, frame, cfg) for c in mc},
                        "roc": roc_set(model, cfg, unit="pct" if kind == "level_gap" else "z", min_hist=min_hist)}
        res["market"] = {"description": spec["market"].get("description"), "value": rnd(market.iloc[-1], 4),
                         "components": {c: component_provenance(c, inp, frame, cfg) for c in kc}}
        g_last = gap.iloc[-1]
        hist = gap.dropna()
        res["history"] = {"n": int(len(hist)), "min_history_weeks": min_hist,
                          "window_start": hist.index[0].date().isoformat() if len(hist) else None,
                          "window_end": hist.index[-1].date().isoformat() if len(hist) else None}
        res["date"] = frame.index[-1].date().isoformat()
        if pd.isna(g_last):
            miss = [c for c in mc + kc if pd.isna(frame[c].iloc[-1])]
            res.update(status=UNAVAILABLE, gap_raw=None, gap_z=None, gap_percentile=None, sign=None,
                       reason=f"Seite fehlt/veraltet: {', '.join(miss) or 'zu wenig Komponenten mit Historie'}")
        else:
            res["gap_raw"] = rnd(g_last, 4)
            res["gap_z"] = rnd(gz.iloc[-1], 4)
            res["gap_percentile"] = rnd(gp.iloc[-1], 4)
            res["gap_roc"] = roc_set(gap, cfg, unit=spec.get("unit") or "", min_hist=min_hist)
            if res["gap_z"] is None:
                res.update(status=INSUFFICIENT_DATA, sign=None,
                           reason=f"Gap-Historie {len(hist)} < {min_hist} Wochen (z/Perzentil nicht belastbar)")
            else:
                res.update(status=OK, sign=int(np.sign(res["gap_z"])))
        res["series_tail"] = {d.date().isoformat(): rnd(v, 4) for d, v in gap.tail(8).items()}
        out[dom] = res
    return out
