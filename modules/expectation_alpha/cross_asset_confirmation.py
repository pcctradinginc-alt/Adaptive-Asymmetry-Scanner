"""modules/expectation_alpha/cross_asset_confirmation.py – bestätigt der Markt die These?

Thesen-Spezifikation = Liste (Signal, erwartetes Vorzeichen, Deadband, Gewicht):
* Kandidaten-These, z. B. Long-Aktie im Sektor S: candidate_signals × Thesenrichtung, dazu die
  Domänen-Signale der im Sektor-Mapping hinterlegten Sensitivitäten (Vorzeichen × Sensitivität × Richtung).
  Widersprüchliche Erwartungen an dasselbe Signal heben sich auf. Das Signal wird dann nicht
  erwartet.
* Domänen-Gap: domain_signals × Vorzeichen des Gaps. Preist der Markt die Modellsicht ein?

Je Signal: s = +1, wenn erwartet × Wert > Deadband (bestätigt); −1, wenn < −Deadband (widerspricht);
sonst 0 (neutral).

Fehlende Signale stehen in missing_inputs und zählen weder als Bestätigung noch als Widerspruch:
Missing ist nicht negativ. confirmation_ratio = n_confirming / n_available; None ohne verfügbare Signale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.expectation_alpha.schemas import OK, UNAVAILABLE, rnd

CYCLICALS, DEFENSIVES = ("XLI", "XLY", "XLB", "XLF"), ("XLP", "XLU", "XLV")


SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLI", "XLV", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC")


def signal_frame(px: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Alle Marktsignale je Handelstag, vektorisiert und nur bis Close t. Fehlt eine Reihe am Tag t,
    ist das Signal NaN und gilt damit als fehlend, nie als 0. Für die Entscheidung wird die letzte Zeile
    genutzt; der WAIT-/Kill-Replay nutzt die Zeile des jeweiligen Tages."""
    if px is None or px.empty:
        return pd.DataFrame()
    nan = pd.Series(np.nan, index=px.index)

    def chg(sym, n=window):
        return px[sym] / px[sym].shift(n) - 1.0 if sym in px else nan

    def diff(sym):
        return px[sym] - px[sym].shift(window) if sym in px else nan

    def ratio_chg(a, b):
        return (px[a] / px[b]) / (px[a] / px[b]).shift(window) - 1.0 if a in px and b in px else nan

    f = pd.DataFrame(index=px.index)
    f["cyc_def_20d"] = pd.concat([chg(s) for s in CYCLICALS], axis=1).mean(axis=1, skipna=False) - \
        pd.concat([chg(s) for s in DEFENSIVES], axis=1).mean(axis=1, skipna=False)
    f["copper_gold_20d"] = ratio_chg("HG=F", "GC=F")
    f["iwm_spy_20d"] = chg("IWM") - chg("SPY")
    f["hyg_lqd_20d"] = ratio_chg("HYG", "LQD")
    f["tnx_20d"] = diff("^TNX")
    f["irx_20d"] = diff("^IRX")
    f["tip_ief_20d"] = ratio_chg("TIP", "IEF")
    f["gld_20d"] = chg("GLD")
    f["tlt_20d"] = chg("TLT")
    f["xle_spy_20d"] = chg("XLE") - chg("SPY")
    f["spy_ret_20d"] = chg("SPY")
    f["vix_term_contango"] = (1.0 - px["^VIX"] / px["^VIX3M"]) if "^VIX" in px and "^VIX3M" in px else nan
    for etf in SECTOR_ETFS:
        f[f"sector_rs_20d:{etf}"] = chg(etf) - chg("SPY")
        f[f"sector_rs_63d:{etf}"] = chg(etf, 63) - chg("SPY", 63)
    return f.replace([np.inf, -np.inf], np.nan)


def market_signals(px: pd.DataFrame, commodity: pd.DataFrame | None, window: int = 20) -> dict:
    """Kurzfrist-Signale am letzten abgeschlossenen Handelstag (+ WTI aus commodity_intelligence, PIT)."""
    f = signal_frame(px, window)
    if f.empty:
        return {"date": None, "window_days": window, "values": {}}
    last = f.index[-1]
    vals = {k: rnd(v, 6) for k, v in f.iloc[-1].items()}
    wti = None
    if commodity is not None and not commodity.empty and "cmd_wti_ret_20d" in commodity:
        c = commodity["cmd_wti_ret_20d"].dropna()
        c = c[c.index <= last]
        if len(c) and (last - c.index[-1]).days <= 7:
            wti = rnd(float(c.iloc[-1]), 6)
    vals["wti_ret_20d"] = wti
    return {"date": last.date().isoformat(), "window_days": window, "values": vals}


def _merge(spec: list[dict]) -> list[dict]:
    """Gleiches Signal aus mehreren Quellen: Vorzeichen summieren. Netto 0 -> mehrdeutig -> nicht erwartet."""
    agg: dict[str, dict] = {}
    for s in spec:
        a = agg.setdefault(s["signal"], {"signal": s["signal"], "expected": 0.0, "deadband": s["deadband"],
                                         "weight": s.get("weight", 1.0), "from": []})
        a["expected"] += s["expected"]
        a["deadband"] = max(a["deadband"], s["deadband"])
        a["from"].append(s.get("from"))
    out, ambiguous = [], []
    for a in agg.values():
        if a["expected"] == 0:
            ambiguous.append(a["signal"])
            continue
        a["expected"] = 1 if a["expected"] > 0 else -1
        out.append(a)
    return sorted(out, key=lambda x: x["signal"]), sorted(ambiguous)


def candidate_spec(direction: int, sector_etf: str | None, sensitivities: dict | None, cfg: dict) -> tuple:
    """Erwartete Bewegungen für eine Aktien-These (Richtung +1 long / −1 short)."""
    cc = cfg.get("confirmation") or {}
    w = cc.get("weights") or {}
    spec = []
    for s in cc.get("candidate_signals") or []:
        name = s["signal"]
        if name == "sector_rs_20d":
            if not sector_etf:
                continue                      # unbekannter Sektor: Signal nicht erwartet (nicht "fehlend")
            name = f"sector_rs_20d:{sector_etf}"
        spec.append({"signal": name, "expected": int(s["sign"]) * direction, "deadband": float(s["deadband"]),
                     "weight": float(w.get(s["signal"], 1.0)), "from": "candidate"})
    for dom, sens in (sensitivities or {}).items():
        for s in (cc.get("domain_signals") or {}).get(dom) or []:
            spec.append({"signal": s["signal"], "expected": int(s["sign"]) * int(sens) * direction,
                         "deadband": float(s["deadband"]), "weight": float(w.get(s["signal"], 1.0)),
                         "from": f"domain:{dom}"})
    return _merge(spec)


def domain_spec(domain: str, gap_sign: int, cfg: dict) -> tuple:
    cc = cfg.get("confirmation") or {}
    w = cc.get("weights") or {}
    spec = [{"signal": s["signal"], "expected": int(s["sign"]) * int(gap_sign), "deadband": float(s["deadband"]),
             "weight": float(w.get(s["signal"], 1.0)), "from": f"gap:{domain}"}
            for s in (cc.get("domain_signals") or {}).get(domain) or []]
    return _merge(spec)


def confirm(spec: list[dict], values: dict, ambiguous: list[str] | None = None) -> dict:
    n_exp = len(spec)
    signals, missing = [], []
    conf = confl = 0
    wsum = wtot = 0.0
    for s in spec:
        v = values.get(s["signal"])
        if v is None:
            missing.append(s["signal"])
            signals.append({**s, "value": None, "state": "MISSING"})
            continue
        x = s["expected"] * float(v)
        st = 1 if x > s["deadband"] else -1 if x < -s["deadband"] else 0
        conf += st == 1
        confl += st == -1
        wsum += s["weight"] * st
        wtot += s["weight"]
        signals.append({**s, "value": v, "state": {1: "CONFIRMING", -1: "CONFLICTING", 0: "NEUTRAL"}[st]})
    n_av = n_exp - len(missing)
    return {"n_expected": n_exp, "n_available": n_av, "n_confirming": int(conf), "n_conflicting": int(confl),
            "confirmation_ratio": rnd(conf / n_av, 4) if n_av else None,
            "conflict_share": rnd(confl / n_av, 4) if n_av else None,
            "weighted_confirmation": rnd(wsum / wtot, 4) if wtot else None,
            "missing_inputs": missing, "ambiguous_signals": ambiguous or [],
            "confidence": rnd(n_av / n_exp, 4) if n_exp else None,
            "status": OK if n_av else UNAVAILABLE, "signals": signals}
