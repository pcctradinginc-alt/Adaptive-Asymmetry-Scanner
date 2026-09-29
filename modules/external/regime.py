"""
modules/external/regime.py – Makro-Regime aus fred_regime_macro (PIT).

Jede Reihe wird zum Stichtag `as_of` so rekonstruiert, wie sie damals bekannt
war: je Beobachtungsperiode die jüngste Vintage mit available_at <= as_of
(ALFRED realtime_start = Veröffentlichungstag). Keine Revisionen aus der
Zukunft (Restatement-Bias), keine unveröffentlichten Perioden (Publication
Lag). Veraltete Reihen (letzte bekannte Periode zu alt) -> None, nie ein
Default-Regime.

Regeln (bewusst einfach, NICHT auf Returns optimiert):
  inflation      CPI-YoY >= 3 %  -> high_inflation, sonst low_inflation
  inflation_trend CPI-YoY heute vs. vor 3 Monaten -> inflation_rising / disinflation
  credit         NFCI-Credit-Subindex > 0 -> credit_tight, sonst credit_loose
  fin_conditions NFCI > 0 -> tight, sonst loose
  liquidity      Fed-Bilanzsumme 13 Wochen: Zuwachs -> expanding, sonst contracting
  dollar         Broad Dollar 63 Handelstage: > 0 -> dollar_up, sonst dollar_down
  oil            WTI 63 Handelstage: > 0 -> oil_up, sonst oil_down
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta

from modules.external.pit import ensure_utc

SOURCE_ID = "fred_regime_macro"
MAX_AGE = {"us_cpi": 75, "nfci_credit": 21, "nfci": 21, "fed_total_assets": 21,
           "usd_broad": 10, "wti": 10}


def pit_series(observations, metric: str, as_of: datetime,
               entity: str | None = None) -> list[tuple[datetime, float]]:
    """(Periode, Wert) aufsteigend, je Periode die zum Stichtag jüngste
    veröffentlichte Vintage; None-Werte ("." bei FRED) ausgeschlossen."""
    as_of = ensure_utc(as_of)
    best: dict = {}
    for o in observations:
        if o.metric != metric or o.available_at is None:
            continue
        if entity is not None and o.entity_id != entity:
            continue
        av = ensure_utc(o.available_at)
        if av > as_of:
            continue
        cur = best.get(o.observation_time)
        if cur is None or av > ensure_utc(cur.available_at):
            best[o.observation_time] = o
    return sorted((t, o.value) for t, o in best.items() if o.value is not None)


def _fresh(series, metric, as_of) -> bool:
    return bool(series) and (ensure_utc(as_of) - ensure_utc(series[-1][0])).days <= MAX_AGE[metric]


def _change(series, lag: int) -> float | None:
    if len(series) <= lag or not series[-1 - lag][1]:
        return None
    return series[-1][1] / series[-1 - lag][1] - 1.0


def _yoy_at(series, idx: int) -> float | None:
    t, v = series[idx]
    prev = [val for (tt, val) in series[:idx + 1] if tt.year == t.year - 1 and tt.month == t.month]
    return v / prev[-1] - 1.0 if prev and prev[-1] else None


def regime_state(observations, as_of: datetime) -> dict:
    """Primitive + Regime-Labels zum Stichtag (fehlend = None)."""
    out: dict = {"cpi_yoy": None, "cpi_yoy_3m_ago": None, "nfci_credit": None, "nfci": None,
                 "fed_assets_13w_chg": None, "usd_63d_chg": None, "wti_63d_chg": None, "labels": {}}
    lab = out["labels"]
    cpi = pit_series(observations, "us_cpi", as_of)
    if _fresh(cpi, "us_cpi", as_of):
        out["cpi_yoy"] = _yoy_at(cpi, len(cpi) - 1)
        if len(cpi) > 3:
            out["cpi_yoy_3m_ago"] = _yoy_at(cpi, len(cpi) - 4)
        if out["cpi_yoy"] is not None:
            lab["inflation"] = "high_inflation" if out["cpi_yoy"] >= 0.03 else "low_inflation"
            if out["cpi_yoy_3m_ago"] is not None:
                lab["inflation_trend"] = ("inflation_rising" if out["cpi_yoy"] > out["cpi_yoy_3m_ago"]
                                          else "disinflation")
    for metric, key, name, pos, neg in (("nfci_credit", "nfci_credit", "credit", "credit_tight", "credit_loose"),
                                        ("nfci", "nfci", "fin_conditions", "tight", "loose")):
        s = pit_series(observations, metric, as_of)
        if _fresh(s, metric, as_of):
            out[key] = s[-1][1]
            lab[name] = pos if s[-1][1] > 0 else neg
    for metric, key, name, lag, pos, neg in (
            ("fed_total_assets", "fed_assets_13w_chg", "liquidity", 13, "expanding", "contracting"),
            ("usd_broad", "usd_63d_chg", "dollar", 63, "dollar_up", "dollar_down"),
            ("wti", "wti_63d_chg", "oil", 63, "oil_up", "oil_down")):
        s = pit_series(observations, metric, as_of)
        if _fresh(s, metric, as_of):
            ch = _change(s, lag)
            out[key] = ch
            if ch is not None:
                lab[name] = pos if ch > 0 else neg
    return out


def regimes_by_date(observations, dates: list[str]) -> dict[str, dict]:
    """Regime-Labels je Tag (Stichtag = Tagesbeginn UTC, d.h. nur bis zum
    Vortag veröffentlichte Werte)."""
    out = {}
    for d in sorted(set(dates)):
        as_of = ensure_utc(datetime.fromisoformat(d)) - timedelta(seconds=1)
        lab = regime_state(observations, as_of)["labels"]
        if lab:
            out[d] = lab
    return out


# ── Strommarkt (entsoe_power) ────────────────────────────────────────────────

ENERGY_SOURCE_ID = "entsoe_power"


def energy_state(observations, as_of: datetime) -> dict:
    """PIT-Energie-Kontext:
      de_power_price_7d   Mittel DE-LU Day-Ahead der letzten 7 bekannten Tage
      de_power_price_z    dieses 7-T-Mittel vs. Verteilung der Tageswerte der
                          vorangehenden 365 Tage (>= 200 Tage nötig)
      de_load_yoy         DE-Last 7-T-Mittel vs. dieselben 7 Tage im Vorjahr
                          (saisonbereinigt; Industrie-Nachfrage-Proxy)
    Labels: power = power_expensive/power_cheap (z > 0), demand = load_up/down."""
    out = {"de_power_price_7d": None, "de_power_price_z": None, "de_load_yoy": None, "labels": {}}
    px = pit_series(observations, "da_price_daily_mean", as_of, entity="DE_LU")
    if px and (ensure_utc(as_of) - ensure_utc(px[-1][0])).days <= 7:
        last7 = [v for _, v in px[-7:]]
        hist = [v for t, v in px[:-7] if (px[-7][0] - t).days <= 365]
        out["de_power_price_7d"] = statistics.fmean(last7)
        if len(hist) >= 200 and statistics.pstdev(hist) > 0:
            out["de_power_price_z"] = (out["de_power_price_7d"] - statistics.fmean(hist)) / statistics.pstdev(hist)
            out["labels"]["power"] = "power_expensive" if out["de_power_price_z"] > 0 else "power_cheap"
    ld = pit_series(observations, "load_daily_mean", as_of, entity="DE")
    if ld and (ensure_utc(as_of) - ensure_utc(ld[-1][0])).days <= 7:
        by_t = dict(ld)
        last = ld[-7:]
        prev = [by_t.get(t - timedelta(days=364)) for t, _ in last]      # gleicher Wochentag
        if all(v is not None for v in prev):
            out["de_load_yoy"] = statistics.fmean(v for _, v in last) / statistics.fmean(prev) - 1.0
            out["labels"]["demand"] = "load_up" if out["de_load_yoy"] > 0 else "load_down"
    return out


def energy_regimes_by_date(observations, dates: list[str]) -> dict[str, dict]:
    out = {}
    for d in sorted(set(dates)):
        lab = energy_state(observations, ensure_utc(datetime.fromisoformat(d)) - timedelta(seconds=1))["labels"]
        if lab:
            out[d] = lab
    return out
