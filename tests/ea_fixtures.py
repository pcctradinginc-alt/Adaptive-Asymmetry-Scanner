"""
tests/ea_fixtures.py – Deterministische Testfixtures für das Expectation-Alpha-Modul
(modules/expectation_alpha/): Marktkurse, PIT-Archiv, Rohstoffdaten, Analysen und Kurs-Bars.

Synthetische Testdaten – nur für Tests, nie Produktion.

Alle Werte entstehen über numpy.random.default_rng(seed). Es gibt keinen Netzwerkzugriff und
kein Schreiben in Dateien. Die einzige Dateilesung ist die schreibgeschützte Symbolliste aus
config/expectation_alpha.yaml (Standardwert von make_px). Die Werte sind keine echten Markt- oder Makrodaten
und dürfen nicht als solche interpretiert werden. Serien- und Quellennamen sind Testbezeichnungen.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from modules.external.pit import AvailabilityPrecision, Observation

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "expectation_alpha.yaml"


# ── Zufallsprozesse (alle über den übergebenen rng) ─────────────────────────

def _gbm(rng: np.random.Generator, n: int, start: float, mu: float, sigma: float) -> np.ndarray:
    """Geometrische Irrfahrt: erster Wert = start, Log-Renditen ~ N(mu, sigma). Immer > 0."""
    r = rng.normal(mu, sigma, n)
    r[0] = 0.0
    return start * np.exp(np.cumsum(r))


def _rw(rng: np.random.Generator, n: int, start: float, step: float,
        lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Additive Irrfahrt: erster Wert = start, Schritte ~ N(0, step), optional auf [lo, hi] geklemmt."""
    steps = rng.normal(0.0, step, n)
    steps[0] = 0.0
    x = start + np.cumsum(steps)
    if lo is not None:
        x = np.clip(x, lo, hi)
    return x


def _ar1(rng: np.random.Generator, n: int, mean: float, sd: float, phi: float = 0.95) -> np.ndarray:
    """AR(1) um `mean` mit stationärer Standardabweichung `sd`."""
    eps = rng.normal(0.0, sd * math.sqrt(1.0 - phi ** 2), n)
    out = np.empty(n)
    x = mean
    for t in range(n):
        x = mean + phi * (x - mean) + eps[t]
        out[t] = x
    return out


def _vix_path(rng: np.random.Generator, n: int) -> np.ndarray:
    """VIX: v = 18 + 0.95 (v_prev - 18) + N(0, 1), auf [9, 80] geklemmt."""
    eps = rng.normal(0.0, 1.0, n)
    out = np.empty(n)
    v = 18.0
    for t in range(n):
        v = min(80.0, max(9.0, 18.0 + 0.95 * (v - 18.0) + eps[t]))
        out[t] = v
    return out


def _fed_funds(rng: np.random.Generator, n: int, start: float = 4.33) -> np.ndarray:
    """Fed Funds: stückweise konstant, Änderung um ±0.25 nach 90–150 Handelstagen (~120).
    Am Boden (<= 0.25) nur aufwärts, an der Decke (>= 8) nur abwärts."""
    out = np.empty(n)
    cur = start
    nxt = int(rng.integers(90, 151))
    for t in range(n):
        if t == nxt:
            if cur <= 0.25:
                step = 0.25
            elif cur >= 8.0:
                step = -0.25
            else:
                step = 0.25 if rng.random() < 0.5 else -0.25
            cur = round(cur + step, 4)
            nxt = t + int(rng.integers(90, 151))
        out[t] = cur
    return out


# ── Zeit- und Wertehilfen ───────────────────────────────────────────────────

def _utc(x) -> datetime:
    ts = pd.Timestamp(x)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _val(v) -> float | None:
    """None und NaN bleiben fehlend (None); nie 0 als Ersatz."""
    return None if v is None or pd.isna(v) else float(v)


def _month_starts(end_ts: pd.Timestamp, years: int) -> pd.DatetimeIndex:
    """Monatsanfänge (1. des Monats, UTC-naiv) von `years` Jahren bis zum Monat von end_ts."""
    start = (end_ts - pd.DateOffset(years=years)).to_period("M").to_timestamp()
    last = end_ts.to_period("M").to_timestamp()
    return pd.date_range(start, last, freq="MS")


@lru_cache(maxsize=1)
def _default_symbols() -> tuple[str, ...]:
    """data.market_symbols aus config/expectation_alpha.yaml (nur Lesezugriff)."""
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    return tuple(cfg["data"]["market_symbols"])


# ── Marktkurse ──────────────────────────────────────────────────────────────

def make_px(end: str = "2026-10-09", years: int = 6, seed: int = 0,
            symbols: list[str] | None = None) -> pd.DataFrame:
    """Synthetische Tagesschlusskurse (Werktage, tz-naiver Index).

    Aktien/ETFs/Futures: geometrische Irrfahrt ab 100, Log-Rendite ~ N(0.0003, 0.012).
    ^TNX/^IRX: Renditeniveau in Prozent ab 4.0 / 4.5, Schritte ~ N(0, 0.03), auf [0.1, 8] geklemmt.
    ^VIX: AR(1) um 18, auf [9, 80] geklemmt. ^VIX3M = VIX * 1.08 + N(0, 0.3), mindestens 9.
    """
    idx = pd.bdate_range(end=end, periods=int(years * 252))
    syms = list(symbols) if symbols is not None else list(_default_symbols())
    n = len(idx)
    rng = np.random.default_rng(seed)
    vix = _vix_path(rng, n) if ("^VIX" in syms or "^VIX3M" in syms) else None

    cols: dict[str, np.ndarray] = {}
    for s in syms:
        if s == "^VIX":
            cols[s] = vix
        elif s == "^VIX3M":
            cols[s] = np.maximum(9.0, vix * 1.08 + rng.normal(0.0, 0.3, n))
        elif s in ("^TNX", "^IRX"):
            start = 4.0 if s == "^TNX" else 4.5
            cols[s] = _rw(rng, n, start, 0.03, 0.1, 8.0)
        else:
            cols[s] = _gbm(rng, n, 100.0, 0.0003, 0.012)
    return pd.DataFrame(cols, index=idx)


# ── PIT-Beobachtungen und Archiv ────────────────────────────────────────────

def make_obs(source_id: str, metric: str, periods: list, values: list, *,
             unit: str, release_lag_days: int, dataset: str = "test",
             series_id: str | None = None,
             revisions: dict | None = None) -> list[Observation]:
    """Eine Observation je (Periode, Wert). available_at = Periode + release_lag_days (UTC).

    revisions: {Periodenindex: (neuer_wert, zusätzliche_Lag_Tage)} hängt je eine spätere
    Vintage an: available_at = ursprüngliches available_at + zusätzliche_Lag_Tage.
    """
    periods = list(periods)
    values = list(values)
    if len(periods) != len(values):
        raise ValueError("periods und values haben unterschiedliche Länge")
    sid = series_id or metric

    def _obs(period_dt: datetime, value, avail: datetime) -> Observation:
        return Observation(
            source_id=source_id,
            dataset=dataset,
            series_id=sid,
            entity_id="US",
            metric=metric,
            value=_val(value),
            unit=unit,
            observation_time=period_dt,
            available_at=avail,
            retrieved_at=avail,
            vintage_time=avail,
            availability_precision=AvailabilityPrecision.EXACT_DATE,
            parser_version="test",
        )

    out: list[Observation] = []
    avails: list[datetime] = []
    for p, v in zip(periods, values):
        pdt = _utc(p)
        avail = pdt + timedelta(days=release_lag_days)
        avails.append(avail)
        out.append(_obs(pdt, v, avail))
    for idx, (new_value, extra_lag) in (revisions or {}).items():
        out.append(_obs(_utc(periods[idx]), new_value, avails[idx] + timedelta(days=extra_lag)))
    return out


def _fred_like_archive(end: str, years: int, seed: int, include_expectations: bool) -> dict[str, list[Observation]]:
    rng = np.random.default_rng(seed)
    end_ts = pd.Timestamp(end)
    start = end_ts - pd.DateOffset(years=years)
    months = _month_starts(end_ts, years)
    nm, km = len(months), np.arange(len(months), dtype=float)
    weds = pd.date_range(start, end_ts, freq="W-WED")
    sats = pd.date_range(start, end_ts, freq="W-SAT")
    bdays = pd.bdate_range(start, end_ts)
    nb = len(bdays)

    arch: dict[str, list[Observation]] = {}

    def add(src: str, obs: list[Observation]) -> None:
        arch.setdefault(src, []).extend(obs)

    # fred_regime_macro
    cpi = 250.0 * (1.0025 ** km) * (1.0 + rng.normal(0.0, 0.001, nm))
    add("fred_regime_macro", make_obs(
        "fred_regime_macro", "us_cpi", months, cpi, unit="index", release_lag_days=45,
        revisions={nm - 3: (cpi[nm - 3] + 0.2, 30)}))  # drittletzte Periode: Revision +0.2 nach 30 Tagen
    add("fred_regime_macro", make_obs(
        "fred_regime_macro", "fed_total_assets", weds, _rw(rng, len(weds), 8e6, 2e4),
        unit="USD_mn", release_lag_days=1))
    add("fred_regime_macro", make_obs(
        "fred_regime_macro", "usd_broad", bdays, _rw(rng, nb, 120.0, 0.15),
        unit="index", release_lag_days=1))
    add("fred_regime_macro", make_obs(
        "fred_regime_macro", "wti", bdays, _gbm(rng, nb, 75.0, 0.0, 0.015),
        unit="USD/bbl", release_lag_days=1))

    # fred_us_macro
    add("fred_us_macro", make_obs(
        "fred_us_macro", "us_indpro", months, 103.0 + 0.03 * km + rng.normal(0.0, 0.2, nm),
        unit="index", release_lag_days=45))
    add("fred_us_macro", make_obs(
        "fred_us_macro", "us_umcsent", months, 70.0 + rng.normal(0.0, 3.0, nm),
        unit="index", release_lag_days=30))

    # fred_world_macro
    add("fred_world_macro", make_obs(
        "fred_world_macro", "us_payems", months, 158000.0 + 150.0 * km + rng.normal(0.0, 100.0, nm),
        unit="thousands", release_lag_days=35))
    add("fred_world_macro", make_obs(
        "fred_world_macro", "us_initial_claims", sats, 220000.0 + rng.normal(0.0, 10000.0, len(sats)),
        unit="persons", release_lag_days=5))
    add("fred_world_macro", make_obs(
        "fred_world_macro", "us_retail_sales", months,
        700000.0 * (1.003 ** km) * (1.0 + rng.normal(0.0, 0.004, nm)),
        unit="USD_mn", release_lag_days=45))
    add("fred_world_macro", make_obs(
        "fred_world_macro", "us_inventory_sales_ratio", months, 1.37 + rng.normal(0.0, 0.01, nm),
        unit="ratio", release_lag_days=45))

    # fred_market_expectations (Marktseite der Erwartungen, Werte in Prozent)
    if include_expectations:
        src = "fred_market_expectations"
        add(src, make_obs(src, "us_breakeven_5y", bdays, _ar1(rng, nb, 2.3, 0.05),
                          unit="percent", release_lag_days=1))
        add(src, make_obs(src, "us_treasury_2y", bdays, _rw(rng, nb, 4.0, 0.03, 0.1, 8.0),
                          unit="percent", release_lag_days=1))
        add(src, make_obs(src, "us_fed_funds_effective", bdays, _fed_funds(rng, nb),
                          unit="percent", release_lag_days=1))
        add(src, make_obs(src, "us_treasury_10y", bdays, _rw(rng, nb, 4.2, 0.03, 0.1, 8.0),
                          unit="percent", release_lag_days=1))
        add(src, make_obs(src, "us_breakeven_10y", bdays, _ar1(rng, nb, 2.25, 0.05),
                          unit="percent", release_lag_days=1))
        add(src, make_obs(src, "us_inflation_5y5y_forward", bdays, _ar1(rng, nb, 2.2, 0.05),
                          unit="percent", release_lag_days=1))
    return arch


def make_archive(end: str = "2026-10-09", years: int = 8, seed: int = 1,
                 include_expectations: bool = True) -> dict[str, list[Observation]]:
    """Synthetisches PIT-Archiv: source_id -> Observations (Metriken siehe Modulkopf)."""
    return _fred_like_archive(end, years, seed, include_expectations)


# ── Rohstoffdaten ───────────────────────────────────────────────────────────

def make_commodity(end: str = "2026-10-09", years: int = 5, seed: int = 2) -> pd.DataFrame:
    """Synthetische Rohstoffsignale je Werktag (RangeIndex, Spalte `date` als ISO-Text)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=int(years * 252))
    n = len(idx)
    dates = list(idx.strftime("%Y-%m-%d"))
    stocks = _ar1(rng, n, 0.0, 0.04)
    ret60 = rng.normal(0.0, 0.12, n)
    ret20 = rng.normal(0.0, 0.07, n)
    return pd.DataFrame({
        "date": dates,
        "cmd_crude_stocks_vs_5y": stocks,
        "cmd_wti_ret_60d": ret60,
        "cmd_wti_ret_20d": ret20,
    })


# ── Analyse-Eintrag und Kurs-Bars ───────────────────────────────────────────

def make_analysis(ticker: str = "AAPL", *, direction: str = "BULLISH", impact: int | None = 6,
                  surprise: int | None = 5, sector: str | None = "Technology",
                  ttm: str | None = "4-8 Wochen", catalyst: str = "Produktstart") -> dict:
    return {
        "ticker": ticker,
        "info": {"sector": sector},
        "deep_analysis": {
            "direction": direction,
            "impact": impact,
            "surprise": surprise,
            "time_to_materialization": ttm,
            "catalyst": catalyst,
        },
    }


def make_bars(close: pd.Series) -> list[tuple]:
    """(date, high, low, close) je Tag, aufsteigend sortiert, NaN übersprungen.
    high = close * 1.01, low = close * 0.99."""
    s = close.dropna().sort_index()
    return [(pd.Timestamp(ts).date(), float(c) * 1.01, float(c) * 0.99, float(c)) for ts, c in s.items()]


def loaders(px: pd.DataFrame, archive: dict, commodity: pd.DataFrame) -> dict[str, Callable]:
    """Lader-Stubs mit der Signatur der echten Lader (cfg bzw. cfg, Ticker)."""
    return {
        "archive": lambda cfg: archive,
        "prices": lambda cfg, t: px,
        "commodity": lambda cfg, t: commodity,
    }


def bars_fn_from(px: pd.DataFrame,
                 extra: dict[str, pd.Series] | None = None) -> Callable[[str, object, object], list[tuple]]:
    """fn(sym, start, end) -> make_bars(Serie), auf start <= date <= end begrenzt.
    `extra` hat Vorrang vor `px`. Unbekanntes Symbol -> []."""
    extra = dict(extra or {})

    def fn(sym, start, end):
        if sym in extra:
            series = extra[sym]
        elif sym in px.columns:
            series = px[sym]
        else:
            return []
        lo, hi = pd.Timestamp(start).date(), pd.Timestamp(end).date()
        return [b for b in make_bars(series) if lo <= b[0] <= hi]

    return fn
