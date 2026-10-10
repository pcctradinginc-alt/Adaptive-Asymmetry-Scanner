"""modules/expectation_alpha/data.py – PIT-Datenzugriff (wiederverwendet, keine eigene Datenhaltung).

* Makro/Markterwartungen: ExternalArchive (ALFRED-Vintages). Sichtbar ist nur, was vor dem
  Stichtag 00:00 UTC verfügbar war (world_model.pit_snapshots).
* Marktschlüsse: nur abgeschlossene Tagesbalken. Vor close_complete_hour_utc gilt der
  Entscheidungstag als unfertig und wird verworfen. Die Grenze wird auch auf injizierte Daten
  angewendet, z. B. aus Tests oder Backfills.
* Commodity: commodity_intelligence.build (PIT, Frische-Grenzen, Qualitätsprüfung). Zeilen nach
  dem Entscheidungstag werden verworfen.

Jeder Lader ist injizierbar (Tests, E2E-Dry-Run). Ein Ladefehler wird als Fehler protokolliert.
Es gibt keinen Default-Wert und keine stille Lücke.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from modules.expectation_alpha.schemas import FeatureValue, RunErrors, OK, STALE, UNAVAILABLE

log = logging.getLogger(__name__)


@dataclass
class Inputs:
    decision_time: datetime
    archive_obs: dict = field(default_factory=dict)       # source_id -> [Observation]
    px: pd.DataFrame = field(default_factory=pd.DataFrame)  # Tagesschlusskurse (Spalten = Symbole)
    commodity: pd.DataFrame = field(default_factory=pd.DataFrame)  # Index = Datum, Spalten cmd_*
    errors: RunErrors = field(default_factory=RunErrors)
    provenance: dict = field(default_factory=dict)         # Quelle -> {loaded, rows, last}

    @property
    def decision_date(self) -> date:
        return self.decision_time.date()


def ensure_utc(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def last_complete_close(decision_time: datetime, close_hour_utc: int = 21) -> date:
    """Letzter Tag, dessen Schlusskurs zur Entscheidungszeit sicher feststeht."""
    t = ensure_utc(decision_time)
    return t.date() if t.hour >= close_hour_utc else t.date() - timedelta(days=1)


def pit_prices(px: pd.DataFrame, decision_time: datetime, close_hour_utc: int = 21) -> pd.DataFrame:
    """Schneidet jede Zeile nach dem letzten abgeschlossenen Handelstag ab (Look-ahead-Schutz)."""
    if px is None or px.empty:
        return pd.DataFrame()
    px = px.copy()
    px.index = pd.DatetimeIndex(pd.to_datetime(px.index)).tz_localize(None).normalize()
    cut = pd.Timestamp(last_complete_close(decision_time, close_hour_utc))
    px = px[px.index <= cut].sort_index()
    px = px[~px.index.duplicated(keep="last")]
    return px.dropna(subset=["SPY"]) if "SPY" in px else px    # Raster = US-Handelstage (wie world_model)


def weekly_grid(px: pd.DataFrame) -> list[pd.Timestamp]:
    """Letzter Handelstag je Woche (wie world_model.build_world) – inklusive laufender Woche."""
    if px is None or px.empty or "SPY" not in px:
        return []
    s = px["SPY"].dropna()
    ser = pd.Series(s.index, index=s.index)
    return list(ser.groupby(ser.index.to_period("W-FRI")).max())


# ── Lader (Default: Netzwerk/Archiv; in Tests injiziert) ─────────────────────
def _default_archive(cfg: dict) -> dict:
    from modules import world_model as wm
    from modules.external.archive import ExternalArchive
    root = (cfg.get("data") or {}).get("archive_root", "outputs/external_data")
    out = wm.load_archive(root)
    src = (cfg.get("data") or {}).get("expectations_source")
    if src:
        try:
            out[src] = ExternalArchive(root).load(src)
        except (OSError, ValueError, KeyError) as e:
            log.warning(f"expectation_alpha: Archivquelle {src} nicht lesbar ({e}) -> Markterwartungen fehlen")
            out[src] = []
    return out


def _default_prices(cfg: dict, decision_time: datetime) -> pd.DataFrame:
    import yfinance as yf
    dc = cfg.get("data") or {}
    start = (decision_time.date() - timedelta(days=int(365.25 * int(dc.get("market_history_years", 6))))).isoformat()
    data = yf.download(list(dc.get("market_symbols") or []), start=start, auto_adjust=True, progress=False,
                       threads=True)
    px = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    return px.dropna(how="all")


def _default_commodity(cfg: dict, decision_time: datetime) -> pd.DataFrame:
    from modules import commodity_intelligence as ci
    start = (decision_time.date() - timedelta(days=int(365.25 * 4))).isoformat()
    df, _status = ci.build(now=decision_time, start=start, write=False)
    return df


def load_inputs(decision_time: datetime, cfg: dict, *, archive_fn=None, prices_fn=None,
                commodity_fn=None) -> Inputs:
    """Alle Rohdaten eines Laufs. Jeder Lader darf scheitern: Fehler -> errors, Daten leer."""
    t = ensure_utc(decision_time)
    inp = Inputs(decision_time=t)
    close_h = int((cfg.get("data") or {}).get("close_complete_hour_utc", 21))
    try:
        inp.archive_obs = (archive_fn or _default_archive)(cfg) or {}
        inp.provenance["archive"] = {k: len(v) for k, v in inp.archive_obs.items()}
    except Exception as e:  # noqa: BLE001 – Datenfehler wird ERROR-Kontext, nie stiller Default
        inp.errors.add("archive", e)
    try:
        raw = prices_fn(cfg, t) if prices_fn else _default_prices(cfg, t)
        inp.px = pit_prices(raw, t, close_h)
        inp.provenance["prices"] = {"rows": int(len(inp.px)), "symbols": sorted(map(str, inp.px.columns)),
                                    "last": inp.px.index[-1].date().isoformat() if len(inp.px) else None,
                                    "adjustment": "auto_adjust (nur Renditen/Verhältnisänderungen genutzt)"}
        if inp.px.empty or "SPY" not in inp.px:
            inp.errors.add("prices", "keine abgeschlossenen Marktschlüsse (SPY) verfügbar")
    except Exception as e:  # noqa: BLE001
        inp.errors.add("prices", e)
    try:
        cdf = commodity_fn(cfg, t) if commodity_fn else _default_commodity(cfg, t)
        if cdf is not None and not cdf.empty:
            cdf = cdf.copy()
            cdf.index = pd.DatetimeIndex(pd.to_datetime(cdf.pop("date") if "date" in cdf else cdf.index))
            cdf = cdf[cdf.index <= pd.Timestamp(t.date())].sort_index()
        inp.commodity = cdf if cdf is not None else pd.DataFrame()
        inp.provenance["commodity"] = {"rows": int(len(inp.commodity)),
                                       "last": inp.commodity.index[-1].date().isoformat()
                                       if len(inp.commodity) else None}
    except Exception as e:  # noqa: BLE001
        inp.errors.add("commodity", e)
        inp.commodity = pd.DataFrame()
    return inp


# ── PIT-Reihen ──────────────────────────────────────────────────────────────
def latest_vintage(observations, metric: str, cut: datetime):
    """Jüngste Periode mit ihrer jüngsten Vintage, die vor `cut` verfügbar war (oder None)."""
    best = None
    for o in observations or []:
        if o.metric != metric or o.value is None or o.available_at is None or o.available_at >= cut:
            continue
        if best is None or o.observation_time > best.observation_time or (
                o.observation_time == best.observation_time and o.available_at >= best.available_at):
            best = o
    return best


def feature_from_obs(name: str, o, *, unit: str, source: str, transformation: str, as_of: datetime,
                     max_age_days: int, value: float | None = None) -> FeatureValue:
    """FeatureValue mit Provenienz der zugrunde liegenden Vintage (value überschreibt o.value)."""
    if o is None:
        return FeatureValue(name, None, unit, source, transformation=transformation, status=UNAVAILABLE)
    age = (as_of - o.observation_time).total_seconds() / 86400.0
    stale = age > max_age_days
    vq = (o.attrs or {}).get("vintage_quality") or (o.attrs or {}).get("revision_status")
    conf = 1.0 if not stale else 0.0
    if vq and "UNKNOWN" in str(vq).upper():
        conf *= 0.8
    v = o.value if value is None else value
    return FeatureValue(name, None if stale else v, unit, source,
                        observed_at=o.observation_time.isoformat(),
                        published_at=(o.source_release_time or o.available_at).isoformat(),
                        available_at=o.available_at.isoformat(), retrieved_at=o.retrieved_at.isoformat(),
                        vintage=(o.vintage_time or o.available_at).isoformat(), transformation=transformation,
                        freshness_days=age, confidence=conf, status=STALE if stale else OK)


def pit_weekly(observations, metric: str, dates: list[pd.Timestamp], max_age_days: int,
               how: str = "level") -> pd.Series:
    """Je Stichtag der zu Tagesbeginn bekannte Wert (Vintage < Stichtag 00:00 UTC), veraltet -> NaN.
    how: level | ann_3m (3M-Rate annualisiert, Prozent)."""
    from modules import world_model as wm
    snaps = wm.pit_snapshots(observations or [], metric, dates)
    out = {}
    for d in dates:
        s = snaps.get(d)
        if s is None or s.empty:
            out[d] = np.nan
            continue
        last_t = s.index[-1]
        if (pd.Timestamp(d).tz_localize("UTC") - pd.Timestamp(last_t)).days > max_age_days:
            out[d] = np.nan
            continue
        v = float(s.iloc[-1])
        if how == "level":
            out[d] = v
        elif how == "ann_3m":
            # exakt 3 Kalendermonate zurück (91 Tage träfen je nach Monatslänge die Periode 4 Monate zurück)
            prev = wm._value_before(s, last_t - pd.DateOffset(months=3))
            out[d] = ((v / prev) ** 4 - 1.0) * 100.0 if prev else np.nan
        else:
            raise ValueError(how)
    return pd.Series(out, dtype=float)
