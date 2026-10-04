"""
modules/commodity_intelligence.py – Commodity-Feature-Engine (RESEARCH/SHADOW, keine Produktionswirkung).

    python -m modules.commodity_intelligence build     (Feature-Store + Status, nach External Data)

Quellen (Archiv outputs/external_data, PIT): fred_regime_macro (WTI, wiederverwendet), fred_commodities,
eia_petroleum_weekly, eia_natural_gas, cftc_cot. Konfiguration: config/commodity_intelligence.yaml.

Ablauf je Handelstag D (Stichtag D 21:00 UTC):
  1. PIT-Rekonstruktion jeder Reihe: je Periode die jüngste Vintage mit available_at <= Stichtag
     (Revisionen nie rückwirkend, unveröffentlichte Perioden nie sichtbar).
  2. Frische: Wert gilt nur bis zur nächsten erwarteten Veröffentlichung (age_days <= Grenze je Frequenz
     + Veröffentlichungsverzug); darüber -> UNAVAILABLE (nie 0, nie neutral, nie stilles Fortschreiben).
  3. Qualität: negative Preise (außer WTI), Extrembewegungen, widersprüchliche Duplikate, falsche
     Frequenz, fehlende Releases, Einheitenwechsel, COT > OI, Mapping-Änderung -> Befund im Status;
     unmögliche Werte/Einheitenwechsel/falsche Frequenz -> Reihe UNAVAILABLE.
  4. Features (Datumsebene): Preis, Fundamental, Positionierung, Divergenzen.
  5. Equity-Exposure-Mapping (versioniert, NON_PIT) erzeugt erst beim Anbinden ans Panel die
     Kreuzfeatures cmdx_<commodity>_<feature> = erwartete Richtung × Datums-Feature (nur gemappte Titel;
     nicht gemappt -> NaN, nie 0).

Ausgaben: outputs/research/feature_store/commodity_daily.csv.gz (Datumsebene + age_days + Revisionsflags),
outputs/research/commodity_intelligence.json (Status, Health, Qualität, Mapping). Kein Feature hebt
Scores oder blockiert Trades; Wirkung nur über Hypothese -> Vertrag -> OOS -> Challenger -> Forward ->
PromotionController -> ProductionIntelligenceAdapter.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from modules.external.pit import ensure_utc
from modules.external.sources.commodities import load_config

log = logging.getLogger(__name__)

ARCHIVE_ROOT = Path("outputs/external_data")
STORE_PATH = Path("outputs/research/feature_store/commodity_daily.csv.gz")
STATUS_PATH = Path("outputs/research/commodity_intelligence.json")
STATUS_HISTORY = Path("outputs/research/commodity_intelligence_history.jsonl")
CUTOFF_HOUR_UTC = 21
GRID_START = "2016-01-01"
FEATURE_VERSION = "cmd-features-1"          # = config features.version (Test prüft Gleichheit)

# Commodity -> Preisreihe (Schlüssel in PRICE_SERIES) und COT-Markt. Gold/Silber: kein offizieller
# kostenloser Preis -> nur Positionierung (Preis UNAVAILABLE).
COMMODITY_PRICE = {"oil": "wti", "natural_gas": "henry_hub", "copper": "copper", "corn": "corn",
                   "wheat": "wheat", "soybeans": "soybeans", "brent": "brent"}
COMMODITY_COT = {"oil": "crude_oil", "natural_gas": "natural_gas", "gold": "gold", "silver": "silver",
                 "copper": "copper", "corn": "corn", "wheat": "wheat", "soybeans": "soybeans"}
FEATURE_GROUPS = ("price", "fundamental", "positioning")


def _cfg() -> dict:
    return load_config()


# ── Reihen-Katalog aus der Config ─────────────────────────────────────────────

def price_series(cfg: dict | None = None) -> dict[str, dict]:
    cfg = cfg or _cfg()
    out = {}
    for k, r in ((cfg.get("fred") or {}).get("reuse") or {}).items():
        out[k] = {"source_id": r["source_id"], "metric": r["metric"], "frequency": r["frequency"],
                  "unit": r["unit"], "lag_days": 7}
    for s in (cfg.get("fred") or {}).get("series") or []:
        out[s["key"]] = {"source_id": "fred_commodities", "metric": s["metric"], "frequency": s["frequency"],
                         "unit": s["unit"], "lag_days": 7 if s["frequency"] == "daily" else 0}
    return out


def fundamental_series(cfg: dict | None = None) -> dict[str, dict]:
    cfg = cfg or _cfg()
    out = {}
    for s in (cfg.get("eia") or {}).get("series") or []:
        if s.get("role", "fundamental") != "fundamental":
            continue
        kind = ("stock" if "stocks" in s["key"] or "storage" in s["key"]
                else "monthly_flow" if s["frequency"] == "monthly" else "flow")
        out[s["key"]] = {"source_id": s["source_id"], "metric": s["metric"], "frequency": s["frequency"],
                         "unit": s["unit"], "commodity": s["commodity"], "kind": kind, "series": s["series"],
                         "lag_days": int((s.get("release") or {}).get("days", 0)),
                         "revised": s.get("revision_status") == "revised_monthly"}
    return out


def crosscheck_pairs(cfg: dict | None = None) -> list[tuple[str, dict, dict]]:
    """EIA-Spot (role price_crosscheck) gegen die FRED-Preisreihe derselben Größe."""
    cfg = cfg or _cfg()
    ps = price_series(cfg)
    pairs = {"eia_wti_spot": "wti", "eia_brent_spot": "brent", "eia_henry_hub_spot": "henry_hub"}
    out = []
    for s in (cfg.get("eia") or {}).get("series") or []:
        if s.get("role") == "price_crosscheck" and pairs.get(s["metric"]) in ps:
            out.append((s["metric"], {"source_id": s["source_id"], "metric": s["metric"]}, ps[pairs[s["metric"]]]))
    return out


# ── Feature-Namen (Registry-Grundlage, deterministisch aus der Config) ────────

def _price_feature_names(key: str, freq: str) -> list[str]:
    if freq == "daily":
        return [f"cmd_{key}_ret_{w}d" for w in (1, 5, 20, 60)] + [
            f"cmd_{key}_mom_60_5", f"cmd_{key}_vol_20d", f"cmd_{key}_z_60d", f"cmd_{key}_dd_252d"]
    return [f"cmd_{key}_ret_1m", f"cmd_{key}_ret_3m", f"cmd_{key}_z_36m"]


def _fund_feature_names(key: str, kind: str) -> list[str]:
    if kind == "stock":
        return [f"cmd_{key}_chg_1w", f"cmd_{key}_chg_4w", f"cmd_{key}_chg_z_52w", f"cmd_{key}_vs_5y"]
    if kind == "flow":
        return [f"cmd_{key}_chg_4w", f"cmd_{key}_z_52w"]
    return [f"cmd_{key}_yoy", f"cmd_{key}_chg_3m"]


COT_FEATURE_SUFFIXES = ("mm_net", "comm_net", "mm_net_pct_oi", "comm_net_pct_oi", "mm_net_chg_1w",
                        "mm_net_chg_4w", "mm_net_chg_13w", "mm_pctile_1y", "mm_pctile_3y", "comm_pctile_1y",
                        "comm_pctile_3y", "oi_chg_4w", "mm_extreme")

DIVERGENCE_FEATURES = {
    "cmd_div_oil_price_inventory": "WTI-20T-Rendite und 4W-Rohölbestandsänderung gleichgerichtet "
                                   "(Preis steigt trotz Lageraufbau / fällt trotz Abbau) = 1",
    "cmd_div_oil_positioning_price": "+1 Managed Money 1J-Perzentil >= 0,8 bei fallendem WTI (überfüllt long), "
                                     "-1 <= 0,2 bei steigendem WTI, sonst 0",
    "cmd_div_gas_storage_price": "Henry-Hub-20T-Rendite und Speicher-Abweichung vom 5J-Mittel gleichgerichtet = 1",
    "cmd_div_copper_positioning_price": "wie Öl-Positionierung, Kupfer (monatlicher Preis, wöchentliche COT)",
}


def date_feature_names(cfg: dict | None = None) -> dict[str, list[str]]:
    cfg = cfg or _cfg()
    out = {"price": [], "fundamental": [], "positioning": [], "divergence": list(DIVERGENCE_FEATURES)}
    for k, s in price_series(cfg).items():
        out["price"] += _price_feature_names(k, s["frequency"])
    for k, s in fundamental_series(cfg).items():
        out["fundamental"] += _fund_feature_names(k, s["kind"])
    for m in (cfg.get("cftc") or {}).get("markets") or {}:
        out["positioning"] += [f"cmd_cot_{m}_{x}" for x in COT_FEATURE_SUFFIXES]
    return out


# Kreuzfeatures (Ticker-Ebene): Exposure-Richtung × ausgewähltes Datums-Feature
CROSS_SPEC = {
    "price": [("oil", "cmd_wti_ret_20d"), ("oil", "cmd_wti_ret_60d"), ("oil", "cmd_wti_z_60d"),
              ("natural_gas", "cmd_henry_hub_ret_20d"), ("natural_gas", "cmd_henry_hub_ret_60d"),
              ("copper", "cmd_copper_ret_3m"), ("corn", "cmd_corn_ret_3m"), ("wheat", "cmd_wheat_ret_3m"),
              ("soybeans", "cmd_soybeans_ret_3m")],
    "fundamental": [("oil", "cmd_crude_stocks_vs_5y"), ("oil", "cmd_crude_stocks_chg_z_52w"),
                    ("oil", "cmd_product_supplied_chg_4w"), ("oil", "cmd_refinery_utilization_z_52w"),
                    ("natural_gas", "cmd_natgas_storage_vs_5y"), ("natural_gas", "cmd_natgas_storage_chg_z_52w"),
                    ("natural_gas", "cmd_lng_exports_yoy")],
    "positioning": [(c, f"cmd_cot_{m}_mm_pctile_1y") for c, m in COMMODITY_COT.items()]
                   + [(c, f"cmd_cot_{m}_mm_net_chg_4w") for c, m in COMMODITY_COT.items()],
    "divergence": [("oil", "cmd_div_oil_positioning_price"), ("oil", "cmd_div_oil_price_inventory"),
                   ("natural_gas", "cmd_div_gas_storage_price"), ("copper", "cmd_div_copper_positioning_price")],
}


COMMODITIES = ("oil", "natural_gas", "gold", "silver", "copper", "corn", "wheat", "soybeans")
# Signal-Sprache der Hypothesen-Fabrik: Exposure-Richtung je Titel (NON_PIT) × Datums-Feature (PIT)
EXPOSURE_COLUMNS = {f"cmdexp_{c}": c for c in COMMODITIES}
SIGNAL_DATE_FEATURES = tuple(sorted({fe for spec in CROSS_SPEC.values() for _, fe in spec}))


def cross_name(commodity: str, feature: str) -> str:
    return f"cmdx_{commodity}__{feature[4:]}"


def cross_features(group: str) -> list[str]:
    return [cross_name(c, f) for c, f in CROSS_SPEC.get(group, [])]


# Gruppe -> abhängige Quellen (Source Health: nur abhängige Komponenten werden UNAVAILABLE)
GROUP_SOURCES = {"price": ["fred_regime_macro", "fred_commodities"],
                 "fundamental": ["eia_petroleum_weekly", "eia_natural_gas"],
                 "positioning": ["cftc_cot"],
                 "divergence": ["fred_regime_macro", "fred_commodities", "eia_petroleum_weekly",
                                "eia_natural_gas", "cftc_cot"]}


def date_feature_sources(feature: str) -> list[str]:
    """Quellen, von denen ein Datums-Feature abhängt (Source Health je Feature, nicht je Gruppe)."""
    f = feature
    if f.startswith("cmd_cot_"):
        return ["cftc_cot"]
    out = []
    if "_wti_" in f or f.startswith("cmd_div_oil"):
        out.append("fred_regime_macro")
    if any(k in f for k in ("_brent_", "_henry_hub_", "_copper_ret", "_corn_", "_wheat_", "_soybeans_")) \
            or f in ("cmd_div_gas_storage_price", "cmd_div_copper_positioning_price"):
        out.append("fred_commodities")
    if any(k in f for k in ("_crude_", "_product_supplied", "_refinery_", "_gasoline_", "_distillate_")) \
            or f == "cmd_div_oil_price_inventory":
        out.append("eia_petroleum_weekly")
    if any(k in f for k in ("_natgas_", "_lng_", "_dry_gas_")) or f == "cmd_div_gas_storage_price":
        out.append("eia_natural_gas")
    if "positioning" in f:
        out.append("cftc_cot")
    return out


def feature_contracts(group: str) -> dict[str, list[str]]:
    """{cmdx-Feature: [source_id, ...]} je Gruppe."""
    return {cross_name(c, fe): date_feature_sources(fe) for c, fe in CROSS_SPEC.get(group, [])}


def feature_descriptions() -> dict[str, dict[str, str]]:
    """{gruppe: {cross_feature: Beschreibung}} für die Alt-Data-Registry."""
    out: dict[str, dict[str, str]] = {}
    for g, spec in CROSS_SPEC.items():
        out[g] = {cross_name(c, f): (f"Erwartete Exposure-Richtung {c} (NON_PIT-Mapping) × {f}"
                                     + (f" – {DIVERGENCE_FEATURES[f]}" if f in DIVERGENCE_FEATURES else ""))
                  for c, f in spec}
    return out


# ── PIT-Rekonstruktion ────────────────────────────────────────────────────────

def load_observations(source_id: str, root: Path | None = None) -> list:
    from modules.external.archive import ExternalArchive
    try:
        return ExternalArchive(str(root or ARCHIVE_ROOT)).load(source_id)
    except Exception as e:  # noqa: BLE001 - fehlendes/defektes Archiv -> Reihe UNAVAILABLE (sichtbar)
        log.warning("commodity: Archiv %s nicht lesbar: %r", source_id, e)
        return []


class PitBook:
    """Inkrementelle PIT-Sicht: Ereignisse nach available_at; je (Periode, Feld) die jüngste Vintage."""

    def __init__(self, events: list[tuple[datetime, datetime, datetime, str, float | None, dict]]):
        self.events = sorted(events, key=lambda e: (e[0], e[1]))
        self.i = 0
        self.state: dict[datetime, dict[str, tuple[datetime, float | None, dict]]] = {}
        self.periods: list[datetime] = []

    def advance(self, t: datetime) -> None:
        ev = self.events
        while self.i < len(ev) and ev[self.i][0] <= t:
            av, vint, period, fld, val, attrs = ev[self.i]
            row = self.state.get(period)
            if row is None:
                row = self.state[period] = {}
                bisect.insort(self.periods, period)
            cur = row.get(fld)
            if cur is None or vint >= cur[0]:
                row[fld] = (vint, val, attrs)
            self.i += 1

    def tail(self, n: int, fld: str = "v") -> tuple[list[datetime], list[float], list[dict]]:
        ps, vs, ats = [], [], []
        for p in reversed(self.periods):
            c = self.state[p].get(fld)
            if c is None or c[1] is None:
                continue
            ps.append(p)
            vs.append(c[1])
            ats.append(c[2])
            if len(ps) >= n:
                break
        return ps[::-1], vs[::-1], ats[::-1]

    def rows(self, n: int, fields: tuple[str, ...]) -> list[tuple[datetime, dict]]:
        out = []
        for p in reversed(self.periods):
            r = self.state[p]
            out.append((p, {f: (r[f][1] if f in r else None) for f in fields}))
            if len(out) >= n:
                break
        return out[::-1]


def _events(obs: list, metric: str | None = None, entity: str | None = None,
            field_of=None) -> list[tuple]:
    ev = []
    for o in obs:
        if metric is not None and o.metric != metric:
            continue
        if entity is not None and o.entity_id != entity:
            continue
        fld = field_of(o) if field_of else "v"
        if fld is None:
            continue
        ev.append((ensure_utc(o.available_at), ensure_utc(o.vintage_time or o.available_at),
                   ensure_utc(o.observation_time), fld, o.value,
                   {"unit": o.unit, "revision_status": (o.attrs or {}).get("revision_status")}))
    return ev


# ── Qualitätsprüfungen ───────────────────────────────────────────────────────

EXPECTED_SPACING = {"daily": (1, 5), "weekly": (6, 8), "monthly": (27, 32)}


def quality_checks(name: str, obs: list, frequency: str, cfg: dict | None = None,
                   unit_expected: str | None = None) -> dict:
    """Befunde einer Reihe über das gesamte Archiv. severe -> Reihe UNAVAILABLE."""
    cfg = cfg or _cfg()
    q = cfg.get("quality") or {}
    issues: list[str] = []
    detail: dict = {}
    if not obs:
        return {"issues": ["NO_DATA"], "severe": True, "detail": {}}
    units = sorted({o.unit for o in obs})
    if len(units) > 1 or (unit_expected and units[0] != unit_expected):
        issues.append("UNIT_CHANGED")
        detail["units"] = units
    latest: dict[datetime, tuple] = {}
    conflicts = 0
    for o in obs:
        k = ensure_utc(o.observation_time)
        v = ensure_utc(o.vintage_time or o.available_at)
        if k in latest and latest[k][0] == v and latest[k][1] != o.value:
            conflicts += 1
        if k not in latest or v >= latest[k][0]:
            latest[k] = (v, o.value)
    if conflicts:
        issues.append("DUPLICATE_CONFLICT")
        detail["duplicate_conflicts"] = conflicts
    periods = sorted(latest)
    vals = [latest[p][1] for p in periods]
    metric = obs[0].metric
    neg = [p for p, v in zip(periods, vals) if v is not None and v < 0]
    if neg and metric not in (q.get("allow_negative") or []) and any(s in metric for s in (
            "spot", "brent", "henry_hub", "copper", "wheat", "corn", "soybeans", "stocks", "storage",
            "production", "imports", "exports", "supplied", "utilization", "cot_")):
        issues.append("NEGATIVE_VALUE")
        detail["negative_example"] = neg[0].date().isoformat()
    if len(periods) >= 10:
        gaps = np.diff(np.array([p.timestamp() for p in periods])) / 86400
        lo, hi = EXPECTED_SPACING.get(frequency, (0, 1e9))
        med = float(np.median(gaps))
        if not (lo <= med <= hi):
            issues.append("WRONG_FREQUENCY")
            detail["median_spacing_days"] = med
        cadence = {"daily": 1, "weekly": 7, "monthly": 31}.get(frequency, 7)
        factor = float(q.get("max_release_gap_factor", 2.0))
        allow = cadence * factor + (4 if frequency == "daily" else 0)
        big = [(periods[i].date().isoformat(), round(float(g), 1)) for i, g in enumerate(gaps) if g > allow]
        if big:
            issues.append("MISSING_RELEASES")
            detail["gaps"] = big[-5:]
            detail["n_gaps"] = len(big)
    lim = (q.get("max_abs_daily_return") or {}).get(metric) if frequency == "daily" else \
        (q.get("max_abs_monthly_return") if frequency == "monthly" else None)
    if lim:
        ext = []
        for (p0, v0), (p1, v1) in zip(zip(periods, vals), zip(periods[1:], vals[1:])):
            if v0 and v1 is not None and v0 > 0 and abs(v1 / v0 - 1) > float(lim):
                ext.append(p1.date().isoformat())
        if ext:
            issues.append("EXTREME_MOVE")                 # Befund, kein Ausschluss (z. B. WTI 20.04.2020)
            detail["extreme_moves"] = ext[-5:]
    severe = bool({"UNIT_CHANGED", "NEGATIVE_VALUE", "WRONG_FREQUENCY", "NO_DATA"} & set(issues))
    return {"issues": issues, "severe": severe, "detail": detail, "n_periods": len(periods),
            "first_period": periods[0].date().isoformat(), "last_period": periods[-1].date().isoformat()}


def crosscheck(eia_obs: list, fred_obs: list, max_rel: float) -> dict:
    """EIA-Spot vs. FRED gleicher Tag (jüngste Vintages): Anteil Abweichungen > max_rel."""
    def last(obs):
        d: dict = {}
        for o in obs:
            k = ensure_utc(o.observation_time).date()
            v = ensure_utc(o.vintage_time or o.available_at)
            if o.value is not None and (k not in d or v >= d[k][0]):
                d[k] = (v, o.value)
        return {k: v[1] for k, v in d.items()}
    a, b = last(eia_obs), last(fred_obs)
    common = sorted(set(a) & set(b))[-500:]
    if not common:
        return {"status": "NO_OVERLAP", "n": 0}
    bad = [k.isoformat() for k in common if b[k] and abs(a[k] / b[k] - 1) > max_rel]
    return {"status": "MISMATCH" if len(bad) > 0.05 * len(common) else "OK", "n": len(common),
            "n_mismatch": len(bad), "examples": bad[-3:]}


# ── Feature-Formeln (reine Funktionen, getestet) ─────────────────────────────

def _ret(v: list[float], k: int) -> float | None:
    if len(v) <= k or v[-1 - k] is None or v[-1 - k] <= 0 or v[-1] is None:
        return None
    return v[-1] / v[-1 - k] - 1.0


def price_features(vals: list[float], frequency: str, min_hist: int = 40) -> dict[str, float | None]:
    """Rendite über Beobachtungsschritte; negative Vorwerte (WTI 2020) -> Rendite None, nie 0."""
    out: dict[str, float | None] = {}
    if frequency == "daily":
        for w in (1, 5, 20, 60):
            out[f"ret_{w}d"] = _ret(vals, w)
        out["mom_60_5"] = (vals[-6] / vals[-61] - 1.0) if len(vals) > 60 and vals[-61] > 0 else None
        r = [vals[i] / vals[i - 1] - 1 for i in range(max(1, len(vals) - 20), len(vals)) if vals[i - 1] > 0]
        out["vol_20d"] = float(np.std(r, ddof=1) * math.sqrt(252)) if len(r) >= 15 else None
        w60 = vals[-60:]
        sd = float(np.std(w60, ddof=1)) if len(w60) >= min_hist else 0.0
        out["z_60d"] = (vals[-1] - float(np.mean(w60))) / sd if sd > 0 else None
        w252 = vals[-252:]
        mx = max(w252) if len(w252) >= 60 else None
        out["dd_252d"] = (vals[-1] / mx - 1.0) if mx and mx > 0 else None
    else:
        out["ret_1m"] = _ret(vals, 1)
        out["ret_3m"] = _ret(vals, 3)
        w = vals[-36:]
        sd = float(np.std(w, ddof=1)) if len(w) >= 24 else 0.0
        out["z_36m"] = (vals[-1] - float(np.mean(w))) / sd if sd > 0 else None
    return out


def _seasonal_dev(periods: list[datetime], vals: list[float], years: int = 5) -> float | None:
    """Abweichung des aktuellen Niveaus vom Mittel derselben Kalenderwoche (±7 T) der Vorjahre (>= 3)."""
    p0, v0 = periods[-1], vals[-1]
    ref = []
    for y in range(1, years + 1):
        target = p0 - timedelta(days=round(365.25 * y))
        i = bisect.bisect_left(periods, target)
        cand = [j for j in (i - 1, i) if 0 <= j < len(periods) and abs((periods[j] - target).days) <= 7]
        if cand:
            j = min(cand, key=lambda j: abs((periods[j] - target).days))
            ref.append(vals[j])
    if len(ref) < 3:
        return None
    m = float(np.mean(ref))
    return (v0 / m - 1.0) if m > 0 else None


def fundamental_features(periods: list[datetime], vals: list[float], kind: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    n = len(vals)
    if kind == "stock":
        out["chg_1w"] = vals[-1] - vals[-2] if n >= 2 else None
        out["chg_4w"] = vals[-1] - vals[-5] if n >= 5 else None
        chg = np.diff(vals[-53:]) if n >= 2 else np.array([])
        sd = float(np.std(chg, ddof=1)) if len(chg) >= 40 else 0.0
        out["chg_z_52w"] = (float(chg[-1]) - float(np.mean(chg))) / sd if sd > 0 else None
        out["vs_5y"] = _seasonal_dev(periods, vals)
    elif kind == "flow":
        if n >= 8 and np.mean(vals[-8:-4]) > 0:
            out["chg_4w"] = float(np.mean(vals[-4:]) / np.mean(vals[-8:-4]) - 1.0)
        else:
            out["chg_4w"] = None
        w = vals[-52:]
        sd = float(np.std(w, ddof=1)) if len(w) >= 40 else 0.0
        out["z_52w"] = (float(np.mean(vals[-4:])) - float(np.mean(w))) / sd if sd > 0 else None
    else:
        p0 = periods[-1]
        prev = [v for p, v in zip(periods, vals) if p.year == p0.year - 1 and p.month == p0.month]
        out["yoy"] = (vals[-1] / prev[-1] - 1.0) if prev and prev[-1] > 0 else None
        out["chg_3m"] = (float(np.mean(vals[-3:]) / np.mean(vals[-6:-3]) - 1.0)
                         if n >= 6 and np.mean(vals[-6:-3]) > 0 else None)
    return out


COT_FIELDS = ("total_open_interest", "producer_merchant_long", "producer_merchant_short", "swap_long",
              "swap_short", "managed_money_long", "managed_money_short", "managed_money_spreading",
              "other_reportables_long", "other_reportables_short")


def _pctile(hist: list[float], x: float) -> float:
    return float(np.mean(np.array(hist) <= x))


def cot_features(rows: list[tuple[datetime, dict]], min_1y: int = 40, min_3y: int = 120,
                 extreme: float = 0.95) -> tuple[dict[str, float | None], int]:
    """rows: [(Report-Datum, {Feld: Wert})] aufsteigend. -> (Features, #OI-inkonsistente Reports)."""
    from modules.external.sources.commodities import oi_consistent
    clean, bad = [], 0
    for p, r in rows:
        if any(r.get(f) is None for f in ("total_open_interest", "managed_money_long", "managed_money_short",
                                          "producer_merchant_long", "producer_merchant_short",
                                          "swap_long", "swap_short")):
            continue
        ok, _ = oi_consistent(r)
        if not ok:
            bad += 1
            continue
        oi = r["total_open_interest"]
        mm = r["managed_money_long"] - r["managed_money_short"]
        cm = (r["producer_merchant_long"] + r["swap_long"]) - (r["producer_merchant_short"] + r["swap_short"])
        clean.append((p, oi, mm, cm, mm / oi, cm / oi))
    out: dict[str, float | None] = {k: None for k in COT_FEATURE_SUFFIXES}
    if not clean:
        return out, bad
    p, oi, mm, cm, mmp, cmp_ = clean[-1]
    out.update(mm_net=mm, comm_net=cm, mm_net_pct_oi=mmp, comm_net_pct_oi=cmp_)
    mmps = [c[4] for c in clean]
    cmps = [c[5] for c in clean]
    for k, lag in (("mm_net_chg_1w", 1), ("mm_net_chg_4w", 4), ("mm_net_chg_13w", 13)):
        out[k] = mmps[-1] - mmps[-1 - lag] if len(mmps) > lag else None
    out["mm_pctile_1y"] = _pctile(mmps[-52:], mmp) if len(mmps) >= min_1y else None
    out["mm_pctile_3y"] = _pctile(mmps[-156:], mmp) if len(mmps) >= min_3y else None
    out["comm_pctile_1y"] = _pctile(cmps[-52:], cmp_) if len(cmps) >= min_1y else None
    out["comm_pctile_3y"] = _pctile(cmps[-156:], cmp_) if len(cmps) >= min_3y else None
    out["oi_chg_4w"] = (oi / clean[-5][1] - 1.0) if len(clean) >= 5 and clean[-5][1] > 0 else None
    pc = out["mm_pctile_3y"]
    out["mm_extreme"] = None if pc is None else (1.0 if pc >= extreme else -1.0 if pc <= 1 - extreme else 0.0)
    return out, bad


def divergence_features(f: dict) -> dict[str, float | None]:
    def same_sign(a, b):
        if a is None or b is None or a == 0 or b == 0:
            return None
        return 1.0 if (a > 0) == (b > 0) else 0.0

    def crowd(pct, ret):
        if pct is None or ret is None:
            return None
        return 1.0 if (pct >= 0.8 and ret < 0) else -1.0 if (pct <= 0.2 and ret > 0) else 0.0
    return {
        "cmd_div_oil_price_inventory": same_sign(f.get("cmd_wti_ret_20d"), f.get("cmd_crude_stocks_chg_4w")),
        "cmd_div_oil_positioning_price": crowd(f.get("cmd_cot_crude_oil_mm_pctile_1y"), f.get("cmd_wti_ret_20d")),
        "cmd_div_gas_storage_price": same_sign(f.get("cmd_henry_hub_ret_20d"), f.get("cmd_natgas_storage_vs_5y")),
        "cmd_div_copper_positioning_price": crowd(f.get("cmd_cot_copper_mm_pctile_1y"), f.get("cmd_copper_ret_1m")),
    }


# ── Feature-Build über das Handelstage-Raster ────────────────────────────────

def _max_age(frequency: str, lag_days: int, cfg: dict) -> int:
    base = int(((cfg.get("features") or {}).get("max_age_days") or {}).get(frequency, 14))
    return base + int(lag_days)


def build(now: datetime | None = None, root: Path | None = None, start: str = GRID_START,
          cfg: dict | None = None, write: bool = True) -> tuple[pd.DataFrame, dict]:
    """-> (Datums-Feature-Frame, Status). Jede Zeile = Stichtag D 21:00 UTC (PIT)."""
    cfg = cfg or _cfg()
    now = ensure_utc(now) or datetime.now(timezone.utc)
    fc = cfg.get("features") or {}
    src_obs: dict[str, list] = {}

    def obs_of(sid):
        if sid not in src_obs:
            src_obs[sid] = load_observations(sid, root)
        return src_obs[sid]

    status: dict = {"generated": now.isoformat(timespec="seconds"), "version": cfg.get("version"),
                    "feature_version": fc.get("version"), "research_status": cfg.get("research_status"),
                    "series": {}, "quality": {}, "crosscheck": {}, "cot": {}, "unavailable": {},
                    "storage_surprise": fc.get("storage_surprise", "EXPECTATION_UNKNOWN")}
    books: dict[str, tuple[PitBook, dict]] = {}
    for key, s in {**{f"price:{k}": v for k, v in price_series(cfg).items()},
                   **{f"fund:{k}": v for k, v in fundamental_series(cfg).items()}}.items():
        obs = [o for o in obs_of(s["source_id"]) if o.metric == s["metric"]]
        qc = quality_checks(key, obs, s["frequency"], cfg, unit_expected=s["unit"])
        status["quality"][key] = qc
        if qc["severe"]:
            status["unavailable"][key] = qc["issues"]
            continue
        books[key] = (PitBook(_events(obs)), s)
    for name, eia_s, fred_s in crosscheck_pairs(cfg):
        status["crosscheck"][name] = crosscheck(
            [o for o in obs_of(eia_s["source_id"]) if o.metric == eia_s["metric"]],
            [o for o in obs_of(fred_s["source_id"]) if o.metric == fred_s["metric"]],
            float((cfg.get("quality") or {}).get("crosscheck_max_rel_diff", 0.02)))
    for name, cc in status["crosscheck"].items():
        if cc.get("status") == "MISMATCH":
            status["quality"].setdefault(f"crosscheck:{name}", {"issues": ["CROSSCHECK_MISMATCH"], "severe": False,
                                                                 "detail": cc})
    cot_books: dict[str, PitBook] = {}
    cot_obs = obs_of("cftc_cot")
    for m in ((cfg.get("cftc") or {}).get("markets") or {}):
        mo = [o for o in cot_obs if o.entity_id == m]
        if not mo:
            status["unavailable"][f"cot:{m}"] = ["NO_DATA"]
            continue
        qc = quality_checks(f"cot:{m}", [o for o in mo if o.metric == "cot_total_open_interest"], "weekly", cfg,
                            unit_expected="contracts")
        status["quality"][f"cot:{m}"] = qc
        if qc["severe"]:
            status["unavailable"][f"cot:{m}"] = qc["issues"]
            continue
        cot_books[m] = PitBook(_events(mo, field_of=lambda o: o.metric[4:] if o.metric.startswith("cot_") else None))

    grid = pd.bdate_range(start, now.date())
    rows = []
    oi_bad: dict[str, int] = {}
    min1, min3 = int(fc.get("min_history_pctile_1y", 40)), int(fc.get("min_history_pctile_3y", 120))
    for d in grid:
        t = datetime.combine(d.date(), time(CUTOFF_HOUR_UTC), timezone.utc)
        if t > now:
            t = now
        r: dict = {"date": d.date().isoformat()}
        for key, (book, s) in books.items():
            book.advance(t)
            name = key.split(":", 1)[1]
            n = 300 if s["frequency"] == "daily" else 320 if s["frequency"] == "weekly" else 72
            ps, vs, ats = book.tail(n)
            if not ps:
                continue
            age = (t - ps[-1]).days
            r[f"age_{name}"] = age
            if age > _max_age(s["frequency"], s.get("lag_days", 0), cfg):
                continue                                   # nicht mehr frisch -> UNAVAILABLE (NaN)
            r[f"rev_{name}"] = 1.0 if (ats[-1] or {}).get("revision_status") in (
                "backfill_latest_vintage", "revised_after_first_seen") else 0.0
            if key.startswith("price:"):
                feats = price_features(vs, s["frequency"], int(fc.get("min_history_pctile_1y", 40)))
            else:
                feats = fundamental_features(ps, vs, s["kind"])
            for k, v in feats.items():
                if v is not None and np.isfinite(v):
                    r[f"cmd_{name}_{k}"] = float(v)
        for m, book in cot_books.items():
            book.advance(t)
            crow = book.rows(170, COT_FIELDS)
            if not crow:
                continue
            age = (t - crow[-1][0]).days
            r[f"age_cot_{m}"] = age
            if age > _max_age("weekly", 6, cfg):
                continue
            feats, bad = cot_features(crow, min1, min3, float(fc.get("positioning_extreme_pctile", 0.95)))
            oi_bad[m] = max(oi_bad.get(m, 0), bad)
            for k, v in feats.items():
                if v is not None and np.isfinite(v):
                    r[f"cmd_cot_{m}_{k}"] = float(v)
        for k, v in divergence_features(r).items():
            if v is not None:
                r[k] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    names = date_feature_names(cfg)
    missing = [c for g in names.values() for c in g if c not in df]
    if missing:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    status["cot"] = {"oi_inconsistent_reports": oi_bad}
    status["coverage"] = coverage_summary(df, names)
    status["latest"] = latest_snapshot(df, names)
    status["mapping"] = mapping_status()
    status["sources"] = source_summary(root)
    status["n_dates"] = int(len(df))
    if write:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(STORE_PATH, index=False, compression="gzip", float_format="%.6g")
        prev = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}
        if prev.get("mapping", {}).get("hash") and prev["mapping"]["hash"] != status["mapping"]["hash"]:
            status["quality"]["mapping"] = {"issues": ["MAPPING_CHANGED"], "severe": False,
                                           "detail": {"previous": prev["mapping"]["hash"],
                                                      "current": status["mapping"]["hash"]}}
        STATUS_PATH.write_text(json.dumps(status, indent=1, default=str, ensure_ascii=False))
        with open(STATUS_HISTORY, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"generated": status["generated"], "unavailable": status["unavailable"],
                                 "coverage": status["coverage"], "mapping_hash": status["mapping"]["hash"]},
                                default=str) + "\n")
    return df, status


def coverage_summary(df: pd.DataFrame, names: dict) -> dict:
    out = {}
    for g, cols in names.items():
        cols = [c for c in cols if c in df]
        if not cols or df.empty:
            out[g] = {"active_features": 0, "share_dates_any": 0.0}
            continue
        nn = df[cols].notna()
        out[g] = {"active_features": int((nn.sum() > 0).sum()), "n_features": len(cols),
                  "share_dates_any": round(float(nn.any(axis=1).mean()), 3),
                  "first_date": (df.loc[nn.any(axis=1), "date"].min() if nn.any(axis=1).any() else None)}
    return out


def latest_snapshot(df: pd.DataFrame, names: dict) -> dict:
    if df.empty:
        return {}
    last = df.iloc[-1]
    keys = ["cmd_wti_ret_20d", "cmd_henry_hub_ret_20d", "cmd_crude_stocks_vs_5y", "cmd_natgas_storage_vs_5y",
            "cmd_cot_crude_oil_mm_pctile_1y", "cmd_cot_natural_gas_mm_pctile_1y", "cmd_cot_gold_mm_pctile_1y"]
    return {"date": last["date"], **{k: (None if k not in last or pd.isna(last[k]) else round(float(last[k]), 4))
                                     for k in keys}}


def source_summary(root: Path | None = None) -> dict:
    """Health der fünf Quellen aus dem Registry-Health-Snapshot (Orchestrator)."""
    from modules.external.registry import load_health
    h = load_health(root or ARCHIVE_ROOT)
    out = {}
    for sid in ("fred_regime_macro", "fred_commodities", "eia_petroleum_weekly", "eia_natural_gas", "cftc_cot"):
        x = h.get(sid) or {}
        out[sid] = {k: x.get(k) for k in ("status", "staleness", "latest_observation", "last_success",
                                           "consecutive_failures", "message")}
        out[sid]["health"] = health_label(x)
    return out


def health_label(h: dict) -> str:
    """HEALTHY / DEGRADED / STALE / BROKEN / UNVALIDATED aus dem Orchestrator-Health."""
    if not h or not h.get("last_success"):
        return "UNVALIDATED"
    st = h.get("status")
    if st in ("FAIL", "SCHEMA_CHANGED", "BLOCKED") or int(h.get("consecutive_failures") or 0) >= 3:
        return "BROKEN"
    if h.get("staleness") == "STALE":
        return "STALE"
    if st in ("WARN", "AUTH_MISSING") or int(h.get("pit_integrity_failures") or 0) > 0:
        return "DEGRADED"
    return "HEALTHY"


# ── Equity-Exposure-Mapping (versioniert, NON_PIT) ───────────────────────────

def mapping_rules(cfg: dict | None = None) -> dict:
    return (cfg or _cfg()).get("exposure_mapping") or {}


def mapping_hash(cfg: dict | None = None) -> str:
    return hashlib.sha256(json.dumps(mapping_rules(cfg), sort_keys=True, default=str).encode()).hexdigest()[:12]


def mapping_status(cfg: dict | None = None) -> dict:
    m = mapping_rules(cfg)
    tickers = sorted({t for r in m.get("ticker_rules") or [] for t in r.get("tickers") or []})
    return {"version": m.get("version"), "hash": mapping_hash(cfg), "point_in_time_status": m.get("point_in_time_status"),
            "non_pit_mapping": m.get("point_in_time_status") != "PIT", "n_ticker_rules": len(tickers),
            "n_industry_rules": len(m.get("industry_rules") or []), "commodities": sorted(
                {r["commodity"] for r in (m.get("ticker_rules") or []) + (m.get("industry_rules") or [])})}


def exposure_table(tickers: list[str], industries: dict[str, str] | None = None,
                   cfg: dict | None = None) -> pd.DataFrame:
    """Zeilen: commodity, ticker, exposure_type, expected_direction, mapping_source, mapping_version,
    confidence, point_in_time_status, non_pit_mapping. Einzeltitel-Regel vor Branchenregel."""
    m = mapping_rules(cfg)
    ver, pit = m.get("version"), m.get("point_in_time_status", "NON_PIT")
    rows = {}
    for r in m.get("industry_rules") or []:
        for t in tickers:
            if (industries or {}).get(t) == r["industry"]:
                rows[(t, r["commodity"])] = {"commodity": r["commodity"], "ticker": t,
                                             "exposure_type": r["exposure_type"],
                                             "expected_direction": int(r["expected_direction"]),
                                             "mapping_source": f"industry:{r['industry']}", "mapping_version": ver,
                                             "confidence": r.get("confidence", "low"), "point_in_time_status": pit,
                                             "non_pit_mapping": pit != "PIT"}
    want = set(tickers)
    for r in m.get("ticker_rules") or []:
        for t in r.get("tickers") or []:
            if t in want:
                rows[(t, r["commodity"])] = {"commodity": r["commodity"], "ticker": t,
                                             "exposure_type": r["exposure_type"],
                                             "expected_direction": int(r["expected_direction"]),
                                             "mapping_source": "ticker_rule", "mapping_version": ver,
                                             "confidence": r.get("confidence", "low"), "point_in_time_status": pit,
                                             "non_pit_mapping": pit != "PIT"}
    cols = ["commodity", "ticker", "exposure_type", "expected_direction", "mapping_source", "mapping_version",
            "confidence", "point_in_time_status", "non_pit_mapping"]
    return pd.DataFrame(list(rows.values()), columns=cols)


def load_date_features(path: Path | None = None) -> pd.DataFrame | None:
    p = Path(path or STORE_PATH)
    if not p.exists():
        return None
    f = pd.read_csv(p, parse_dates=["date"])
    f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    return f.sort_values("date")


def attach_panel(panel: pd.DataFrame, groups: list[str] | None = None, path: Path | None = None,
                 date_features: pd.DataFrame | None = None, availability_cols: dict | None = None) -> pd.DataFrame:
    """Kreuzfeatures cmdx_* je (date, ticker) + Verfügbarkeit je Gruppe. Datums-Feature mit Stichtag
    <= Panel-Datum (höchstens 4 Tage alt: Wochenenden/Feiertage). Nicht gemappter Titel -> NaN."""
    groups = groups or ["price", "fundamental", "positioning", "divergence"]
    availability_cols = availability_cols or {g: f"alt_cmd_{g}_available" for g in groups}
    f = date_features if date_features is not None else load_date_features(path)
    out = panel.copy()
    feats = {g: cross_features(g) for g in groups}
    if f is None or f.empty:
        log.warning("commodity: Feature-Store fehlt -> Kreuzfeatures NaN, Verfügbarkeit 0")
        return out.assign(**{c: np.nan for g in groups for c in feats[g]},
                          **{fe: np.nan for g in groups for _, fe in CROSS_SPEC[g]},
                          **{c: np.nan for c in EXPOSURE_COLUMNS},
                          **{availability_cols[g]: 0.0 for g in groups})
    need = sorted({fe for g in groups for _, fe in CROSS_SPEC[g]})
    f = f[["date"] + [c for c in need if c in f]].rename(columns={"date": "_fdate"})
    f["_fdate"] = pd.to_datetime(f["_fdate"]).dt.tz_localize(None).astype("datetime64[ns]")
    base = out.drop(columns=[c for c in need if c in out]).reset_index(drop=True)
    base["_row"] = np.arange(len(base))
    base["_pdate"] = pd.to_datetime(base["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    mrg = pd.merge_asof(base.sort_values("_pdate"), f.sort_values("_fdate"), left_on="_pdate", right_on="_fdate",
                        direction="backward", tolerance=pd.Timedelta(days=4))
    mrg = mrg.sort_values("_row").reset_index(drop=True)
    tickers = sorted(mrg["ticker"].astype(str).unique())
    industries = None
    if "industry" in mrg:
        industries = mrg.drop_duplicates("ticker").set_index("ticker")["industry"].astype(str).to_dict()
    ex = exposure_table(tickers, industries)
    # Exposure-Spalten für die Signal-Sprache: gemappt -> erwartete Richtung; nicht gemappt bei bekanntem
    # Sektor -> 0 ("kein Mapping", wie exp_<sektor>); Sektor unbekannt/fehlt -> NaN (nie geraten).
    known = (mrg["sector"].notna() & ~mrg["sector"].astype(str).isin(["", "nan", "unknown", "None"])
             if "sector" in mrg else pd.Series(False, index=mrg.index))
    for col, c in EXPOSURE_COLUMNS.items():
        dmap = ex[ex["commodity"] == c].set_index("ticker")["expected_direction"].to_dict()
        d = mrg["ticker"].astype(str).map(dmap).astype(float)
        mrg[col] = d.where(d.notna(), np.where(known, 0.0, np.nan))
    for g in groups:
        av = np.zeros(len(mrg))
        for c, fe in CROSS_SPEC[g]:
            col = cross_name(c, fe)
            dmap = ex[ex["commodity"] == c].set_index("ticker")["expected_direction"].to_dict()
            direction = mrg["ticker"].astype(str).map(dmap).astype(float)        # nicht gemappt -> NaN
            vals = mrg[fe] if fe in mrg else pd.Series(np.nan, index=mrg.index)
            mrg[col] = direction * vals
            av = np.maximum(av, mrg[col].notna().to_numpy(dtype=float))
        mrg[availability_cols[g]] = av
    return mrg.drop(columns=[c for c in ("_row", "_pdate", "_fdate") if c in mrg])


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "evaluate"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if a.cmd == "evaluate":
        import os
        import pickle
        cache = os.environ.get("ML_PANEL_CACHE")
        from modules import ml_research as ml
        if cache and Path(cache).exists():
            with open(cache, "rb") as fh:
                panel = pickle.load(fh)  # noqa: S301 – eigene, im selben Job erzeugte Datei
        else:
            panel = ml.build_research_panel("full")
        if any(c not in panel for c in EXPOSURE_COLUMNS):
            panel = attach_panel(panel)
        rep = evaluate(panel)
        print(json.dumps({g: {"verdict": r.get("verdict"), "reason": r.get("verdict_reason")}
                          for g, r in rep["groups"].items()}, indent=1, ensure_ascii=False))
        print("incremental_value:", rep["incremental_value"])
        return 0
    if a.cmd == "build":
        df, st = build()
        print(json.dumps({"n_dates": st["n_dates"], "coverage": st["coverage"], "unavailable": st["unavailable"],
                          "sources": {k: v["health"] for k, v in st["sources"].items()}}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ── Inkrementeller Wert: BASE vs. BASE+COMMODITY (Research, keine Produktionswirkung) ──

EVAL_PATH = Path("outputs/research/commodity_evaluation.json")
EVAL_GROUPS = ("price", "fundamental", "positioning", "divergence", "all")


def _liquidity_bucket(panel: pd.DataFrame) -> pd.Series:
    if "log_dollar_vol" not in panel:
        return pd.Series("nicht verfügbar", index=panel.index)
    r = panel.groupby("date")["log_dollar_vol"].rank(pct=True)
    return pd.Series(np.where(r.isna(), "unknown", np.where(r <= 1 / 3, "low", np.where(r <= 2 / 3, "mid", "high"))),
                     index=panel.index)


def _exposure_label(panel: pd.DataFrame) -> pd.Series:
    """Rohstoff-Exposure je Zeile (erste gemappte Commodity mit Richtung != 0, sonst 'none')."""
    lab = pd.Series("none", index=panel.index, dtype=object)
    for col, c in EXPOSURE_COLUMNS.items():
        if col in panel:
            m = (lab == "none") & panel[col].notna() & (panel[col] != 0)
            lab[m] = np.where(panel.loc[m, col] > 0, f"{c}:+", f"{c}:-")
    return lab


def breakdown_positions(pos_b: pd.DataFrame, pos_v: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Δ Netto-Rendite je market_cap_bucket / sector / commodity_exposure / regime / liquidity_bucket."""
    key = panel[["date", "ticker"]].copy()
    key["sector"] = panel["sector"] if "sector" in panel else "unknown"
    key["commodity_exposure"] = _exposure_label(panel)
    key["regime"] = (np.where(panel["vix"].isna(), "unknown", np.where(panel["vix"] >= 20, "vix_ge_20", "vix_lt_20"))
                     if "vix" in panel else "unknown")
    key["liquidity_bucket"] = _liquidity_bucket(panel)
    key["market_cap_bucket"] = (panel["market_cap_bucket"] if "market_cap_bucket" in panel
                                else "nicht verfügbar (V1-Panel ohne PIT-Market-Cap)")
    key = key.drop_duplicates(["date", "ticker"])
    dims = ("market_cap_bucket", "sector", "commodity_exposure", "regime", "liquidity_bucket")
    out: dict = {d: {} for d in dims}
    for name, pos in (("base", pos_b), ("variant", pos_v)):
        if pos is None or pos.empty:
            continue
        p = pos[["date", "ticker", "ret"]].merge(key, on=["date", "ticker"], how="left")
        for d in dims:
            for k, g in p.groupby(d):
                out[d].setdefault(str(k), {})[name] = round(float(g["ret"].mean()), 5)
                out[d][str(k)][f"n_{name}"] = int(len(g))
    for d in out.values():
        for v in d.values():
            if "base" in v and "variant" in v:
                v["delta"] = round(v["variant"] - v["base"], 5)
    return out


def evaluate(panel: pd.DataFrame, write: bool = True) -> dict:
    """Je Gruppe (+ 'all'): gleiche Walk-Forward-Spec mit/ohne Commodity-Kreuzfeatures auf identischen Zeilen,
    Population = Titel mit Exposure-Mapping (Kreuzfeatures sind nur dort definiert). Metriken netto nach Kosten:
    Expectancy, Sharpe, IC, Precision@K, Brier, ECE, MaxDD, Stabilität (Δ je Jahr), Bootstrap mit
    Bonferroni über Gruppen × Baselines (Protokoll config/alt_data_protocol.yaml, unverändert)."""
    from modules import meta_learning as meta
    from modules import ml_research as ml
    from modules.alt_data import evaluate as ae
    AP = ae.AP
    reg = ml.load_registry()
    st, _ = ml.check_registry(reg)
    specs = {s["id"]: s for s in reg.get("models") or [] if s["id"] in AP["evaluation"]["baselines"]
             and st.get(s["id"]) == "valid"}
    exp_cols = [c for c in EXPOSURE_COLUMNS if c in panel]
    mapped = panel[exp_cols].fillna(0).ne(0).any(axis=1) if exp_cols else pd.Series(False, index=panel.index)
    # Walk-Forward auf dem VOLLEN Panel (Top-Dezil braucht den Querschnitt, MIN_CROSS_SECTION); Kreuzfeatures
    # existieren nur für gemappte Titel (sonst NaN -> Modell-Imputation). Auswahl/Abdeckung: nur gemappte Titel.
    pop = panel[mapped].copy()
    n_tests = max(1, len(specs) * len(EVAL_GROUPS))
    rep = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "protocol": AP["version"],
           "population": "PIT-Panel; Commodity-Kreuzfeatures nur für Titel mit Exposure-Mapping (NON_PIT exposure-v1)",
           "n_rows_population": int(len(pop)), "n_tickers_population": int(pop["ticker"].nunique()) if len(pop) else 0,
           "mapped_row_share": round(float(mapped.mean()), 4) if len(panel) else 0.0,
           "non_pit_mapping": True, "groups": {}}
    dev = AP["evaluation"]["dev_years"]
    for g in EVAL_GROUPS:
        feats = (sum((cross_features(x) for x in ("price", "fundamental", "positioning", "divergence")), [])
                 if g == "all" else cross_features(g))
        feats = [f for f in feats if f in pop]
        res: dict = {"population": rep["population"], "n_rows": int(len(pop)),
                     "commodities": sorted({c for x in (CROSS_SPEC if g == "all" else {g: CROSS_SPEC[g]}).values()
                                            for c, _ in x}), "baselines": {}}
        if pop.empty or not feats:
            res.update(verdict="REJECT", verdict_reason="keine gemappten Titel oder keine Kreuzfeatures im Panel")
            rep["groups"][g] = res
            continue
        screen = ae.feature_screen(pop, feats, AP["feature_selection"]["selection_years"])
        chosen = ae.select_features(screen)
        dpop = pop[pop["date"].dt.year.isin(dev)]
        coverage = float(dpop[feats].notna().any(axis=1).mean()) if len(dpop) else 0.0
        res.update(screen=screen, selected_features=chosen, coverage_dev=round(coverage, 3))
        if not chosen or coverage < AP["evaluation"]["min_coverage"]:
            res.update(verdict="REJECT", verdict_reason=("keine nicht-redundante Feature mit ausreichender Abdeckung"
                                                         if not chosen else f"Abdeckung {coverage:.0%} zu gering"))
            rep["groups"][g] = res
            continue
        for bid, spec in specs.items():
            ob = ae._oos(panel, spec)
            ov = ae._oos(panel, {**spec, "id": f"{bid}+cmd_{g}", "extra_features": chosen})
            if ob.empty or ov.empty:
                res["baselines"][bid] = {"status": "no_data"}
                continue
            common = ob[["date", "ticker"]].merge(ov[["date", "ticker"]], on=["date", "ticker"])
            ob, ov = ob.merge(common, on=["date", "ticker"]), ov.merge(common, on=["date", "ticker"])
            mb, sb, pb = ae._metrics(ob)
            mv, sv, pv = ae._metrics(ov)
            boot = meta.bootstrap_delta(sv, sb, n=AP["evaluation"]["bootstrap_n"], seed=AP["evaluation"]["bootstrap_seed"],
                                        alpha=AP["evaluation"]["alpha_one_sided"] / n_tests)
            years = {str(y): round(float(sv[sv.index.year == y].mean() - sb[sb.index.year == y].mean()), 5)
                     for y in dev if len(sb[sb.index.year == y]) and len(sv[sv.index.year == y])}
            res["baselines"][bid] = {"base": mb, "with_commodity": mv, "with_source": mv, "bootstrap": boot,
                                     "delta": {k: (None if mb.get(k) is None or mv.get(k) is None
                                                   else round(mv[k] - mb[k], 6)) for k in mb},
                                     "delta_by_year": years, "stability_share_years_positive":
                                         round(float(np.mean([v > 0 for v in years.values()])), 3) if years else None,
                                     "breakdown": breakdown_positions(pb, pv, panel)}
        res["verdict"], res["verdict_reason"] = ae.decide({"baselines": {
            k: {**v, "breakdown": {"sector": (v.get("breakdown") or {}).get("sector", {})}}
            for k, v in res["baselines"].items() if v.get("bootstrap")}}, coverage) if any(
            v.get("bootstrap") for v in res["baselines"].values()) else ("REJECT", "keine auswertbare Baseline")
        res["oos_summary"] = {bid: {"delta_monthly_mean": (b.get("bootstrap") or {}).get("delta_monthly_mean"),
                                    "ci": (b.get("bootstrap") or {}).get("ci_monthly_mean")}
                              for bid, b in res["baselines"].items() if b.get("bootstrap")}
        rep["groups"][g] = res
    rep["incremental_value"] = incremental_verdict(rep)
    if write:
        EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_PATH.write_text(json.dumps(rep, indent=1, default=str, ensure_ascii=False))
    return rep


def incremental_verdict(rep: dict) -> str:
    """NONE / HISTORICAL_ONLY / NO RELATIONSHIP. Forward-Evidenz kommt ausschließlich aus Verträgen."""
    gs = rep.get("groups") or {}
    if any(r.get("verdict") == "KEEP" for r in gs.values()):
        return "HISTORICAL_ONLY"              # historisch inkrementell, prospektiv unbestätigt
    if gs and all(r.get("verdict") == "REJECT" for r in gs.values()):
        return "NO_RELATIONSHIP"
    return "NONE"


# ── Report (Wochen-/Monatsbericht, rein lesend) ──────────────────────────────

NO_EVIDENCE_LINE = "Commodity Intelligence: RESEARCH ONLY – no validated incremental alpha"
VALIDATED_STATES = ("FORWARD_VALIDATED", "GUARDED_PRODUCTION", "LIMITED_PRODUCTION", "FULL_PRODUCTION")


def _jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for x in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(x))
        except ValueError:
            continue
    return out


def _js(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def report_summary(out_dir: Path | None = None, now: datetime | None = None) -> dict:
    """Health, aktive Reihen, Hypothesen (neu/getestet/verworfen), Challenger, Forward, inkrementeller Wert."""
    out_dir = Path(out_dir or "outputs")
    now = ensure_utc(now) or datetime.now(timezone.utc)
    res = out_dir / "research"
    st = _js(res / STATUS_PATH.name)
    ev = _js(res / EVAL_PATH.name)
    plan = _js(res / "factory_plan.json")
    results = (_js(res / "factory_results.json").get("results") or {})
    challengers = [c for c in _jl(res / "factory_challengers.jsonl") if (c.get("spec") or {}).get("commodity")]
    pstate = (_js(out_dir / "intelligence" / "promotion_state.json").get("hypotheses") or {})
    week_ago = (now - timedelta(days=7)).isoformat()
    ideas = [h for h in plan.get("ideas") or [] if h.get("commodity") or str(h.get("domain", "")).startswith("commodity")]
    cres = {k: r for k, r in results.items() if (r.get("spec") or {}).get("commodity")
            or str((r.get("spec") or {}).get("domain", "")).startswith("commodity")}
    fwd = {}
    for c in challengers:
        hid = c["hypothesis_id"]
        p = next((v for k, v in pstate.items() if hid in k), None)
        fwd[hid] = {"state": (p or {}).get("state") or "PENDING_FORWARD",
                    "n": ((p or {}).get("evidence") or {}).get("n_observations"),
                    "influence": (p or {}).get("influence_level") or "NONE"}
    validated = [h for h, v in fwd.items() if v["state"] in VALIDATED_STATES]
    srcs = st.get("sources") or {}
    cov = st.get("coverage") or {}
    return {
        "generated": st.get("generated"), "research_status": st.get("research_status") or "RESEARCH_ONLY",
        "health": {k: v.get("health") for k, v in srcs.items()},
        "active_features": {g: (v or {}).get("active_features") for g, v in cov.items()},
        "unavailable": st.get("unavailable") or {},
        "quality_issues": {k: v.get("issues") for k, v in (st.get("quality") or {}).items() if v.get("issues")},
        "mapping": st.get("mapping") or {},
        "hypotheses_new": sum(1 for h in ideas if str(h.get("created_at", "")) >= week_ago[:10]),
        "hypotheses_planned": len(ideas),
        "hypotheses_data_gap": sum(1 for h in ideas if h.get("plan_status") == "DATA_GAP"),
        "hypotheses_tested": len(cres),
        "hypotheses_rejected": sum(1 for r in cres.values() if r.get("status") in ("REJECTED", "NOT_ROBUST")),
        "challengers": len(challengers), "forward": fwd, "validated": validated,
        "incremental_value": ev.get("incremental_value") or "NOT_EVALUATED",
        "ablation": {g: {"verdict": r.get("verdict"), "reason": r.get("verdict_reason")}
                     for g, r in (ev.get("groups") or {}).items()},
        "headline": (f"Commodity Intelligence: {len(validated)} forward-validierte Hypothese(n) – Einfluss nur über "
                     f"PromotionController" if validated else NO_EVIDENCE_LINE),
    }


def report_rows(s: dict) -> list[tuple[str, str]]:
    def kv(d):
        return ", ".join(f"{k} {v}" for k, v in d.items()) if d else "nicht verfügbar"
    return [
        ("Status", s.get("headline") or NO_EVIDENCE_LINE),
        ("Source Health", kv(s.get("health") or {})),
        ("Aktive Features je Gruppe", kv(s.get("active_features") or {})),
        ("UNAVAILABLE", kv({k: "/".join(v) for k, v in (s.get("unavailable") or {}).items()}) if s.get("unavailable")
         else "keine"),
        ("Qualitätsbefunde", kv({k: "/".join(v) for k, v in (s.get("quality_issues") or {}).items()})
         if s.get("quality_issues") else "keine"),
        ("Exposure-Mapping", f"{(s.get('mapping') or {}).get('version', 'n/a')} "
                             f"({(s.get('mapping') or {}).get('point_in_time_status', 'NON_PIT')}, nicht point-in-time)"),
        ("Hypothesen", f"neu {s.get('hypotheses_new', 0)} · geplant {s.get('hypotheses_planned', 0)} · DATA_GAP "
                       f"{s.get('hypotheses_data_gap', 0)} · getestet {s.get('hypotheses_tested', 0)} · verworfen "
                       f"{s.get('hypotheses_rejected', 0)}"),
        ("Prospective Challenger", str(s.get("challengers", 0))),
        ("Forward-Evidenz", kv({k: f"{v['state']} (n={v['n'] or 0}, Einfluss {v['influence']})"
                                for k, v in (s.get("forward") or {}).items()}) if s.get("forward") else "keine"),
        ("Inkrementeller Wert (BASE vs. BASE+COMMODITY)", f"{s.get('incremental_value')} – " + (
            kv({g: v["verdict"] for g, v in (s.get("ablation") or {}).items()}) if s.get("ablation") else "noch nicht bewertet")),
    ]


# ── Entscheidungszeitpunkt (nur lesend): Commodity-Merkmale für Verträge im Decision-/Shadow-Ledger ──
# Damit ein registrierter Commodity-Vertrag PROSPEKTIV ausgewertet werden kann (Forward-Evidenz zum
# Entscheidungszeitpunkt eingefroren), braucht die Regelauswertung dieselben PIT-Merkmale wie die
# Research-Seite. Kein Einfluss: wirkt nur über einen vom PromotionController freigegebenen Vertrag.

_SNAP_CACHE: dict = {}


def decision_snapshot(now: datetime | None = None, root: Path | None = None) -> dict:
    """Datums-Features zum Stichtag (PIT aus dem Archiv, Frische-Grenzen wie im Feature-Store).
    Fehler/fehlende Daten -> {} (Merkmale bleiben None, Verträge nicht auswertbar – nie 0)."""
    now = ensure_utc(now) or datetime.now(timezone.utc)
    key = (now.date().isoformat(), str(root or ARCHIVE_ROOT))
    if key in _SNAP_CACHE:
        return _SNAP_CACHE[key]
    snap: dict = {}
    try:
        df, st = build(now=now, root=root, start=(now.date() - timedelta(days=7)).isoformat(), write=False)
        if not df.empty:
            row = df.iloc[-1]
            feats = {f: float(row[f]) for f in SIGNAL_DATE_FEATURES if f in row and pd.notna(row[f])}
            ver = hashlib.sha256(json.dumps({"date": row["date"], "f": feats, "unavailable": sorted(st["unavailable"])},
                                            sort_keys=True, default=str).encode()).hexdigest()[:12]
            snap = {"date": row["date"], "features": feats, "commodity_data_version": ver,
                    "mapping_version": mapping_status().get("version"), "unavailable": sorted(st["unavailable"])}
    except Exception as e:  # noqa: BLE001 – Research-Kontext darf die Pipeline nie brechen
        log.warning(f"commodity: Entscheidungs-Snapshot nicht ableitbar ({type(e).__name__}: {e})")
    _SNAP_CACHE[key] = snap
    return snap


def decision_features(ticker: str | None, snap: dict | None, ticker_known: bool = True) -> dict:
    """Merkmale je Kandidat: cmdexp_<rohstoff>, Datums-Features cmd_*, Kreuzfeatures cmdx_*.
    Nicht gemappt: cmdexp 0 nur bei bekanntem Titel (sonst None), cmdx None. Fehlender Snapshot: alles None."""
    feats = (snap or {}).get("features") or {}
    out: dict = {f: feats.get(f) for f in SIGNAL_DATE_FEATURES}
    ex = exposure_table([ticker]) if ticker else exposure_table([])
    dirs = {r.commodity: int(r.expected_direction) for r in ex.itertuples()}
    for col, c in EXPOSURE_COLUMNS.items():
        out[col] = float(dirs[c]) if c in dirs else (0.0 if ticker_known and ticker else None)
    for spec in CROSS_SPEC.values():
        for c, fe in spec:
            v = feats.get(fe)
            out[cross_name(c, fe)] = (dirs[c] * v) if (c in dirs and v is not None) else None
    out["commodity_data_version"] = (snap or {}).get("commodity_data_version")
    out["commodity_mapping_version"] = mapping_status().get("version") if snap else None
    return out
