"""
modules/external/context.py – Point-in-Time-Snapshot des externen Kontexts
+ Anhängen an einzelne Kandidaten. SHADOW-ONLY: nichts hier darf jemals eine
Produktions-Entscheidung verändern (siehe modules/external/policy.py für die
harte Grenze).

build_external_context(now, archive=None, registry=None):
    Baut GENAU EINEN Snapshot für `now` aus archivierten Beobachtungen
    (Point-in-Time: available_at <= now, siehe modules.external.pit). Jede
    Familie (road_freight/maritime/weather) wird unabhängig berechnet — ein
    Fehler in einer Familie darf die anderen nie verhindern (Ergebnis ist
    dann ein partieller Kontext, es wird NIE geraten/erfunden).

attach_candidate_context(candidate, snapshot):
    Reine Funktion (kein I/O): baut candidate["external_context"] aus einem
    bereits gebauten Snapshot + candidate["info"] (yfinance sector/industry)
    + candidate ticker + Katalysator-Text.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.external import features as feat
from modules.external.pit import ensure_utc, utc_now
from modules.external.sources import road_freight_features as rff
from modules.external.sources import weather_features as wf
from modules.external.sources.weather import (
    load_industry_exposure, load_weather_exposures, resolve_weather_relevance,
)

FEATURE_VERSIONS = {
    "road_freight": "v1",
    "maritime": "v1",
    "weather": "v1",
    "divergences": "v1",
    "context": "v1",
}

DEFAULT_ARCHIVE_ROOT = "outputs/external_data"

# Quellen je Familie (siehe config/external_sources/*.yaml für die vollen
# Registry-Einträge; hier reicht die source_id, um das Archiv zu befragen).
ROAD_FREIGHT_SOURCES = ["destatis_truck_toll", "bts_freight_tsi", "eurostat_road_freight",
                         "estat_jp_truck"]
MARITIME_SOURCES = ["imf_portwatch_ports", "imf_portwatch_chokepoints"]
WEATHER_SOURCES = ["nws_forecast", "nws_alerts", "ncei_normals", "nhc_storms"]

CHOKEPOINT_SLUGS = {
    "suez": ("suez", "suez canal"),
    "panama": ("panama", "panama canal"),
    "hormuz": ("hormuz", "strait of hormuz"),
    "malacca": ("malacca", "strait of malacca"),
    "bab_el_mandeb": ("bab el mandeb", "bab-el-mandeb"),
}

SEVERE_ALERT_EVENTS = wf.SEVERE_ALERT_EVENTS


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt is not None else None


def _safe(fn, *args, default=None, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _load_observations(archive, source_id: str, now: datetime) -> list:
    return _safe(archive.as_of, source_id, now, default=[]) or []


def _series_for(observations, metric: str, entity_id: str | None = None):
    """(observation_time, value)-Liste, aufsteigend, None-Werte ausgeschlossen."""
    pts = []
    for o in observations:
        if o.metric != metric:
            continue
        if entity_id is not None and o.entity_id != entity_id:
            continue
        if o.value is None:
            continue
        pts.append((o.observation_time, o.value))
    pts.sort(key=lambda p: p[0])
    return pts


def _aggregate_sum_series(observations, metric: str):
    """Summe aller Entities je observation_time (past-only, per Definition
    der Observation-Liste selbst — kein zusätzlicher Filter nötig)."""
    by_time: dict = {}
    for o in observations:
        if o.metric != metric or o.value is None:
            continue
        by_time[o.observation_time] = by_time.get(o.observation_time, 0.0) + o.value
    return sorted(by_time.items(), key=lambda p: p[0])


def _mean_of(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def _entity_zscores(observations, metric: str, window_days: int = 365) -> dict[str, float | None]:
    by_entity: dict[str, list] = {}
    for o in observations:
        if o.metric != metric or o.value is None:
            continue
        by_entity.setdefault(o.entity_id, []).append((o.observation_time, o.value))
    out = {}
    for entity_id, pts in by_entity.items():
        pts.sort(key=lambda p: p[0])
        z = feat.rolling_zscore(pts, window_days=window_days, min_periods=5)
        out[entity_id] = z[-1] if z else None
    return out


def _zscore_for_metric(observations, metric: str, window_days: int = 365) -> float | None:
    series = _series_for(observations, metric)
    z = feat.rolling_zscore(series, window_days=window_days, min_periods=5)
    return z[-1] if z else None


def _region_entry(z, source_id):
    return {"source_id": source_id, "z": z, "is_fresh": z is not None, "age_days": None}


def _chokepoint_zscore(observations, slug: str, metric: str = "n_total",
                        window_days: int = 365) -> float | None:
    aliases = CHOKEPOINT_SLUGS.get(slug, (slug,))
    for o in observations:
        pass
    matched_entities = {
        o.entity_id for o in observations
        if o.entity_id and o.entity_id.strip().lower() in aliases
    }
    if not matched_entities:
        return None
    zs = _entity_zscores(observations, metric, window_days=window_days)
    vals = [zs[e] for e in matched_entities if zs.get(e) is not None]
    return _mean_of(vals)


# ── Familien-Feature-Blöcke ─────────────────────────────────────────────────

def _build_road_freight(archive, now: datetime, errors: list) -> dict:
    out = {"us_z": None, "eu_z": None, "asia_z": None, "global_z": None,
           "breadth": None, "de_z_1y": None, "de_acceleration": None,
           "us_tsi_yoy": None, "us_tsi_z": None,
           "eu_sources": [], "eu_source_count": 0,
           "us_combined": feat.combine_states([]), "eu_combined": feat.combine_states([]),
           "asia_combined": feat.combine_states([])}
    try:
        de_obs = _load_observations(archive, "destatis_truck_toll", now)
        us_obs = _load_observations(archive, "bts_freight_tsi", now)
        eu_obs = _load_observations(archive, "eurostat_road_freight", now)
        jp_obs = _load_observations(archive, "estat_jp_truck", now)

        out["de_z_1y"] = _safe(rff.de_truck_z_1y, de_obs)
        out["de_acceleration"] = _safe(rff.de_truck_acceleration, de_obs)
        out["us_tsi_yoy"] = _safe(rff.us_freight_tsi_yoy, us_obs)
        out["us_z"] = _safe(rff.us_freight_tsi_z, us_obs)
        out["us_tsi_z"] = out["us_z"]

        # ── US: bts_freight_tsi (+ trucking-Komponente, falls vorhanden) ────
        us_sources = []
        if us_obs:
            us_sources.append(_region_entry(out["us_z"], "bts_freight_tsi"))
            us_metrics = {o.metric for o in us_obs}
            if "us_trucking" in us_metrics:
                trucking_z = _zscore_for_metric(us_obs, "us_trucking")
                us_sources.append(_region_entry(trucking_z, "bts_freight_tsi_trucking"))
        out["us_combined"] = feat.combine_states(us_sources)

        # ── EU: DE (destatis) + jede eurostat-Länderserie/EU-Aggregat als
        # EIGENE Quelle -- DE zählt als EINE von mehreren EU-Quellen, NIE
        # stillschweigend als "die" EU (siehe eu_source_count/eu_sources
        # unten, die das dokumentieren). Kein Fallback mehr: freight_eu_z
        # ist der Mittelwert der tatsächlich vorhandenen EU-Quellen.
        eu_sources = []
        if de_obs:
            eu_sources.append(_region_entry(out["de_z_1y"], "destatis_truck_toll"))
        eu_country_entities = sorted({
            o.entity_id for o in eu_obs if o.metric == "road_freight_ths_t"
        })
        for eid in eu_country_entities:
            # Eurostat liefert Jahreswerte: 12-Jahres-Fenster (>= 5 Basispunkte)
            z = _safe(rff.eu_road_freight_z, eu_obs, eid, "road_freight_ths_t", 365 * 12)
            eu_sources.append(_region_entry(z, f"eurostat_road_freight:{eid}"))
        out["eu_sources"] = [e["source_id"] for e in eu_sources]
        out["eu_source_count"] = len(eu_sources)
        out["eu_combined"] = feat.combine_states(eu_sources)
        out["eu_z"] = _mean_of([e["z"] for e in eu_sources])

        # ── ASIA: estat_jp_truck (maritime-unabhängige Quellen only) ────────
        # Japan: keine Default-Metrik bekannt -> nur wenn ein plausibles
        # 'index'-Metric tatsächlich vorhanden ist (nie erfinden).
        jp_metrics = {o.metric for o in jp_obs}
        asia_z = None
        for candidate_metric in ("index_sa", "index", "truck_index"):
            if candidate_metric in jp_metrics:
                asia_z = _safe(rff.jp_truck_z, jp_obs, candidate_metric)
                if asia_z is not None:
                    break
        out["asia_z"] = asia_z
        asia_sources = []
        if jp_obs:
            asia_sources.append(_region_entry(asia_z, "estat_jp_truck"))
        out["asia_combined"] = feat.combine_states(asia_sources)

        out["global_z"] = _mean_of([out["us_z"], out["eu_z"], out["asia_z"]])

        breadth_vals = {"us": out["us_z"], "eu": out["eu_z"], "asia": out["asia_z"]}
        out["breadth"] = feat.breadth(breadth_vals, threshold=0.5, min_valid=2)
    except Exception as e:  # noqa: BLE001 - Familie darf nie den Snapshot brechen
        errors.append(f"road_freight: {e!r}")
    return out


def _build_maritime(archive, now: datetime, errors: list) -> dict:
    out = {"global_z": None, "container_z": None, "drybulk_z": None, "tanker_z": None,
           "negative_breadth": None, "valid_port_count": 0,
           "suez_z": None, "panama_z": None, "hormuz_z": None,
           "malacca_z": None, "bab_el_mandeb_z": None}
    try:
        port_obs = _load_observations(archive, "imf_portwatch_ports", now)
        choke_obs = _load_observations(archive, "imf_portwatch_chokepoints", now)

        for metric_key, out_key in (
            ("portcalls_total", "global_z"),
            ("portcalls_container", "container_z"),
            ("portcalls_dry_bulk", "drybulk_z"),
            ("portcalls_tanker", "tanker_z"),
        ):
            series = _aggregate_sum_series(port_obs, metric_key)
            z = feat.rolling_zscore(series, window_days=365, min_periods=5)
            out[out_key] = z[-1] if z else None

        entity_z = _entity_zscores(port_obs, "portcalls_total", window_days=365)
        out["valid_port_count"] = sum(1 for v in entity_z.values() if v is not None)
        # Mindestabdeckung aus config/port_universe.yaml (Default 8 Häfen):
        # Breite nie aus nur ein, zwei Häfen; unterhalb -> None (fehlend != 0).
        try:
            import yaml
            _fc = (yaml.safe_load(open("config/port_universe.yaml")) or {}).get("feature_config", {})
            _min_ports = int(_fc.get("min_valid_ports_for_breadth", 8))
        except Exception:
            _min_ports = 8
        breadth = feat.breadth(entity_z, threshold=0.5, min_valid=_min_ports)
        out["negative_breadth"] = breadth.get("negative_breadth") if breadth.get("breadth_valid") else None

        for slug in CHOKEPOINT_SLUGS:
            out[f"{slug}_z"] = _chokepoint_zscore(choke_obs, slug)
    except Exception as e:  # noqa: BLE001
        errors.append(f"maritime: {e!r}")
    return out


def _build_weather(archive, now: datetime, errors: list) -> dict:
    out = {"disruption_index": None, "hdd_anomaly": None, "cdd_anomaly": None,
           "forecast_revision_hdd": None, "forecast_revision_cdd": None,
           "active_tropical_system": None}
    try:
        forecast_obs = _load_observations(archive, "nws_forecast", now)
        alert_obs = _load_observations(archive, "nws_alerts", now)
        normals_obs = _load_observations(archive, "ncei_normals", now)
        storm_obs = _load_observations(archive, "nhc_storms", now)

        as_of_date = ensure_utc(now).date()
        temp_daily = wf.daily_temperature_aggregates(forecast_obs)
        normals = wf.normals_by_month_day(normals_obs)

        hdd_anoms, cdd_anoms = [], []
        for entity in temp_daily:
            a = wf.anomalies_for_day(temp_daily, entity, as_of_date, normals)
            if a.get("hdd_anomaly") is not None:
                hdd_anoms.append(a["hdd_anomaly"])
            if a.get("cdd_anomaly") is not None:
                cdd_anoms.append(a["cdd_anomaly"])
        out["hdd_anomaly"] = _mean_of(hdd_anoms)
        out["cdd_anomaly"] = _mean_of(cdd_anoms)

        # Forecast-Revisionen: Durchschnitt über alle Entities/'tmean'
        # (Tagesmittel, siehe weather.py: aggregate_daily_forecast).
        # HDD-Revision ist die invertierte Temperatur-Revision (mehr Wärme
        # -> weniger Heizbedarf) -- dokumentierte Vereinfachung.
        entities = {o.entity_id for o in forecast_obs}
        temp_revisions = []
        for entity in entities:
            for rev in wf.revisions_for_metric(forecast_obs, entity, "tmean"):
                if rev.get("revision") is not None:
                    temp_revisions.append(rev["revision"])
        mean_rev = _mean_of(temp_revisions)
        out["forecast_revision_hdd"] = None if mean_rev is None else -mean_rev
        out["forecast_revision_cdd"] = mean_rev

        pc = wf.pc_insurance_features([], alert_obs, storm_obs)
        out["active_tropical_system"] = pc.get("active_tropical_system")
        # Distanz der AKTUELLEN Sturmposition (NHC lat/lon) zur nächsten
        # kuratierten US-Küsten-Expositionsregion. Ohne Position -> None.
        out["tropical_min_distance_km"] = _tropical_min_distance_km(storm_obs)

        alert_entities = {o.entity_id for o in alert_obs if o.metric == "alert_count"}
        if alert_entities:
            severe_entities = {
                o.entity_id for o in alert_obs
                if o.metric == "alert_count" and o.attrs.get("event") in SEVERE_ALERT_EVENTS
                and (o.value or 0) > 0
            }
            out["disruption_index"] = len(severe_entities) / len(alert_entities)
    except Exception as e:  # noqa: BLE001
        errors.append(f"weather: {e!r}")
    return out


# Methodischer Schwellenwert (nicht gegen Renditen optimiert): ein tropisches
# System zählt nur dann als operatives Risiko, wenn seine aktuelle Position
# innerhalb dieser Distanz zu einer kuratierten US-Küsten-Expositionsregion liegt.
TROPICAL_NEAR_EXPOSURE_KM = 1000.0


def _tropical_min_distance_km(storm_obs) -> float | None:
    try:
        from modules.external.sources.weather import haversine_km, load_weather_locations
        regions = load_weather_locations().get("coastal_exposure_regions", []) or []
    except Exception:
        return None
    pos: dict[str, dict] = {}
    for o in storm_obs or []:
        if o.metric in ("lat", "lon") and o.value is not None:
            pos.setdefault(o.series_id, {})[o.metric] = o.value
    best = None
    for p_ in pos.values():
        if "lat" not in p_ or "lon" not in p_:
            continue
        for r in regions:
            try:
                d = haversine_km(float(p_["lat"]), float(p_["lon"]), float(r["lat"]), float(r["lon"]))
            except (KeyError, TypeError, ValueError):
                continue
            best = d if best is None or d < best else best
    return best


def _weather_operational_risk(disruption_index: float | None, active_storm: bool | None,
                              tropical_min_distance_km: float | None = None) -> str:
    # Ein aktiver Sturm allein ist kein Risiko für US-Exposition (z.B.
    # Ostpazifik-Hurrikane): nur mit Position nahe einer Expositionsregion.
    near_storm = bool(active_storm) and tropical_min_distance_km is not None \
        and tropical_min_distance_km <= TROPICAL_NEAR_EXPOSURE_KM
    if disruption_index is None and not near_storm:
        return "UNKNOWN" if not active_storm else "LOW"
    score = disruption_index or 0.0
    if near_storm:
        score = max(score, 0.6)
    if score >= 0.5:
        return "HIGH"
    if score >= 0.2:
        return "MEDIUM"
    return "LOW"


def _sources_summary(archive, now: datetime, source_ids: list[str]) -> dict:
    out = {}
    for sid in source_ids:
        obs = _load_observations(archive, sid, now)
        if not obs:
            out[sid] = {"status": "NO_DATA", "latest_observation": None,
                        "available_at_max": None, "vintage_ids": []}
            continue
        latest_obs_time = max(o.observation_time for o in obs)
        available_at_max = max(o.available_at for o in obs)
        vintages = [
            {"identity_key": o.identity_key(),
             "vintage_time": _iso(o.vintage_time or o.available_at)}
            for o in obs[:20]
        ]
        out[sid] = {
            "status": "OK",
            "latest_observation": _iso(latest_obs_time),
            "available_at_max": _iso(available_at_max),
            "vintage_ids": vintages,
        }
    return out


def _configured_archive_root() -> str:
    try:
        from modules.config import cfg
        arc = getattr(getattr(cfg, "external_context", None), "archive", None)
        root = getattr(arc, "root", None) if arc else None
        return root or DEFAULT_ARCHIVE_ROOT
    except Exception:
        return DEFAULT_ARCHIVE_ROOT


def _canonical_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_external_context(now: datetime | None = None, archive=None, registry=None) -> dict:
    """Baut GENAU EINEN PIT-Snapshot für `now`. Fehler in einer Familie
    reduzieren den Snapshot nie auf 'nichts' -- jede Familie ist unabhängig
    try/except-geschützt. Wirft NIE."""
    now = ensure_utc(now) or utc_now()
    errors: list[str] = []

    if archive is None:
        try:
            from modules.external.archive import ExternalArchive
            archive = ExternalArchive(_configured_archive_root())
        except Exception as e:  # noqa: BLE001
            errors.append(f"archive_init: {e!r}")
            archive = None

    road = _build_road_freight(archive, now, errors) if archive is not None else {}
    maritime = _build_maritime(archive, now, errors) if archive is not None else {}
    weather = _build_weather(archive, now, errors) if archive is not None else {}

    div = feat.divergence(road.get("global_z"), maritime.get("global_z"), threshold=0.5)
    road_shipping_divergence_z = div.get("divergence_z")
    road_shipping_agreement = div.get("agreement")

    possible_weather_confounded = None
    if road.get("global_z") is not None and weather.get("disruption_index") is not None:
        possible_weather_confounded = bool(
            road["global_z"] <= -0.5 and weather["disruption_index"] >= 0.5
        )

    # ISM/PMI-Umfrageserie existiert (Stand dieser Codebase) nicht in
    # modules/macro_context.py -> NIE erfinden, nur begründet None liefern.
    hard_vs_survey_reason = "no_survey_series_in_macro_context"
    hard_vs_survey = None

    # Regionale Zustände kommen JEWEILS aus combine_states über die eigenen
    # Quellen dieser Region (siehe _build_road_freight: us_combined/
    # eu_combined/asia_combined) -- NIE aus der globalen Confidence.
    us_combined = road.get("us_combined") or feat.combine_states([])
    eu_combined = road.get("eu_combined") or feat.combine_states([])
    asia_combined = road.get("asia_combined") or feat.combine_states([])

    us_state = us_combined.get("state", "UNKNOWN")
    eu_state = eu_combined.get("state", "UNKNOWN")
    asia_state = asia_combined.get("state", "UNKNOWN")
    maritime_state = feat.classify_state(maritime.get("global_z"))
    container_state = feat.classify_state(maritime.get("container_z"))
    drybulk_state = feat.classify_state(maritime.get("drybulk_z"))
    tanker_state = feat.classify_state(maritime.get("tanker_z"))

    def _entry(z, source_id):
        return {"source_id": source_id, "z": z, "is_fresh": z is not None, "age_days": None}

    # Global-Freight-State: Combine über die REGIONALEN Zustände (nicht über
    # einen Mittelwert der Roh-Z-Scores) -- is_fresh richtet sich danach, ob
    # die jeweilige Region überhaupt eine frische Quelle hatte; dadurch erbt
    # combine_states' eingebaute Regel ("<=1 frische Quelle -> confidence
    # <=0.3") korrekt: nur 1 von 3 Regionen frisch -> niedrige Global-Confidence.
    combined_freight = feat.combine_states([
        {"source_id": "us_region", "z": road.get("us_z"),
         "is_fresh": us_combined.get("fresh_source_count", 0) > 0,
         "age_days": us_combined.get("data_age_days")},
        {"source_id": "eu_region", "z": road.get("eu_z"),
         "is_fresh": eu_combined.get("fresh_source_count", 0) > 0,
         "age_days": eu_combined.get("data_age_days")},
        {"source_id": "asia_region", "z": road.get("asia_z"),
         "is_fresh": asia_combined.get("fresh_source_count", 0) > 0,
         "age_days": asia_combined.get("data_age_days")},
    ])
    combined_global = feat.combine_states([
        _entry(road.get("global_z"), "road_freight_composite"),
        _entry(maritime.get("global_z"), "maritime_composite"),
    ])

    weather_risk = _weather_operational_risk(
        weather.get("disruption_index"), weather.get("active_tropical_system"),
        weather.get("tropical_min_distance_km"))

    supply_chain = {
        "us_freight_state": us_state,
        "us_freight_confidence": us_combined.get("confidence"),
        "eu_freight_state": eu_state,
        "eu_freight_confidence": eu_combined.get("confidence"),
        "asia_freight_state": asia_state,
        "asia_freight_confidence": asia_combined.get("confidence"),
        "global_freight_state": combined_freight.get("state"),
        "global_freight_confidence": combined_freight.get("confidence"),
        "global_maritime_state": maritime_state,
        "container_state": container_state,
        "dry_bulk_state": drybulk_state,
        "tanker_state": tanker_state,
        "shipping_negative_breadth": maritime.get("negative_breadth"),
        "weather_operational_risk": weather_risk,
        "road_shipping_agreement": road_shipping_agreement,
        "possible_weather_confounded_freight": possible_weather_confounded,
        "hard_data_vs_survey_divergence": hard_vs_survey,
    }

    primitives = {
        "freight_global_z": road.get("global_z"),
        "freight_us_z": road.get("us_z"),
        "freight_eu_z": road.get("eu_z"),
        "freight_asia_z": road.get("asia_z"),
        # Breite nur, wenn die Mindestabdeckung erfüllt ist (fehlend != 0)
        "freight_breadth": ((road.get("breadth") or {}).get("positive_breadth")
                            if (road.get("breadth") or {}).get("breadth_valid") else None),
        "de_truck_z_1y": road.get("de_z_1y"),
        "de_truck_acceleration": road.get("de_acceleration"),
        "us_freight_tsi_yoy": road.get("us_tsi_yoy"),
        "us_freight_tsi_z": road.get("us_tsi_z"),
        "shipping_global_z": maritime.get("global_z"),
        "shipping_container_z": maritime.get("container_z"),
        "shipping_drybulk_z": maritime.get("drybulk_z"),
        "shipping_tanker_z": maritime.get("tanker_z"),
        "shipping_negative_breadth": maritime.get("negative_breadth"),
        "shipping_valid_port_count": maritime.get("valid_port_count"),
        "suez_z": maritime.get("suez_z"),
        "panama_z": maritime.get("panama_z"),
        "hormuz_z": maritime.get("hormuz_z"),
        "malacca_z": maritime.get("malacca_z"),
        "bab_el_mandeb_z": maritime.get("bab_el_mandeb_z"),
        "weather_disruption_index": weather.get("disruption_index"),
        "hdd_anomaly": weather.get("hdd_anomaly"),
        "cdd_anomaly": weather.get("cdd_anomaly"),
        "forecast_revision_hdd": weather.get("forecast_revision_hdd"),
        "forecast_revision_cdd": weather.get("forecast_revision_cdd"),
        "active_tropical_system": weather.get("active_tropical_system"),
        "road_shipping_divergence_z": road_shipping_divergence_z,
        "road_shipping_agreement": road_shipping_agreement,
        "possible_weather_confounded_freight": possible_weather_confounded,
        "hard_data_vs_survey_divergence": hard_vs_survey,
    }

    all_source_ids = ROAD_FREIGHT_SOURCES + MARITIME_SOURCES + WEATHER_SOURCES
    sources = _sources_summary(archive, now, all_source_ids) if archive is not None else {}

    available_at_values = [
        ensure_utc(s.get("available_at_max")) for s in sources.values()
        if s.get("available_at_max")
    ]
    max_available_at_used = max(available_at_values) if available_at_values else None

    quality = {
        "errors": errors,
        "families_with_data": [
            name for name, block in (("road_freight", road), ("maritime", maritime),
                                      ("weather", weather))
            if any(v is not None for v in block.values())
        ],
        "hard_data_vs_survey_divergence_reason": hard_vs_survey_reason,
    }

    inputs_for_hash = {
        "as_of": _iso(now),
        "sources": {k: v.get("vintage_ids") for k, v in sources.items()},
        "feature_versions": FEATURE_VERSIONS,
    }
    snapshot_id = _canonical_hash(inputs_for_hash)

    snapshot = {
        "snapshot_id": snapshot_id,
        "created_at": _iso(utc_now()),
        "as_of": _iso(now),
        "feature_versions": dict(FEATURE_VERSIONS),
        "sources": sources,
        "road_freight": road,
        "maritime_freight": maritime,
        "weather": weather,
        "real_economy": {},   # Platzhalter -- kein realer Realwirtschafts-Konnektor in dieser Version
        "supply_chain": supply_chain,
        "divergences": {
            "road_shipping_divergence_z": road_shipping_divergence_z,
            "road_shipping_agreement": road_shipping_agreement,
        },
        "quality": quality,
        "point_in_time": {
            "as_of": _iso(now),
            "max_available_at_used": _iso(max_available_at_used),
            "rule": "available_at<=as_of",
        },
        "primitives": primitives,
        "states": {
            "us_freight_state": us_state,
            "eu_freight_state": eu_state,
            "asia_freight_state": asia_state,
            "us_freight_confidence": us_combined.get("confidence"),
            "eu_freight_confidence": eu_combined.get("confidence"),
            "asia_freight_confidence": asia_combined.get("confidence"),
            "us_freight_source_count": us_combined.get("source_count"),
            "eu_freight_source_count": eu_combined.get("source_count"),
            "asia_freight_source_count": asia_combined.get("source_count"),
            "global_maritime_state": maritime_state,
            "global_maritime_confidence": combined_global.get("confidence"),
            "global_freight_state": combined_freight.get("state"),
            "global_freight_confidence": combined_freight.get("confidence"),
        },
    }
    _archive_root = getattr(archive, "root", None) if archive is not None else None
    if _archive_root is not None:
        # Nur speichern, wenn wir einen echten Dateisystem-Root kennen (das
        # tatsächliche ExternalArchive oder ein Test-Double mit .root). Ein
        # Archiv-Objekt ohne .root (z.B. ein reines as_of()-Stub) speichert
        # NICHT stillschweigend in den Code-Default -- sonst würde ein Test-
        # Double unbeabsichtigt echte Dateien im Repo anlegen.
        _save_snapshot_immutable(snapshot, root=_archive_root)
    return snapshot


def _save_snapshot_immutable(snapshot: dict, root: str | Path | None = None) -> Path | None:
    root = root or _configured_archive_root()
    try:
        as_of = ensure_utc(snapshot["as_of"]) or utc_now()
        month = as_of.strftime("%Y-%m")
        path = Path(root) / "snapshots" / month / f"{snapshot['snapshot_id']}.json"
        if path.exists():
            return path  # unveränderlich -- niemals überschreiben
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, default=str))
        return path
    except Exception:
        return None


# ── Kandidaten-Anhang ────────────────────────────────────────────────────────

def _resolve_ticker_exposure(ticker: str | None, yf_sector: str | None, yf_industry: str | None,
                              industry_cfg: dict, exposures_cfg: dict) -> dict:
    """Wie resolve_weather_relevance(), aber für alle drei Relevanz-Achsen
    (road_freight/maritime/weather). ticker_overrides in weather_exposures.yaml
    kennen nur weather (Hub-Codes) -> für road_freight/maritime fällt ein
    Ticker-Override auf 'unknown' für diese beiden Achsen zurück, WENN er
    keine Industrie-Zuordnung hat (dokumentierte Grenze der aktuellen
    Config-Struktur)."""
    industries = industry_cfg.get("industries") or {}
    industry_map = industry_cfg.get("yfinance_industry_map") or {}
    sector_fallback = industry_cfg.get("sector_fallback") or {}

    override = (exposures_cfg.get("ticker_overrides") or {}).get(ticker) if ticker else None

    industry_name = None
    level = "unknown"
    if yf_industry and yf_industry in industry_map:
        industry_name = industry_map[yf_industry]
        level = "industry"
    elif yf_sector and yf_sector in sector_fallback:
        industry_name = sector_fallback[yf_sector]
        level = "sector"

    rel = industries.get(industry_name) or {} if industry_name else {}
    road_rel = rel.get("road_freight_relevance")
    maritime_rel = rel.get("maritime_relevance")
    weather_rel = rel.get("weather_relevance")

    if override:
        weather_rel = "HIGH"
        level = "ticker"

    if industry_name is None and not override:
        level = "unknown"

    return {
        "industry": industry_name,
        "level": level,
        "road_freight_relevance": road_rel,
        "maritime_relevance": maritime_rel,
        "weather_relevance": weather_rel,
        "exposure_source": level,
        "hub_codes": list(override.get("hub_codes") or []) if override else None,
    }


def _resolve_catalyst_relevance(catalyst_text: str | None, industry_cfg: dict) -> dict | None:
    if not catalyst_text:
        return None
    mapping = industry_cfg.get("catalyst_relevance") or {}
    text_low = catalyst_text.lower()
    # keine Direction/Klassifikation erfinden -- nur Keyword-Match auf die
    # deklarierten Catalyst-Typ-Namen selbst (nicht auf Freitext-Heuristiken
    # jenseits des Namens).
    for catalyst_type, rel in mapping.items():
        needle = catalyst_type.replace("_", " ")
        if needle in text_low or catalyst_type in text_low:
            return {"catalyst_type": catalyst_type, **{k: v for k, v in rel.items() if k != "notes"}}
    return None


def attach_candidate_context(candidate: dict, snapshot: dict | None,
                              exposures_cfg: dict | None = None,
                              industry_cfg: dict | None = None) -> dict | None:
    """Reine Funktion: baut candidate['external_context'] aus `snapshot`.
    Gibt None zurück, wenn snapshot None ist (z.B. Snapshot-Bau ist
    fehlgeschlagen) -- die Pipeline hängt dann gar kein Feld an."""
    if snapshot is None:
        return None

    info = candidate.get("info") or {}
    ticker = candidate.get("ticker")
    yf_sector = info.get("sector")
    yf_industry = info.get("industry")

    exposures_cfg = exposures_cfg if exposures_cfg is not None else _safe(load_weather_exposures, default={})
    industry_cfg = industry_cfg if industry_cfg is not None else _safe(load_industry_exposure, default={})

    exposure = _resolve_ticker_exposure(ticker, yf_sector, yf_industry, industry_cfg, exposures_cfg)

    catalyst_text = None
    da = candidate.get("deep_analysis") or {}
    catalyst_text = da.get("catalyst")
    catalyst_rel = _resolve_catalyst_relevance(catalyst_text, industry_cfg)
    exposure["catalyst_relevance"] = catalyst_rel

    primitives = dict(snapshot.get("primitives") or {})
    states = snapshot.get("states") or {}
    divergences = snapshot.get("divergences") or {}
    road_freight_block = snapshot.get("road_freight") or {}

    return {
        "snapshot_id": snapshot.get("snapshot_id"),
        "feature_version": dict(snapshot.get("feature_versions") or {}),
        "available_at": snapshot.get("point_in_time", {}).get("max_available_at_used"),
        "primitives": primitives,
        "states": {
            "us_freight_state": states.get("us_freight_state"),
            "us_freight_confidence": states.get("us_freight_confidence"),
            "us_freight_source_count": states.get("us_freight_source_count"),
            "eu_freight_state": states.get("eu_freight_state"),
            "eu_freight_confidence": states.get("eu_freight_confidence"),
            "eu_freight_source_count": states.get("eu_freight_source_count"),
            "asia_freight_state": states.get("asia_freight_state"),
            "asia_freight_confidence": states.get("asia_freight_confidence"),
            "asia_freight_source_count": states.get("asia_freight_source_count"),
            "global_maritime_state": states.get("global_maritime_state"),
            "global_maritime_confidence": states.get("global_maritime_confidence"),
            "global_freight_state": states.get("global_freight_state"),
            "global_freight_confidence": states.get("global_freight_confidence"),
        },
        "ticker_exposure": exposure,
        "divergences": dict(divergences),
        # Dokumentiert die tatsächliche Zusammensetzung von freight_eu_z:
        # eu_source_count==1 mit eu_sources==["destatis_truck_toll"] bedeutet
        # "nur Deutschland", NIE stillschweigend als "EU" ausgegeben.
        "quality": {
            "eu_source_count": road_freight_block.get("eu_source_count"),
            "eu_sources": list(road_freight_block.get("eu_sources") or []),
        },
        "relation": {"relation": "NEUTRAL", "materiality": "NONE", "confidence": 0.0,
                     "mechanism": "", "relevant_sources": [], "source": "skipped",
                     "reason": "shadow_analysis_not_run"},
        "policy": {"mode": None, "score_delta": 0.0, "veto": False},
    }
