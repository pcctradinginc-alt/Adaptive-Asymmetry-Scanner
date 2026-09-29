"""PortWatch-Erweiterungen im Kontext: aktive Hafenstörungen (PIT),
Länder-Export-z / Asien-Export-z, Industrie-z und das kandidatenbezogene
industry_shipping_z. Keine Netzwerkzugriffe."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from modules.external import context as ctxmod
from modules.external.archive import ExternalArchive
from modules.external.pit import AvailabilityPrecision, Observation

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, tzinfo=UTC)


def _o(source, entity, metric, value, obs_time, avail, series="s", attrs=None):
    return Observation(source_id=source, dataset="d", series_id=series, entity_id=entity,
                       metric=metric, value=value, unit="x", observation_time=obs_time,
                       available_at=avail, retrieved_at=avail,
                       availability_precision=AvailabilityPrecision.CONSERVATIVE_DATE,
                       parser_version="1", attrs=attrs or {})


def _disruption(eid, port, level, start, end, avail):
    return [_o("imf_portwatch_disruptions", port, "disruption_alert_level", level, start, avail, str(eid)),
            _o("imf_portwatch_disruptions", port, "disruption_end_ts",
               end.timestamp() if end else None, start, avail, str(eid))]


def test_active_disruptions_respect_window_level_and_pit():
    obs = (_disruption(1, "port1", 3, NOW - timedelta(days=2), NOW + timedelta(days=3), NOW - timedelta(days=2))
           + _disruption(2, "port9", 2, NOW - timedelta(days=1), None, NOW - timedelta(days=1))
           + _disruption(3, "port1", 1, NOW - timedelta(days=1), NOW + timedelta(days=1), NOW)    # GREEN
           + _disruption(4, "port2", 3, NOW - timedelta(days=30), NOW - timedelta(days=20), NOW)  # vorbei
           + _disruption(5, "port2", 3, NOW + timedelta(days=1), NOW + timedelta(days=4), NOW))   # Zukunft
    res = ctxmod.active_port_disruptions(obs, NOW, {"port1", "port2"})
    assert res == {"disruption_active_events": 2, "disruption_active_ports": 2,
                   "disruption_curated_ports": 1, "disruption_curated_max_level": 3}


def test_disruption_extension_is_seen_only_after_its_vintage(tmp_path):
    arch = ExternalArchive(root=tmp_path)
    start = NOW - timedelta(days=5)
    arch.store_observations(_disruption(7, "port1", 3, start, NOW - timedelta(days=1), start))
    later = NOW + timedelta(hours=1)
    arch.store_observations(_disruption(7, "port1", 3, start, NOW + timedelta(days=5), later))
    before = ctxmod.active_port_disruptions(arch.as_of("imf_portwatch_disruptions", NOW), NOW, {"port1"})
    after = ctxmod.active_port_disruptions(arch.as_of("imf_portwatch_disruptions", later), later, {"port1"})
    assert before["disruption_curated_ports"] == 0        # Verlängerung zu NOW noch unbekannt
    assert after["disruption_curated_ports"] == 1


def _port_series(entity, metric, n_days, last_value, n_ports=10, last_n_ports=None):
    rows = []
    for i in range(n_days):
        t = NOW - timedelta(days=n_days - i + 2)
        v = 100.0 + (i % 5) if i < n_days - 1 else last_value
        n = last_n_ports if (i == n_days - 1 and last_n_ports is not None) else n_ports
        rows.append(_o("imf_portwatch_ports", entity, metric, v, t, t + timedelta(days=1),
                       series="aggregate", attrs={"aggregate": True, "n_ports": n}))
    return rows


def test_country_and_industry_z_use_only_complete_days():
    obs = (_port_series("COUNTRY:CHN", "export_total", 40, 150.0)
           + _port_series("COUNTRY:KOR", "export_total", 40, 60.0)
           # jüngster Tag unvollständig (2 von 10 Häfen) -> verworfen
           + _port_series("COUNTRY:TWN", "export_total", 40, 5.0, last_n_ports=2)
           + _port_series("INDUSTRY:Transportation", "portcalls_total", 40, 150.0))
    cz = ctxmod._aggregate_entity_z(obs, "COUNTRY:", "export_total")
    assert cz["CHN"] > 3 and cz["KOR"] < -3
    assert abs(cz["TWN"]) < 3
    iz = ctxmod._aggregate_entity_z(obs, "INDUSTRY:", "portcalls_total")
    assert set(iz) == {"Transportation"} and iz["Transportation"] > 3


def test_snapshot_exposes_new_primitives_and_candidate_industry_z(tmp_path):
    arch = ExternalArchive(root=tmp_path)
    arch.store_observations(_port_series("COUNTRY:CHN", "export_total", 40, 150.0)
                            + _port_series("COUNTRY:KOR", "export_total", 40, 150.0)
                            + _port_series("INDUSTRY:Transportation", "portcalls_total", 40, 150.0)
                            + _disruption(1, "port1", 3, NOW - timedelta(days=1), None, NOW - timedelta(days=1)))
    snap = ctxmod.build_external_context(NOW, archive=arch)
    p = snap["primitives"]
    assert p["asia_export_z"] > 3
    assert p["port_disruption_active_events"] == 1
    assert "Transportation" not in (snap["maritime_freight"]["industry_mapping_unmatched"] or [])
    cand = {"ticker": "F", "info": {"sector": "Consumer Cyclical", "industry": "Auto Manufacturers"}}
    cfg = {"industries": {"Autos": {}}, "yfinance_industry_map": {"Auto Manufacturers": "Autos"},
           "portwatch_industry_map": {"Autos": ["Transportation"]}}
    ctx = ctxmod.attach_candidate_context(cand, snap, exposures_cfg={}, industry_cfg=cfg)
    assert ctx["primitives"]["industry_shipping_z"] > 3
    other = ctxmod.attach_candidate_context({"ticker": "X", "info": {}}, snap, exposures_cfg={}, industry_cfg=cfg)
    assert other["primitives"]["industry_shipping_z"] is None


def test_empty_archive_new_primitives_are_none(tmp_path):
    snap = ctxmod.build_external_context(NOW, archive=ExternalArchive(root=tmp_path))
    p = snap["primitives"]
    assert p["asia_export_z"] is None
    assert p["port_disruption_active_events"] == 0


def test_chokepoint_z_matches_portwatch_names_in_attrs():
    """Regression 2026-09-29: Entities heißen chokepoint<N>, Name in attrs.port_name
    -> vorher nie ein Treffer, alle Chokepoint-z immer None."""
    names = {"chokepoint1": "Suez Canal", "chokepoint2": "Panama Canal", "chokepoint4": "Bab el-Mandeb Strait",
             "chokepoint5": "Malacca Strait", "chokepoint6": "Strait of Hormuz", "chokepoint9": "Dover Strait",
             "chokepoint20": "Makassar Strait"}
    obs = []
    for eid, name in names.items():
        for i in range(30):
            t = NOW - timedelta(days=32 - i)
            v = 50.0 + (i % 4) if i < 29 else 80.0
            obs.append(_o("imf_portwatch_chokepoints", eid, "n_total", v, t, t, attrs={"port_name": name}))
    for slug in ctxmod.CHOKEPOINT_SLUGS:
        z = ctxmod._chokepoint_zscore(obs, slug)
        assert z is not None and z > 3, slug
    matched = {slug: {o.entity_id for o in obs
                      if any(a in ctxmod._norm_name(o.attrs["port_name"]) for a in
                             [ctxmod._norm_name(x) for x in ctxmod.CHOKEPOINT_SLUGS[slug]])}
               for slug in ctxmod.CHOKEPOINT_SLUGS}
    assert all(len(v) == 1 for v in matched.values()), matched


def _poll(entity, n, t):
    return _o("nws_alerts", entity, "alerts_polled", float(n), t, t, series="poll")


def _alert_count(entity, event, n, t):
    return _o("nws_alerts", entity, "alert_count", float(n), t, t, series=f"count:{event}",
              attrs={"event": event})


def test_weather_index_zero_when_polled_without_alerts_and_none_when_stale():
    t = NOW - timedelta(hours=2)
    obs = [_poll("ATL", 0, t), _poll("HOU", 0, t)]
    assert ctxmod.weather_disruption_index(obs, NOW) == 0.0
    assert ctxmod.weather_disruption_index(obs, NOW + timedelta(days=3)) is None
    assert ctxmod.weather_disruption_index([], NOW) is None


def test_weather_index_uses_only_latest_poll():
    severe = next(iter(ctxmod.SEVERE_ALERT_EVENTS))
    old, new = NOW - timedelta(days=5), NOW - timedelta(hours=1)
    obs = [_poll("ATL", 1, old), _alert_count("ATL", severe, 1, old),     # alte Warnung, vorbei
           _poll("ATL", 0, new), _poll("HOU", 1, new), _alert_count("HOU", severe, 1, new)]
    assert ctxmod.weather_disruption_index(obs, NOW) == 0.5


def test_alerts_parser_writes_poll_row_even_without_alerts():
    from modules.external.sources import weather as w
    obs = w.parse_alerts_response({"features": []}, "ATL", ["Hurricane Warning"], NOW)
    assert [(o.metric, o.value) for o in obs] == [("alerts_polled", 0.0)]
