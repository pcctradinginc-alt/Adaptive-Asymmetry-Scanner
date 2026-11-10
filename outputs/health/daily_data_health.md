# Daily Data Health Report – 2026-11-10

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.959

Status: HEALTHY 46 · DEGRADED 2 · STALE 2 · BROKEN 0 · UNVALIDATED 13

## Neue Änderungen

- recovery: entsoe_power DEGRADED → HEALTHY
- degradation: clinicaltrials_sponsors HEALTHY → UNVALIDATED (nicht aktiv: status_override=DEFERRED)
- degradation: usaspending_contracts HEALTHY → UNVALIDATED (nicht aktiv: status_override=DEFERRED)
- degradation: sec_13f_holdings HEALTHY → UNVALIDATED (nicht aktiv: status_override=DEFERRED)
- recovery: destatis_truck_toll DEGRADED → HEALTHY
- degradation: nhc_storms HEALTHY → DEGRADED (Ingest WARN: DQ: SCALE_SHIFT; Data-Quality-Befund schwerwiegend)
- degradation: noaa_storm_events HEALTHY → UNVALIDATED (nicht aktiv: status_override=DEFERRED)

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| finra_short_interest | STALE | NON_CRITICAL | 0 | 2026-10-15 | – | veraltet (Frequenz-Grenze der Registry) |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |
| nhc_storms | DEGRADED | IMPORTANT | 0 | – | – | Ingest WARN: DQ: SCALE_SHIFT; Data-Quality-Befund schwerwiegend |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['destatis_truck_toll', 'estat_jp_truck', 'finra_short_interest', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._