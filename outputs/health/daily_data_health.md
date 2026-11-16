# Daily Data Health Report – 2026-11-16

**Safe Mode (Daten): AKTIV** – gewichtete Data Quality 0.89 – 2 wichtige Quellen ausgefallen aus 1 unabhängigen Ausfall-Domänen: ['imf_portwatch_chokepoints', 'imf_portwatch_ports'] (Domänen: ['imf_portwatch_arcgis']; Safe-Mode-Schwelle unverändert)

Status: HEALTHY 45 · DEGRADED 2 · STALE 0 · BROKEN 3 · UNVALIDATED 13

## Neue Änderungen

- degradation: options_chain_yfinance HEALTHY → DEGRADED (leere Antwort: keine Verfallstermine)

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| imf_portwatch_chokepoints | BROKEN | IMPORTANT | 5 | 2026-11-08 | – | Ingest FAIL: UPSTREAM_RATE_LIMITED host=services9.arcgis.com: Kontingent in diesem Lauf bereits erschöpft (Unable to perform query. Too many |
| imf_portwatch_disruptions | BROKEN | NON_CRITICAL | 5 | 2026-08-14 | – | Ingest FAIL: UPSTREAM_RATE_LIMITED host=services9.arcgis.com: Kontingent in diesem Lauf bereits erschöpft (Unable to perform query. Too many |
| imf_portwatch_ports | BROKEN | IMPORTANT | 5 | 2026-11-06 | – | Ingest FAIL: [UPSTREAM_RATE_LIMITED] ArcGIS-Query-Fehler trotz Wartezeit (höchstens 3 Versuche, Wartebudget 130s je Host): {'code': 429, 'me |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |
| options_chain_yfinance | DEGRADED | CRITICAL | 1 | – | implied_vol, option_quotes, options_design, roi_precheck | leere Antwort: keine Verfallstermine |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['destatis_truck_toll', 'estat_jp_truck', 'finra_short_interest', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports', 'nhc_storms', 'sec_13f_holdings', 'usaspending_contracts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._