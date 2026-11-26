# Daily Data Health Report – 2026-11-26

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.992

Status: HEALTHY 48 · DEGRADED 2 · STALE 0 · BROKEN 0 · UNVALIDATED 13

## Neue Änderungen

- degradation: eia_petroleum_weekly HEALTHY → DEGRADED (Ingest FAIL: SourceTimeout('Zeitbudget 300s überschritten'))
- recovery: finra_short_interest DEGRADED → HEALTHY

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| eia_petroleum_weekly | DEGRADED | NON_CRITICAL | 1 | 2026-11-17 | cmdx_oil__crude_stocks_chg_z_52w, cmdx_oil__crude_stocks_vs_5y, cmdx_oil__div_oil_price_inventory, cmdx_oil__product_supplied_chg_4w | Ingest FAIL: SourceTimeout('Zeitbudget 300s überschritten') |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['destatis_truck_toll', 'entsoe_power', 'estat_jp_truck', 'finra_short_interest', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports', 'nhc_storms', 'usaspending_contracts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._