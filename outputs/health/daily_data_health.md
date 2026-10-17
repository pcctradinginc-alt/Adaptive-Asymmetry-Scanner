# Daily Data Health Report – 2026-10-17

**Safe Mode (Daten): aus** – gewichtete Data Quality 1.0

Status: HEALTHY 42 · DEGRADED 5 · STALE 0 · BROKEN 0 · UNVALIDATED 8

## Neue Änderungen

- degradation: eia_natural_gas HEALTHY → DEGRADED (Ingest FAIL: SourceTimeout('Zeitbudget 300s überschritten'))
- recovery: noaa_storm_events DEGRADED → HEALTHY

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| bea_release_calendar | DEGRADED | NON_CRITICAL | 0 | 2027-11-04 | – | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| bls_release_calendar | DEGRADED | NON_CRITICAL | 0 | 2026-12-30 | – | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| eia_natural_gas | DEGRADED | NON_CRITICAL | 1 | 2026-10-06 | cmdx_natural_gas__div_gas_storage_price, cmdx_natural_gas__lng_exports_yoy, cmdx_natural_gas__natgas_storage_chg_z_52w, cmdx_natural_gas__natgas_storage_vs_5y | Ingest FAIL: SourceTimeout('Zeitbudget 300s überschritten') |
| fomc_calendar | DEGRADED | NON_CRITICAL | 0 | 2027-10-27 | – | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| sec_13f_holdings | DEGRADED | NON_CRITICAL | 0 | 2026-08-31 | – | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['eia_natural_gas', 'fred_market_expectations', 'noaa_storm_events', 'sec_companyfacts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._