# Daily Data Health Report – 2026-10-03

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.932

Status: HEALTHY 26 · DEGRADED 3 · STALE 0 · BROKEN 1 · UNVALIDATED 9

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| bts_open_data_tsi | BROKEN | IMPORTANT | 2 | 2024-04-01 | – | Schema geändert (Orchestrator); Data-Quality-Befund schwerwiegend; veraltet (Frequenz-Grenze der Registry) |
| imf_portwatch_ports | DEGRADED | IMPORTANT | 0 | 2026-09-25 | – | Ingest WARN: 10682 observations (kuratierte Häfen + GLOBAL/Gruppen-Aggregate aus 607110 Rohbeobachtungen); 0 unresolved ports: [] | volume g |
| sec_companyfacts | DEGRADED | NON_CRITICAL | 0 | 2026-09-28 | xbrl_accruals, xbrl_asset_growth, xbrl_rev_yoy, xbrl_share_change | 1 Beobachtungen mit Periode in der Zukunft (Datenfehler); Ausreißer-Anteil 3.0% |
| vix_fred | DEGRADED | CRITICAL | 2 | – | vix, risk_gates | Timeout: FetchError: HTTPSConnectionPool(host='fred.stlouisfed.org', port=443): Read timed out. (read timeout=20); Antwortzeit 66.16 s > 15  |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): keine

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._