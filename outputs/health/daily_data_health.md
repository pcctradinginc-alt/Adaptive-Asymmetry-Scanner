# Daily Data Health Report – 2026-10-03

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.991

Status: HEALTHY 29 · DEGRADED 1 · STALE 0 · BROKEN 0 · UNVALIDATED 9

## Neue Änderungen

- recovery: imf_portwatch_ports DEGRADED → HEALTHY
- recovery: bts_open_data_tsi DEGRADED → HEALTHY

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| sec_companyfacts | DEGRADED | NON_CRITICAL | 0 | 2026-09-28 | xbrl_accruals, xbrl_asset_growth, xbrl_rev_yoy, xbrl_share_change | 1 Beobachtungen mit Periode in der Zukunft (Datenfehler) |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['bts_open_data_tsi']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._