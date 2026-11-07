# Daily Data Health Report – 2026-11-07

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.912

Status: HEALTHY 48 · DEGRADED 5 · STALE 1 · BROKEN 0 · UNVALIDATED 9

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| estat_jp_truck | STALE | NON_CRITICAL | 0 | 2026-03-01 | – | veraltet (Frequenz-Grenze der Registry) |
| bts_open_data_tsi | DEGRADED | IMPORTANT | 2 | 2026-08-01 | – | Ingest FAIL: SODA-Zeilenabruf fehlgeschlagen (bw6n-ddqk): 503 für https://data.bts.gov/resource/bw6n-ddqk.json; Data-Quality-Befund schwerwi |
| destatis_truck_toll | DEGRADED | IMPORTANT | 2 | 2026-09-01 | – | Ingest FAIL: GENESIS meldet Status 7000: Für die Teiltabelle mit der Metadatenkombination:
Indizes aus der Lkw-Mauterhebung (42191)
Lkw-Maut |
| imf_portwatch_chokepoints | DEGRADED | IMPORTANT | 0 | 2026-10-25 | – | RECOVERING 1/2: kein neuer erfolgreicher Abruf seit der letzten Prüfung, Bestätigung ausstehend |
| imf_portwatch_ports | DEGRADED | IMPORTANT | 0 | 2026-10-30 | – | RECOVERING 1/2: kein neuer erfolgreicher Abruf seit der letzten Prüfung, Bestätigung ausstehend |
| ncei_normals | DEGRADED | NON_CRITICAL | 0 | – | – | Missing-Rate 35% |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['estat_jp_truck', 'imf_portwatch_chokepoints', 'imf_portwatch_disruptions', 'imf_portwatch_ports']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._