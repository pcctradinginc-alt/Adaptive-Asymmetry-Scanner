# Daily Data Health Report – 2026-10-14

**Safe Mode (Daten): aus** – gewichtete Data Quality 0.912

Status: HEALTHY 28 · DEGRADED 7 · STALE 0 · BROKEN 1 · UNVALIDATED 8

## Nicht gesunde Quellen

| Quelle | Status | Kritikalität | Fehler in Folge | letzter Datenstand | betroffene Features/Modelle/Entscheidungen | Grund |
|---|---|---|---|---|---|---|
| fred_market_expectations | BROKEN | IMPORTANT | 4 | – | – | Ingest FAIL: FRED ALFRED-Abruf fehlgeschlagen (DGS2): 400 für https://api.stlouisfed.org/fred/series/observations (nicht retrybar): {"error_ |
| fred_commodities | DEGRADED | NON_CRITICAL | 0 | 2026-10-06 | cmdx_copper__copper_ret_3m, cmdx_copper__div_copper_positioning_price, cmdx_corn__corn_ret_3m, cmdx_natural_gas__div_gas_storage_price | Ingest WARN: 7384 ALFRED-Vintage-Beobachtungen (Brent/HenryHub/Kupfer/Weizen/Mais/Soja) | DQ: OUT_OF_RANGE; Data-Quality-Befund schwerwiegen |
| gleif_lei | DEGRADED | NON_CRITICAL | 0 | – | – | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| noaa_storm_events | DEGRADED | NON_CRITICAL | 0 | 2025-01-01 | – | Data-Quality-Befund schwerwiegend; Duplikate None / Konflikte 932 |
| sec_companyfacts | DEGRADED | NON_CRITICAL | 0 | 2026-10-06 | xbrl_accruals, xbrl_asset_growth, xbrl_rev_yoy, xbrl_share_change | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| sec_form345 | DEGRADED | NON_CRITICAL | 0 | 2026-10-08 | sec_8k_count_30d_z, sec_8k_negative_90d, sec_exec_change_90d, sec_filing_delay_z | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| sec_submissions | DEGRADED | NON_CRITICAL | 0 | 2026-10-14 | sec_8k_count_30d_z, sec_8k_negative_90d, sec_exec_change_90d, sec_filing_delay_z | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |
| ted_awards | DEGRADED | NON_CRITICAL | 0 | 2026-10-14 | ted_any_award_365d, ted_award_value_365d, ted_awards_90d, ted_awards_z | RECOVERING 1/2: vollständiger Abruf ok, Bestätigung ausstehend |

## Fallbacks

- keine

## Blockierte Entscheidungspfade / deaktivierte Signale

- blockiert: keine
- deaktivierte Signale: keine
- nicht verfügbare Features: keine
- Cache (stale) genutzt: keine

Wiederkehrende Instabilität (30 T): ['sec_companyfacts']

_Regel: Ein fehlendes Signal ist besser als ein Signal aus falschen oder unbekannt alten Daten._