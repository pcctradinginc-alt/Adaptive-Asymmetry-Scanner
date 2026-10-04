# Commodity Intelligence (RESEARCH / SHADOW)

**Kernregel:** Nur offizielle, kostenlose Quellen: EIA Open Data API v2, FRED/ALFRED und CFTC COT. Lieber UNAVAILABLE als unzuverlässige, gescrapte oder unklare Daten.

Rohstoffdaten sind erklärende Research-Features für US-Aktien und ihre Optionssetups. Es wird **nicht** angenommen, dass sie Alpha liefern. Das System kann lernen:
- NO RELATIONSHIP (Ablation REJECT, Hypothese REJECTED),
- RELATIONSHIP DECAYED (`alpha_decay`, Forward-Demotion).

Konfiguration: `config/commodity_intelligence.yaml`. Registry: `config/external_sources/commodities.yaml`.
Konnektoren: `modules/external/sources/commodities.py`. Features: `modules/commodity_intelligence.py`.

## Architektur: Wiederverwendung, keine zweite Pipeline
| Schritt | Bestehende Komponente |
|---|---|
| Abruf, Budget, PIT-Integrität, Data Quality, Archiv | `modules/external/orchestrator.run_ingestion` → `ExternalArchive` (append-only Vintages) |
| FRED/ALFRED | `FredCommodityConnector` erbt `FredUsMacroConnector`. WTI wird aus `fred_regime_macro` wiederverwendet. |
| Fälligkeit / keine unnötigen Calls | `Connector.is_due()`: der Orchestrator überspringt eine Quelle, solange die jüngste veröffentlichte Periode schon im Archiv liegt. |
| Source Health | Registry-Quellen erscheinen automatisch in `modules/source_health` (`research_only` → nie globaler Safe Mode) |
| Feature-Store / Panel | `modules/alt_data/registry.SOURCES` (`commodity_price` / `_fundamental` / `_positioning` / `_divergence`, attach `date_level`). `feature_store.attach` arbeitet mit `feature_contracts` je Feature. |
| Ablation | `alt_data.evaluate` (volle Population, Protokoll unverändert) plus `commodity_intelligence.evaluate` (Breakdowns) |
| Hypothesen | `config/research_domains.yaml` (Commodity-Domänen und -Familien), `hypothesis_factory._commodity_idea`, `DIVERGENCES` |
| Gedächtnis | `research_memory.commodity_entries` (Fingerprint → Ähnlichkeitssperre) |
| Promotion | Hypothese → Vertrag → OOS → Prospective Challenger → Forward → PromotionController → Adapter |

## Secrets
Werden nur über die Umgebung gelesen, nie im Code:

| Umgebungsvariable | Quelle (GitHub Secret) |
|---|---|
| `EIA_API_KEY` | `EIA_KEY` (Fallback `EIA_KEY` direkt) |
| `FRED_API_KEY` | `FRED_API_KEY` oder `FRED_KEY` |

CFTC braucht keinen Key. Ohne Key: `AUTH_MISSING`, und es erfolgt kein Call. Fehlertexte werden maskiert (`redact_secrets`).

## Quellen, Release-Regeln, PIT
`available_at = min(konservative Release-Regel, retrieved_at)`, Präzision `CONSERVATIVE_DATE`. Ausnahme FRED: ALFRED-`realtime_start` (`EXACT_DATE`).

| Quelle | Serien | Frequenz | Release-Regel (konservativ) |
|---|---|---|---|
| eia_petroleum_weekly | WCESTUS1, WCRFPUS2, WCRIMUS2, WCREXUS2, WPULEUS3, WRPUPUS2, WGTSTUS1, WDISTUS1 | wöchentlich (Fr) | Fr + 6 T, 16:00 UTC (WPSR Mi 10:30 ET, Feiertag Do 11:00 ET) |
| eia_petroleum_weekly | RWTC, RBRTE (nur Crosscheck gegen FRED) | täglich | Tag + 9 T |
| eia_natural_gas | NW2_EPG0_SWO_R48_BCF (Speicher) | wöchentlich | Fr + 7 T, 16:00 UTC (Do 10:30 ET) |
| eia_natural_gas | RNGWHHD (Crosscheck) | täglich | Tag + 9 T |
| eia_natural_gas | N9070US2 (Trockengas), N9133US2 (LNG-Export) | monatlich, revidiert | Monatsbeginn + 100 T |
| fred_commodities | DCOILBRENTEU, DHHNGSP, PCOPPUSDM, PWHEAMTUSDM, PMAIZMTUSDM, PSOYBUSDM | täglich / monatlich | ALFRED-Vintages |
| fred_regime_macro | DCOILWTICO (WTI, wiederverwendet) | täglich | ALFRED-Vintages |
| cftc_cot | Disaggregated Futures Only (8 Märkte) | wöchentlich (Di) | Di + 3 T, 21:00 UTC; Feiertagswoche + 6 T (Mo) |

Weitere Regeln:
- **Revisionen** nach dem ersten archivierten Wert gelten erst ab dem Abruf (`revised_after_first_seen`, `available_at = vintage_time = Abruf`). Ein Erstimport revidierter Monatsreihen ist als `backfill_latest_vintage` markiert; die Spalte `rev_<serie>` steht im Feature-Store.
- **COT-Shutdown-Rückstände** (2018-12-18 bis 2019-01-29, verspätet veröffentlicht) sind ausgeschlossen.
- **Gold und Silber:** keine offizielle kostenlose Preisreihe (LBMA wurde aus FRED entfernt). Preis-Features sind daher UNAVAILABLE. Die COT-Positionierung bleibt verfügbar.
- **COT-Mapping:** Contract Market Code **und** Namensprüfung. Bei Abweichung oder fehlendem Markt ist der Markt UNAVAILABLE.
- **Forward Fill:** nur bis `max_age_days` je Frequenz plus Veröffentlichungsverzug. Bei Überschreitung ist das Feature UNAVAILABLE (NaN). `age_<serie>` wird je Stichtag protokolliert.

## Features (`cmd-features-1`, Stichtag D 21:00 UTC)
**Preis** (WTI, Brent, Henry Hub täglich; Kupfer und Agrar monatlich):
- täglich: `ret_{1,5,20,60}d`, `mom_60_5`, `vol_20d`, `z_60d`, `dd_252d`;
- monatlich: `ret_1m`, `ret_3m`, `z_36m`.
- Ist der Vorwert ≤ 0 (WTI 2020), ist die Rendite None, nie 0.

**Fundamental:**
- Bestände: `chg_1w`, `chg_4w`, `chg_z_52w`, `vs_5y` (gleiche Kalenderwoche der Vorjahre, mindestens 3 Jahre).
- Flüsse: `chg_4w`, `z_52w`.
- Monatsreihen: `yoy`, `chg_3m`.
- `storage_surprise` = **EXPECTATION_UNKNOWN**. Es gibt keine offizielle kostenlose Konsens-Erwartung, also auch kein Ersatz durch eine Schätzung.

**Positionierung** je Markt:
- `mm_net`, `comm_net` (Producer/Merchant + Swap), beide auch in % des OI;
- `mm_net_chg_{1,4,13}w`;
- `mm_pctile_{1y,3y}`, `comm_pctile_{1y,3y}` (mindestens 40 bzw. 120 Wochen, sonst None);
- `oi_chg_4w`, `mm_extreme`.
- Reports mit COT > OI werden verworfen und gezählt.

**Divergenz** (nur Research): Preis gegen Bestand (Öl), Positionierung gegen Preis (Öl, Kupfer), Speicher gegen Preis (Gas).

**Equity-Exposure** (`exposure-v1`, **NON_PIT**, `non_pit_mapping=true`):
- Felder: commodity, ticker, exposure_type, expected_direction (+1/−1/0), mapping_source, mapping_version, confidence, point_in_time_status.
- Kreuzfeatures `cmdx_<rohstoff>__<feature>` = Richtung × Datums-Feature. Nicht gemappte Titel haben NaN (nie 0).
- Signal-Spalten `cmdexp_<rohstoff>`: gemappt → Richtung; Sektor bekannt, aber nicht gemappt → 0; Sektor unbekannt → NaN.
- Eine Mapping-Änderung (Hash) erzeugt den Befund `MAPPING_CHANGED`.

## Datenqualität
- Befunde je Reihe: negative Werte (außer WTI), Extrembewegungen (nur Befund), widersprüchliche Duplikate, falsche Frequenz, fehlende Releases, Einheitenwechsel, Crosscheck EIA gegen FRED (2 %), COT > OI, Mapping-Änderung.
- **Schwer** (Reihe UNAVAILABLE): Einheitenwechsel, unmögliche negative Werte, falsche Frequenz, keine Daten.
- Einheiten sind explizit je Serie festgelegt. Es gibt keine stillen Umrechnungen; eine API-Einheitsänderung führt zu `SCHEMA_CHANGED`.

## Source Health
- Status HEALTHY / DEGRADED / STALE / BROKEN / UNVALIDATED (`commodity_intelligence.health_label`, plus der tägliche Source Health Check).
- Bei einem Ausfall werden **nur die abhängigen Features** UNAVAILABLE (`feature_contracts`). Beispiel: Fällt CFTC aus, sind Positionierung und Divergenz mit COT betroffen; Preise und EIA nicht.
- `research_only: true`: Diese Quellen zählen nie für den globalen Safe Mode.

## Research, ML, Promotion
- **Fabrik:** neun Commodity-Familien und zwei Divergenzen. Signal `cmdexp_<rohstoff> * sign(feature − center)`.
  - Population, Horizont (20 T), Baseline (Querschnittsmittel), OOS (Walk-Forward + Locked + Forward) und BH-Mehrfachtest liegen fest.
  - Die LLM oder Vorlage liefert nur den Mechanismus. Readiness setzt das Flag `non_pit_mapping`.
- **Surprise Engine:** EIA-Konsens fehlt → DATA_GAP. Commodity-Abweichungen gehen nur als Hypothesen in die Fabrik.
- **Ablation** (`python -m modules.commodity_intelligence evaluate`, `ml_research.yml`, nur full):
  - Vergleich BASE gegen BASE+COMMODITY je Gruppe und gesamt, auf identischen Zeilen.
  - Metriken netto nach Kosten: Expectancy, Sharpe, IC, Precision@K, Brier, ECE, MaxDD, Δ je Jahr (Stabilität), Bootstrap mit Bonferroni.
  - Breakdown je market_cap_bucket (im V1-Panel nicht verfügbar), sector, commodity_exposure, regime, liquidity_bucket.
  - Ergebnis `incremental_value`: NONE / HISTORICAL_ONLY / NO_RELATIONSHIP.
- **Gedächtnis:** domain, commodity, equity_population, mechanism, features, result, n, OOS, forward, failure_reason, fingerprint. Beinahe-Duplikate werden über `similarity` blockiert.
- **Promotion:**
  - Start ist RESEARCH/SHADOW, Einfluss NONE.
  - Leiter: NONE → RERANK_ONLY → SCORE_LIMITED. `hypothesis_contract.validate_commodity` lehnt alles darüber ab, ebenso production_class weight/universe.
  - Ein Champion-Wechsel bleibt eine menschliche Entscheidung.
  - Kein Commodity-Feature hebt Scores oder blockiert Trades. Produktionsmodule importieren das Modul nicht (Test).

## Workflows
| Workflow | Schritt |
|---|---|
| `external_data.yml` (täglich 06:17) | Ingestion aller Commodity-Quellen. `is_due` ruft nur ab, wenn ein Release fällig ist: COT wöchentlich, EIA wöchentlich/monatlich, FRED täglich. |
| `source_health.yml` (täglich 12:41) | Health inklusive Commodity-Quellen |
| `ml_research.yml` | `commodity_intelligence build` (Feature-Store aus dem Archiv) → Panel → Fabrik/Lab → `alt_data.evaluate` → `commodity_intelligence evaluate` |
| Montagsbericht §10 / Monatsbericht | COMMODITY INTELLIGENCE. Ohne Evidenz: „Commodity Intelligence: RESEARCH ONLY – no validated incremental alpha“ |

## Grenzen (ehrlich)
- **Exposure-Mapping** ist von heute (NON_PIT). Historische Tests überschätzen die Trennschärfe, wenn Firmen ihr Geschäft geändert haben.
- **Abdeckung:** Das Mapping erfasst wenige S&P-500-Titel (Einzeltitel-Regeln; Branchenregeln nur, wenn das Panel `industry` liefert). Das geschützte Alt-Data-Protokoll (Mindestabdeckung 30 % des Panels) wird die Gruppen deshalb in `alt_data.evaluate` voraussichtlich als REJECT einstufen. Die eigene Ablation misst die Abdeckung innerhalb der gemappten Titel.
- **Monatliche EIA-Reihen:** Historische Werte sind der heutige revidierte Stand (Flag gesetzt). Echte Erstveröffentlichungen entstehen erst prospektiv.
- **Feldnamen und Serien-IDs** sind nicht live verifiziert (Sandbox ohne Netz). Abweichungen schlagen laut fehl (SCHEMA_CHANGED / UNAVAILABLE), nie still.
