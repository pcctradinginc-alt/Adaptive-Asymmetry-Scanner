# Alternative Data – Architektur (Phase 1: Entity Resolution + SEC Deep Events)

Status: **SHADOW / RESEARCH**. Keine Alternative-Data-Quelle hat Produktionseinfluss.
Kein Orderpfad. Safe Mode hat immer Vorrang.

## Kette

```
offizielle API ──► Rohdaten (Archiv/Event-Speicher, append-only, retrieved_at)
                 ──► normalisiert (pit.Observation: available_at, precision, provenance)
                 ──► Entity-Mapping (modules/entity_resolution, valid_from/valid_to, confidence)
                 ──► Feature Store (outputs/research/feature_store/<quelle>.csv.gz, feature_version)
                 ──► ml_research-Panel (alt_data.feature_store.attach, NaN statt 0)
                 ──► Hypothesen-Verträge (config/alt_hypotheses.yaml, spec_hash, unveränderlich)
                 ──► Research-Lab (gleiche Prüfkette, BH über alle Hypothesen)
                 ──► inkrementelle Walk-Forward-Bewertung (alt_data.evaluate, Protokoll alt-v1)
                 ──► Prospective Challenger (alt_forward_ledger.jsonl ab forward_start)
                 ──► Meta-Learning / PromotionController (nur nach Forward-Validierung, Mensch gibt frei)
```

| Schicht | Modul / Datei | Regel |
|---|---|---|
| Quellenvertrag | `config/external_sources/sec_entity.yaml` | Pflichtfelder der Registry (Lizenz, Rate-Limit, PIT-Präzision, Revisionen); `status_override: OWN_WORKFLOW` → im allgemeinen Orchestrator DEFERRED |
| Ingestion | `modules/external/sources/sec_ingest.py`, `modules/entity_resolution/build.py` | inkrementell, fehlertolerant, Fair-Access-Rate-Limit, Zustand in `state.json` |
| Event-Speicher | `outputs/external_data/sec/{insider,filings}/<jahr>.csv.gz` | append-only, Dedupe über `series_id`, deterministisches gzip |
| Entity-Map | `outputs/entity/entity_map.jsonl` | append-only open/close; keine Rückdatierung |
| Features | `modules/external/sources/sec_features.py` (`sec-f1`) | nur `available_at <= Stichtag 21:00 UTC`; außerhalb belegter Abdeckung NaN |
| Registry | `modules/alt_data/registry.py` | einzig zulässige Alt-Features; `ml_research.feature_list(extra_features)` lehnt alles andere ab |
| Panel-Join | `modules/alt_data/feature_store.py` | `merge_asof` rückwärts (Feature-Stichtag <= Panel-Datum, max. 7 T); fehlende Datei → NaN + Verfügbarkeit 0 |
| Bewertung | `modules/alt_data/evaluate.py`, `config/alt_data_protocol.yaml` (Hash gepinnt) | Baseline vs. Baseline+Quelle, Block-Bootstrap, Bonferroni |
| Verträge | `modules/alt_data/contracts.py`, `config/alt_hypotheses.yaml` | Pflichtfelder, `spec_hash`, INVALID_MODIFIED bei Nachänderung |
| Bericht | `reports/weekly.py` Abschnitt 17 | Scoreboard, Verdikt, Forward-Kohorten |

## Workflows

* `.github/workflows/alt_data.yml` – wöchentlich (Fr): Entity-Build + SEC-Ingestion, commit von `outputs/entity/` und `outputs/external_data/sec/`.
* `.github/workflows/ml_research.yml` – baut vor dem ML-Lauf die Feature-Tabelle aus den eingecheckten Events (`sec_ingest --features-only`), nach dem Research-Lab `python -m modules.alt_data.evaluate` (nur `full`).

Beide teilen die Concurrency-Gruppe `research-write` (nie parallel schreibend).

## Nicht verhandelbare Regeln (umgesetzt + getestet)

| Regel | Umsetzung | Test |
|---|---|---|
| `available_at <= prediction_time` | Cutoff 21:00 UTC; Form 345 Bulk: FILING_DATE + 1 T (CONSERVATIVE_DATE) | `tests/test_sec_events.py` |
| fehlend ≠ 0 | NaN + `alt_sec_available = 0`; echte Null nur innerhalb belegter Abdeckung | `test_feature_store_attach_missing_source_is_nan_not_zero` |
| keine Rückdatierung von Konzernstrukturen | `EntityStore.upsert` verweigert `as_of` vor letztem `valid_from` | `tests/test_entity_resolution.py` |
| keine Nachänderung von Hypothesen | `spec_hash` + append-only Registry | `test_contracts_valid_hash_frozen_and_modification_rejected` |
| Champion unberührt | `extra_features` nur in Varianten, nie in Registry-Champion | `test_registry_and_extra_features_guard` |
| keine LLM-Werte als Ground Truth | Labels ausschließlich Kursdaten; Alt-Features ausschließlich aus offiziellen Rohdaten | Code-Review, keine LLM-Aufrufe in `modules/alt_data`, `entity_resolution`, `sec_*` |

## Netz

Die Entwicklungs-Sandbox blockiert SEC/GLEIF/Wikipedia; Tests laufen mit Fixtures. Die
Live-Verifikation (echte Abrufe, echte Zahlen) erfolgt ausschließlich in GitHub Actions.
