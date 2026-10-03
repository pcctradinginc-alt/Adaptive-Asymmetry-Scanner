# SystemState: kanonischer Zustand, abgestufte Drift, Source-Health-Abhängigkeiten

Stand: 2026-10-03. Teil 1 des Architektur-Auftrags (Schritte 1–5). Es werden keine neuen Intelligence-Komponenten gebaut, sondern Konsistenz hergestellt.

## 1. Audit: Wo lagen die Zustände?

| Zustand | vorher | Problem |
|---|---|---|
| Safe Mode | `outputs/research/safe_mode.json` (Meta-Cognition, wöchentlich), `meta_state.json.safe_mode` (**immer false**), `machine_state.json.safe_mode` (Kopie), `outputs/health/data_safe_mode.json` (täglich) | Der Montagsbericht las nur `safe_mode.json`, der Scanner nur Data Health. HC, Adapter und Promotion kombinierten beides. Gleichzeitig widersprüchliche Meldungen waren also möglich. |
| Drift | `meta_learning.json.drift.feature_drift_flag` (binär) → Safe-Mode-Trigger | Ein einzelnes Merkmal knapp außerhalb von p01/p99 (Live: `tnx` 5,28 > 4,78) blockierte das ganze System. |
| Data Health | `outputs/health/source_health_snapshot.json` und Orchestrator-Health | doppelt bewertet: Meta-Cognition zählte FAIL/STALE erneut |
| Champion | `config/model_registry.yaml champion: null` (ML), Regelwerk `config.yaml` | keine Version in Entscheidungen |
| Promotion | `outputs/intelligence/promotion_state.json` | keine Verknüpfung mit Drift |
| Model Health | `meta_learning.json.model_intelligence`, Kalibrierung, Disagreement | verstreut |

## 2. SystemState (`modules/system_state.py`)

Es gibt **eine** Ableitung (`derive`) aus den Komponenten. Das Ergebnis wird atomar und versioniert persistiert: `outputs/state/system_state.json` plus append-only `system_state_history.jsonl`. Die Version steigt nur bei inhaltlicher Änderung (Fingerprint).

Felder:

- `safe_mode`, `safe_mode_reason`
- `data_health`, `model_health`, `drift_state`
- `champion_version` (`scanner_rules@<config-hash>` bzw. ML-Champion)
- `promotion_state`
- `known`
- `updated_at`, `code_version`, `state_version`, `inputs` (Hash je Eingabe)

| Komponente | Eingabe | Beitrag zu Safe Mode |
|---|---|---|
| model_health | `safe_mode.json`: nur `components.model` (Kalibrierung, Disagreement, World Model, Performance, defekte Artefakte) | jeder Grund |
| drift_state | `modules/drift.py` | nur **SEVERE** |
| data_health | Source-Health-Snapshot | globaler Daten-Safe-Mode |
| unbekannt | fehlende oder unlesbare Eingabe, Snapshot älter als 30 h | **fail-closed**: Safe Mode aktiv |

**Leser** (alle über `current()` bzw. `safe_mode_view()`):

- Pipeline (`stats.system_state`)
- HC-Scanner (inkl. Confidence-Multiplikator)
- ProductionIntelligenceAdapter (inkl. Boost-Deckel)
- PromotionController (pausiert ab Drift MODERATE)
- Weekly Report
- Meta-Learning (`meta_active` nur ohne Safe Mode)
- Meta-Cognition (`machine_state.safe_mode`)

**Fortschreibung:** täglicher Source Health Check, Meta-Cognition und Montagsbericht. Jeder Workflow committet `outputs/state/`.

`safe_mode.json` ist nur noch eine Komponente (`canonical: false`). `meta_state.safe_mode` ist entfernt. Ein Test (`test_no_component_reads_legacy_safe_mode_flags`) verhindert neue direkte Leser.

## 3. Abgestufte Drift (`modules/drift.py`, `config/drift_policy.yaml`, gepinnt)

| Komponente | NORMAL | MILD | MODERATE | SEVERE |
|---|---|---|---|---|
| Feature (Überschreitung von [p01, p99] relativ zur Spannweite) | innerhalb | ≤ 5 % | ≤ 30 % | > 30 % **und** ≥ 2 Merkmale außerhalb, oder ≥ 50 % der Merkmale außerhalb |
| Modell (Anteil „deteriorating") | < 15 % | ≥ 15 % | ≥ 30 % | ≥ 50 % (= gepinnte `model_drift_share`, nicht gelockert) |
| Daten (gewichtete Data Quality) | ≥ 0,90 | < 0,90 | < 0,75 | < 0,60 oder unbekannt |

Gesamt = schlechteste Komponente.

| Stufe | Confidence | positive Boosts | Gewichtserhöhung | Promotion | Safe Mode |
|---|---|---|---|---|---|
| NORMAL | × 1,0 | voll | ja | ja | nein |
| MILD | × 0,9 (HC: p → 0,5 geschrumpft) | voll | ja | ja | nein |
| MODERATE | × 0,75 | × 0,5 | nein | pausiert | nein |
| SEVERE | × 0,5 | keine | nein | pausiert | **ja** |

Die Lockerung gegenüber dem binären Trigger gilt nur für Feature-Drift, die aus einem einzelnen Merkmal kommt. Das war ausdrücklicher Auftrag. MILD und MODERATE schränken weiterhin ein.

**Live-Stand 2026-10-03:**

- `tnx` ist MODERATE (13 % über p99) und blockiert nicht mehr.
- Modell-Drift: Anteil „deteriorating" 0,333 → MODERATE (SEVERE erst ab 0,50, unverändert gepinnt).
- Gesamt **MODERATE**, Safe Mode **aus** (`outputs/state/system_state.json`, state_version 6):
  Confidence × 0,75, positive Boosts × 0,5, keine Gewichtserhöhung, Promotion pausiert.
  Frühere Fassung dieses Dokuments nannte SEVERE (3 von 6 Modellen); maßgeblich ist immer die
  Datei, nie dieses Dokument.

## 4. Source Health als harte Voraussetzung

- **Feature→Quelle für jedes Panel- und Alt-Feature:** `feature_sources` in `config/source_health.yaml` plus Alt-Registry. Generierte Doku: `docs/FEATURE_DEPENDENCIES.md`. Ein Test prüft Vollständigkeit und Aktualität.
- **Quelle STALE, BROKEN (inkl. SCHEMA_CHANGED) oder UNVALIDATED:** Die Features sind unavailable.
  - Feature Store: NaN und Verfügbarkeit 0 ab dem Ausfall.
  - Adapter: Wert `None`, Verträge mit diesen Features werden deaktiviert.
  - HC: gesperrt, wenn Modellmerkmale fehlen.
  - Data Quality und Confidence sinken über die Daten-Drift.
- **Kritische Pfade:** blockiert oder über einen explizit getesteten Fallback (VIX → FRED, Optionen → yfinance) mit `source_primary`/`source_actual`.
- **Live-Fix:** Ein einzelner SEC-XBRL-Fakt mit Periodenende nach der Einreichung machte die Quelle BROKEN.
  - Der Parser verwirft solche Fakten jetzt.
  - Eine Periode in der Zukunft gilt als Datenfehler: DEGRADED, ab mehr als 20 % Anteil BROKEN.
  - `available_at > retrieved_at` bleibt eine PIT-Verletzung (BROKEN).

## 5. Ein Safe-Mode-Begriff (Härtung 2026-10-03)

- Tages-Stats der Pipeline: `stats.system_state.active` ist der Safe Mode. Die Daten-Komponente
  heißt dort `stats.data_health.data_safe_mode` (vorher ebenfalls `safe_mode` → zwei
  gleichnamige Flags in einer Datei; die Warnung im Log las das falsche).
- `meta_learning.safe_mode_check` → `meta_fallback_check` (`fallback_to_reference`): Rückfall
  des (nicht promoteten) Meta-Modells auf die Referenz, kein System-Safe-Mode.
- Tests: `tests/test_pipeline_orchestration.py` (Fehlerpfade von `pipeline.main()`),
  `tests/test_hardening.py`.
