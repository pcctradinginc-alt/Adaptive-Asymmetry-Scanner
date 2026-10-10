# Robustheit & Lern-Integrität V1 (2026-10-10, SHADOW/RESEARCH)

Fünf Konzepte wurden geprüft. Implementiert wurden nur echte Lücken; bestehende Logik wird nicht dupliziert.
Keine der neuen Metriken hat Produktionsgewicht. Einfluss ist nur über die bestehende Leiter möglich:
Vertrag → Forward-Evidenz → `PromotionController` → `ProductionIntelligenceAdapter`.

## A) Bereits vorhanden (wiederverwendet, nicht dupliziert)
| Konzept | Bestand |
|---|---|
| Familien-Dämpfung | nur die EA-Rohzählung `n_confirming/n_available` (`cross_asset_confirmation`); Familien fehlten |
| LLM-Claims | fehlten. Der Deep-Analysis-Prompt bleibt unangetastet (Produktion) |
| Consume-once | EA-, Final-MC- und V2-Ledger je Schlüssel idempotent. Voll neu berechnet (zählen je Berechnung genau einmal): Pearson-Gewichte, Kalibrierung, Robust-PPO-Retrain (deterministisch) und PPO-Retrain (nur bei aktivem Veto, aus). **Lücken:** inkrementelle `feature_stats`/`feature_stats_external` und 7 exakte Altduplikate in `closed_trades` |
| Lead-Lag | `factor_monitor` (Spearman-IC, Fisher-CI, Walk-Forward) nur für Champion-Features |
| Warm-up | teilweise: `expanding_z` ab `min_history_weeks`, `pit_weekly` mit `max_age_days`; RoC, Regime-Unsicherheit und Cross-Asset-Signale ohne Guard |

## B) Implementiert
1. **Familienbewusste Bestätigung** (`modules/evidence_families.py`)
   - Familiengewicht `W_f = Σ_{i<k} dⁱ` (d = `confirmation.family_dampening` = 0,5, also 1 / 1,5 / 1,75 …). Das Gewicht ist reihenfolgeunabhängig.
   - Effektiv bestätigend je Familie: `W_f · n_conf / k`.
   - Fehlende Evidenz (state None) zählt weder im Zähler noch im Nenner.
   - Ausgabe:
     - `raw_confirmation_count` (Rohzählung bleibt erhalten);
     - `effective_confirmation`, `effective_confirmation_ratio`, `effective_conflict_share`;
     - `family_count`, `families_missing`, `family_breakdown`.
   - Die Familienzuordnung ist ökonomisch, nicht gefittet (`config/expectation_alpha.yaml` `signal_families`/`sector_families`). Beispiel: `xle_spy_20d` und `sector_rs_20d:XLE` gehören zur gleichen Familie.
   - Die EA-Research-Entscheidung (TRADE/WAIT/ABSTAIN), der WAIT-Trigger und die Kill Condition nutzen effektive Werte. Erforderlich sind mindestens 3 Familien; Dämpfung und Mindestfamilien sind im Kill-Spec eingefroren.
   - Die registrierten Vertragsfeatures (`ea_confirmation_ratio`, EA002) bleiben roh. Damit ändert sich kein Vertragshash.
   - Config-Version `ea-v1.1` (vor `forward_start`, ohne Outcomes).
2. **Maschinell prüfbare Claims** (`modules/expectation_alpha/claims.py`, `claim_extraction.py`)
   - Das LLM strukturiert nur den Deep-Analysis-Text in einem separaten Shadow-Call (Workflow `ea_claims`, Kostenbereich `shadow`, drosselbar). Ohne API-Key, ohne Modell oder unter dem Budget-Guard wird der Schritt übersprungen: SKIPPED, kein harter Fallback.
   - Taxonomie `claims-v1` mit 21 Typen. Jeder Claim hat `claim_type`, `entity`, `metric`, `direction`, `value_if_known`, `source_reference`/`evidence_id` und `confidence`.
   - `value_if_known` bleibt nur erhalten, wenn die Zahl wörtlich im Quelltext steht. Erfundene Werte werden verworfen.
   - Die Prüfung ist deterministisch. Sie nutzt nur Evidenz mit `available_at < decision_time`:
     - SEC-XBRL: je Periode jüngste Einreichung, Totband 1 %, maximal 200 Tage alt;
     - 8-K-Items: 14-Tage-Fenster; ein fehlendes Filing widerspricht nie.
   - Status VERIFIED / UNVERIFIED / CONTRADICTED. Nicht prüfbare Typen (Guidance, Orders, Capex, other) bleiben immer UNVERIFIED. Das LLM verifiziert sich nie selbst.
   - Ausgabe: `n_claims`, `n_verified`, `n_unverified`, `n_contradicted`, `verified_claim_fraction`.
   - Nur Logging; kein Score und keine Entscheidung. Ein Ausfall ergibt ERROR-Platzhalter und bricht den Lauf nie.
3. **Consume-once-Lernen** (`modules/learning_ledger.py`, `feedback.py`)
   - Schlüssel `observation_id:horizon:learning_target`, z. B. `T…:close:feature_stats` oder `shadow:<id>:45d:feature_stats_external`.
   - Die Markierung liegt im selben, atomar geschriebenen Dokument wie der gelernte Zustand (`history.json`). Update und Markierung werden daher gemeinsam gespeichert oder gehen gemeinsam verloren; das ist crash- und retry-sicher.
   - Bei einem Fehler im Update werden die betroffenen Abschnitte zurückgesetzt, und es wird keine Markierung gesetzt.
   - Verschiedene Horizonte oder Lernziele derselben Beobachtung sind getrennt erlaubt.
   - Doppelte `active_trades` werden beim Close in `closed_trades_quarantine` verschoben (Grund `duplicate_close`).
   - Exakte Altduplikate in `closed_trades` (gleiche Trade-ID, gleicher Close-Tag, gleicher Outcome) wandern nach `closed_trades_quarantine`. Sie werden nicht gelöscht, und der Vorgang ist idempotent.
   - Das Log meldet `duplicate_learning_skips`.
4. **Warm-up-Guards** (`future_state`, `regime_change`, `expectation_gap`, `cross_asset_confirmation`)
   - Mindestpunkte = Lag + 1:
     - Δ1M: d1+1;
     - Δ3M, velocity, acceleration: d3+1;
     - change_of_change: d3+d1+1;
     - z und Perzentil: `min_history_weeks`+1.
   - Status je Feld (`history_status`, `insufficient_history_fields`).
   - Regime-Unsicherheit erst ab 6 verfügbaren Dimensionen.
   - Cross-Asset-Signale: 20T ab 21 Handelstagen, 63T ab 64 Handelstagen; Grund `missing_reason` = INSUFFICIENT_HISTORY oder UNAVAILABLE.
   - Eine lange, konstante Historie gilt als UNAVAILABLE („ohne Streuung“), nicht als Warm-up.
   - Nie wird ersatzweise 0, „neutral“, ein künstliches Extrem-z oder ein langer Forward-Fill geliefert.
5. **Lead-Lag-Diagnostik** (`modules/expectation_alpha/lead_lag.py`, Evaluation + Montagsbericht)
   - Je Feature und Familie auf den vorab festen Horizonten 20/60/120/250:
     - `n`, `independent_signal_days`;
     - Spearman-IC mit Fisher-CI;
     - `forward_return_spread` (Terzil bzw. binär) mit Block-Bootstrap-CI über Signaltage;
     - Trefferquote, MAE/MFE, Regime-Aufschlüsselung.
   - Der Featurewert stammt aus dem eingefrorenen Snapshot. Outcomes zählen nur mit `exit_date >` Entscheidungstag; ERROR-Zeilen zählen nie.
   - Es werden immer alle Horizonte berichtet. Es gibt keine Auswahl, kein Gewicht und keine Strategieänderung.
   - Unter 30 Beobachtungen lautet der Status NEED_MORE_DATA, ohne Effektaussage.

Kompaktes Lauf-Log (`runs.jsonl` / Feedback-Log):
- `effective_vs_raw_confirmation`;
- `family_diversity`;
- `claims` (inkl. `verified_claim_fraction`);
- `insufficient_history_count`;
- `duplicate_learning_skips`;
- `lead_lag_status`.

## C) Geänderte Dateien
**Neu:**
- `modules/evidence_families.py`, `modules/learning_ledger.py`;
- `modules/expectation_alpha/claims.py`, `claim_extraction.py`, `lead_lag.py`;
- Tests: `test_evidence_families.py` (9), `test_ea_claims.py` (10), `test_ea_claim_extraction.py` (45), `test_learning_consume_once.py` (11), `test_ea_warmup.py` (15), `test_ea_lead_lag.py` (7).

**Geändert:**
- `modules/expectation_alpha/`: `cross_asset_confirmation`, `timing`, `thesis`, `ledger`, `__init__`, `future_state`, `regime_change`, `expectation_gap`, `schemas`, `evaluation`;
- `feedback.py`, `reports/weekly.py`;
- `config/expectation_alpha.yaml`, `config/cost_policy.yaml`;
- `tests/test_expectation_alpha.py`, `tests/test_expectation_alpha_integration.py`.

## D) Tests / E2E
- Neue Tests: 97, alle grün. Volle Suite: siehe Commit bzw. `docs/EXPECTATION_ALPHA_V1_REPORT.md` 6a.
- Lint (`pyflakes`): alle neuen und geänderten Dateien sauber. Die eine verbliebene Warnung (`reports/weekly.py:329`) besteht schon auf `main`.
- Kein Typecheck im Repo konfiguriert.
- E2E-Shadow-Lauf (echtes Archiv, echter Commodity-Build, echte SEC-Stores, synthetische Preise, LLM gestubbt):
  - Anreicherung 5,5 s, 4 ABSTAIN;
  - effective vs. raw: 18,58 / 20 (0,93); `family_diversity` 7–8 (rates 2 → 1,5; commodities 2 → 1,5);
  - Claims: 12 gesamt, davon 6 VERIFIED, 5 UNVERIFIED, 1 CONTRADICTED; der erfundene Wert 99,9 wurde verworfen;
  - `insufficient_history_count` 3; `lead_lag_status` NEED_MORE_DATA;
  - `runs.jsonl` enthält alle Schlüssel.
- Champion-Vergleich:
  - Der Integrationstest `pipeline.main()` mit EA `off` vs. `shadow` ergibt identische Kandidaten, Stats und Rejects.
  - Duplikat-Quarantäne auf dem echten `history.json`: 120 → 113 closed_trades (AAPL/META/LLY/MRK, Close 2026-05-26). Alle 7 Duplikate sind nicht-verlässliche Outcomes, deshalb bleiben die Pearson-Gewichte unverändert (0,35/0,45/0,20). Ein zweiter Lauf verschiebt 0.

## E) PIT- und Lernstatus
- Claims: nur Evidenz mit `available_at < decision_time`, keine revidierten Werte. Fehlendes Filing ≠ Widerspruch.
- Lead-Lag: Feature aus dem eingefrorenen Snapshot, Outcome erst nach Ablauf des Horizonts.
- Confirmation: Familien und Dämpfung sind in Kill und Trigger eingefroren und werden beim Replay nicht neu konfiguriert.
- Consume-once gilt für alle inkrementellen Akkumulatoren. Voll neu berechnete Größen zählen jede (deduplizierte) Beobachtung genau einmal.

## F) Offene Risiken
- Die bestehenden `feature_stats`-Bins enthalten historische Doppelzählungen. Sie lassen sich nicht sauber zurückrechnen (QuasiML ist SHADOW, kein Champion-Einfluss).
- Die Quarantäne der 7 Altduplikate greift erst im nächsten CI-Feedback-Lauf. `outputs/history.json` wurde lokal nicht verändert.
- XBRL-Tag-Abdeckung: Bei NVDA und JPM sind die Umsatz-Tags veraltet (2020/2014). Claims bleiben dort UNVERIFIED; das ist eine Datenlücke, kein Widerspruch.
- Die Claim-Extraktion braucht `ANTHROPIC_API_KEY` und Budget, sonst SKIPPED. Der Inflations- und Zins-Gap braucht weiterhin `FRED_API_KEY`.
- Die Familienzuordnung ist ein ökonomischer Prior (nicht validiert). d = 0,5 ist gesetzt, nicht optimiert.

## G) Neue Research-Features OHNE Alpha-Evidenz
Null Forward-Beobachtungen für:
- `effective_confirmation_ratio` und `family_count`;
- `verified_claim_fraction` und die Claim-Status;
- alle Lead-Lag-Ergebnisse;
- die Warm-up-Status als Filter.

Sie werden nur protokolliert. Ein Einfluss erfordert zuerst einen präregistrierten Vertrag und danach Forward-Evidenz über den PromotionController.
