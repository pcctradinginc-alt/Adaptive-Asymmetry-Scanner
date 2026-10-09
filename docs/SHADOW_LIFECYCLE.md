# Shadow-Lifecycle – status-/horizontbasierte Retention (2026-10-09)

## Root Cause (vorher)

**Erzeugung:** `pipeline.py` legt Schatten-Trades an drei Stellen in `history["shadow_trades"]` an:
- Final-MC-Survivors (`reject_reason=final_mc_survivor`);
- ROI-Gate-Rejects mit `fail_gates` je Tier;
- Score-/Korrelations-Rejects `score_<n>` bzw. `correlation`.

**Bewertung:** `feedback.evaluate_shadow_trades`, zweimal pro Werktag.
- Bewertet wurde nur, wenn `today − entry ≥ learning.close_after_days` (45 Tage) war.
- Ein Horizont, Bewertung per Live-Quote am Bewertungstag – auch wenn das Wochen nach Fälligkeit lag.

**300er-Grenze:** in `feedback.evaluate_shadow_trades`:
`if len(shadows) > 300: archive_shadow_trades(shadows[:-300]); history["shadow_trades"] = shadows[-300:]`
- Verdrängt wurde FIFO nach Position, ohne Rücksicht auf den Status.
- Bis 2026-10-03 gab es kein Archiv. Seitdem gibt es `outputs/shadow_trades_archive.jsonl`, aber **kein Prozess las es**.

**Warum Pending-Trades verschwanden:** Der Zufluss stieg von etwa 10 auf 28–47 Schatten-Trades pro Handelstag. Die 300 Einträge reichen damit nur noch etwa 9,7 Handelstage, die Bewertung braucht aber 45 Kalendertage.
- Am 08.10. wurden 41 von 47 verdrängten Records ohne Outcome archiviert. Sie waren 36–42 Tage alt, also knapp vor ihrer Fälligkeit.
- Alle 107 Archiv-Records dieser Woche waren im Mittel 66 Tage alt.
- Aus der Git-Historie (323 Stände von `history.json`) fehlte nur 1 Record komplett (`score_44`, Juni, bereits bewertet).

**Downstream-Consumer der Outcomes:**
- `factor_monitor`, `trade_memory`
- `engine_monitor` (parallel_tests)
- `learning_health` (shadow_lifecycle)
- `reports/weekly.py` (`roi_subgate_evidence`)
- Monatsbericht (Shadow-Statistik)

## Neu

| Teil | Inhalt |
|---|---|
| `modules/shadow_ledger.py` | append-only `outputs/intelligence/shadow_ledger/records/<YYYY-MM>.jsonl` (unveränderlicher Original-Snapshot) und `events/<YYYY-MM>.jsonl` (EVALUATED / RETRY / UNAVAILABLE / LEGACY_OUTCOME / ARCHIVED) |
| Lifecycle (abgeleitet) | PENDING → MATURED / MATURED_RETRY_REQUIRED → PARTIALLY_EVALUATED → EVALUATED / OUTCOME_UNAVAILABLE → ARCHIVED |
| Integritäts-Invariante | Archivieren oder Verdrängen eines ungeklärten Records → `ShadowIntegrityError`; der Job wird rot, nichts wird verworfen |
| Horizonte | 20 / 45 / 60 Tage, je Horizont: Fälligkeit, Status, Underlying-Return (richtungsbereinigt, Schlusskurse ≤ Fälligkeit), MFE, MAE, Options-/Spread-Return (nur Live-Quote ≤ 4 Tage nach Fälligkeit), Outcome-Qualität, evaluated_at |
| Retry | RETRY mit `retry_count` / `last_error` / `last_attempt` / `next_attempt`. Nach 7 Versuchen und frühestens 14 Tagen nach Fälligkeit: OUTCOME_UNAVAILABLE, nie geschätzt |
| Ansicht `history.json` | alle Records mit ungeklärtem 45-T-Horizont plus die letzten 300. Der Export ins Archiv erfolgt erst nach Auflösung |
| Gate-Attribution | `gate_group` aus unveränderten `fail_gates`: ROI_ONLY / LIQUIDITY_ONLY / EXECUTION_ONLY / MULTI_GATE / ROI_GATE_UNSPECIFIED / FINAL_MC_SURVIVOR / SCORE_REJECT |
| Recovery | `scripts/recover_shadow_ledger.py [--git]`: Ansicht + Archiv + Git-Historie, unverändert. Vorhandene Legacy-Outcomes werden als LEGACY_OUTCOME übernommen |
| Reporting | Montagsbericht §1 „SHADOW LIFECYCLE HEALTH“ und „GATE LEARNING“ (n < 30 → NEED_MORE_DATA). LEARNING_HEALTH ist `DEGRADED`, wenn `LOST_BEFORE_EVALUATION > 0` |

**PIT-Regeln:**
- Der Snapshot wird nie verändert.
- Underlying-Outcomes nutzen nur Schlusskurse bis zum Fälligkeitstag (split-bereinigt, keine Dividendenrevision).
- Spätere Bewertungen nutzen keine heutigen Optionsquotes für vergangene Horizonte.

Gate-Schwellen, Production Policy, Alpha-Logik, Forward-Starts und Candidate-Snapshots sind unverändert. Gate-Learning erhält nur bessere Evidenz, es gibt keine automatische Policy-Anpassung.
