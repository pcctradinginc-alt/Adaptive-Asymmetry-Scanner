# Source Scoreboard

Datei: `outputs/research/source_scoreboard.json` (von `alt_data.evaluate`), Weekly Report Abschnitt 17.

| Spalte | Herkunft |
|---|---|
| coverage | Anteil Panel-Zeilen der Dev-Jahre mit verfügbaren Quellen-Features |
| freshness | 1 − Alter der letzten Beobachtung / 120 T (aus `health.json`) |
| data_quality | 1 − Fehlerquote der letzten Ingestion |
| active_features | Selektion nach Protokoll (Redundanz/Abdeckung/Auswahl-IC) |
| oos_value | Komponente `incremental_value` = clip(0,5 + mittl. Δ Monatsrendite / 1 %) |
| forward_value | aus Forward-Kohorten, sobald ≥ 26 vorliegen (bis dahin leer) |
| source_value_score | gewichtete Summe (Gewichte im Protokoll) |
| status | KEEP → HISTORICALLY_VALIDATED, MODIFY → SHADOW, REJECT → REJECTED |

## Source Value Score

Deterministisch aus Messwerten, keine LLM-Schätzung:

| Komponente | Gewicht |
|---|---|
| incremental_value | 0,40 |
| coverage | 0,15 |
| freshness | 0,10 |
| stability (Anteil Jahre mit Δ > 0) | 0,10 |
| mapping_quality (Entity-Bericht) | 0,10 |
| api_reliability (1 − Fehlerquote) | 0,10 |
| revision_risk (1 = keine Revisionen; SEC-Filings sind unveränderlich) | 0,05 |

Fehlende Health-/Entity-Daten zählen 0 für die jeweilige Komponente; der Score ist dann konservativ niedrig.
