# Audit 2026-11-01: Performance-Reporting und Lernqualität nach Evidenzklassen

Historische Rohdaten (`outputs/history.json`, Journale) sind unverändert. Es gibt keine Umklassifizierung, keine
Änderung an Gates, Schwellen, Sizing, Modellen, Experimenten oder Forward-Journalen.

## Phase 1 – Antworten

| # | Frage | Befund (reproduziert auf `outputs/history.json`) |
|---|---|---|
| 1 | Verifizierte Optionspreise? | **7** `RELIABLE`-Trades: 5 Spreads `spread_quote` und 2 Long Calls `option_quote`. Alle wurden im Sep./Okt. geschlossen, jeder Oktober-Trade (5) ist RELIABLE. |
| 2 | Rekonstruiert / geschätzt? | **40** `RECONSTRUCTED` (`outcome_method_reconstructed = delta_approx`, alle im Juni geschlossen). `APPROXIMATED`: 0. |
| 3 | Ohne dokumentierte Preisquelle? | **70** `UNKNOWN`: Altbestand ohne `outcome_method` (Owner-Entscheidung 2026-10-04). |
| 4 | 116 statt 117? | **Defekt** in `monthly_report.dedup_closed_trades`: Der Schlüssel war nur (Ticker, Entry-Tag). Dadurch wurde einer der beiden UNH-Trades vom 2026-04-11 verworfen. Der Report zählte 116; `paper_performance_analysis` und `trade_memory` zählen 117. |
| 5 | Verschiedene Kontrakte zusammengefasst? | **Ja**, genau dieser Fall: UNH Put 310 (−83,2 %) und Call 330 (+258,8 %). Behoben: Die Identität umfasst jetzt Strategie, Strike(s), Verfall und Symbol. |
| 6 | Doppelt gezählt? | Nein. Exakte Kopien (LLY ×4, META, MRK, AAPL) liegen in `closed_trades_quarantine` (learning_ledger) und werden nirgends gezählt. Alle Lernpfade filtern mit `is_reliable_outcome`. Strukturelles Risiko ohne heutigen Fall: `promotion_controller.resolve_outcomes` und der Wochenreport ordnen über (Ticker, Tag) zu. Für die betroffenen Fälle gilt `outcome_method` „unknown“, also keine Evidenz. Nicht geändert, weil CODEOWNERS-geschützt und ohne Auswirkung. |
| 7 | −95 % vs. −99 %? | Der Monatsreport nutzte beide Schwellen: rollierend ≤ −95 %, Monat/Gesamt ≤ −99 %. Dadurch zählte ein Outcome von −95,3 % einmal mit und einmal nicht (27 vs. 26). Jetzt gilt eine Definition, `outcomes.TOTAL_LOSS_THRESHOLD = −0,95`, wie im Challenger-Protokoll und in `exit_sim`. Das Ops-Monitoring nutzt bewusst ≤ −99 % **netto** (nach Gebühren; durch einen eigenen Test festgeschrieben) und wird hier nicht verändert. |
| 8 | Fürs Lernen zugelassen? | Nur `RELIABLE` (`modules/outcomes.is_reliable_outcome`). Das gilt für Feedback-Gewichte (ab 5), RL, Evidence-Ledger, Kalibrierung, Abstention und Failure-Analyse. Für das Options-Lernmodell zählen nur Ledger-Labels mit Provenienz `VERIFIED`. Monatsreport und Vorschläge fließen in kein Lernen. Einen Lernpfad mit nicht verifizierten Outcomes gibt es nicht. |

**Monatsreport Oktober reproduziert (alter Code):** 116 Trades, Win-Rate 31 %, Ø +2,8 %, Median −48,1 %, ohne Top-3 −9,1 %, letzte 30 −20,7 %, Oktober 5 Trades −19,3 %.

**Korrigiert (117 Trades, alle Klassen):** Win-Rate 32 %, Ø +5,0 %, Median −46,8 %, ohne Top-3 −6,7 %.

**Verifiziert (RELIABLE):** n = 7, Win-Rate 14 %, Ø −32,8 %, Median −46,8 %, PF 0,18. Damit `INSUFFICIENT_EVIDENCE`.

## Statistik

- 104 der 117 Trades wurden im ersten 45-Tage-Fenster eröffnet (11.04.–21.05.). Sie überlappen und sind voneinander abhängig.
- Der alte iid-Bootstrap lieferte für 60 Trades aus einem einzigen Fenster ein 90-%-KI von ±6 %. Das war scheinbare Präzision.
- Jetzt: KI per Bootstrap über ganze, nicht überlappende Haltefenster (`evidence_core.group_bootstrap_ci`). Bei weniger als 5 Fenstern gibt es kein KI.
- „Signifikant besser/schlechter“ erscheint nur, wenn beide Hälften gültige Fenster-KIs haben.
- SPY dient nur noch als Marktkontext. Es wird keine „Edge“-Differenz mehr berechnet, weil Prämienkapital, Hebel, Haltedauer und Ausführung nicht vergleichbar sind.

## Phase 4 – Verlustursachen (bestehender Failure Analyzer)

| Ebene | Befund |
|---|---|
| Beobachtet | 6 RELIABLE-Verluste. Primäre Ursachen: `signal_wrong` 2, `structure_decay` 2, `timing_too_early` 2. Merkmal `entry_cost_high` bei 5 von 6. |
| Plausibel | Hohe Einstiegskosten (Spread) und Zeitwertverlust der Spreads; 5 von 5 RELIABLE-Spreads endeten mit Verlust. |
| Statistisch gestützt | Keine Aussage möglich (n = 6, 2 unabhängige Fenster). |
| Ungeprüft / Datenlücke | 71 nicht verifizierte Verluste ohne Ursachenanalyse. IV-Veränderung je Trade nicht erfasst. RELIABLE-Trades ohne Sektor (`sector_etf` fehlt). Regime nur grob (alle 7 im Zustand `low_vol`). |

Aus 7 RELIABLE-Outcomes wird keine Optimierung abgeleitet.

## Phase 5 – Retrospektive Vorschläge

| Vorschlag | Evidenzbasis | Bewertung |
|---|---|---|
| Impact-Floor 4 → 6 (+10 pp) | 57 Trades, davon 0 RELIABLE (29 UNKNOWN, 28 RECONSTRUCTED). Neue Trades tragen kein `deep_analysis.impact`. | Best-of-Auswahl aus etwa 20 Schwellenvergleichen (Mehrfachtest), fast nur eine abhängige Kohorte (April/Mai). Im Research Memory gibt es die gegenläufige Abstention-Regel `impact > 6.0` als REJECTED. **Keine Challenger-Vorlage**, keine Registrierung, kein Testslot. Prospektiv prüfbar nur nach menschlicher Freigabe über `features.impact` im Kandidaten-Ledger. |
| Surprise-Floor 3 → 5 (+6 pp) | 63 Trades, 0 RELIABLE | Wie beim Impact-Floor; die Abstention-Regel `surprise > 6.0` ist REJECTED. |
| Long Calls vs. Spreads | Explorativ: 73 Long vs. 44 Spreads. RELIABLE: 2 Long Calls (Ø +2,1 %) vs. 5 Spreads (Ø −46,7 %) | `INSUFFICIENT_EVIDENCE`. Wird bereits prospektiv im Options-Lernmodell geprüft (Ausdrucksformen single/spread auf VERIFIED-Labels). Kein neues Experiment nötig. |
| Sektor XLK (n = 17, Ø +145,9 %) | Nur 58 Trades haben `sector_etf`, 0 RELIABLE | Getrieben von der Tech-Rally im April/Mai (eine abhängige Kohorte), nicht verifiziert. Kein Vorschlag. |

Bestehende Challenger (`challengers.yaml`: `final_mc_dte_shadow`, `trade_score_61`, `ppo_robust_shadow`) und Forward-Experimente bleiben unverändert.

## Änderungen

- **`modules/outcomes.py`:** `TOTAL_LOSS_THRESHOLD`, `is_total_loss` (eine Definition).
- **`monthly_report.py`:**
  - Deduplizierung je Kontrakt (`trade_identity`) und einheitliche Totalverlust-Definition.
  - Neue Report-Struktur: A Verifizierte Performance (RELIABLE, nach Struktur/Sektor/Regime, alle Kennzahlen je Klasse, `INSUFFICIENT_EVIDENCE`); B Historische Performance mit dem Banner „EXPLORATIV“; C Forschungs- und Lernfortschritt aus vorhandenen Dateien (fehlend wird als „fehlt“ angezeigt, nie als 0).
  - Neu sind außerdem die Verlustursachen nach Evidenzstärke, das Haltefenster-KI, SPY nur als Kontext und ein Betreff nach Evidenzqualität.
  - Gate-Vorschläge zeigen ihre Evidenzbasis. Eine Challenger-Vorlage gibt es nur bei mindestens 20 RELIABLE-Outcomes und mindestens 5 unabhängigen Fenstern.
- **`backtest_thresholds.trade_rows`:** zusätzliche Felder `outcome_class` und `entry_date`; die Berechnung ist unverändert.
- **Tests:** `tests/test_performance_evidence.py` (16 Tests). 14 davon schlagen auf dem alten `main` nachweislich fehl.
