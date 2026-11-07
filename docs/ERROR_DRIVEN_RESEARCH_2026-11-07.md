# Fehlergetriebene Forschung: nächste Stufe des autonomen Lernens (2026-11-07)

Die Stufe arbeitet nur in der Forschung: keine Orders, kein LLM, keine Produktionswirkung, kein automatischer Merge.
Grundlage ist `main` bei `e9e35846` mit echten Daten aus `outputs/` vom 07.11.

## Kreislauf in den bestehenden Modulen

Es gibt keine neue Architektur. Ein Lauf ist `python -m modules.research_director --errors`. Er läuft wöchentlich in
`ml_research.yml` vor dem Director, und sein Ausfall macht den CI-Lauf rot.

| Schritt | Umsetzung |
|---|---|
| 1 Fehleranalyse | `research_director.verified_forecasts`: Jede Ensemble-Prognose im Candidate Ledger wird gegen das `ml_predictions`-Archiv geprüft. Sie muss vor dem Signal erstellt worden und höchstens 8 Tage alt sein. `forecast_errors` berechnet je Stichtag Rang der realisierten Rendite minus Rang der Prognose; als *unerklärter Teil* gilt das Residuum nach Rang-OLS auf die Prognose, zusätzlich eine Fassung nach Größenkontrolle. |
| 2 Hypothesen | `error_patterns` prüft einen festen Suchraum aus `config/error_research_protocol.yaml`: 23 Kandidaten, nämlich 12 Ledger-Merkmale, 5 externe PIT-Merkmale mit festen Lags (Pageviews, FINRA) und 6 Paare über Quellenfamilien hinweg. `error_spec` friert die falsifizierbare Spezifikation ein: H0/H1, Baseline, Metrik, Falsifikationsbedingung, Forward-Kriterien, Hash. |
| 3 Datenlücken | Zu kurze Historie laufender Quellen, zu wenige unabhängige Fenster und Segmente mit signifikant größerem unerklärtem Fehler (`error_segments`, BH) werden auf Katalog-Dimensionen abgebildet. Dazu kommen die Quellen mit gemessener Aufnahmeprüfung. Nur Segmentlücken heben `active_learning`. |
| 4 Priorisierung | `error_priority` = `priority()` (EIG × Wert × Neuheit × Zuverlässigkeit ÷ Kosten) × Gedächtnis-Prior (`memory_prior`: Beta-Posterior aller Ergebnisse der Quellenfamilie, auch aus Paaren). Höchstens 2 neue Hypothesen je Lauf und 24 insgesamt. |
| 5 Test | `research_lab`: unabhängige Wochen- bzw. 4-Wochen-Fenster (`period_blocks`). Darauf folgt der Walk-Forward-Screen nach einem Fenster Embargo (`ledger_walk_forward`, falsifiziert nur). Den Abschluss bildet genau eine Forward-Prüfung auf den ersten 4 fertigen Fenstern nach der Registrierung (`ledger_forward_test`), mit BH über alle Forward-Tests. Gesehene Daten (bis `seen_until`) bestätigen nie. |
| 6 Optionen | Nur für erfolgreiche Aktienhypothesen: `option_check` auf den geprüften Netto-Labels von `option_learner` (Kauf Ask, Verkauf Bid, Kommission). Verglichen wird das obere Drittel des Signals mit „alles handeln“ und „nie handeln“, per Block-Bootstrap (`option_learner._paired`). |
| 7 Gedächtnis | `research_memory.error_entries` (Quelle `error_research`) legt jede Hypothese, jedes durch Größe erklärte Muster (NOT_ROBUST) und jede Datenlücke ab. Der `error_key` verhindert eine erneute Registrierung, die Familie steuert den Prior. Erfolg zählt nur nach prospektiver Bestätigung (FORWARD_VALIDATED). `meta_cognition` zeigt den Track. |

Zusätzlich gilt: `research_lab` bestätigt Director-Hypothesen nicht mehr über den kontaminierten Locked-Holdout. Bisher
galt das nur für den Explorer. Aktuell ist keine Director-Hypothese angenommen, der Bestand ändert sich also nicht.

## Echter End-to-End-Nachweis (as of 2026-11-07)

| Stufe | Ergebnis |
|---|---|
| Gemessener Fehler | 8.976 Ledger-Prognosen. Davon sind **8.590 VERIFIED**: Alle 51.498 ML-Werte entsprechen einer Archivprognose, die vor dem Signal erstellt wurde. 386 haben weniger als 3 Modelle (28.09., vor dem ersten Archivlauf). 5 T: **7.342 fällige Fehler** an 21 Stichtagen in 5 unabhängigen Wochenfenstern. Der Ensemble-Rang-IC je Fenster liegt bei **−0,078** (t −1,09; Fenster: +0,19, −0,22, −0,13, −0,06, −0,17). 20 T: 2.963 Fehler, aber 0 fertige 28-Tage-Fenster, also Datenlücke. |
| Muster | 14 der 23 Kandidaten sind testbar (Entdeckung 29.09.–18.10., 3 Fenster). 1 wiederkehrend, 1 durch Größe erklärt, 11 nicht wiederkehrend, 10 mit zu wenigen Daten (alle externen Merkmale und Paare sowie `dollar_volume`). BH ist über alle 14 Tests nicht signifikant; das wird berichtet. |
| Hypothese | **ERR-d6695229**: `sentiment_score` sagt den Teil der 5-Tage-Querschnittsrendite voraus, den das Ensemble nicht erklärt. Residual-IC +0,036 (t 3,04, 3 Fenster), an 82 % der Stichtage gleiches Vorzeichen, nach Größenkontrolle +0,033 (t 2,33). Hash `0b87a198…`, Forward ab 08.11. |
| Kritische Prüfung | `news_count*sentiment_score`: +0,022 fällt nach Größenkontrolle auf +0,009 (t 0,90). Gilt als **durch Größe erklärt** und geht als NOT_ROBUST ins Gedächtnis. |
| Wissenschaftlicher Test | Der Walk-Forward-Screen hat nach dem Embargo nur 1 Fenster (26.10.–01.11.). Residual-IC dort **−0,021**, also umgekehrtes Vorzeichen. Mit einem Fenster gibt es nach Protokoll kein Urteil (INCONCLUSIVE_WF); es ist aber ein Warnsignal. Die einzige Forward-Prüfung folgt nach 4 Fenstern, frühestens im Lauf am 12.12. |
| Optionen | Nicht erreicht, die Aktienhypothese ist nicht erfolgreich. Ein Funktionsnachweis auf gesehenen Daten (explorativ, nicht gespeichert): oberes Drittel nach Sentiment +3,2 pp gegenüber „alles handeln“, **−9,8 pp netto je 5 Tage** gegenüber „nie handeln“ (KI [−17,2; −0,6]), 3 Fenster, Status INSUFFICIENT. |
| Dokumentiert | `outputs/research/error_research.json`, `outputs/research/error_hypotheses.jsonl` (append-only). Gedächtnis: 1 PENDING_FORWARD, 1 NOT_ROBUST, 13 DATA_GAP. |
| Neue Entscheidung | **Lauf 2** mit demselben Datenstand: keine Neu-Registrierung (`skip: ERR-d6695229`). Der Familien-Prior für „sentiment“ sinkt von 1,0 auf **0,833**, belegt durch `ERRPAT-news_count*sentiment_score-5d:NOT_ROBUST`. **Datenbeschaffung**: Der unerklärte Fehler ist bei Technology (+0,067, t 12,0), Communication Services (+0,038, t 3,7) und hoher Short-Quote (+0,018, t 6,7) signifikant größer (BH über 13 Segmente). Damit rückt `cboe_put_call` (Positionierung) von Rang 7 auf **5**. `earnings_calendar_history` und `sec_financial_statements` sind als Quellen für Fundamentaldaten benannt; der Zugang ist ungeprüft, die Freigabe liegt beim Menschen. |

Reproduzieren: `python -m modules.research_director --errors`. Der Lauf ist deterministisch und liest nur den Ledger,
das Prognose-Archiv, das externe Archiv, den Katalog und das Gedächtnis.

## Kritische Selbstprüfung

- **Artefakte vermieden:** Die erste Probe nahm den Rohfehler (Rang y − Rang p) und Tages-t-Werte. Dabei zeigten 8
  von 12 Ledger-Merkmalen |t| ≥ 2 (z. B. `dollar_volume` t −35, `news_count` t −12). Der Rohfehler enthält aber die
  Prognose selbst, und überlappende 5-Tage-Fenster sind nicht unabhängig. Nach Residualisierung und Fenster-Blöcken
  bleibt 1 Muster übrig.
- **Zwei Protokolländerungen, beide vor der ersten Registrierung und im Protokoll vermerkt:**
  1. BH als Entdeckungsbedingung hätte bei 3 Fenstern |t| ≥ 11,8 verlangt (Trennschärfe 0). Ersetzt durch den Screen
     p < 0,10. BH wird weiter berichtet.
  2. Die Größenkontrolle ließ einen reinen Größen-Stellvertreter durch (synthetischer Test). Neu ist die Bedingung
     „mindestens 50 % der Stärke“. Der Wert wurde vorab gewählt, nicht an echten Daten.

  Bestätigungsschwellen wurden nicht verändert.
- **Synthetischer Kreislauftest** (`tests/test_error_driven_research.py`): Ein dauerhaft wirkendes Merkmal wird
  registriert und forward bestätigt; die Options-Prüfung ergibt dann OPTION_EDGE, nur auf Forward-Signalen. Ein Merkmal
  mit Regimewechsel wird forward verworfen und senkt den Familien-Prior. Ein Größen-Stellvertreter gilt als erklärt,
  Rauschen als nicht wiederkehrend. Eine nachträglich geänderte Registrierung wird INVALID_MODIFIED, das Budget wird
  eingehalten.

## Messbar verbessert

| | Vorher (`main` e9e35846) | Nachher |
|---|---|---|
| Fehlerquelle der Forschung | nur historische Ensemble-Fehler (Blind Spots, Holdout kontaminiert) | 7.342 forward geprüfte Prognosefehler |
| Herkunft der Prognose geprüft | nein | 8.590 von 8.976, 0 Abweichungen |
| Unabhängige Perioden | Tages-t | nicht überlappende Haltefenster mit Embargo |
| Scheinmuster (Ledger-Merkmale, \|t\| ≥ 2 roh) | 8 von 12 | 1 registriert, 1 als Größe erkannt |
| Bestätigung | Locked-Holdout (kontaminiert) für Director-Ideen möglich | nur prospektiv, einmalige Prüfung, BH über alle Forward-Tests |
| Rückkopplung | Ähnlichkeitssperre | Schlüsselsperre und Familien-Prior aus Ergebnissen (sentiment 1,0 → 0,833) |
| Datenlücken aus Fehlern | keine | 13, davon 3 Segmentlücken, die die Quellen-Priorität ändern |

## Wo Daten oder Evidenz fehlen

- Es gibt nur 5 unabhängige 5-Tage-Fenster und kein fertiges 20-Tage-Fenster. Der Walk-Forward-Screen hat ein einziges
  Fenster, die erste Forward-Prüfung kommt frühestens am 12.12.
- Pageviews und FINRA-Short-Volume liegen PIT erst ab 15.10. vor, also mit 0–1 Entdeckungsfenstern. Alle 9 externen
  bzw. Paar-Kandidaten sind noch nicht prüfbar.
- Der unerklärte Fehler bei Technology und Communication Services deutet auf fehlende Fundamental- bzw. Earnings-Daten
  hin. Die freien Katalogquellen sind im Zugang nicht verifiziert.
- Das Ensemble hat forward keinen belegten Skill (5 T: IC −0,078; 20 T: nicht messbar).
- Optionen: Explorativ verliert selbst das beste Aktiensignal netto gegenüber „nie handeln“.
- **Kein Alpha-Nachweis.** Kein Ergebnis dieser Stufe ist nach Kosten belegt.

## Offen

- Fabrik-Hypothesen können weiterhin über den kontaminierten Locked-Holdout „accepted“ werden, bevor die
  Robustheitsbatterie greift. Eine Korrektur braucht eine Änderung am Fabrik-Ablauf und gehört in einen eigenen PR.
- Eine historische Replikation über das Panel ist nicht umgesetzt: Nur 2 der 12 Ledger-Merkmale haben ein
  Panel-Gegenstück.
