# Options-Feature- und Stichproben-Audit vom 21.11.2026

**Scope:** Nur Auswertung, Dokumentation und vorgeschlagene menschliche Task-Entscheidungen. Kein Eingriff in Modell, Feature-Set, historische Ledger, Handelslogik, Risikolimits, API, Secrets oder Orders.

## Datenbasis und Befunde

Quelle: `outputs/research/option_learner/feature_diagnostics.json` (Stand 2026-11-21), `status_5d.json`, `status_20d.json` und unveränderter `option_learner.features()`-Lesepfad.

| Kohorte | Verifizierte Options-Labels | Eindeutige Kandidaten | impact | dealer_oi_ratio |
| --- | ---: | ---: | ---: | ---: |
| 5d Directional | 507 | 410 | 507/507 (100 %) | 478/507 (94,28 %) |
| 5d Research Census | 980 | 411 | 0/980 (0 %) | 2/980 (0,20 %) |
| 5d gesamt | 1.487 | 821 | 507/1.487 (34,10 %) | 480/1.487 (32,28 %) |
| 20d Directional | 445 | 386 | 445/445 (100 %) | 416/445 (93,48 %) |
| 20d Research Census | 47 | 38 | 0/47 (0 %) | 0/47 (0 %) |
| 20d gesamt | 492 | 424 | 445/492 (90,45 %) | 416/492 (84,55 %) |

*Fünf und zwanzig Tage sind unterschiedliche Reifekohorten; Options-Labels können mehrere Kontraktformen desselben Kandidaten enthalten. Die 1.487 Labels sind keine 1.487 unabhängigen Experimente.*

Quellprüfung (5d): `reader_consistency = CONSISTENT`, `reader_mismatches = {}`, `source_rows_unmatched = 0`. Für `impact`: 507 numerische Quellwerte, 507 extrahiert, 980 fehlende Schlüssel, 0 Mismatches. Für `dealer_oi_ratio`: 480 numerische Quellwerte, 480 extrahiert, 978 fehlende Schlüssel und 29 Nullwerte, 0 Mismatches. Keine numerischen Quellwerte stehen nur auf Top-Level.

## Kausaldiagnostik

- `option_learner.features()` liest numerische Werte bereits aus `row["features"]`. Die gemeldete niedrige 5d-Gesamtquote ist kein belegter falscher Reader-Pfad.
- Die `research_census`-Stichprobe wird ohne verpflichtende Deep Analysis mit LLM-Richtung erzeugt; bestimmte teure oder nachgelagerte Features sind dort definitionsgemäß kaum verfügbar. `directional`-Kandidaten durchlaufen einen anderen Pipelinepfad.
- `MIN_FEATURE_COVERAGE = 0.5` prüft die **gemischte Trainingspopulation**. Damit kann ein Feature mit 100 % in Directionals in der gemischten 5d-Population wegfallen. Das ist ein Modell- und Forschungsdesignproblem, **kein** Nachweis, dass der Reader defekt ist. Keine Schwelle stillschweigend ändern, keine Nullen ergänzen und historische Daten nicht rekonstruieren.
- In der 5d-Walk-Forward-Stichprobe liegen nur **92 gewählte Fälle an sieben Kalendertagen** vor: **1 Directional**, **91 Research Census**, 92 bullisch. Das reicht nicht für eine Aussage über produktive Directional-Signale oder bearishen Einsatz.
- `candidate_ledger.flush()` priorisiert bereits Directionals und vergibt das Restbudget an Census. Weitere **Census**-Snapshots erhöhen die Zahl produktiver Directionals nicht. Die bestehende Research-Budgeterweiterung ist außerdem ausdrücklich provider-plan-gesteuert. Keine zusätzliche Last ohne gesonderte Genehmigung.

## Entscheidung zu drei vorgeschlagenen Maßnahmen

| Task | Vorgeschlagene Entscheidung | Begründung |
| --- | --- | --- |
| `IMP-8c83b133` | REJECTED (Reader-/Backfill-Fix) | `impact` wird aus vorhandenen Quellwerten verlustfrei gelesen |
| `IMP-8ea10f58` | REJECTED (Reader-/Backfill-Fix) | `dealer_oi_ratio` wird aus vorhandenen Quellwerten verlustfrei gelesen |
| `IMP-79e87220` | REJECTED (zusätzliche Snapshot-Slots als Abhilfe) | Die Stichprobenverzerrung kommt von fehlenden Directional-Signalen, nicht von zu wenigen Census-Snapshots |

**REJECTED bewertet die vorgeschlagene technische Abhilfe, nicht die beobachtete Forschungsgrenze.** Die Feature- und Repräsentativitätswarnungen bleiben weiterhin Gegenstand der Evidenz- und Freigabeprüfung. Die `config/improvement_task_decisions.yaml`-Einträge unterdrücken nur erneute E-Mails zu genau diesen drei Task-IDs, sobald der PR ausdrücklich freigegeben und gemergt ist.

## Sinnvolle separate Forschung (nicht Bestandteil dieses PRs)

Einen präspezifizierten, ausschließlich im Research-Schattenlauf arbeitenden **Cohort-Challenger** gegen die eingefrorene Basis testen: `directional` getrennt von `research_census` berichten und gegebenenfalls getrennt trainieren oder unter bekannten Stichproben-Propensities korrekt gewichten. Zeitliche Point-in-Time-Verfügbarkeit pro Feature und Abhängigkeit mehrfacher Options-Labels pro Kandidat explizit prüfen. Vorab festgelegte train/test-Trennung, ausreichende unabhängige Haltefenster, Netto-Ausführungskosten und echtes prospektives Forward-Journal sind erforderlich. Keine Änderung der Champion-Strategie ohne externe Freigabe und valide Evidenz.

## Verifizierung und Grenzen

Dieser PR enthält lediglich ein YAML-Entscheidungsprotokoll und ein Auditdokument; er verändert keinen ausführbaren Code. Auditwerte sind aus dem bereits veröffentlichten, reproduzierbaren Offline-Diagnoseartefakt vom 21.11.2026 übernommen. Die angehängten Entscheidungen sind Vorschläge zur menschlichen Freigabe; kein Auto-Merge.
