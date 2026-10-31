# Autonomous Research Improvement (Options-Lernmodell)

Stand 2026-10-31 (inkl. Review: Netto-Alpha, Abdeckung, sequenzielles Alpha, Vorschlagsstatus). Erweitert die vorhandenen Bausteine `research_director`, `meta_cognition`, `option_learner` und `research_memory` sowie den Workflow `ml_research.yml`. Eine zweite Architektur gibt es nicht. Tests: `tests/test_research_director.py`, `tests/test_option_learner.py`.

## Gap-Analyse (vorher)

| Fähigkeit | Vorhanden | Was fehlte wirklich |
|---|---|---|
| Lücken erkennen | `meta_cognition` (Machine Intelligence State), `research_director` (Befunde, Active Learning über `config/data_catalog.yaml`), `learning_health` | Alles bezog sich auf das Aktien-Panel. Messbare Lücken des Options-Lernmodells wurden nirgends erkannt, z. B. das Merkmal `news_novelty` mit 0 % Abdeckung, 33 Merkmale auf 359 Labels, nicht bewertbare gewählte Formen, einseitige Evidenz und nie getestete Alternativen. |
| Vorschläge mit Nutzen, Kosten, Falsifikation | `hypothesis_factory` (H0/H1, Failure Condition, DATA_GAP) für Aktien-Signale; Director-Kandidaten mit Priorität | Keine Vorschläge für Merkmale, Modellvarianten, Auswahlregel oder Datenquellen des Options-Lernmodells, und keine vorab festgelegte Falsifikation auf Options-Nettoergebnissen. |
| Isolierte Experimente | `research_lab`, Factory-Batterie, `autonomous_alpha` explorer (alle historisch, Aktien) | Das Options-Lernmodell hatte genau eine Spezifikation. Es gab keine Varianten auf denselben Kandidaten. |
| Fairer Vergleich mit eingefrorener Referenz | `option_learner.compare_policies` (lernend gegen eingefrorene Referenz derselben Version) | Kein Vergleich Variante gegen Basis und keine Mehrfachtest-Korrektur über Varianten. |
| Dokumentation, Änderungsvorschlag | `research_memory` (append-only Sync), `autonomous_alpha` registry, menschlicher PR für Promotion | Options-Ergebnisse kamen nicht ins Research Memory. Es gab keinen begründeten Änderungsvorschlag für das Options-Lernmodell. |

Bewusst nicht gebaut: kein neuer Orchestrator, kein LLM, kein Code-Generator, kein zweiter Experiment-Speicher und keine automatische Datenbeschaffung.

## Kreislauf

1. **Lücken erkennen** – `research_director.option_gaps()` liest `status_<h>d.json`. Jede Lücke hat Befund, Metrik, Wert, Schwelle und Schweregrad (0–1). Ohne Messung gibt es keine Lücke.
   - **Daten:** Merkmal unter 50 % befüllt (erst ab 50 Labels); gewählte Formen zu unter 80 % bewertbar.
   - **Repräsentativität:** Warnungen aus dem Walk-Forward.
   - **Modell:** weniger als 20 Labels je Merkmal; NOT_VALIDATED; Auswahl ohne Mehrwert; Wahl schlechter als die Standardform; noch keine Alternative getestet.
   - **Evidenz:** NEED_MORE_DATA mit den fehlenden Mengen; fehlender Status.
   - Die Lücken erscheinen auch in `meta_cognition` und im Leistungszeugnis.
2. **Vorschläge** – `research_director.option_proposals()` erzeugt je Lücke Vorschläge mit Änderung, Hypothese und Mechanismus.
   - Dazu kommen erwarteter Nutzen (Priorität aus dem Schweregrad, keine erfundene Renditeprognose), Kosten (API, LLM, bezahlte Daten), Datenverfügbarkeit, vorab festgelegte Falsifikation und Freigabebedarf.
   - Datenquellen kommen aus `config/data_catalog.yaml` mit Aufnahmeprüfung (`onboarding_checklist`). Sie werden nur vorgeschlagen, wenn dem Modell nachweislich Güte fehlt. Bezahlte Quellen sind gekennzeichnet.
3. **Isolierte Experimente** – Automatisch laufen nur Varianten aus `config/option_learner_variants.yaml`.
   - Allowlist: genau eine Änderung, freigegebener Wert (Regularisierung, Mindestprognose, Merkmalsgruppe weglassen), keine Kosten.
   - `option_learner.register_variants()` registriert append-only in `variants.jsonl`: höchstens 3 je Horizont, je Methode eine, nie eine bereits falsifizierte Änderung erneut. Start ist der Folgetag, es zählen also nur künftige Signale.
   - Im täglichen Scanner laufen die Varianten **nach** der Basis auf denselben Kandidaten und Labels. Sie schreiben nur in `variants/<id>.jsonl` und ins Modellinstanz-Register.
   - Basis-Journal, Freigabe und Mail bleiben unberührt; ein Fehler einer Variante wird nur protokolliert.
4. **Fairer, sequenziell korrekter Vergleich** – `option_learner.evaluate_variants()`:
   - **Basis des Vergleichs:** identische, fällige Kandidaten mit geprüften Netto-Ergebnissen; gepaart je Tag, gleiches Budget je Kandidat. Primär gegen die Basis-Lernversion, zusätzlich gegen deren eingefrorene Referenz und gegen Nicht-Handeln.
   - **Prüfzeitpunkte:** Urteile gibt es nur an den bei der Registrierung festgeschriebenen Prüfzeitpunkten `look_days` (1/3, 2/3 und 1 von `max_days`).
     - Prüfung k nutzt nur Kandidaten bis zu ihrem Stichtag und erst, wenn diese alle fällig sind. Die wöchentlichen Auswertungen zwischen zwei Prüfungen werten also dieselben Daten aus und erzeugen keine zusätzlichen Tests.
     - Eine verpasste frühere Prüfung wird zuerst nachgeholt. Schließt sie die Variante, entfällt die spätere.
   - **Dauerhaftes Fehlerbudget (`test_budget`):** `total_alpha` 0,05 wird auf `max_tests` 20 gleich große Bonferroni-Slots verteilt.
     - Jede Registrierung erhält bei der Anmeldung genau einen Slot (`alpha_slot`, `slot_alpha` = 0,0025) und verteilt ihn nach O'Brien-Fleming auf ihre drei Prüfungen (`alpha_spending`). Familienweise Fehlerrate über alle jemals getesteten Varianten und Prüfungen ≤ Summe der Slots ≤ 0,05.
     - Die Budgetdefinition gilt ab der ersten Registrierung aus dem append-only Register. Verbraucht ist jede jemals registrierte Variante, auch geschlossene, erneut registrierte und solche alter Basisversionen. Neustart, Modellwechsel oder Konfigurationsänderung setzen nichts zurück.
     - Spätere Registrierungen ändern frühere Ergebnisse nicht. Bei erschöpftem Budget wird nichts mehr registriert; eine Variante ohne gültigen Slot bleibt RESEARCH_ONLY (nie eine Erfolgsfreigabe).
     - Das KI kommt aus dem Bootstrap-Standardfehler über unabhängige Haltefenster.
   - **Abdeckung (gegen Selektionsverzerrung):** Der Nenner sind ALLE Ledger-Kandidaten im vorregistrierten Prüfzeitraum [Start, Stichtag], je Aktie und Tag einer, unabhängig von Entscheidungen oder bekannten Ergebnissen.
     - Als Ausfälle zählen fehlende (`NO_VARIANT_DECISION`) oder ungültige Variantenentscheidungen (z. B. `LATE_PREDICTION`), fehlende Basis-Entscheidungen (`NO_BASE_DECISION`) und fehlende geprüfte Entry-/Exit-Ergebnisse (z. B. `EXIT_NOT_ARCHIVED`, `NO_LABEL`).
     - Unter 80 % gemeinsamer Abdeckung lautet das Urteil INSUFFICIENT_COVERAGE; eine positive Freigabe und einen Übernahmevorschlag gibt es dann nicht.
   - **Eingefrorene Prüfberichte (`look_reports/<variante>_look<k>_v<n>.json`, create-only):**
     - Inhalt: Experiment- und Basisversion, Modellinstanzen, Prüfzeitpunkt und Datenstichtag, Entscheidungs- und Label-IDs, Fallzahl, Abdeckung und Ausschlüsse, Testbudget (Slot, Teil-Alpha), Ergebnis, Freigabestatus sowie Hashes der Eingangsdaten und des Berichts.
     - Gleiche Eingangsdaten bestätigen den bestehenden Bericht. Nachgelieferte oder korrigierte Daten erzeugen eine neue Fassung `_v<n+1>` mit `supersedes` und `correction_reason`; frühere Fassungen bleiben unverändert. Maßgeblich ist die neueste.
   - **Zwei getrennte Urteile:**
     - relativ: BETTER_THAN_BASE oder WORSE_THAN_BASE;
     - wirtschaftlich gegenüber Nicht-Handeln: POSITIVE_EXECUTABLE_ALPHA, NEGATIVE_NET oder NOT_PROVEN.
   - **Status:**
     - Einen Übernahmevorschlag (PROPOSE_ADOPTION, an der letzten Prüfung CONFIRMED_ADOPTION) gibt es nur bei beidem.
     - „Nur relativ besser“ ergibt BETTER_THAN_BASE_ONLY bzw. an der letzten Prüfung BETTER_THAN_BASE_NOT_PROFITABLE. Das ist kein Erfolg.
     - Geschlossen werden Varianten als FALSIFIED_WORSE, FALSIFIED_NO_EFFECT oder BASE_CHANGED.
5. **Dokumentation und Änderungsvorschlag:**
   - `research_memory` übernimmt Varianten-Experimente (`option_experiment`; ein Erfolg nur bei relativem UND wirtschaftlichem Vorteil) und Vorschläge mit Freigabebedarf (`option_proposal`, APPROVAL_REQUIRED).
   - Bei PROPOSE_ADOPTION entsteht `change_proposals/<id>.{md,json}`. Die ursprüngliche Begründung bleibt eingefroren (`original`, `original_markdown`).
   - Der aktuelle Evidenzstatus wird bei jeder Auswertung oben im Vorschlag und in `current_status` nachgeführt, Wechsel append-only in `status_history`:
     - VALID: Vorteil an der letzten Prüfung belegt;
     - WITHDRAWN: Vorteil nicht mehr belegt;
     - SUPERSEDED: Basisversion geändert.
   - Umsetzung nur per menschlichem PR.

**Freigabe der Basis (Folge-PR):** `release_gate` verlangt seit 2026-10-31 zusätzlich eine sequenzielle Bestätigung des Forward-Tests an vorab festgelegten Prüfzeitpunkten (`forward_sequential`, siehe `docs/OPTION_LEARNER.md` Punkt 8). Das tägliche nominale Prüfen allein gibt nicht mehr frei.

## Automatik

| Workflow | Schritt | Wirkung |
|---|---|---|
| `ml_research.yml` (Samstag, monatlich) | `option_learner evaluate` → `research_director --options` | Varianten bewerten und ggf. schließen, Lücken, Vorschläge, Registrierung, Research Memory, Leistungszeugnis |
| `scanner.yml` (täglich) | `option_learner recommend` | Basis wie bisher, danach aktive Varianten isoliert |

## Sicherheitsgrenzen

- Keine Änderung an `main`, Champion, Produktionsregeln, Risikolimits oder API-Zugängen. Änderungsvorschläge sind Dateien unter `outputs/`; umgesetzt wird nur per menschlichem PR.
- Kein generierter Code: Varianten sind Daten (eine Änderung aus der Allowlist) und werden gegen `validate_overrides` sowie die registrierte Experimentversion geprüft.
- Keine bezahlten Daten, keine Marktdaten-API und keine LLM-Kosten. Datenquellen, Quote-Erfassung und Stichprobe sind immer Vorschläge mit Freigabepflicht.
- Unzureichende Evidenz führt zu NEED_MORE_DATA bzw. AWAITING_LOOK (vor der ersten Prüfung) oder INSUFFICIENT_COVERAGE (zu viele fehlende Ergebnisse).
- Die Varianten-Experimente beeinflussen keine Empfehlung: Die Freigabe (`release_gate`) betrachtet nur die Basis.
