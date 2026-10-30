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
   - **Prüfzeitpunkte:** Urteile gibt es nur an den bei der Registrierung festgeschriebenen Prüfzeitpunkten `look_days` (1/3, 2/3 und 1 von `max_days`). Prüfung k nutzt nur Entscheidungen bis zu ihrem Stichtag und erst, wenn diese alle fällig sind. Die wöchentlichen Auswertungen zwischen zwei Prüfungen werten also dieselben eingefrorenen Daten aus und erzeugen keine zusätzlichen Tests.
   - **Alpha-Budget:** Das Teil-Alpha je Prüfung folgt der O'Brien-Fleming-Ausgabefunktion (`alpha_spending`; frühe Prüfungen sehr streng, Summe = Alpha). Dazu kommt Bonferroni über alle für Basisversion und Horizont registrierten Varianten. Das KI kommt aus dem Bootstrap-Standardfehler über unabhängige Haltefenster.
   - **Abdeckung (gegen Selektionsverzerrung):** Gezählt werden alle fälligen Kandidaten einer Prüfung, dazu die gemeinsame Ergebnisabdeckung und die Ausschlussgründe (nicht im Ledger, keine Basis-Entscheidung, fehlendes geprüftes Ergebnis einer gehandelten Wahl). Unter 80 % lautet das Urteil INSUFFICIENT_COVERAGE; eine positive Freigabe gibt es dann nicht.
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

**Bekannte Grenze:** Die Freigabe der Basis-Empfehlungen (`release_gate`, seit #144) prüft den Forward-Test täglich ohne sequenzielle Korrektur. Das kann ein eigener Folgeschritt mit demselben Alpha-Budget werden.

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
