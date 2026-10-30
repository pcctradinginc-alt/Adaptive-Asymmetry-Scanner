# Autonomous Research Improvement (Options-Lernmodell)

Stand 2026-10-31. Erweitert die vorhandenen Bausteine `research_director`, `meta_cognition`, `option_learner` und `research_memory` sowie den Workflow `ml_research.yml`. Eine zweite Architektur gibt es nicht. Tests: `tests/test_research_director.py`, `tests/test_option_learner.py`.

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
4. **Fairer Vergleich** – `option_learner.evaluate_variants()` vergleicht jede Variante auf identischen, fälligen Kandidaten mit geprüften Netto-Ergebnissen.
   - Gepaart je Tag, gleiches Budget je Kandidat, Block-Bootstrap über unabhängige Haltefenster.
   - Primär gegen die Basis-Lernversion, zusätzlich gegen deren eingefrorene Referenz.
   - Bonferroni über alle für Basisversion und Horizont registrierten Varianten.
   - **Verdikte:** NEED_MORE_DATA bei zu wenig Evidenz, VARIANT_BETTER, VARIANT_WORSE, NO_SIGNIFICANT_DIFFERENCE.
   - **Schließen** (append-only Ereignis `closed`): FALSIFIED_WORSE, FALSIFIED_NO_EFFECT (nach `max_days` ohne Vorteil), BASE_CHANGED.
5. **Dokumentation und Änderungsvorschlag:**
   - `research_memory` übernimmt Varianten-Experimente (`option_experiment`) und Vorschläge mit Freigabebedarf (`option_proposal`, APPROVAL_REQUIRED).
   - Bei VARIANT_BETTER entsteht einmalig `change_proposals/<id>.{md,json}`: konkrete Code-Änderung, Evidenz, Falsifikation und Prüfpunkte. Umsetzung nur per menschlichem PR.

## Automatik

| Workflow | Schritt | Wirkung |
|---|---|---|
| `ml_research.yml` (Samstag, monatlich) | `option_learner evaluate` → `research_director --options` | Varianten bewerten und ggf. schließen, Lücken, Vorschläge, Registrierung, Research Memory, Leistungszeugnis |
| `scanner.yml` (täglich) | `option_learner recommend` | Basis wie bisher, danach aktive Varianten isoliert |

## Sicherheitsgrenzen

- Keine Änderung an `main`, Champion, Produktionsregeln, Risikolimits oder API-Zugängen. Änderungsvorschläge sind Dateien unter `outputs/`; umgesetzt wird nur per menschlichem PR.
- Kein generierter Code: Varianten sind Daten (eine Änderung aus der Allowlist) und werden gegen `validate_overrides` sowie die registrierte Experimentversion geprüft.
- Keine bezahlten Daten, keine Marktdaten-API und keine LLM-Kosten. Datenquellen, Quote-Erfassung und Stichprobe sind immer Vorschläge mit Freigabepflicht.
- Unzureichende Evidenz führt zu NEED_MORE_DATA.
- Die Varianten-Experimente beeinflussen keine Empfehlung: Die Freigabe (`release_gate`) betrachtet nur die Basis.
