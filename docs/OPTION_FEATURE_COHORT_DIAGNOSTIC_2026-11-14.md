# Diagnose der Optionen-Feature-Abdeckung – 2026-11-14

**Scope:** Nur Forschung und Datenqualität. Keine Änderung von
\`modules/option_learner.py\`, \`LEDGER_FEATURES\`, Modellgewichten,
Feature-Imputation, Handelskosten, Risikolimits, Produktionsregeln oder
historischen Options-Buchungen.

## Ausgangsbefund (eingecheckte Research-Statusdateien am 14.11.2026)

| Merkmal | 5 Handelstage, 963 geprüfte Labels | 20 Handelstage, 356 geprüfte Labels |
| --- | ---: | ---: |
| \`surprise\` | 48,70 % | 100,00 % |
| \`options_skew_ratio\` | 48,91 % | 100,00 % |
| \`options_skew_25d\` | 48,70 % | 99,44 % |
| \`impact\` (Kontrolle) | 48,70 % | 100,00 % |
| \`dealer_oi_ratio\` (Kontrolle) | 45,90 % | 94,66 % |

**Code-Befund:** \`option_learner.features()\` liest schon heute
\`row["features"][name]\`; ein Namens- oder Top-Level-Lesefehler für die drei
genannten Features ist daraus **nicht ersichtlich**.
\`pipeline.note_research_features()\` schreibt Skew-Werte unter
\`features\`, nach \`enrich_with_alpha_sources\` und
dem vorgelagerten Economics-first-Gate.
\`surprise\` und \`impact\` werden erst nach \`deep_analysis\` gespeichert.
Seit 30.10. werden zusätzliche \`research_census\`-Kandidaten ohne zwingende
LLM-Richtungsanalyse für Options-Snapshots ausgewählt.
Das ist ein konkreter **Selektions- und Pipeline-Stufen-Mechanismus**,
nicht automatisch ein Softwaredefekt.
Die 5-/20-Tage-Datensätze haben unterschiedliche Verfalls-/Reifegrenzen,
sind daher weder paarweise identisch noch unabhängig.

Die exakten Anteile dieses Mechanismus an den beobachteten Lücken sind
ohne die folgende Archiv-Rekonstruktion **noch nicht bewiesen**.

## Reproduzierbare Analyse

\`\`\`sh
python -m modules.option_feature_diagnostics --as-of 2026-11-14
\`\`\`

Der read-only Befehl lädt mit dem **unveränderten**
\`option_learner._load\` exakt die \`VERIFIED\`-Labels samt archivierter
Einstiegs- und Exit-Quotes. Er schreibt nur:

- \`outputs/research/option_learner/feature_diagnostics.json\`
- \`outputs/research/option_learner/feature_diagnostics.md\`

Abgedeckt sind: 5-/20-Tage-Nenner, Label- und eindeutige
Kandidatenanzahl, \`directional\` vs. \`research_census\`,
Signalzeitraum vor/nach 30.10., einzelne Signaltage, Stufen-/Gate-Gründe,
Optionsstruktur sowie fehlendes/null/ungültiges Quellfeld.
Die Auswertung zählt zusätzlich, ob ein gültiger Quellwert vom vorhandenen
Reader **wirklich verloren** geht, ob ein numerischer Wert fälschlich nur
auf Top-Level abgelegt wurde und ob die Resultate mit
\`status_5d.json\`/\`status_20d.json\` einschließlich Label-Manifest
übereinstimmen. Fehlerhafte oder andere Status-Stichtage werden offen
als nicht reproduziert bzw. nicht vergleichbar markiert.

**Wichtig:** Ein Kandidat kann mehrere Optionsformen beitragen.
Diese Labels werden zwar vom bestehenden Trainingscode einzeln gezählt,
sind aber statistisch nicht unabhängig; deshalb wird auch die Anzahl
eindeutiger Kandidaten ausgewiesen.

## PIT-Grenze und Entscheidungsregel

Die bestehenden Ledger-Features enthalten **keine individuelle
Erfassungszeit je Merkmal**. Aus dem Ledger allein lässt sich damit nicht
beweisen, dass jedes Skew-Feature vor der tatsächlichen Entscheidung
verfügbar war. \`decision_frozen_at\` ist keine Erfassungszeit für jedes
einzelne Feature. Der Report kennzeichnet das explizit als
\`NOT_RECORDED\`; es wird kein nachträglicher Zeitstempel erfunden.

**Keine automatische Fehlerkorrektur aus einer 49-%-Quote.** Zuerst
Quellen-Audit und Gruppenvergleich; nur bei nachgewiesenem Read-Path-Fehler
ein separater, versionierter Änderungs-PR. Nie stillschweigend historische
Features umschreiben, Features künstlich mit Null füllen oder die
50-%-Schwelle ändern. Neue Feature-Definitionen erfordern eigene
Experimentversion und prospektive Tests. Kein Alpha-Nachweis aus
Abdeckungsstatistiken.

Der Wochenlauf erzeugt den Diagnosebericht unmittelbar nach den
kanonischen beiden Options-Lernmodell-Statusläufen; seine Ausgabe wird vom
bestehenden Research-Push archiviert. Ein technischer Diagnose-Ausfall
steht in der Stufen-Bilanz als **DEGRADED**, nicht als Handelserlaubnis.
