# Phase 2 — FD3 gepaarte Forward-Tests (Research/Shadow)

Basis: FD3-Lernprognosen aus PR #183 und unveränderliche Aktienergebnislabels.
Diese Phase trifft **keine Handelsentscheidung** und ist **keine Alpha-Freigabe**.

## Aufbau

1. Nur zuvor eingefrorene Prognosen aus `predictions.jsonl`, denen
   ein hinreichend späteres, ebenfalls eingefrorenes Ergebnis aus
   `outcomes.jsonl` zugeordnet werden kann, werden bewertet.
2. Die Baseline und der Challenger verwenden exakt dieselben Aktien,
   Signalzeitpunkte, 20-Handelstage-Labels und hypothetischen Aktienkosten.
3. Pro Stichtag werden das gepaarte Rank-IC-Delta und die Änderung einer
   festen Top-Dezil-Aktienrendite ausgewiesen.
4. Die Zeitblöcke dürfen sich **bis zum tatsächlichen Label-Ende** nicht
   überlappen. Weniger als acht unabhängige Blöcke: nur DATA_GAP.
5. Diagnostische, vorab definierte Auswertungen nach 8, 16 und 24
   unabhängigen Blöcken: zeitlich innerhalb eines Stichtags permutierte
   Outcome-Labels, deterministischer RNG-Seed, Bootstrap-KI und lokale
   Bonferroni-Korrektur für vier Features × drei Looks.
6. Erstmalige Look-Ergebnisse werden mit Protokoll-Hash und eigener
   Hash-Kette dauerhaft eingefroren. Eine heimliche Änderung der
   Statistikmethodik führt zum Fehler, nicht zum stillen Neutest.
7. Alle Evidenz bleibt `DIAGNOSTIC_SHADOW_ONLY`. Phase 2 bucht
   **kein** Alpha-Kontingent im übergreifenden `alpha_ledger`,
   dessen Registrierung allein einen formalen Alpha-Test ermöglichen würde.
   Somit gibt es hier **keinen FORWARD_CONFIRMED-Status**.

## Evidenz

- `outputs/research/feature_discovery_v3/paired_looks.jsonl`
- `outputs/research/feature_discovery_v3/paired_validation.json`

## Ausführen

```bash
python -m modules.research_feature_validation
python -m pytest -q tests/test_research_feature_validation.py
```

Der bestehende `ml_research.yml` führt Phase 2 nach der
Feature-Store-Aktualisierung aus; kein Markt-API-Abruf, keine neuen Modelle,
keine LLM-Kosten.

## Weiteres

Eine statistische Verbesserung der **Aktien**prognose ist kein Beleg für
options-natives Alpha. Die spätere Phase 4 darf nur tatsächlich archivierte
Bid-/Ask-Entscheidungen und nachgewiesene RTH-Exit-Quotes verwenden.
Fehlende Quotes bleiben Datenlücken. Eine unabhängige Bestätigung unter
**globalem** Alpha-Fehlerbudget benötigt eine gesonderte freigegebene Studie.
