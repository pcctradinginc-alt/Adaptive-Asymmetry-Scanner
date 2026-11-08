# Phase 3 – Autonome Forschungspriorisierung (FD3)

**Ziel:** Nächste Forschungsmaßnahmen aus empirisch beobachtbarer
Point-in-Time-Abdeckung und prospektiven Forward-Diagnosen auswählen.
Keine autonome Code-Modifikation oder Produktionsfreigabe.

## Entscheidungskreislauf

1. `status.json` aus FD3: Datenabdeckung je Feature/Quelle.
2. `paired_validation.json` aus Phase 2: unabhängige Forward-Fenster,
   dokumentierte negative Kontrollen und gepaarte Aktienergebnisse.
3. `research_memory.jsonl`: bereits dokumentierte Forschungsrichtungen,
   Datenlücken und Falsifikationen. Keine positive Finanzwirkung wird
   daraus abgeleitet.
4. Bei `COLLECT_PIT`: noch nicht ausreichende Datenbasis / Verknüpfung
   herstellen. Keine simulierten Ersatzdaten.
5. Bei `CONTINUE_FORWARD`: existierende eingefrorene Studie weiter
   beobachten, keine nachträgliche Feature-Änderung.
6. Bei `DESIGN_OPTION_TRIAL`: nur nach positivem **diagnostischem**
   Aktienresultat eine *getrennte* Optionsstudie vorschlagen.
   Erforderlich: menschliche Freigabe, eigenes Protokoll, echte
   archivierte RTH-Quotes und globale Alpha-Ledger-Zuteilung.
7. Bei `AUDIT_STABILITY`/`REVIEW_LOW_VALUE`: auf Stabilität und
   wirtschaftlichen Zusatznutzen prüfen, keine automatische Quellabschaltung.

Es werden höchstens **zwei** Aktionen pro Lauf ausgewählt. Das Prioritätsmaß
kombiniert gemessene Abdeckung, unabhängige Forward-Fenster, Unsicherheit,
Recherchekosten und mögliche vorangegangene Falsifikationen. Es ist bewusst
**kein** geschätztes Alpha oder Trainiertes-RL-Reward.

Die gewählten Maßnahmen werden append-only in einer Hash-Kette gespeichert.
Erneute gleiche Situationen erzeugen keine weiteren Aufgaben. Eine Manipulation
des Journals lässt die Auswertung fehlschlagen.

## Dateien

- `modules/research_feature_director.py` – Auswahl und Audit-Journal.
- `modules/research_memory.py` – Einordnung der Phase-3-Dossiers als
  unbestätigte Forschungsaufgaben, **niemals FORWARD_VALIDATED**.
- `outputs/research/feature_discovery_v3/director_plan.json`
- `outputs/research/feature_discovery_v3/director_decisions.jsonl`
- `tests/test_research_feature_director.py`

Der bestehende wöchentliche ML-Research-Workflow führt Phase 3 unmittelbar
nach den Phase-2-Kontrollen aus. Keine neuen API-Aufrufe, Brokerverbindungen,
Handelsregeln, Risikoänderungen oder Trainingsjobs.

**Formal bestätigtes Alpha erfordert weiterhin eine separat registrierte
prospektive Prüfung und eine options-native nettoökonomische Abnahme.**
