# Active Learning

Modul: `modules/research_director.py` (`active_learning`,
`onboarding_checklist`). Katalog: `config/data_catalog.yaml`. Ausgabe:
`outputs/research/active_learning.json`.

## Frage
Welche zusätzliche Information würde die Unsicherheit am stärksten reduzieren?

## Vorgehen
- **Unsicherheit je Informationsdimension:**
  - aus dem World Model: `unavailable` bzw. `no_data` = 1, sonst die
    Unsicherheit der Dimension;
  - aus den Blind Spots, z.B. extreme Bewegungen: Event-Risiko.
- **Erwarteter Informationsgewinn** = Unsicherheit × Machbarkeit. Machbar ist
  eine Quelle nur mit historischer Abdeckung ≥ 2016 und intaktem Zeitstempel.
- **Priorität** = EIG × Zuverlässigkeit ÷ Beschaffungskosten.
- **Neue Quellen sind nie automatisch vertrauenswürdig.** Jede Quelle
  durchläuft eine Aufnahmeprüfung:
  - provenance
  - license (bei `REVIEW_REQUIRED` ist eine manuelle Prüfung nötig)
  - historical coverage
  - timestamp integrity
  - revision analysis
  - missing-data analysis
  - leakage audit
  - inkrementeller OOS-Test

## Aktueller Stand
Integriert sind die ALFRED-Reihen PAYEMS, ICSA, RSAFS und ISRATIO
(`fred_world_macro`).

| Kandidat | Status |
|---|---|
| SEC-EDGAR-Earnings-Termine (8-K 2.02, Filing-Zeitstempel) | offen |
| SEC-XBRL-Fundamentaldaten | offen |
| FINRA Short Interest | offen |
| Cboe Put/Call | offen |
| I/B/E/S-Revisionen | offen (kommerziell, REVIEW_REQUIRED) |

PortWatch und Destatis-Maut werden nur vorwärts archiviert (`forward_only`).

Die priorisierte Liste steht jede Woche im Weekly Report (Abschnitt 16).
