# Audit-Remediation-Plan

Grundlage: `docs/FORENSIC_ACCEPTANCE_AUDIT.md` (2026-10-02). **Nichts davon ist
umgesetzt.** Für jeden Punkt sind Abnahmekriterium und Test genannt. Wird ein
mit xfail markierter Defekt behoben, muss der xfail in
`tests/test_forensic_audit.py` entfernt werden. Wegen `strict=True` schlägt der
Test sonst fehl.

## P0: macht Ergebnisse ungültig

| ID | Maßnahme | Abnahme |
|---|---|---|
| P0-1 (F04) | **CI-Testlauf** auf `push` und `pull_request`: `pytest -q` und die Audit-Tests als Pflicht-Check; Branch-Schutz auf `main` mit Pflicht-Check und Pflicht-Review (CODEOWNERS). | PR ohne grünen Check nicht mergebar; Agent-Merges ohne Review technisch unmöglich |
| P0-2 (F01) | **Survivorship-freies Universum:** historische Indexmitgliedschaft je Stichtag (PIT-Konstituenten inkl. Delistings und Kursen delisteter Titel). Bis dahin trägt jede Research-Zahl den Vermerk „survivorship-biased“. | `test_A5_research_universe_is_survivorship_free` grün; Neuberechnung aller WF-, Meta- und Validierungszahlen |
| P0-3 (F02) | **Locked Holdout für KONTAMINIERT erklären.** Neuer Holdout nur **vorwärts** ab einem festen Datum, nach jeder Code-Änderung höchstens einmal auswerten, Zähler mit Sperre. `prob_maps` nicht auf dem Holdout fitten. | Registry verweigert eine zweite Auswertung; `prob_map.fold != holdout` |
| P0-4 (F03) | **Abstinenz-Regel neu einstufen:** HYP-ABST-001 von ACCEPTED auf INCONCLUSIVE (in-sample abgeleitet). Belastbar wird sie nur durch vorwärts gesammelte Kohorten ab der Registrierung (2026-09-29), präregistriert mit Mindest-n. Docs korrigieren (`NEXT_INTELLIGENCE_VALIDATION.md`, `machine_state`). | Keine Doku nennt 2019–2020 mehr „ungesehen“; Forward-Ledger der Regel existiert |
| P0-5 (F06) | **Protokolle wirklich schützen:** Änderungen an `config/*_protocol.yaml` und an den Hash-Pins nur per menschlichem Review (CODEOWNERS erzwungen, siehe P0-1). | Branch-Protection-Regel aktiv |

## P1: wichtige Funktion fehlt oder ist falsch

| ID | Maßnahme | Abnahme |
|---|---|---|
| P1-1 (F11) | Weekly §1 „Safe Mode“ aus `safe_mode.json` lesen; §3 Kalibrierung des **aktiven** Ensembles zeigen; Datenstand (Commit oder Zeitstempel der Artefakte) im Kopf ausweisen; Workflows mit `ref: main` auschecken oder Weekly per `workflow_run` nach ML auslösen. | Test: Weekly-Abschnitte widerspruchsfrei bei aktivem Safe Mode |
| P1-2 (F07) | Safe Mode fail-closed: fehlendes oder beschädigtes `safe_mode.json` blockiert HC. | 2 xfails grün |
| P1-3 (F10) | Dedup-Zustand nur bei erfolgreichem Versand fortschreiben; sonst Retry. | xfail `test_A16_failed_delivery_is_retried` grün |
| P1-4 (F12) | Wahrscheinlichkeitsmodell neu bewerten. Solange Brier ≈ Zufall und die Buckets nicht monoton sind: keine Prozentangaben in Alerts, nur „Rang-Top-Dezil“ und ein klarer Hinweis. | Bucket-Monotonie-Test auf OOS (Vorwärtsdaten) |
| P1-5 (F09) | `overall_calibration` berücksichtigt überkonfidente Buckets mit n ≥ 100; der festschreibende Test wird korrigiert. | xfail `test_A14_…` grün |
| P1-6 (F13) | Entscheidung durch den Menschen: Soll der ML-/Intelligenz-Stack den täglichen Scanner beeinflussen (z.B. Abstinenz als Gate der täglichen Mail)? Wenn nein: Doku und Weekly müssen klar sagen, dass er die tägliche Mail **nicht** steuert. | dokumentierte Entscheidung |
| P1-7 | Doku-Fehler korrigieren: B = Meta-Learning (nicht „Basis-Faktormodell“); „Champion“ = keiner registriert; Meta-Learning-Urteil auf den Stand des letzten Laufs bringen bzw. Instabilität ausweisen. | Review |
| P1-8 | Produktions-Scanner testen: options_designer (30 %), risk_gates (34 %), prescreener (40 %), mirofish (40 %), email_reporter (20 %); negative Fälle für jedes Gate. | Coverage ≥ 70 % für diese Module |
| P1-9 | Echte Paper-Performance des täglichen Scanners als Hauptmaßstab: letzte 3 Monate PF 0,42, letzte 4 Wochen 0/5 Gewinner. Ursachenanalyse vor weiterem Ausbau. | Bericht |

## P2: Robustheit

| ID | Maßnahme |
|---|---|
| P2-1 (F05) | `lagged_calibration`: Vorjahreszeilen mit `label_end` ≥ Testbeginn ausschließen (Purge). |
| P2-2 (F08) | Safe-Mode-Auslöser für STALE-Quellen hoher Kritikalität und für beschädigte Modell- oder Registry-Dateien. |
| P2-3 | Sektor-Zuordnung PIT (historische GICS) oder als Bias kennzeichnen; KG-Kanten mit `valid_from`. |
| P2-4 | Prediction Memory: echte `model_versions` (`spec_hash`) statt `null`. |
| P2-5 | Reproduzierbarkeitsjob: zwei Läufe auf eingefrorenem Panel-Snapshot (Artefakt) und Vergleich der Kernmetriken mit Toleranz. Heute fehlt ein Daten-Snapshot; Kurse werden jedes Mal neu geladen (`auto_adjust`). |
| P2-6 | Konfidenz-Buckets: MFE und MAE je Bucket speichern (fehlen). |
| P2-7 | Embargo zusätzlich zum Purge prüfen (Feature-Lookback-Überlappung). |

## P3: Verbesserung

| ID | Maßnahme |
|---|---|
| P3-1 | Dead Code entfernen oder anschließen: `news_fetcher.py`, `reddit_signals.py`. |
| P3-2 | KG: nur behalten, wenn gemessene Kanten entstehen; sonst den HC-Check entfernen (heute wirkungslos). |
| P3-3 | Analogie-Engine: OOS-Nutzen messen (mit oder ohne Analogie-Gate) oder als ungeprüft kennzeichnen. |
| P3-4 | Self-Play: Rollen als getrennte, protokollierte Prüfschritte mit eigener Begründung je Kandidat. |
| P3-5 | Coverage für `pipeline.py` und `feedback.py` messen. |

## Reihenfolge
1. P0-1
2. P0-5
3. P0-3 und P0-4 (Dokumentation und Einstufung, sofort möglich)
4. P1-1 bis P1-3 (kleine, klar testbare Fixes)
5. P0-2: größter Aufwand; erfordert eine PIT-Konstituenten-Quelle mit
   geklärter Lizenz, bis dahin REVIEW_REQUIRED
6. Rest
