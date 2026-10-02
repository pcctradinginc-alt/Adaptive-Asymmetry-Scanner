# Gesamtvalidierung der Intelligenz-Komponenten

> **Korrekturhinweis (Audit 2026-10-02, `docs/FORENSIC_ACCEPTANCE_AUDIT.md`).**
> Die Zahlen unten wurden mit Survivorship-Bias erhoben (F01, inzwischen
> behoben). Sie sind deshalb neu zu messen.
>
> Weitere Korrekturen:
> - **Locked Holdout:** KONTAMINIERT (F02). Locked-Werte sind nur informativ.
> - **Abstinenz-„Bestätigung“ 2019–2020:** nicht ungesehen (F03). Sie zählt
>   nicht. Die Regel ist INCONCLUSIVE und nur vorwärts bestätigbar.
> - **Variante B:** ist „A + Meta-Learning (Regime-Gewichte)“, nicht ein
>   Basis-Faktormodell.
> - **„Champion“ A:** ist eine Referenz; die Registry hat keinen Champion.
> - **Reproduzierbarkeit:** Einzelurteile kippen zwischen Läufen ohne
>   Daten-Snapshot (F14).


Modul: `modules/next_intelligence.py`. Protokoll (hash-gepinnt):
`config/next_protocol.yaml`.
Ausgaben: `outputs/research/next_validation.{json,md}`.
Lauf: CI 36595328836 (2026-09-29).

Leitfrage: *Erkennt das System besser als zuvor, wann es eine echte
asymmetrische Chance hat, und wann es besser nichts tut?*

## Design
- **Universum:** S&P 500, PIT-Feature-Store, wöchentliche Kohorten.
- **Ziel:** 20-Tage-Überrendite gegenüber dem Querschnitt, abzüglich
  2 × 10 bp Kosten.
- **Auswahl:** jeweils das Top-Dezil.
- **Walk-Forward:** gepurgt, OOS 2021 bis 2025-06.
- **Locked Holdout:** ab 2025-07-01. **KONTAMINIERT** (28 Auswertungen, Ergebnisse in Verdikten zitiert; Audit F02).
- **Statistik:**
  - monatlicher Block-Bootstrap;
  - Bonferroni über 6 Komponenten (α = 0,0083 einseitig);
  - Ausreißertest ohne die besten 5 % der Monate.

| Variante | Beschreibung |
|---|---|
| A | Referenz: statisches Ensemble (Baseline; kein registrierter Champion) |
| B | A + Meta-Learning (Regime-Gewichte) |
| C | A + Meta-Learning + World-Model-Zustand |
| D / E | A + Causal-Sektor-Tilts (zwei Gewichtungen) |
| F | A ohne kontrafaktisch fragile Positionen |
| A_blindspot | A ohne Unknown-Unknown-Cluster |
| A_abstention | A, aber nur handeln bei VIX ≥ 20 **oder** SPY < SMA200 |
| G | Kombination aller Komponenten, die einzeln bestehen (hier nur Abstinenz) |

## Ergebnisse
| Variante | CAGR | Sharpe | MaxDD | Hit | Expectancy | HC-Hit (n) | Locked |
|---|---|---|---|---|---|---|---|
| A | 2,2 % | 0,227 | −18,8 % | 47,4 % | 0,27 % | 48,6 % (288) | 1,9 % |
| B | 1,5 % | 0,191 | −17,2 % | 47,2 % | 0,22 % | 45,0 % (980) | 1,2 % |
| C | 1,9 % | 0,208 | −17,0 % | 47,4 % | 0,28 % | 44,9 % (988) | 1,2 % |
| D | 2,4 % | 0,255 | −15,8 % | 47,6 % | 0,31 % | 40,9 % (728) | 1,3 % |
| E | 2,4 % | 0,252 | −15,7 % | 47,5 % | 0,31 % | 41,3 % (744) | 1,3 % |
| F | 1,6 % | 0,231 | −13,2 % | 48,0 % | 0,16 % | 43,3 % (616) | 0,7 % |
| A_blindspot | 1,5 % | 0,249 | −10,2 % | 47,8 % | 0,06 % | 47,8 % (157) | 0,5 % |
| **G = A_abstention** | **8,4 %** | **0,677** | **−12,8 %** | **50,9 %** | **1,07 %** | **61,3 % (269)** | **3,4 %** |

Bei G ist das System nur in 40 % der Wochen aktiv.

## Komponenten gegen A (Monats-Δ, 95-%-CI)
| Komponente | Δ | CI | ohne Top-5 % | Verdikt |
|---|---|---|---|---|
| C Meta + World | −0,03 % | [−0,63 %, +0,63 %] | −0,32 % | **REJECT** (siehe `META_LEARNING_VALIDATION.md`) |
| D/E Causal-Tilts | +0,00 % | [−0,60 %, +0,66 %] | −0,26 % | **REJECT** |
| F Counterfactual | −0,10 % | [−0,79 %, +0,59 %] | −0,42 % | **REJECT** |
| Blind-Spot-Filter | −0,11 % | [−0,99 %, +0,77 %] | −0,49 % | **REJECT** |
| Abstinenz | **+0,48 %** | [−0,26 %, +1,31 %] | +0,06 % | **KEEP (Shadow), NEED MORE DATA** |
| Decision Intelligence | Sharpe 0,229 → 0,252, ES −8,2 % → −7,7 % | – | – | **KEEP (knapp)** |

**Abstinenz 2019–2020 (KONTAMINIERT, zählt nicht – Audit F03).** Die Regel ist
vorab festgelegt und wurde nicht angepasst.
- Aktive Kohorten (n = 52): +3,02 %.
- Inaktive Kohorten (n = 53): +0,24 %.
- Differenz: t = 3,4. Die Jahre waren vor der Regel gesehen, daher **keine Bestätigung**.

**Ablation (Phase 19):** G enthält nach Regel nur Komponenten, die einzeln
gegen A bestehen. Für jede verworfene Komponente gilt daher: „G ohne X“ = G.
Ihr Beitrag wurde stattdessen inkrementell gemessen, also „A mit X“ gegen A.

| Komponente | gemessener Beitrag | behalten? |
|---|---|---|
| Abstinenz (aus World-Model-Regime) | G ohne sie: Expectancy −0,80 pp | **ja** (Shadow, NEED MORE DATA) |
| World Model als Modell-Input (C) | −0,03 %/Monat, n.s. | nein (nur Bericht) |
| Causal Engine (D) | +0,00 %/Monat, n.s. | nein |
| Knowledge Graph (E) | +0,00 %/Monat, n.s.; keine Messkanten | nein (nur Evidenz) |
| Counterfactuals (F) | −0,10 %/Monat; Locked 1,9 % → 0,7 % | nein (nur HC-Diagnose) |
| Unknown-Unknown-Filter | −0,11 %/Monat; Locked 1,9 % → 0,5 % | nein (nur Diagnose) |
| Active Learning / Director | keine Renditegröße; 5 Hypothesen ohne Ressourcenverschwendung verworfen | ja (Prozess) |
| Self-Play (Lab-Prüfkette) | verhindert 2 Varianten-Tests; keine Renditegröße | ja (Governance) |
| Meta-Cognition / Safe Mode | verhindert HC-Alerts außerhalb des Trainingsbandes; nicht OOS-renditemessbar | ja (Risikokontrolle) |

Prozess- und Governance-Komponenten verändern kein Ranking. Sie werden daran
gemessen, wie viele Fehlentscheidungen sie verhindern, nicht an der Rendite.

**Stress** (nur OOS-Fenster):

| Fenster | A | G |
|---|---|---|
| Zinsschock 2022 | +0,58 % | +0,58 % |
| Regionalbanken 2023 | −0,72 % | −0,72 % |
| Zollschock 2025 | +8,6 % | +8,6 % |

In allen drei Fenstern war G aktiv. Covid 2020 und Q4 2018 liegen außerhalb
der OOS-Basis; 2019–2020 ist durch die Abstinenz-Bestätigung abgedeckt.

## 10-Punkte-Gate für die Promotion von G
| Kriterium | Ergebnis |
|---|---|
| 1 kein Leakage | ✅ |
| 2 reproduzierbar | ✅ |
| 3 Mindest-Kohorten | ❌ nur 40 % aktiv |
| 4 Kalibrierung nicht schlechter | ✅ |
| 5 risikoadjustiert signifikant | ❌ CI-Untergrenze < 0 |
| 6 kein Regime-Kollaps | ❌ ein Regime negativ |
| 7 HC-Hit höher | ✅ |
| 8 Drawdown | ✅ |
| 9 nicht ausreißergetrieben | ✅ |
| 10 Komplexität lohnt | ✅ |

## Entscheidung: **KEEP_CHAMPION**
- Die Referenz A bleibt das Research-Ranking (kein registrierter Champion; die tägliche Mail nutzt ihn nicht, F13). Es gibt keine automatische
  Promotion; über eine Promotion entscheidet nur ein Mensch.
- **Abstinenz:** Die Regel ist die einzige Komponente mit konsistentem
  Mehrwert in-sample: Sharpe ×3, Hit +3,5 pp, HC-Hit +12,7 pp. Die Bestätigung
  2019–2020 und Locked sind aber kontaminiert (F02, F03). Status INCONCLUSIVE;
  bindend ist die Vorwärts-Bestätigung ab 2026-09-29.
  - Sie ist als `HYP-ABST-001` (ACCEPTED) in der Hypothesen-DB.
  - Der HC-Scanner nutzt sie als Regime-Gate: Liegt kein Stress-Regime vor,
    gibt es keine High-Confidence-Alerts.
  - Für eine volle Promotion fehlen Signifikanz (Kriterium 5) und
    Kohortenzahl. Status: **NEED MORE DATA**, Neubewertung mit jedem Lauf.
- **Konsequenz für „wann nichts tun“:** In ruhigen Aufwärtsmärkten (VIX < 20
  und SPY über SMA200) hat das Ranking historisch keinen nutzbaren Vorsprung.
  Das System soll dort schweigen.

## Reproduktion (Nachlauf CI 37021907362, 2026-10-02)
Neu trainiert mit einer weiteren Woche Daten.

| Variante | Sharpe | MaxDD | Hit | HC-Hit (n) |
|---|---|---|---|---|
| A | 0,262 | −17,9 % | 47,6 % | 46,4 % (356) |
| G | 0,706 | −12,4 % | 51,2 % | 62,4 % (340) |

- Abstinenz in-sample erneut gleiche Richtung (kontaminiert, F03): 2019–2020 aktiv +3,02 % gegen inaktiv
  +0,34 %, t = 3,27. G gegen A: Monats-Δ +0,48 % (CI [−0,26 %, +1,29 %]).
- Gate und Entscheidung sind unverändert: **KEEP_CHAMPION**, Abstinenz
  **NEED MORE DATA**.
- **Decision Intelligence kippt auf MODIFY:** Sharpe 0,264 → 0,242 bei
  besserem ES und MaxDD. Der Vorteil war also nicht robust. Die Auswahl bleibt
  nur eine Risiko-Information im HC-Scanner (Portfolio-Nutzen), ohne
  Ranking-Einfluss.
- Das Basismodell B zeigt diesmal eine höhere Sharpe (0,468), aber nur 22
  HC-Fälle. Ein Wechsel ist daraus nicht ableitbar; die Frage bleibt in der
  Champion/Challenger-Registry.
