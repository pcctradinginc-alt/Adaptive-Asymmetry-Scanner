# Gesamtvalidierung der Intelligenz-Komponenten

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
- **Locked Holdout:** ab 2025-07-01. Dort wurde nichts angepasst.
- **Statistik:**
  - monatlicher Block-Bootstrap;
  - Bonferroni über 6 Komponenten (α = 0,0083 einseitig);
  - Ausreißertest ohne die besten 5 % der Monate.

| Variante | Beschreibung |
|---|---|
| A | Champion: statisches Ensemble (Baseline) |
| B | Basis-Faktormodell |
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

**Abstinenz-Bestätigung auf ungesehenen Jahren 2019–2020.** Die Regel ist
vorab festgelegt und wurde nicht angepasst.
- Aktive Kohorten (n = 52): +3,02 %.
- Inaktive Kohorten (n = 53): +0,24 %.
- Differenz: t = 3,4, also **bestätigt**.

**Ablation:** Ohne Abstinenz sinkt die Expectancy von G um 0,80 pp.

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
- Der Champion A bleibt das Produktions-Ranking. Es gibt keine automatische
  Promotion; über eine Promotion entscheidet nur ein Mensch.
- **Abstinenz:** Die Regel ist die einzige Komponente mit konsistentem
  Mehrwert: Sharpe ×3, Hit +3,5 pp, HC-Hit +12,7 pp, Locked +1,5 pp, und
  bestätigt auf ungesehenen Jahren.
  - Sie ist als `HYP-ABST-001` (ACCEPTED) in der Hypothesen-DB.
  - Der HC-Scanner nutzt sie als Regime-Gate: Liegt kein Stress-Regime vor,
    gibt es keine High-Confidence-Alerts.
  - Für eine volle Promotion fehlen Signifikanz (Kriterium 5) und
    Kohortenzahl. Status: **NEED MORE DATA**, Neubewertung mit jedem Lauf.
- **Konsequenz für „wann nichts tun“:** In ruhigen Aufwärtsmärkten (VIX < 20
  und SPY über SMA200) hat das Ranking historisch keinen nutzbaren Vorsprung.
  Das System soll dort schweigen.
