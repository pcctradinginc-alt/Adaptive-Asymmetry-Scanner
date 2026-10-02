# Meta-Learning-Validierung

Lauf vom 2026-09-29, 15:18 UTC, `meta-v1`, Panel-Hash `776a327d3d005250`. Die
Rohdaten stehen in `outputs/research/meta_learning.{json,md}`, das
Protokoll ist `config/meta_protocol.yaml` (vorab registriert, gepinnt).

## 1. Ausgangszustand
- Sechs registrierte Basismodelle: Momentum-Regel, Elastic Net, drei
  HistGBM-Modelle und zwei Spezialisten.
- Keines besteht allein die Promotionskriterien; t liegt zwischen 1,0 und 1.9.
- Einen Champion gibt es nicht. Die Referenz ist das **statische,
  gleichgewichtete Rang-Ensemble**.

## 2. Architektur
```
Basis-OOS je Fold (Training < Testjahr) ─► Meta-Merkmale je Stichtag
  (Regime, 13/52-Wochen-IC nur aus fertigen Labels, Kalibrierungs-Steigung,
   Uneinigkeit) ─► Meta-Walk-Forward (Meta-Training < Meta-Testjahr)
  ─► Varianten ─► Kennzahlen/Bootstrap/Ablation ─► Gate ─► aktives Ensemble
```
Umsetzung in `modules/meta_learning.py`, aufbauend auf dem Feature-Store, der
Registry und den Kennzahlen von `modules/ml_research.py`.

## 3. Komponenten
| Komponente | Umsetzung |
|---|---|
| Dynamische Modellgewichte | Ridge je Modell: IC(t) ~ Regime + Historie + Uneinigkeit. Gewicht ∝ max(ŷ, 0), dazu P(Mehrwert) = Φ(ŷ/σ) und eine Konfidenz. |
| Stacking | HistGBM auf Rängen, Uneinigkeit, Regime, Historie und Sektor |
| Vergleichsvarianten | bestes Einzelmodell ex ante, statisch, IC-gewichtet |
| Model Memory | `prediction_memory`, append-only |
| Failure-Profile | IC je Segment, t über Monate |
| Uneinigkeits-Test | fünf Maße |
| Kalibrierungs-Buckets | isotonische Abbildung aus dem Vorjahr |
| High-Confidence-Regel | OOS-kalibriert |
| Safe Mode | Rückfall auf die Referenz |

## 4. Daten
- 513 Titel (heutige S&P-500-Liste), wöchentliche Stichtage 2014–2026.
- Basis-OOS ab 2019, Meta-Testjahre 2021 bis 2025-06, Locked-Holdout ab
  2025-07-01.
- Bewertungsgröße: Top-Dezil minus Querschnittsmittel, 20 Handelstage, netto
  10 bp je Seite.

## 5. Leakage-Schutz
- Basis-Training endet mit dem letzten Label vor dem Fold.
- Meta-Training nutzt nur Basis-OOS-Zeilen mit einem Label-Ende vor dem
  Meta-Testjahr.
- Historische Modellleistung stammt nur aus Stichtagen, deren Label vor dem
  Stichtag feststand.
- Die Kalibrierung verwendet nur Vorjahres-OOS.
- Automatische Prüfung je Fold: **OK**.

## 6. Design
Jährlicher Walk-Forward. Der Block-Bootstrap über Monate (2000 Ziehungen) ist
einseitig, mit Bonferroni über zwei Meta-Learner (α = 0,025).

## 7. Baselines und 8. Ergebnisse (Meta-Testjahre 2021 bis 2025-06, 230 Kohorten, 11.587 Positionen)
| Variante | CAGR | Sharpe | MaxDD | PF | Hit | Expectancy | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| **statisch (Referenz)** | 3,0 % | 0,28 | −16,0 % | 1,09 | 47,7 % | +0,33 % | 0,2518 | 0,046 |
| bestes Einzelmodell ex ante | −1,0 % | −0,01 | −28,2 % | 1,00 | 46,4 % | 0,00 % | 0,2522 | 0,047 |
| IC-gewichtet | 0,0 % | 0,06 | −24,4 % | 1,01 | 47,0 % | +0,03 % | 0,2518 | 0,047 |
| **Meta-Regime-Gewichte** | 2,4 % | 0,26 | −18,8 % | 1,08 | 47,3 % | +0,30 % | 0,2519 | 0,046 |
| Meta-Stacking | −1,9 % | −0,05 | −29,1 % | 1,00 | 46,6 % | −0,02 % | 0,2517 | 0,046 |

**Locked-Holdout** (2025-07 bis 2026-08, 3.111 Positionen, einmalig):

| Variante | Expectancy | Sharpe |
|---|---|---|
| statisch | +1,91 % | 1,59 |
| Meta-Regime | +2,37 % | 1,90 |
| Stacking | +2,03 % | 2,67 |

Das Holdout fiel in eine günstige Phase für alle Varianten.

**Ergebnis nach Regime** (statisch; Meta-Regime ähnlich):

| Regime | Expectancy | Trefferquote |
|---|---|---|
| VIX < 20 | −0,24 % | 45 % |
| VIX ≥ 20 | **+1,22 %** | 51 % |
| Aufwärtstrend | −0,04 % | 46 % |
| Abwärtstrend | **+1,53 %** | 53 % |

## 9. Ablationen (Expectancy)
| Variante | Expectancy | Differenz zur Vollversion |
|---|---|---|
| Meta-Regime-Gewichte (voll) | +0,30 % | – |
| ohne Regime-Merkmale | **+0,44 %** | +0,14 |
| ohne Failure Memory (Historie) | +0,26 % | −0,04 |
| ohne Uneinigkeit | +0,25 % | −0,05 |
| ohne dynamische Gewichtung (= statisch) | +0,33 % | – |
| Stacking ohne Failure Memory | −0,21 % | – |

- Die Regime-Merkmale **schaden** den Modellgewichten; das Rauschen der
  Gewichtsschätzung überwiegt.
- Historische Analogien und alternative Daten: n/a, sie sind nicht Teil der
  Basismodelle.

**Uneinigkeit (G):** Keines der Maße ist empirisch belegt, t liegt zwischen
−2,8 und 0,5. Hohe Übereinstimmung geht **nicht** mit höherer Trefferquote
einher. Die Uneinigkeit wird deshalb nicht in den Score eingebaut.

**Failure-Profile (F):** Nur ein Segment ist signifikant:
`hgb_asym20_v1` funktioniert bei Consumer Cyclical (IC 0,042, t 2,16).
Ansonsten gibt es keine Segmente mit |t| ≥ 2.

## 10. Kalibrierung
Die Buckets zu P(20d-Überrendite ggü. SPY > 0) sind **überkonfident**:

| Bucket | Trefferquote |
|---|---|
| 55–60 % | 42 % |
| 60–65 % | 44 % |

Werte über 70 % kommen praktisch nicht vor.

Ursache ist ein Zielkonflikt:
- Die Wahrscheinlichkeit bezog sich auf „schlägt SPY“.
- In 2023/24 lag die Basisrate wegen der Mega-Cap-Dominanz deutlich unter 50 %.
- Das Ergebnis der Trades wird dagegen relativ zum Querschnitt gemessen.

**Maßnahme (MODIFY):** Das Wahrscheinlichkeitsziel ist jetzt „Netto-Rendite
relativ zum Querschnitt > 0“, also genau die Größe, auf die gehandelt wird.
Details in `NEXT_INTELLIGENCE_VALIDATION.md`.

Unsicherheitsintervalle: Die Abdeckung steigt von roh 64,9 % auf 78,8 % nach
konformaler Korrektur. P(>+10 %) hat auch rekalibriert keinen Skill (−0,14).

## 11. Meta-Learning gegen Champion
| Metrik | Δ Meta-Regime − statisch |
|---|---|
| Δ Sharpe | −0,03 |
| Δ CAGR | −0,6 Pp. |
| Δ MaxDD | −2,7 Pp. |
| Δ Expectancy | −0,03 Pp. |
| Δ PF | −0,01 |
| Δ Hit | −0,4 Pp. |
| Δ Brier | +0,0001 |
| Δ ECE | +0,0001 |
| Δ Prec@K | −0,9 Pp. |
| Bootstrap-KI der Monatsdifferenz | [−0,50 %; +0,40 %] |

- Robustheit: Mit 4-wöchentlichem Rebalancing und in beiden Liquiditätshälften
  ist das Ergebnis nicht besser als statisch.
- Nach Jahren: Das Meta-Learning ist nur in 2 von 5 Jahren besser, und in der
  zweiten Hälfte des Zeitraums negativ.

## 12. Risiken
- Survivorship: Es wird das heutige Universum verwendet.
- Die Stichprobe ist gering: 53 Monate.
- Der Locked-Holdout liegt in einem günstigen Regime.

## 13. Bekannte Schwächen
- Die Modellgewichte werden auf verrauschten wöchentlichen IC geschätzt.
- Momentum wird aktuell zu 85 % gewichtet, obwohl der Trend „deteriorating“ ist
  (t −2,65). Das ist ein Beleg für die Instabilität.

## 14. Entscheidung: **REJECT** (instabil)

> Nachtrag Audit 2026-10-02 (F14): Zwei Läufe mit gleichem Code und gleicher
> Datenperiode ergaben einmal NEED_MORE_DATA (Δ Sharpe +0,21) und einmal
> REJECT (Meta-Sharpe 0,013). Ohne eingefrorenen Daten-Snapshot sind
> Meta-Urteile nicht reproduzierbar. Inzwischen gibt es Replay mit Snapshot
> (`repro.yml`). Zusätzlich war der Survivorship-Bias (F01) wirksam. Das Urteil
> ist auf dem PIT-Universum neu zu erheben.
Nicht bestanden:
- Δ Sharpe > 0
- Bootstrap-Untergrenze > 0
- mindestens 60 % der Jahre positiv
- beide Hälften positiv
- Robustheit ohne Ausreißer

Bestanden:
- Leakage
- Stichprobe
- keine Regime-Verschlechterung
- Kalibrierung nicht schlechter
- Locked

Der bisherige Champion bleibt: das **statische Ensemble**. Das Meta-Learning
bleibt im Schattenbetrieb und wird monatlich neu geprüft.

## 15. Nächste Verbesserungen
1. **Regime-Abstinenz statt Regime-Gewichtung:** Laut Ergebnis funktioniert das
   Ensemble nur bei Stress (VIX ≥ 20, Abwärtstrend). Das wird als Hypothese
   registriert und auf ungesehenen Daten geprüft: Basis-OOS 2019–2020, danach
   Forward.
2. Das Wahrscheinlichkeitsziel auf die gehandelte Größe umstellen (erledigt, s.o.).
3. Das World Model als Meta-Merkmal testen (Challenger C in
   `NEXT_INTELLIGENCE_VALIDATION.md`).
4. `rs_63` entfernen (Duplikat von `mom_3m`), in der nächsten Modellgeneration.
