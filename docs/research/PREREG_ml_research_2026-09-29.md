# Präregistrierung: ML-Research-Kreislauf (Shadow)

Registriert: 2026-09-29, 16:00 UTC, vor dem ersten Lauf auf echten Daten.
Code: `modules/ml_research.py`. Registry: `config/model_registry.yaml`.
Tests: `tests/test_ml_research.py`.

## Zweck
Der Kreislauf soll kontrolliert und ohne Selbsttäuschung prüfen, ob heute
beobachtbare Konstellationen ein besseres Chance-Risiko-Verhältnis der folgenden
20 Handelstage vorhersagen. Eine Produktionswirkung gibt es nicht: Scoring, Gates
und PPO bleiben unverändert. Eine Promotion ist immer ein menschlicher PR.

## Daten und Feature-Store
- **Universum:** die heutige S&P-500-Liste (`modules/universe.py`). Das bedeutet
  einen **Survivorship-Bias**. Deshalb misst die Hauptgröße relativ zum
  Querschnittsmittel desselben Stichtags, nicht absolut.
- **Stichtage:** der letzte Handelstag jeder Woche ab 2014-06. Das Training
  beginnt 2015-01.
- **Aktienmerkmale** (nur Daten bis Close_t, als Querschnittsrang je Stichtag):
  - Momentum 12-1, 3 Monate, 1 Monat und 5 Tage;
  - relative Stärke zum SPY über 63 Tage;
  - Volatilität 20 und 60 Tage sowie deren Verhältnis;
  - relatives Volumen (5 zu 60 Tage) und Dollar-Volumen;
  - Abstand zum 52-Wochen-Hoch;
  - maximale Tagesrendite der letzten 21 Tage;
  - Beta über 126 Tage.
- **Markt- und Makromerkmale:**
  - VIX und dessen Veränderung über 21 Tage;
  - SPY-Trend (Abstand zur 200-Tage-Linie) und SPY-Momentum über 63 Tage;
  - 10-jährige Rendite und Zinskurve (10 Jahre minus 3 Monate);
  - CPI-Veränderung zum Vorjahr, Veränderung der Fed-Bilanz über 13 Wochen,
    Dollar und WTI über 63 Tage.

  Die Makrowerte sind ALFRED-Vintages, sichtbar erst ab dem Tag nach der
  Veröffentlichung. NFCI-Credit ist ausgeschlossen, weil die Reihe erst ab 2025
  vorliegt und sonst nur im Locked-Holdout existieren würde.
- **Alternative Daten** (PortWatch, Maut, Wetter, ENTSO-E) sind noch **nicht**
  enthalten. Ihre archivierte Historie ist zu kurz für jahresweisen Walk-Forward
  und hat keine Ticker-Zuordnung mit Belegkraft. Sie kommen als eigene, neu
  registrierte Challenger hinzu, sobald mindestens 3 Jahre PIT-Historie vorliegen.

## Labels
- Entry zum Open von t+1, Exit zum Close von t+h, mit h = 20 und 60.
- `fwd_xs_h`: Rendite minus SPY im selben Fenster.
- `mfe_h` / `mae_h`: maximales High bzw. minimales Low über t+1 bis t+h relativ
  zum Entry.
- `asym_h = (mfe_h + mae_h) / (σ20 · √h)`: Chance minus Risiko, normiert auf die
  Volatilität. Das ist das Asymmetrie-Ziel und enthält keine frei gewählten
  Gewichte.

## Modelle (fest registriert)

| id | Rolle | Modell | Ziel |
|---|---|---|---|
| momentum_12_1 | Benchmark | Regel (Rang Momentum 12-1) | – |
| enet_xs20_v1 | Challenger | Elastic Net | fwd_xs_20 |
| hgb_xs20_v1 | Challenger | Gradient Boosting (sklearn HistGBM) | fwd_xs_20 |
| hgb_asym20_v1 | Challenger | Gradient Boosting (sklearn HistGBM) | asym_20 |

Es gibt kein Deep Learning und keine neuen Abhängigkeiten. HistGBM entspricht
LightGBM in sklearn.

## Validierung (drei Ebenen plus Forward)
1. **Training:** Nur Zeilen, deren Label **vor** dem Teststart endet (Purge
   gegen überlappende Labels).
2. **Validierung:** Hyperparameter aus dem kleinen, festen Grid werden nur auf
   dem Jahr vor dem Testjahr gewählt (Kriterium: mittlerer Rank-IC).
3. **Walk-Forward-Test:** jährlich, Testjahre 2019 bis zum Locked-Start. Auch die
   Test-Labels müssen vor `locked_from` enden.
4. **Locked-Holdout** ab 2025-07-01: Er wird nie zur Auswahl benutzt. Jede
   Auswertung wird je Modell-id im `ml_registry_log.json` gezählt.
5. **Forward-Shadow:** Wöchentliche Prognosen landen in
   `outputs/research/ml_predictions/`. Gezählt werden nur Prognosen, die nach
   `registered_at` erzeugt wurden. Die realisierten Werte werden nachgetragen.

## Kennzahlen
- **Hauptgröße:** Top-Dezil minus Querschnittsmittel je Wochenkohorte, netto
  10 bp je Seite. Mittelwert und Median, Hit-Rate.
- **Statistik:** t-Wert und annualisierte Sharpe über Monatsmittel (die
  Kohorten überlappen), Max-Drawdown, Anteil positiver Testjahre.
- **Robustheit:** Stress mit 25 bp je Seite, Long-Short-Spread, Rank-IC mit
  t-Wert über Monate, Asymmetrie mean(MFE) / |mean(MAE)| des Top-Dezils
  gegenüber dem Universum.
- **Attribution:**
  - Permutations-Wichtigkeit (IC-Verlust) je Testjahr, aggregiert nach Feature
    und Gruppe;
  - Rank-IC innerhalb jedes Sektors;
  - Nettoergebnis nach VIX-Regime und SPY-Trend.

## Entscheidungsregel (fest, `promotion_criteria`)
Die Empfehlung `promote_recommended` gilt nur, wenn **alle** Bedingungen erfüllt
sind:
- Walk-Forward netto > 0 mit t ≥ 2;
- mindestens 60 % der Testjahre positiv;
- auch bei 25 bp noch > 0;
- Rank-IC > 0 mit t ≥ 2;
- Sharpe über der Benchmark plus 0,10;
- Max-Drawdown nicht schlechter als die Benchmark;
- Locked-Holdout netto > 0;
- mindestens 26 fertige Forward-Kohorten mit netto > 0.

Sonst lautet das Verdikt `running_forward` oder `rejected_so_far`. Auch
`promote_recommended` ist nur eine **Empfehlung**. Champion wird ein Modell erst
per menschlichem PR. Eine spätere Nutzung im Scoring braucht einen weiteren,
eigenen PR mit Begründung.

## Nicht umgesetzt (bewusst)
- **Automatische Promotion:** widerspricht der Governance.
- **Meta-Learning nach Regime** (dynamische Modellgewichte): Das ist erst
  sinnvoll, wenn mindestens zwei Modelle einzeln belegt sind. `factor_monitor`
  prüft Regime-Abhängigkeit bereits und findet derzeit keine.
- **Sektormodelle:** Pro Sektor gibt es nur etwa 20 bis 70 Titel. Das ist zu
  wenig für getrennte Modelle. Die Sektor-Attribution zeigt zuerst, ob sich das
  überhaupt lohnen würde.
