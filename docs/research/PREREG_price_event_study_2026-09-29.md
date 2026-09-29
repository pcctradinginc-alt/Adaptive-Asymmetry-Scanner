# Präregistrierung: Historische Volumen-Event-Studie (2026-09-29)

Festgelegt **vor** dem ersten Datenlauf. Änderungen nach Sichtung der Ergebnisse
nur als neue, datierte Version mit Begründung (nie stillschweigend).

## Zweck
Die preisbasierten Annahmen des Scanners (News-Proxy über abnormales Volumen,
Unter- vs. Überreaktion, relative Stärke, Stärke des Volumenschubs) historisch
PIT-sauber prüfen. Die Evidenzbasis ist heute zu klein und selektionsverzerrt
(184 Trades an 52 Tagen).

## Daten und Universum
- Tagesdaten (OHLCV, split-/dividendenbereinigt) über yfinance, 2014-01-01 bis heute.
  Universum: heutige S&P-500- und Nasdaq-100-Liste (`modules/universe.py`).
- **Bekannter Bias:** Survivorship, weil nur die heutigen Mitglieder enthalten sind.
  Long-only-Ergebnisse sind nach oben verzerrt. Maßgeblich sind deshalb
  **marktbereinigte** Renditen (minus SPY im selben Fenster) und das Vorzeichen-
  bzw. Long/Short-Ergebnis.
- Regime: ^VIX (Schlusskurs am Event-Tag), SPY über/unter SMA200 (bis Event-Tag).

## Event (fest)
Tag t mit `Volumen_t ≥ 2,0 × Mittel(Volumen_{t−20..t−1})`, `Close_t ≥ 5 $` und
`Mittel(Dollarvolumen_{t−20..t−1}) ≥ 20 Mio $`. Höchstens ein Event je Ticker in 5
Handelstagen (Cluster-Entdopplung).

## Zeitliche Regeln
Alle Features nutzen Daten bis einschließlich Close_t. Der Entry ist **Open_{t+1}**,
der Exit Close_{t+h} mit h ∈ {1, 5, 20, 60} Handelstagen.
Zielgröße: `r = (Close_{t+h}/Open_{t+1} − 1) − (SPY analog)`.

## Kosten
Basis: 10 bp je Seite (20 bp Round-Trip) für diese liquiden Large Caps.
Sensitivität: 25 bp je Seite. Short-Seite ohne Leihkosten, deshalb ist Long-only
separat ausgewiesen.

## Hypothesen (Richtung vorab festgelegt)
- **H1 Drift:** Die Richtung der Event-Tag-Überrendite (gegen SPY) setzt sich fort.
  Signal: `sign(ev_ret_adj)`. Grundlage sind die Scanner-Prämisse (Unterreaktion)
  und die Literatur zum News-Drift bzw. High-Volume-Return-Premium.
- **H1L Long-only:** nur positive Events, long (der Scanner handelt überwiegend
  Long Calls).
- **H2 Überreaktions-Gate:** Große 2-Tages-Bewegung relativ zu σ
  (`|z2| = |r_2d| / (σ20·√2)`) schwächt die Fortsetzung ab. Prüfung: signierte Rendite
  im unteren gegenüber dem oberen Terzil von |z2|. Die Terzilgrenzen stammen nur
  aus den Trainingsjahren.
- **H3 Relative Stärke:** Bei positiven Events liefert eine positive 35-Tage-Stärke
  gegenüber SPY eine höhere Forward-Rendite (Analogon zum Sektor-Momentum-Gate).
- **H4 Volumenstärke:** Ein höheres relatives Volumen verstärkt die Drift.

## Validierung
- **Walk-Forward** jahresweise: Test-Jahre 2019 bis heute, Training jeweils alle
  Vorjahre ab 2015. H1/H1L haben keine freien Parameter (jedes Jahr ist OOS). Bei
  H2–H4 kommen Terzilgrenzen und Richtung ausschließlich aus dem Training.
- **Statistik:**
  - Signierte Trade-Renditen werden je Entry-Tag gemittelt (Kohorte), dann je
    Monat. Der t-Wert wird über Monate berechnet (Cluster gegen Überlappung).
  - Kennzahlen: Hit-Rate, Mittel und Median je Trade, Sharpe und Sortino
    (Monatskohorten, annualisiert), Max-Drawdown der Monatskohorten-Kurve,
    Anzahl Events und Tage, Turnover (ein Round-Trip je Event).

## Entscheidungsregel (fest)
Eine Hypothese gilt nur als **gestützt**, wenn beim Horizont h = 20 **alle** Kriterien
erfüllt sind:
1. OOS-Mittel netto (Basis-Kosten) > 0 mit t ≥ 2,0.
2. Positiv in ≥ 60 % der Test-Jahre.
3. Gleiches Vorzeichen in beiden VIX-Regimen (VIX < 20 / ≥ 20) und beiden
   Trend-Regimen.
4. Auch bei 25 bp je Seite noch > 0.

Nur gestützte Hypothesen dürfen als **prospektiver Challenger** vorgeschlagen werden.
Promotion in die Produktion bleibt eine menschliche Entscheidung.
