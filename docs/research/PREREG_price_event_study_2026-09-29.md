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

---
## Ergebnis (Lauf 2026-09-29 08:47 UTC, unverändert gegenüber Präregistrierung)

Datenbasis: 514 von 514 Tickern geladen, 42.869 Events (2014-02 bis 2026-09), Test-Jahre
2019–2026. Details in `outputs/research/price_event_study.{json,md}`.

**Keine Hypothese gestützt.** Werte bei h = 20, marktbereinigt:

| Hypothese | netto (10 bp/Seite) | brutto | t (Monate) | Jahre positiv |
|---|---|---|---|---|
| H1 Drift (Vorzeichen des Event-Tags) | −0,35 % | −0,15 % | −2,4 | 1/8 |
| H1L Long nach positivem Event | −0,23 % | −0,03 % | −0,6 | 1/8 |
| H2 kleine Bewegung (Gate lässt passieren) | −0,47 % | −0,27 % | −3,6 | 0/8 |
| H2 große Bewegung (Gate verwirft) | −0,06 % | +0,14 % | −0,9 | 3/8 |
| H3 positive relative Stärke | −0,16 % | +0,05 % | −1,1 | 4/8 |
| H4 hohes relatives Volumen | −0,39 % | −0,19 % | −2,3 | 0/8 |

Das Vorzeichen ist über Jahre, VIX- und Trend-Regime konsistent. Die Brutto-Effekte sind
winzig, die Kosten bestimmen das Ergebnis.

**Schlussfolgerungen:**
1. Die Unterreaktions-/Drift-Prämisse des Scanners ist **aus Preisen allein nicht
   belegt**. Nach Volumen-Events folgt eher eine leichte Umkehr.
2. Die Annahme des Überreaktions-Gates („kleine Bewegung = noch nicht eingepreist =
   besser“) ist historisch **umgekehrt**. Das gilt für die Event-Richtung; die
   Produktion nutzt die LLM-Richtung, deshalb gibt es **keine Produktionsänderung**
   ohne direkten Beleg.
3. Die Umkehr als Strategie ist **nicht** präregistriert (post hoc) und brutto zu
   klein für 20 bp Kosten. Sie wird nicht verfolgt.
4. Folge: Wert kann nur aus der LLM-News-Beurteilung stammen. Das wird jetzt
   prospektiv und gepaart gemessen (`factor_monitor.llm_value_add`: LLM-Richtung
   gegen die naive Preis-Baseline `sign(scan_day_ret)` auf denselben Ledger-Zeilen).
