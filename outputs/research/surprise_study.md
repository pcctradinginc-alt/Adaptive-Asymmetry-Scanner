# Surprise Engine – Walk-Forward-Studie (OOS ab 2019)

Events: 21917 · Ticker: 588 · Zeitraum: ['2014-02-06', '2026-10-02']
Präregistrierung: docs/research/PREREG_surprise_engine_2026-10-03.md
Renditen marktbereinigt (minus SPY), netto 10 bp/Seite; PIT-Universum (S&P 500 inkl. entfernter Titel).

| Hypothese | h | n | Mittel | t | Jahre + | 25bp | Lag | p | BH | Placebo p | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_pead | 20 | 13930 | -0.00137 | 0.34676 | 0.375 | -0.00437 | -0.00109 | 0.7288 | nein | None | REJECT |
| S1_pead | 60 | 13471 | -0.00429 | -0.70526 | 0.25 | -0.00729 | -0.00374 | 0.4806 | nein | – |  |
| S1L_pead_long | 20 | 5367 | 0.00037 | 0.06234 | 0.5 | -0.00263 | 0.00128 | 0.9503 | nein | 0.0693 | REJECT |
| S1L_pead_long | 60 | 5215 | -0.00148 | -0.28194 | 0.625 | -0.00448 | -0.00137 | 0.778 | nein | – |  |
| S2_unpriced | 20 | 9880 | -0.00147 | 0.81574 | 0.375 | -0.00447 | -0.0002 | 0.4146 | nein | None | REJECT |
| S2_unpriced | 60 | 9542 | 0.00211 | 0.91461 | 0.5 | -0.00089 | 0.00351 | 0.3604 | nein | – |  |
| S2L_unpriced_long | 20 | 5361 | -0.00135 | 0.60091 | 0.5 | -0.00435 | -9e-05 | 0.5479 | nein | None | REJECT |
| S2L_unpriced_long | 60 | 5174 | -0.0051 | -0.79189 | 0.5 | -0.0081 | -0.00296 | 0.4284 | nein | – |  |
| S3_reaction_only | 20 | 14595 | -0.0018 | -2.58993 | 0.25 | -0.0048 | -0.00247 | 0.0096 | ja | None | REJECT |
| S3_reaction_only | 60 | 14123 | -0.00334 | -2.54366 | 0.25 | -0.00634 | -0.00405 | 0.011 | ja | – |  |
| S4_disagreement_long | 20 | 4993 | -0.00141 | 0.6567 | 0.5 | -0.00441 | -0.00025 | 0.5114 | nein | None | REJECT |
| S4_disagreement_long | 60 | 4792 | -0.00588 | -0.77872 | 0.375 | -0.00888 | -0.00488 | 0.4361 | nein | – |  |

## Entscheidungen (h=20)

- **S1_pead** (Richtung der fundamentalen Überraschung setzt sich fort (PEAD), long/short): REJECT – OOS-Mittel/t nicht ausreichend (mean=-0.00137, t=0.34676); nur 0.375 der Jahre positiv; bei 25 bp/Seite nicht positiv; nicht signifikant nach Benjamini-Hochberg (q=0.1); Placebo nicht übertroffen (p=None); Lag-Test (5 T später) nicht positiv; Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv
- **S1L_pead_long** (Oberes Trainings-Terzil der fundamentalen Überraschung, nur long): REJECT – OOS-Mittel/t nicht ausreichend (mean=0.00037, t=0.06234); nur 0.5 der Jahre positiv; bei 25 bp/Seite nicht positiv; nicht signifikant nach Benjamini-Hochberg (q=0.1); Placebo nicht übertroffen (p=0.0693); Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv
- **S2_unpriced** (Unpriced Surprise (Perzentil Fundamental − Perzentil Reaktion), oberes vs. unteres Terzil): REJECT – OOS-Mittel/t nicht ausreichend (mean=-0.00147, t=0.81574); nur 0.375 der Jahre positiv; bei 25 bp/Seite nicht positiv; nicht signifikant nach Benjamini-Hochberg (q=0.1); Placebo nicht übertroffen (p=None); Lag-Test (5 T später) nicht positiv; Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv
- **S2L_unpriced_long** (Unpriced Surprise oberes Terzil, nur long (produktionsnah: Long Calls)): REJECT – OOS-Mittel/t nicht ausreichend (mean=-0.00135, t=0.60091); nur 0.5 der Jahre positiv; bei 25 bp/Seite nicht positiv; nicht signifikant nach Benjamini-Hochberg (q=0.1); Placebo nicht übertroffen (p=None); Lag-Test (5 T später) nicht positiv; Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv
- **S3_reaction_only** (KONTROLLE: Richtung der Kursreaktion allein (ohne Fundamentaldaten)): REJECT – OOS-Mittel/t nicht ausreichend (mean=-0.0018, t=-2.58993); nur 0.25 der Jahre positiv; bei 25 bp/Seite nicht positiv; Lag-Test (5 T später) nicht positiv; Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv
- **S4_disagreement_long** (Positive Überraschung, aber negative Marktreaktion -> long): REJECT – OOS-Mittel/t nicht ausreichend (mean=-0.00141, t=0.6567); nur 0.5 der Jahre positiv; bei 25 bp/Seite nicht positiv; nicht signifikant nach Benjamini-Hochberg (q=0.1); Placebo nicht übertroffen (p=None); Lag-Test (5 T später) nicht positiv; Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv

## Datenlücken (nie simuliert)

- historische implizite Volatilität/Implied Move je Meldung (Optionsarchiv)
- Positionierung (Short-Interest-Historie, Dealer-Gamma historisch)
- Prediction Markets (keine kostenlose PIT-Historie)
- PIT-Sektorzuordnung (Sektor-Analyse nur mit heutiger Zuordnung, beschreibend)
