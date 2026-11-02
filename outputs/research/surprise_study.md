# Surprise Engine – Walk-Forward-Studie (OOS ab 2019)

Events: 21953 · Ticker: 591 · Zeitraum: ['2014-02-06', '2026-11-02']
Präregistrierung: docs/research/PREREG_surprise_engine_2026-10-03.md
Renditen marktbereinigt (minus SPY), netto 10 bp/Seite; PIT-Universum (S&P 500 inkl. entfernter Titel).

| Hypothese | h | n | Mittel | t | Jahre + | 25bp | Lag | p | BH | Placebo p | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_pead | 20 | 13916 | -0.00135 | 0.59166 | 0.375 | -0.00435 | -0.00109 | 0.5541 | nein | None | REJECT |
| S1_pead | 60 | 13849 | -0.00475 | -0.66575 | 0.25 | -0.00775 | -0.00395 | 0.5056 | nein | – |  |
| S1L_pead_long | 20 | 5359 | 0.00037 | 0.12669 | 0.5 | -0.00263 | 0.00117 | 0.8992 | nein | 0.0297 | REJECT |
| S1L_pead_long | 60 | 5357 | -0.00267 | -0.09657 | 0.5 | -0.00567 | -0.00229 | 0.9231 | nein | – |  |
| S2_unpriced | 20 | 9879 | -0.00154 | 0.56789 | 0.375 | -0.00454 | -0.00028 | 0.5701 | nein | None | REJECT |
| S2_unpriced | 60 | 9816 | 0.00199 | 1.60563 | 0.5 | -0.00101 | 0.00328 | 0.1084 | nein | – |  |
| S2L_unpriced_long | 20 | 5356 | -0.00141 | 0.52622 | 0.5 | -0.00441 | -0.00018 | 0.5987 | nein | None | REJECT |
| S2L_unpriced_long | 60 | 5335 | -0.00544 | -0.4155 | 0.5 | -0.00844 | -0.00328 | 0.6778 | nein | – |  |
| S3_reaction_only | 20 | 14581 | -0.00179 | -2.65661 | 0.125 | -0.00479 | -0.00245 | 0.0079 | ja | None | REJECT |
| S3_reaction_only | 60 | 14511 | -0.00346 | -2.85262 | 0.25 | -0.00646 | -0.00406 | 0.0043 | ja | – |  |
| S4_disagreement_long | 20 | 4988 | -0.00139 | 0.59887 | 0.5 | -0.00439 | -0.00025 | 0.5493 | nein | None | REJECT |
| S4_disagreement_long | 60 | 4959 | -0.00628 | -0.45951 | 0.375 | -0.00928 | -0.00507 | 0.6459 | nein | – |  |

## Entscheidungen (h=20)

- **S1_pead** (Richtung der fundamentalen Überraschung setzt sich fort (PEAD), long/short): REJECT – OOS-Mittel/t nicht ausreichend (mean=-0.00135, t=0.59166); nur 0.375 der Jahre positiv; bei 25 bp/Seite nicht positiv; nicht signifikant nach Benjamini-Hochberg (q=0.1); Placebo nicht übertroffen (p=None); Lag-Test (5 T später) nicht positiv; Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv
- **S1L_pead_long** (Oberes Trainings-Terzil der fundamentalen Überraschung, nur long): REJECT – OOS-Mittel/t nicht ausreichend (mean=0.00037, t=0.12669); nur 0.5 der Jahre positiv; bei 25 bp/Seite nicht positiv; nicht signifikant nach Benjamini-Hochberg (q=0.1); Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv
- **S2_unpriced** (Unpriced Surprise (Perzentil Fundamental − Perzentil Reaktion), oberes vs. unteres Terzil): REJECT – OOS-Mittel/t nicht ausreichend (mean=-0.00154, t=0.56789); nur 0.375 der Jahre positiv; bei 25 bp/Seite nicht positiv; nicht signifikant nach Benjamini-Hochberg (q=0.1); Placebo nicht übertroffen (p=None); Lag-Test (5 T später) nicht positiv; Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv
- **S2L_unpriced_long** (Unpriced Surprise oberes Terzil, nur long (produktionsnah: Long Calls)): REJECT – OOS-Mittel/t nicht ausreichend (mean=-0.00141, t=0.52622); nur 0.5 der Jahre positiv; bei 25 bp/Seite nicht positiv; nicht signifikant nach Benjamini-Hochberg (q=0.1); Placebo nicht übertroffen (p=None); Lag-Test (5 T später) nicht positiv; Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv
- **S3_reaction_only** (KONTROLLE: Richtung der Kursreaktion allein (ohne Fundamentaldaten)): REJECT – OOS-Mittel/t nicht ausreichend (mean=-0.00179, t=-2.65661); nur 0.125 der Jahre positiv; bei 25 bp/Seite nicht positiv; Lag-Test (5 T später) nicht positiv; Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv
- **S4_disagreement_long** (Positive Überraschung, aber negative Marktreaktion -> long): REJECT – OOS-Mittel/t nicht ausreichend (mean=-0.00139, t=0.59887); nur 0.5 der Jahre positiv; bei 25 bp/Seite nicht positiv; nicht signifikant nach Benjamini-Hochberg (q=0.1); Placebo nicht übertroffen (p=None); Lag-Test (5 T später) nicht positiv; Replikation (Universumshälften / Zeithälften) nicht durchgehend positiv; nicht in allen Regimen positiv

## Gegenhypothese S5_reaction_reversal – prospektiver Locked Holdout 2026-10-05 bis 2027-10-04

Earnings-Reaktion kehrt sich um: gegen die Richtung der abnormalen Reaktion positionieren (Gegenhypothese zu S3; prospektiver Locked Holdout, Meldungen ab Registrierung)

LOCKED: 289 Meldungen gesammelt; einzige Auswertung ab 2027-11-15 (spec_hash e605109c98ed). Keine Zwischen-Looks.

## Datenlücken (nie simuliert)

- historische implizite Volatilität/Implied Move je Meldung (Optionsarchiv)
- Positionierung (Short-Interest-Historie, Dealer-Gamma historisch)
- Prediction Markets (keine kostenlose PIT-Historie)
- PIT-Sektorzuordnung (Sektor-Analyse nur mit heutiger Zuordnung, beschreibend)
- Konsens-Erwartung EIA-Rohöllager/Gasspeicher (keine offizielle kostenlose Quelle) -> storage_surprise EXPECTATION_UNKNOWN; Commodity-Abweichungen gehen nur als Hypothesen in die Fabrik (commodity_intelligence)
