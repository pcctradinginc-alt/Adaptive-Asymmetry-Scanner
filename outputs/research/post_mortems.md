# Post-mortems (jüngste geschlossene Trades)

Deterministisch aus trade_memory (feste Regeln, keine Signifikanzaussage). Ähnliche Fälle: nur solche,
die vor dem Entry bereits abgeschlossen waren.

## MO · 2026-09-24 → 2026-10-16 · WIN 51 %
- Strategie LONG_CALL, Haltedauer 22 T, Exit: take_profit, Outcome verlässlich: ja
- Underlying (signiert) 8 %, Option/Underlying 6.22, Zwischenhoch 51 %
- Vorhersage: MC-Trefferquote 0.613, Impact 4.0
- Ursache: — 
- Lehre: Gewinn mit Underlying in Signalrichtung – Signal und Struktur trugen.
- Ähnliche frühere Fälle: n=10, Gewinnanteil 20 %, Median -18 %

## AMZN · 2026-08-31 → 2026-10-15 · LOSS -32 %
- Strategie BULL_CALL_SPREAD, Haltedauer 45 T, Exit: max_holding_period, Outcome verlässlich: ja
- Underlying (signiert) -2 %, Option/Underlying 17.69, Zwischenhoch -20 %
- Vorhersage: MC-Trefferquote 0.724, Impact 4.0
- Ursache: timing_too_early ['timing_too_early', 'weak_adverse_move']
- Lehre: Schwache Gegenbewegung, erwarteter Zeithorizont länger als die Haltedauer – zu früh.
- Ähnliche frühere Fälle: n=10, Gewinnanteil 40 %, Median -30 %

## KDP · 2026-10-15 → 2026-10-15 · LOSS -18 %
- Strategie BULL_CALL_SPREAD, Haltedauer 0 T, Exit: stop_loss, Outcome verlässlich: ja
- Underlying (signiert) 1 %, Option/Underlying -16.61, Zwischenhoch -24 %
- Vorhersage: MC-Trefferquote 0.741, Impact 4.0
- Ursache: structure_decay ['structure_decay']
- Lehre: Underlying in Signalrichtung, Option trotzdem im Verlust – Zeitwert-/IV-Verfall der Struktur.
- Ähnliche frühere Fälle: n=10, Gewinnanteil 30 %, Median -36 %

## SNPS · 2026-10-06 → 2026-10-08 · LOSS -51 %
- Strategie BULL_CALL_SPREAD, Haltedauer 2 T, Exit: stop_loss, Outcome verlässlich: ja
- Underlying (signiert) -2 %, Option/Underlying 30.39, Zwischenhoch -38 %
- Vorhersage: MC-Trefferquote 0.693, Impact 4.0
- Ursache: timing_too_early ['timing_too_early', 'weak_adverse_move', 'low_data_confidence']
- Lehre: Schwache Gegenbewegung, erwarteter Zeithorizont länger als die Haltedauer – zu früh.
- Ähnliche frühere Fälle: n=10, Gewinnanteil 10 %, Median -61 %

## ADM · 2026-09-21 → 2026-09-30 · LOSS -59 %
- Strategie BULL_CALL_SPREAD, Haltedauer 9 T, Exit: stop_loss, Outcome verlässlich: ja
- Underlying (signiert) -5 %, Option/Underlying 10.86, Zwischenhoch -13 %
- Vorhersage: MC-Trefferquote 0.684, Impact 4.0
- Ursache: signal_wrong ['signal_wrong', 'low_data_confidence']
- Lehre: Underlying lief deutlich gegen die Signalrichtung – das Signal war falsch.
- Ähnliche frühere Fälle: n=10, Gewinnanteil 20 %, Median -57 %

## BBY · 2026-09-02 → 2026-09-29 · LOSS -74 %
- Strategie BULL_CALL_SPREAD, Haltedauer 27 T, Exit: stop_loss, Outcome verlässlich: ja
- Underlying (signiert) 1 %, Option/Underlying -50.04, Zwischenhoch 19 %
- Vorhersage: MC-Trefferquote 0.750, Impact 4.0
- Ursache: structure_decay ['structure_decay']
- Lehre: Underlying in Signalrichtung, Option trotzdem im Verlust – Zeitwert-/IV-Verfall der Struktur.
- Ähnliche frühere Fälle: n=10, Gewinnanteil 50 %, Median 23 %

## APO · 2026-08-28 → 2026-09-23 · LOSS -51 %
- Strategie BULL_CALL_SPREAD, Haltedauer 26 T, Exit: stop_loss, Outcome verlässlich: nein
- Underlying (signiert) -8 %, Option/Underlying 6.40, Zwischenhoch -20 %
- Vorhersage: MC-Trefferquote 0.723, Impact 4.0
- Ursache: unreliable_outcome ['unreliable_outcome']
- Lehre: Outcome nicht verlässlich bewertet – keine Lernaussage möglich.
- Ähnliche frühere Fälle: n=10, Gewinnanteil 20 %, Median -57 %

## ABNB · 2026-08-07 → 2026-09-21 · LOSS -35 %
- Strategie BULL_CALL_SPREAD, Haltedauer 45 T, Exit: max_holding_period, Outcome verlässlich: nein
- Underlying (signiert) -5 %, Option/Underlying 7.59, Zwischenhoch 23 %
- Vorhersage: MC-Trefferquote 0.750, Impact 5.0
- Ursache: unreliable_outcome ['unreliable_outcome']
- Lehre: Outcome nicht verlässlich bewertet – keine Lernaussage möglich.
- Ähnliche frühere Fälle: n=10, Gewinnanteil 20 %, Median -65 %

## ALL · 2026-08-06 → 2026-09-10 · LOSS -50 %
- Strategie BULL_CALL_SPREAD, Haltedauer 35 T, Exit: stop_loss, Outcome verlässlich: nein
- Underlying (signiert) -8 %, Option/Underlying 6.61, Zwischenhoch -6 %
- Vorhersage: MC-Trefferquote 0.720, Impact 6.0
- Ursache: unreliable_outcome ['unreliable_outcome']
- Lehre: Outcome nicht verlässlich bewertet – keine Lernaussage möglich.
- Ähnliche frühere Fälle: n=10, Gewinnanteil 10 %, Median -100 %

## AAPL · 2026-07-15 → 2026-08-29 · LOSS -16 %
- Strategie BULL_CALL_SPREAD, Haltedauer 45 T, Exit: max_holding_period, Outcome verlässlich: nein
- Underlying (signiert) -1 %, Option/Underlying 12.38, Zwischenhoch 21 %
- Vorhersage: MC-Trefferquote 0.675, Impact 4.0
- Ursache: unreliable_outcome ['unreliable_outcome']
- Lehre: Outcome nicht verlässlich bewertet – keine Lernaussage möglich.
- Ähnliche frühere Fälle: n=10, Gewinnanteil 20 %, Median -31 %

