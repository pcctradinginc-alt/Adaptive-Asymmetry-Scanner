# Failure-Analyse (Trade-Gedächtnis)

Fälle gesamt: 118 (Gewinner 37, Verlierer 78). Auswertung über die letzten 100 Verlusttrades je Gruppe.

> Hinweis Stichprobengröße: Die Anteile sind deskriptiv; bei wenigen Dutzend Trades sind Unterschiede zwischen Ursachen statistisch nicht belastbar.
> Hinweis Outcome-Klassen: Nur RELIABLE (Quote-basiert) wird klassifiziert. UNKNOWN (Altbestand ohne Preismethode), RECONSTRUCTED (`delta_approx`) und APPROXIMATED werden nicht weiter klassifiziert und getrennt ausgewiesen.
> Outcome-Klassen (geschlossen): {'UNKNOWN': 70, 'RECONSTRUCTED': 40, 'RELIABLE': 8}

### RELIABLE Verlusttrades (n = 7, mittleres Outcome -0.407)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| structure_decay | 3 | 43 % | -0.322 | 43 % |
| signal_wrong | 2 | 29 % | -0.530 | 29 % |
| timing_too_early | 2 | 29 % | -0.412 | 29 % |
| (entry_cost_high) | 0 | 0 % | – | 71 % |
| (low_data_confidence) | 0 | 0 % | – | 29 % |
| (weak_adverse_move) | 0 | 0 % | – | 29 % |
| (regime_changed) | 0 | 0 % | – | 14 % |

### Nicht-RELIABLE Verlusttrades (UNKNOWN/RECONSTRUCTED/APPROXIMATED, explorativ) (n = 71, mittleres Outcome -0.680)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| unreliable_outcome | 71 | 100 % | -0.680 | 100 % |

### Gewinner vs. Verlierer (nur RELIABLE-Outcomes)

| Feature | Ø Gewinner | Ø Verlierer | Differenz | Effektstärke | n (G/V) |
|---|---:|---:|---:|---:|---:|
| impact | 4.000 | 4.143 | -0.143 | -0.43 | 1/7 |
| surprise | 4.000 | 3.000 | 1.000 | 3.02 | 1/7 |
| mismatch | 1.266 | 2.049 | -0.783 | -0.79 | 1/7 |
| z_score | 0.547 | 0.419 | 0.128 | 0.64 | 1/7 |
| sigma_30d | 0.015 | 0.028 | -0.013 | -1.04 | 1/7 |
| price_move_48h | 0.008 | 0.011 | -0.003 | -0.42 | 1/7 |
| eps_drift | 0.000 | 0.006 | -0.006 | -0.43 | 1/7 |
| quick_mc_hit_rate | 0.507 | 0.643 | -0.136 | -2.25 | 1/7 |

Kleine Stichprobe: Anteile sind deskriptiv, keine Signifikanzaussage.
