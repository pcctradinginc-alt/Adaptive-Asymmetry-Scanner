# Failure-Analyse (Trade-Gedächtnis)

Fälle gesamt: 117 (Gewinner 37, Verlierer 77). Auswertung über die letzten 100 Verlusttrades je Gruppe.

> Hinweis Stichprobengröße: Die Anteile sind deskriptiv; bei wenigen Dutzend Trades sind Unterschiede zwischen Ursachen statistisch nicht belastbar.
> Hinweis Outcome-Klassen: Nur RELIABLE (Quote-basiert) wird klassifiziert. UNKNOWN (Altbestand ohne Preismethode), RECONSTRUCTED (`delta_approx`) und APPROXIMATED werden nicht weiter klassifiziert und getrennt ausgewiesen.
> Outcome-Klassen (geschlossen): {'UNKNOWN': 70, 'RECONSTRUCTED': 40, 'RELIABLE': 7}

### RELIABLE Verlusttrades (n = 6, mittleres Outcome -0.467)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| signal_wrong | 2 | 33 % | -0.530 | 33 % |
| structure_decay | 2 | 33 % | -0.460 | 33 % |
| timing_too_early | 2 | 33 % | -0.412 | 33 % |
| (entry_cost_high) | 0 | 0 % | – | 83 % |
| (low_data_confidence) | 0 | 0 % | – | 33 % |
| (weak_adverse_move) | 0 | 0 % | – | 33 % |

### Nicht-RELIABLE Verlusttrades (UNKNOWN/RECONSTRUCTED/APPROXIMATED, explorativ) (n = 71, mittleres Outcome -0.680)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| unreliable_outcome | 71 | 100 % | -0.680 | 100 % |

### Gewinner vs. Verlierer (nur RELIABLE-Outcomes)

| Feature | Ø Gewinner | Ø Verlierer | Differenz | Effektstärke | n (G/V) |
|---|---:|---:|---:|---:|---:|
| impact | 4.000 | 4.167 | -0.167 | -0.48 | 1/6 |
| surprise | 4.000 | 3.000 | 1.000 | 2.86 | 1/6 |
| mismatch | 1.266 | 2.116 | -0.850 | -0.80 | 1/6 |
| z_score | 0.547 | 0.410 | 0.137 | 0.64 | 1/6 |
| sigma_30d | 0.015 | 0.030 | -0.015 | -1.29 | 1/6 |
| price_move_48h | 0.008 | 0.012 | -0.004 | -0.52 | 1/6 |
| eps_drift | 0.000 | 0.006 | -0.006 | -0.48 | 1/6 |
| quick_mc_hit_rate | 0.507 | 0.647 | -0.140 | -2.17 | 1/6 |

Kleine Stichprobe: Anteile sind deskriptiv, keine Signifikanzaussage.
