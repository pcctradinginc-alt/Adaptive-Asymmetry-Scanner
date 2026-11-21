# Failure-Analyse (Trade-Gedächtnis)

Fälle gesamt: 194 (Gewinner 53, Verlierer 138). Auswertung über die letzten 100 Verlusttrades je Gruppe.

> Hinweis Stichprobengröße: Die Anteile sind deskriptiv; bei wenigen Dutzend Trades sind Unterschiede zwischen Ursachen statistisch nicht belastbar.
> Hinweis Outcome-Klassen: Nur RELIABLE (Quote-basiert) wird klassifiziert. UNKNOWN (Altbestand ohne Preismethode), RECONSTRUCTED (`delta_approx`) und APPROXIMATED werden nicht weiter klassifiziert und getrennt ausgewiesen.
> Outcome-Klassen (geschlossen): {'UNKNOWN': 70, 'RECONSTRUCTED': 40, 'RELIABLE': 8}

### RELIABLE Verlusttrades (n = 38, mittleres Outcome -0.660)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| underlying_unknown | 31 | 82 % | -0.718 | 82 % |
| structure_decay | 3 | 8 % | -0.322 | 8 % |
| signal_wrong | 2 | 5 % | -0.530 | 5 % |
| timing_too_early | 2 | 5 % | -0.412 | 5 % |
| (regime_changed) | 0 | 0 % | – | 55 % |
| (entry_cost_high) | 0 | 0 % | – | 26 % |
| (low_data_confidence) | 0 | 0 % | – | 5 % |
| (weak_adverse_move) | 0 | 0 % | – | 5 % |

### Nicht-RELIABLE Verlusttrades (UNKNOWN/RECONSTRUCTED/APPROXIMATED, explorativ) (n = 100, mittleres Outcome -0.552)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| unreliable_outcome | 100 | 100 % | -0.552 | 100 % |

### Gewinner vs. Verlierer (nur RELIABLE-Outcomes)

| Feature | Ø Gewinner | Ø Verlierer | Differenz | Effektstärke | n (G/V) |
|---|---:|---:|---:|---:|---:|
| impact | 4.556 | 4.342 | 0.213 | 0.38 | 9/38 |
| surprise | 3.333 | 3.132 | 0.202 | 0.47 | 9/38 |
| mismatch | 2.107 | 2.197 | -0.090 | -0.07 | 9/38 |
| z_score | 0.490 | 0.429 | 0.061 | 0.22 | 9/38 |
| sigma_30d | 0.022 | 0.031 | -0.010 | -0.89 | 9/38 |
| price_move_48h | 0.011 | 0.014 | -0.003 | -0.33 | 9/38 |
| eps_drift | 0.000 | 0.001 | -0.001 | -0.18 | 9/38 |
| quick_mc_hit_rate | 0.535 | 0.590 | -0.055 | -0.82 | 9/38 |

Kleine Stichprobe: Anteile sind deskriptiv, keine Signifikanzaussage.
