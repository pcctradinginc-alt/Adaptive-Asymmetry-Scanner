# Failure-Analyse (Trade-Gedächtnis)

Fälle gesamt: 116 (Gewinner 37, Verlierer 76). Auswertung über die letzten 100 Verlusttrades je Gruppe.

> Hinweis Stichprobengröße: Die Anteile sind deskriptiv; bei wenigen Dutzend Trades sind Unterschiede zwischen Ursachen statistisch nicht belastbar.
> Hinweis Outcome-Klassen: Nur RELIABLE (Quote-basiert) wird klassifiziert. UNKNOWN (Altbestand ohne Preismethode), RECONSTRUCTED (`delta_approx`) und APPROXIMATED werden nicht weiter klassifiziert und getrennt ausgewiesen.
> Outcome-Klassen (geschlossen): {'UNKNOWN': 70, 'RECONSTRUCTED': 40, 'RELIABLE': 6}

### RELIABLE Verlusttrades (n = 5, mittleres Outcome -0.467)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| structure_decay | 2 | 40 % | -0.460 | 40 % |
| timing_too_early | 2 | 40 % | -0.412 | 40 % |
| signal_wrong | 1 | 20 % | -0.592 | 20 % |
| (low_data_confidence) | 0 | 0 % | – | 40 % |
| (weak_adverse_move) | 0 | 0 % | – | 40 % |

### Nicht-RELIABLE Verlusttrades (UNKNOWN/RECONSTRUCTED/APPROXIMATED, explorativ) (n = 71, mittleres Outcome -0.680)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| unreliable_outcome | 71 | 100 % | -0.680 | 100 % |

### Gewinner vs. Verlierer (nur RELIABLE-Outcomes)

| Feature | Ø Gewinner | Ø Verlierer | Differenz | Effektstärke | n (G/V) |
|---|---:|---:|---:|---:|---:|
| impact | 4.000 | 4.000 | 0.000 | – | 1/5 |
| surprise | 4.000 | 3.000 | 1.000 | 2.68 | 1/5 |
| mismatch | 1.266 | 2.082 | -0.816 | -0.72 | 1/5 |
| z_score | 0.547 | 0.384 | 0.163 | 0.72 | 1/5 |
| sigma_30d | 0.015 | 0.027 | -0.012 | -1.29 | 1/5 |
| price_move_48h | 0.008 | 0.009 | -0.001 | -0.23 | 1/5 |
| eps_drift | 0.000 | 0.008 | -0.008 | -0.54 | 1/5 |
| quick_mc_hit_rate | 0.507 | 0.641 | -0.134 | -2.03 | 1/5 |

Kleine Stichprobe: Anteile sind deskriptiv, keine Signifikanzaussage.
