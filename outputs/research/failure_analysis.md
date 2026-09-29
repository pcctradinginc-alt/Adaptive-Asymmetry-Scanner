# Failure-Analyse (Trade-Gedächtnis)

Fälle gesamt: 177 (Gewinner 55, Verlierer 119). Auswertung über die letzten 100 Verlusttrades je Gruppe.

> Hinweis Stichprobengröße: Die Anteile sind deskriptiv; bei wenigen Dutzend Trades sind Unterschiede zwischen Ursachen statistisch nicht belastbar.
> Hinweis unzuverlässige Outcomes: Trades mit `delta_approx` haben nur geschätzte Renditen und werden nicht weiter klassifiziert.

### Zuverlässige Verlusttrades (n = 92, mittleres Outcome -0.489)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| underlying_unknown | 52 | 57 % | -0.395 | 57 % |
| signal_wrong | 23 | 25 % | -0.696 | 25 % |
| timing_too_early | 11 | 12 % | -0.574 | 12 % |
| weak_adverse_move | 5 | 5 % | -0.400 | 17 % |
| exit_gave_back | 1 | 1 % | -0.163 | 1 % |
| (low_data_confidence) | 0 | 0 % | – | 17 % |
| (regime_changed) | 0 | 0 % | – | 8 % |
| (structure_decay) | 0 | 0 % | – | 1 % |

### Unzuverlässige Verlusttrades (delta_approx) (n = 27, mittleres Outcome -0.797)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| unreliable_outcome | 27 | 100 % | -0.797 | 100 % |

### Gewinner vs. Verlierer (nur zuverlässige Outcomes)

| Feature | Ø Gewinner | Ø Verlierer | Differenz | Effektstärke | n (G/V) |
|---|---:|---:|---:|---:|---:|
| impact | 5.071 | 5.011 | 0.061 | 0.08 | 42/92 |
| surprise | 4.119 | 3.880 | 0.239 | 0.23 | 42/92 |
| mismatch | 3.524 | 3.365 | 0.159 | 0.09 | 42/92 |
| z_score | 0.309 | 0.329 | -0.020 | -0.06 | 42/92 |
| sigma_30d | 0.027 | 0.030 | -0.003 | -0.23 | 42/92 |
| price_move_48h | 0.012 | 0.012 | 0.000 | 0.00 | 27/72 |
| eps_drift | 0.003 | 0.016 | -0.012 | -0.20 | 42/92 |
| iv_rank | 66.867 | 68.850 | -1.983 | -0.07 | 11/22 |
| quick_mc_hit_rate | 0.669 | 0.617 | 0.052 | 0.43 | 38/88 |

Kleine Stichprobe: Anteile sind deskriptiv, keine Signifikanzaussage.
