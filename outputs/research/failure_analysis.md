# Failure-Analyse (Trade-Gedächtnis)

Fälle gesamt: 178 (Gewinner 55, Verlierer 120). Auswertung über die letzten 100 Verlusttrades je Gruppe.

> Hinweis Stichprobengröße: Die Anteile sind deskriptiv; bei wenigen Dutzend Trades sind Unterschiede zwischen Ursachen statistisch nicht belastbar.
> Hinweis unzuverlässige Outcomes: Trades mit `delta_approx` haben nur geschätzte Renditen und werden nicht weiter klassifiziert.

### Zuverlässige Verlusttrades (n = 93, mittleres Outcome -0.494)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| underlying_unknown | 51 | 55 % | -0.396 | 55 % |
| signal_wrong | 24 | 26 % | -0.692 | 26 % |
| timing_too_early | 11 | 12 % | -0.574 | 12 % |
| weak_adverse_move | 5 | 5 % | -0.400 | 17 % |
| exit_gave_back | 1 | 1 % | -0.163 | 1 % |
| structure_decay | 1 | 1 % | -0.738 | 2 % |
| (low_data_confidence) | 0 | 0 % | – | 18 % |
| (regime_changed) | 0 | 0 % | – | 8 % |

### Unzuverlässige Verlusttrades (delta_approx) (n = 27, mittleres Outcome -0.797)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| unreliable_outcome | 27 | 100 % | -0.797 | 100 % |

### Gewinner vs. Verlierer (nur zuverlässige Outcomes)

| Feature | Ø Gewinner | Ø Verlierer | Differenz | Effektstärke | n (G/V) |
|---|---:|---:|---:|---:|---:|
| impact | 5.071 | 4.989 | 0.082 | 0.10 | 42/93 |
| surprise | 4.119 | 3.871 | 0.248 | 0.24 | 42/93 |
| mismatch | 3.524 | 3.349 | 0.176 | 0.10 | 42/93 |
| z_score | 0.309 | 0.328 | -0.019 | -0.06 | 42/93 |
| sigma_30d | 0.027 | 0.030 | -0.003 | -0.23 | 42/93 |
| price_move_48h | 0.012 | 0.012 | 0.000 | 0.01 | 27/73 |
| eps_drift | 0.003 | 0.015 | -0.012 | -0.20 | 42/93 |
| iv_rank | 66.867 | 68.850 | -1.983 | -0.07 | 11/22 |
| quick_mc_hit_rate | 0.669 | 0.619 | 0.050 | 0.42 | 38/89 |

Kleine Stichprobe: Anteile sind deskriptiv, keine Signifikanzaussage.
