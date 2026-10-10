# Failure-Analyse (Trade-Gedächtnis)

Fälle gesamt: 113 (Gewinner 36, Verlierer 74). Auswertung über die letzten 100 Verlusttrades je Gruppe.

> Hinweis Stichprobengröße: Die Anteile sind deskriptiv; bei wenigen Dutzend Trades sind Unterschiede zwischen Ursachen statistisch nicht belastbar.
> Hinweis Outcome-Klassen: Nur RELIABLE (Quote-basiert) wird klassifiziert. UNKNOWN (Altbestand ohne Preismethode), RECONSTRUCTED (`delta_approx`) und APPROXIMATED werden nicht weiter klassifiziert und getrennt ausgewiesen.
> Outcome-Klassen (geschlossen): {'UNKNOWN': 77, 'RECONSTRUCTED': 40, 'RELIABLE': 3}

### RELIABLE Verlusttrades (n = 3, mittleres Outcome -0.613)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| signal_wrong | 1 | 33 % | -0.592 | 33 % |
| structure_decay | 1 | 33 % | -0.738 | 33 % |
| timing_too_early | 1 | 33 % | -0.508 | 33 % |
| (low_data_confidence) | 0 | 0 % | – | 67 % |
| (weak_adverse_move) | 0 | 0 % | – | 33 % |

### Nicht-RELIABLE Verlusttrades (UNKNOWN/RECONSTRUCTED/APPROXIMATED, explorativ) (n = 71, mittleres Outcome -0.680)

| Primäre Ursache | n | Anteil | mittl. Outcome | Label-Anteil (Multi-Label) |
|---|---:|---:|---:|---:|
| unreliable_outcome | 71 | 100 % | -0.680 | 100 % |

### Gewinner vs. Verlierer (nur RELIABLE-Outcomes)

Keine vergleichbaren Features.

Kleine Stichprobe: Anteile sind deskriptiv, keine Signifikanzaussage.
