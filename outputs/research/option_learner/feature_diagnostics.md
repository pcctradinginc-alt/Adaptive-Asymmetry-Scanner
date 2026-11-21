# Options-Feature-Diagnose – 2026-11-21

Research- und Observability-Bericht; **keine Modelländerung, keine Alpha-Bestätigung**.

Diagnose: **READ_PATH_CONSISTENT__COHORT_AND_STAGE_COMPOSITION_TO_REVIEW**

| Merkmal | 5-T-Labels | 5-T-Abdeckung | 20-T-Labels | 20-T-Abdeckung |
|---|---:|---:|---:|---:|
| impact | 507/1487 | 0.341 | 445/492 | 0.9045 |
| surprise | 507/1487 | 0.341 | 445/492 | 0.9045 |
| options_skew_ratio | 509/1487 | 0.3423 | 445/492 | 0.9045 |
| options_skew_25d | 504/1487 | 0.3389 | 443/492 | 0.9004 |
| dealer_oi_ratio | 480/1487 | 0.3228 | 416/492 | 0.8455 |

## 5 Tage – 1487 archivverifizierte Labels
Bestandsstatus: REPRODUCED; Reader: CONSISTENT.

| Kandidatengruppe | Labels | surprise | skew_ratio | skew_25d |
|---|---:|---:|---:|---:|
| directional | 507 | 1.0 | 1.0 | 0.9901 |
| research_census | 980 | 0.0 | 0.002 | 0.002 |

| Zeitraum | Labels | surprise | skew_ratio | skew_25d |
|---|---:|---:|---:|---:|
| before_research_census | 446 | 1.0 | 1.0 | 0.9955 |
| from_research_census_start | 1041 | 0.0586 | 0.0605 | 0.0576 |

Quellen-Audit (pro archivverifiziertem Label):

| Merkmal | Kein Feld | Null | Ungültig | Reader-Abweichung |
|---|---:|---:|---:|---:|
| impact | 980 | 0 | 0 | 0 |
| surprise | 980 | 0 | 0 | 0 |
| options_skew_ratio | 978 | 0 | 0 | 0 |
| options_skew_25d | 978 | 5 | 0 | 0 |
| dealer_oi_ratio | 978 | 29 | 0 | 0 |

## 20 Tage – 492 archivverifizierte Labels
Bestandsstatus: REPRODUCED; Reader: CONSISTENT.

| Kandidatengruppe | Labels | surprise | skew_ratio | skew_25d |
|---|---:|---:|---:|---:|
| directional | 445 | 1.0 | 1.0 | 0.9955 |
| research_census | 47 | 0.0 | 0.0 | 0.0 |

| Zeitraum | Labels | surprise | skew_ratio | skew_25d |
|---|---:|---:|---:|---:|
| before_research_census | 443 | 1.0 | 1.0 | 0.9955 |
| from_research_census_start | 49 | 0.0408 | 0.0408 | 0.0408 |

Quellen-Audit (pro archivverifiziertem Label):

| Merkmal | Kein Feld | Null | Ungültig | Reader-Abweichung |
|---|---:|---:|---:|---:|
| impact | 47 | 0 | 0 | 0 |
| surprise | 47 | 0 | 0 | 0 |
| options_skew_ratio | 47 | 0 | 0 | 0 |
| options_skew_25d | 47 | 2 | 0 | 0 |
| dealer_oi_ratio | 47 | 29 | 0 | 0 |

## Interpretation

Genaue Aufschlüsselungen nach Signaltag, Ablehnungsgrund und Optionsstruktur stehen in der begleitenden JSON-Datei. Die Gruppen sind beschreibend und keine kausale Identifikation.
Einzelne Features besitzen im Ledger keine separat belegten Erfassungszeitpunkte: die Point-in-Time-Korrektheit ist damit **nicht unabhängig nachgewiesen**.
Historische Daten, Modelle und Optionen bleiben unverändert.
