# Expectation Alpha – SHADOW – Research, keine Handelsempfehlung

Stand 2026-10-15T22:33:08+00:00 · config 9772eff4a0fef81d

Population EA_NEWS_CANDIDATE: n=148, Signaltage 2, Ereignis-Cluster 127, Status {'ABSTAIN': 139, 'WAIT': 9}

## NEWS × CONTEXT (Underlying, netto, 60 Handelstage)

| Gruppe | n | Status | Mittel | Median | Treffer | MAE | MFE | CI95 |
|---|---|---|---|---|---|---|---|---|
| A | 0 | NEED_MORE_DATA | – | – | – | – | – | – |
| B | 0 | NEED_MORE_DATA | – | – | – | – | – | – |
| C | 0 | NEED_MORE_DATA | – | – | – | – | – | – |
| D | 0 | NEED_MORE_DATA | – | – | – | – | – | – |
| E | 0 | NEED_MORE_DATA | – | – | – | – | – | – |
| X | 0 | NEED_MORE_DATA | – | – | – | – | – | – |

## Werte (gepaart, netto)

- abstention: Δ –
- wait: n=0 NEED_MORE_DATA Δ –, ohne Einstieg None
- expression.thesis_quality: n=0 NEED_MORE_DATA Mittel –
- expression.expression_quality: n=0 NEED_MORE_DATA Mittel –
- expression.rule_vs_default: n=0 NEED_MORE_DATA Mittel –
- kill_management: n=0 NEED_MORE_DATA Δ –

## Verträge EA001–EA007

| Vertrag | Zustand | n | Tage | Spanne | Regime | Δ | CI | nächste Anforderung |
|---|---|---|---|---|---|---|---|---|
| EA001_EXPECTATION_GAP@v1 | PROSPECTIVE_CHALLENGER | 0 | 0 | 0 | 0 | – | [None, None] | n 0/100; unabhängige Tage 0/40; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Cluster 0/20 |
| EA002_CONFIRMATION@v1 | PROSPECTIVE_CHALLENGER | 0 | 0 | 0 | 0 | – | [None, None] | n 0/100; unabhängige Tage 0/40; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Cluster 0/20 |
| EA003_ACCELERATION@v1 | PROSPECTIVE_CHALLENGER | 0 | 0 | 0 | 0 | – | [None, None] | n 0/100; unabhängige Tage 0/40; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Cluster 0/20 |
| EA004_WAIT@v1 | PROSPECTIVE_CHALLENGER | 0 | 0 | 0 | 0 | – | [None, None] | n 0/60; unabhängige Tage 0/30; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/40; Treffer-Cluster 0/20 |
| EA005_CONTEXT_FILTER@v1 | PROSPECTIVE_CHALLENGER | 0 | 0 | 0 | 0 | – | [None, None] | n 0/100; unabhängige Tage 0/40; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Cluster 0/20 |
| EA006_ABSTENTION@v1 | PROSPECTIVE_CHALLENGER | 0 | 0 | 0 | 0 | – | [None, None] | n 0/100; unabhängige Tage 0/40; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/60; Treffer-Cluster 0/20 |
| EA007_EXPRESSION@v1 | PROSPECTIVE_CHALLENGER | 0 | 0 | 0 | 0 | – | [None, None] | n 0/60; unabhängige Tage 0/30; Kalenderspanne 0/180 T; Regel-Treffer 0/30; Treffer-Tage 0/15; nur 0 Marktregime beobachtet; Ereignis-Cluster 0/40; Treffer-Cluster 0/20 |

## Lead-Lag-Diagnostik (Research, Horizonte vorab: [20, 60, 120, 250], keine Auswahl) – Status NEED_MORE_DATA

| Feature | Familie | IC 20T (n) | IC 60T (n) | IC 120T (n) | IC 250T (n) |
|---|---|---|---|---|---|
| macro_alignment | expectation_gap | – (0) | – (0) | – (0) | – (0) |
| gap_abs_z | expectation_gap | – (0) | – (0) | – (0) | – (0) |
| gap_aligned | expectation_gap | – (0) | – (0) | – (0) | – (0) |
| gap_percentile | percentile | – (0) | – (0) | – (0) | – (0) |
| gap_acceleration_aligned | rate_of_change | – (0) | – (0) | – (0) | – (0) |
| sector_alignment | relative | – (0) | – (0) | – (0) | – (0) |
| confirmation_ratio_raw | cross_asset | – (0) | – (0) | – (0) | – (0) |
| confirmation_ratio_effective | cross_asset | – (0) | – (0) | – (0) | – (0) |
| confirmation_family_count | cross_asset | – (0) | – (0) | – (0) | – (0) |
| regime_uncertainty | regime | – (0) | – (0) | – (0) | – (0) |
| regime_transition_probability | regime_transition | – (0) | – (0) | – (0) | – (0) |
| verified_claim_fraction | claims | – (0) | – (0) | – (0) | – (0) |
| news_impact | news | – (0) | – (0) | – (0) | – (0) |
| news_surprise | news | – (0) | – (0) | – (0) | – (0) |

Fehlerklassen (Shadow-TRADE): – · je Status: –

Hinweis: SHADOW. Keine Handelsempfehlung, kein Einfluss auf Champion, Scores, Gates oder Sizing.
