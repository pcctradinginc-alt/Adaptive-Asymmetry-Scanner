# Gate- und Optionsökonomie – 2026-11-12

**READ-ONLY SHADOW – keine Handels- oder Gate-Freigabe.**

## Economics-first: letzte verfügbare Läufe

| Datum | Sektor | Quote-Probes | Budget-untersucht: nein | Earnings-Reject | Haiku | Sonnet | Quick-MC | ROI OK | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-11-12 | 208 | 8 | 200 | 0 | 8 | 3 | 1 | 0 | 0 |
| 2026-11-11 | 173 | 8 | 165 | 1 | 7 | 3 | 1 | 0 | 0 |
| 2026-11-10 | 175 | 8 | 167 | 1 | 7 | 2 | 1 | 0 | 0 |
| 2026-11-09 | 145 | 8 | 137 | 2 | 6 | 3 | 1 | 0 | 0 |
| 2026-11-06 | 132 | 8 | 124 | 4 | 3 | 2 | 1 | 0 | 0 |
| 2026-11-05 | 366 | 8 | 358 | 2 | 5 | 3 | 0 | 0 | 0 |
| 2026-11-04 | 378 | 8 | 370 | 1 | 5 | 1 | 0 | 0 | 0 |
| 2026-11-03 | 140 | 8 | 132 | 3 | 5 | 1 | 0 | 0 | 0 |
| 2026-11-02 | 123 | 8 | 115 | 2 | 6 | 1 | 0 | 0 | 0 |
| 2026-10-30 | 291 | 8 | 283 | 0 | 8 | 2 | 0 | 0 | 0 |

Budget-bedingt nicht untersuchte Kandidaten sind **keine** nachgewiesenen
ökonomischen Rejects. Für unbepreiste Aktien existieren keine realen Options-Outcomes.

## ROI-Reject-Shadow: echte RTH-Bid-/Ask-Marks

| Horizont (Kalendertage) | Registriert | Markiert | Voll ausgereift | Davon markiert | Quote-Abdeckung | Ø Netto* | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| 5 | 0 | 0 | 0 | 0 | n/v | n/v | NEED_MORE_MATURED_QUOTES |
| 20 | 0 | 0 | 0 | 0 | n/v | n/v | NEED_MORE_MATURED_QUOTES |
| 45 | 0 | 0 | 0 | 0 | n/v | n/v | NEED_MORE_MATURED_QUOTES |

*Netto-Mittelwerte nur ab 20 verwertbaren Marks an mindestens 10 unabhängigen
Signaltagen. Rein deskriptiv, kein Alpha-Nachweis.
Voll ausgereift: Signalzeitpunkt plus Horizont, maximal zulässiger
Entry-Lag und maximal zulässiger Exit-Lag.

## Reject-Gründe (Gruppen können sich überlappen)

| Gate | Registrierte Kontrakte | 5d markiert | 20d markiert | 45d markiert |
|---|---:|---:|---:|---:|

Fest definierte Marks sind nicht mit Take-Profit-/Stop-Loss-Exits
gleichzusetzen. Unbeobachtete Quotes bleiben UNKNOWN.
Es erfolgt keine Änderung an Champion, Gates, Sizing, PPO oder Promotion.


## Kandidaten-Forward-Renditen (Aktien, nicht Optionen)

| Ablehnungsgrund | Signale | Richtung bekannt | Aktienmark 5d | Ø 5d* | Aktienmark 20d | Ø 20d* |
|---|---:|---:|---:|---:|---:|---:|
| prescreen_no | 3941 | 0 | 3915 | 0.5% | 3309 | 2.1% |
| economics_first_quote_probe_budget | 2768 | 0 | 2099 | n/v | 0 | n/v |
| sector_momentum_weak | 972 | 1 | 845 | 0.8% | 467 | 2.8% |
| earnings_gate | 439 | 43 | 439 | -0.3% | 281 | 0.5% |
| mismatch_below_min | 347 | 347 | 343 | 0.3% | 294 | -0.5% |
| bearish_disabled | 313 | 313 | 310 | -0.1% | 286 | 0.7% |
| time_budget_exceeded | 253 | 0 | 252 | n/v | 252 | n/v |
| pre_mc_sigma_too_low | 204 | 0 | 204 | 1.0% | 189 | 6.8% |
| options_design_roi_or_edge | 168 | 168 | 165 | 0.2% | 145 | -2.5% |
| deep_analysis_veto_or_invalid | 162 | 0 | 162 | -0.4% | 144 | -3.8% |
| mc_below_threshold | 101 | 101 | 100 | 1.0% | 90 | 5.0% |
| prescreen_bearish_prefilter | 39 | 0 | 39 | n/v | 0 | n/v |
| impact_x_surprise_below_floor | 36 | 36 | 36 | 0.1% | 30 | 2.0% |
| economics_first_earnings_gate | 22 | 0 | 18 | n/v | 0 | n/v |
| unlabeled_after_options_design | 20 | 20 | 20 | n/v | 20 | n/v |

*Mittelwerte erst ab 20 Marks an 10 unabhängigen Signaltagen; nicht kausal.
Richtungsunbekannte Budget-Signale sind keine Long-Call-Prognosen.
Aktienrenditen dürfen nicht als verpasste Optionsgewinne gelten.