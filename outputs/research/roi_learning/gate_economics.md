# Gate- und Optionsökonomie – 2026-11-18

**READ-ONLY SHADOW – keine Handels- oder Gate-Freigabe.**

## Economics-first: letzte verfügbare Läufe

| Datum | Sektor | Quote-Probes | Budget-untersucht: nein | Earnings-Reject | Haiku | Sonnet | Quick-MC | ROI OK | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-11-17 | 95 | 8 | 87 | 2 | 5 | 1 | 0 | 0 | 0 |
| 2026-11-16 | 43 | 8 | 35 | 1 | 6 | 1 | 1 | 0 | 0 |
| 2026-11-13 | 80 | 8 | 72 | 0 | 8 | 2 | 0 | 0 | 0 |
| 2026-11-12 | 208 | 8 | 200 | 0 | 8 | 3 | 1 | 0 | 0 |
| 2026-11-11 | 173 | 8 | 165 | 1 | 7 | 3 | 1 | 0 | 0 |
| 2026-11-10 | 175 | 8 | 167 | 1 | 7 | 2 | 1 | 0 | 0 |
| 2026-11-09 | 145 | 8 | 137 | 2 | 6 | 3 | 1 | 0 | 0 |
| 2026-11-06 | 132 | 8 | 124 | 4 | 3 | 2 | 1 | 0 | 0 |
| 2026-11-05 | 366 | 8 | 358 | 2 | 5 | 3 | 0 | 0 | 0 |
| 2026-11-04 | 378 | 8 | 370 | 1 | 5 | 1 | 0 | 0 | 0 |

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
| prescreen_no | 3954 | 0 | 3930 | 0.5% | 3893 | 1.9% |
| economics_first_quote_probe_budget | 2962 | 0 | 2768 | -0.4% | 350 | n/v |
| sector_momentum_weak | 1050 | 1 | 972 | 0.3% | 603 | 2.0% |
| earnings_gate | 439 | 43 | 439 | -0.3% | 439 | -0.1% |
| mismatch_below_min | 349 | 349 | 347 | 0.3% | 339 | -0.5% |
| bearish_disabled | 314 | 314 | 312 | -0.1% | 303 | 0.5% |
| time_budget_exceeded | 253 | 0 | 252 | n/v | 252 | n/v |
| pre_mc_sigma_too_low | 204 | 0 | 204 | 1.0% | 204 | 6.9% |
| options_design_roi_or_edge | 169 | 169 | 168 | 0.1% | 164 | -2.2% |
| deep_analysis_veto_or_invalid | 164 | 0 | 162 | -0.4% | 160 | -3.4% |
| mc_below_threshold | 101 | 101 | 101 | 0.9% | 99 | 4.8% |
| prescreen_bearish_prefilter | 39 | 0 | 39 | n/v | 37 | n/v |
| impact_x_surprise_below_floor | 36 | 36 | 36 | 0.1% | 36 | 2.2% |
| economics_first_earnings_gate | 25 | 0 | 22 | -5.1% | 5 | n/v |
| unlabeled_after_options_design | 20 | 20 | 20 | n/v | 20 | n/v |

*Mittelwerte erst ab 20 Marks an 10 unabhängigen Signaltagen; nicht kausal.
Richtungsunbekannte Budget-Signale sind keine Long-Call-Prognosen.
Aktienrenditen dürfen nicht als verpasste Optionsgewinne gelten.