# Gate- und Optionsökonomie – 2026-10-29

**READ-ONLY SHADOW – keine Handels- oder Gate-Freigabe.**

## Economics-first: letzte verfügbare Läufe

| Datum | Sektor | Quote-Probes | Budget-untersucht: nein | Earnings-Reject | Haiku | Sonnet | Quick-MC | ROI OK | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-10-29 | 375 | 8 | 367 | 1 | 7 | 3 | 0 | 0 | 0 |
| 2026-10-28 | 358 | 8 | 350 | 5 | 3 | 1 | 0 | 0 | 0 |

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
| prescreen_no | 3908 | 0 | 3518 | 0.5% | 1706 | n/v |
| economics_first_quote_probe_budget | 717 | 0 | 0 | n/v | 0 | n/v |
| sector_momentum_weak | 643 | 1 | 499 | 1.1% | 229 | n/v |
| earnings_gate | 439 | 43 | 329 | 0.2% | 31 | n/v |
| mismatch_below_min | 340 | 340 | 312 | 0.4% | 97 | n/v |
| bearish_disabled | 305 | 305 | 293 | 0.0% | 111 | n/v |
| time_budget_exceeded | 253 | 0 | 252 | n/v | 252 | n/v |
| pre_mc_sigma_too_low | 204 | 0 | 195 | 1.1% | 124 | n/v |
| options_design_roi_or_edge | 164 | 164 | 153 | 0.2% | 91 | n/v |
| deep_analysis_veto_or_invalid | 160 | 0 | 150 | -0.3% | 58 | n/v |
| mc_below_threshold | 100 | 100 | 92 | 1.0% | 38 | n/v |
| prescreen_bearish_prefilter | 37 | 0 | 16 | n/v | 0 | n/v |
| impact_x_surprise_below_floor | 36 | 36 | 32 | -0.2% | 14 | n/v |
| unlabeled_after_options_design | 20 | 20 | 20 | n/v | 17 | n/v |
| pre_mc_sigma_unavailable | 9 | 0 | 7 | n/v | 0 | n/v |

*Mittelwerte erst ab 20 Marks an 10 unabhängigen Signaltagen; nicht kausal.
Richtungsunbekannte Budget-Signale sind keine Long-Call-Prognosen.
Aktienrenditen dürfen nicht als verpasste Optionsgewinne gelten.