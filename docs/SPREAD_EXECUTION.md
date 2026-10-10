# Spread Execution / Liquidity – Root Cause und SHADOW-Gate (2026-10-09)

## Root Cause: echter Codepfad vor diesem PR

| Schritt | Code | Preis |
|---|---|---|
| Kontraktwahl | `options_designer._find_best_option` | Filter `risk.max_bid_ask_ratio` nur auf das **Long-Leg**; Short-Leg ohne Spannenprüfung |
| Entry-Preis | `options_designer` / `pipeline.build_trade_record` | `net_debit = long.ask − short.bid` (Leg-by-Leg-NBBO = combo_ask); `entry_quote` speichert nur das Long-Leg |
| ROI | `options_designer._compute_roi` | Kostenbasis = net_debit (executable), Friktion aber `2 × (ask − bid)/ask` **nur des Long-Legs** |
| Expected PnL | `_compute_roi` | `roi_net = roi_gross − 2·spread_pct − vega_loss − commission` |
| Mark-to-Market / Stop | `feedback.compute_outcome` → `get_current_spread_price` | `long.bid − short.ask` (executable, combo_bid); der Docstring sprach fälschlich von Mid |
| Exit | `feedback.check_exit_rules` | Spread-SL −50 % auf executable value |
| Mid / Fair Value / Combo-Bid/Ask / Quote-Frische | – | existierten für Spreads nicht |

**Inkonsistenz:** Entry und Stop rechnen executable (Bid/Ask), der ROI modelliert die Execution-Kosten aber nur über die Long-Leg-Spanne. Die Spannen beider Legs addieren sich, während der Debit eine Differenz ist. Bei einem Spread mit kleinem Netto-Debit unterschätzt das die echte Friktion um ein Vielfaches. Der Entry-Verlust steht dann ab der ersten Beobachtung im Stop-PnL.

## SNPS-Rekonstruktion

Bull Call Spread 500/550, Verfall 2027-03-19.

**Kurse beim Entry (2026-10-06):**

| Leg | Bid | Ask |
|---|---|---|
| Long 500 | 64,20 | 70,30 |
| Short 550 | 45,10 | 48,60 |

**Combo-Werte:**

| Kennzahl | Wert |
|---|---|
| combo_ask = Entry (Leg-by-Leg) | 25,20 |
| combo_bid = sofort ausführbarer Exit | 15,60 |
| combo_mid = Fair Value | 20,40 |
| relative Combo-Spanne | 47 % |
| **Immediate Liquidation Loss** | (25,20 − 15,60) / 25,20 = **38,1 %** (Bucket `>=30%`) |
| QUOTE_QUALITY | VERY_WIDE |

Der ROI-Friktionsansatz hat 2 × 8,68 % = **17,4 %** eingerechnet, tatsächlich waren es 38,1 %.

**Verlauf:**
- Underlying 506,21 → 497,75 (**−1,7 %**).
- Executable PnL −50,8 % → Produktions-Stop (−50 %) am 2026-10-08.
- Fair Value (Mid) und Leg-Quotes zum Stop-Zeitpunkt wurden **nicht gespeichert**; genau diese Lücke schließt das Monitoring. Bei einer Combo-Spanne von 47 % liegt der Fair Value deutlich über dem executable value, der Stop war also überwiegend execution-getrieben, nicht ökonomisch. Eine genaue Zahl liegt nicht vor.

## Neu (SHADOW_ONLY, `config/spread_execution.yaml` v`spread-exec-v1`)

`modules/spread_execution.py`:
- **Combo-Preise:** `combo_bid`, `combo_ask`, `combo_mid`, `fair_value`, `assumed_entry` (Policy `NATURAL_LEG_BY_LEG` = bisherige Annahme, keine Preisverbesserung), `actual_entry_fill` (Broker-Fill, derzeit nicht vorhanden), `executable_exit_value`.
- **Liquidationsverlust:** `immediate_liquidation_loss_pct` und Research-Buckets `<5 / 5–10 / 10–15 / 15–20 / 20–30 / >=30 %`.
- **QUOTE_QUALITY:** HEALTHY, WIDE, VERY_WIDE, STALE, INVALID, UNAVAILABLE. Bei INVALID oder UNAVAILABLE entsteht keine PnL-Zahl.
- **Netto-ROI:** `gross_expected_roi`, `expected_entry_friction`, `expected_exit_friction`, `expected_roundtrip_cost`, `net_expected_roi` (nach Combo-Friktion), zum Vergleich `production_roi_net` und `production_friction_model`.
- **`SPREAD_EXECUTION_LIQUIDITY_GATE`:** liefert ein Urteil PASS / FLAG (ab 15 %) / WOULD_REJECT (ab 30 % oder bei unbrauchbarer Quote). Wirkt nur im Ledger und im Report.
- **Stop-Shadow:** Fair-Value-PnL, Executable-PnL und Underlying-Bewegung werden getrennt geführt. Mögliche Urteile:
  - EXECUTION_DRIVEN: Stop durch Bid/Ask-Struktur bei stabilem Underlying;
  - STOP_CONFIRMED: echter Einbruch, bestätigt über 2 Beobachtungen;
  - NO_RELIABLE_QUOTE: veraltete oder ungültige Quote.
  
  Der Produktions-Stop (−50 % executable) bleibt **unverändert**.
- **Ledger:** `outputs/intelligence/spread_execution/<YYYY-MM>.jsonl` (entry_candidate, roi_rejected_candidate, monitor, exit). Zusätzlich: Candidate Ledger (`spread_execution`), Trade-Record (`spread_execution_entry`, `spread_monitor`, `spread_stop_shadow`, `spread_execution_exit`), Montagsbericht §20.
- **Shadow-Analyse:** `scripts/spread_execution_analysis.py` (Wochenlauf) → `outputs/research/spread_execution_analysis.json`.

## Shadow-Ergebnis (85 historische Spreads mit Leg-Quotes, explorativ)

| Bucket | n | mit Outcome | RELIABLE | Status | Ø Outcome | Stop-Quote | False-Stop-Quote |
|---|---|---|---|---|---|---|---|
| <5 % | 2 | 2 | 0 | NEED_MORE_DATA | +32 % | 0 % | – |
| 5–10 % | 11 | 10 | 0 | NEED_MORE_DATA | −1 % | 20 % | 0 % |
| 10–15 % | 10 | 10 | 0 | NEED_MORE_DATA | −61 % | 20 % | 0 % |
| 15–20 % | 6 | 5 | 0 | NEED_MORE_DATA | +12 % | 20 % | 0 % |
| 20–30 % | 12 | 9 | 1 | NEED_MORE_DATA | −22 % | 22 % | 0 % |
| >=30 % | 44 | 18 | 2 | NEED_MORE_DATA | −45 % | 22 % | 50 % |

**52 %** aller historischen Spreads lagen bereits beim Entry im Extrem-Bucket `>=30%`. Jeder Bucket hat n < 30, deshalb gibt es keine Schwellenempfehlung.

**Produktionswirkung: keine.** Eine Freischaltung braucht prospektive Evidenz je Bucket, ein separates Owner-Approval und einen eigenen PR.
