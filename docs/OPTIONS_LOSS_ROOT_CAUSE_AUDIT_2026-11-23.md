# Optionsverlust-Attribution und P0/P1-Integritätsaudit — 23.11.2026

**Status: READ-ONLY / RESEARCH ONLY / NOT PROMOTABLE.** Keine Änderung an Champion-Regeln, Research-/Promotion-Gates, Kapital, Risikolimits, Datenarchiven, Trading-Empfehlungen oder Marktpreisannahmen. Alle Zahlen sind aus committen Dateien auf `main` und aus GitHub Actions abgerufen. Annahmen nicht als Ursachenbeweise ausgeben.

## P0 – GitHub-Run-Reconciliation

- Watchdog-Run [45237982395](https://github.com/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/actions/runs/45237982395), 23.11. 18:24 UTC: `source_health=RECOVERY_TRIGGERED`, `scanner=MISSED` mangels Pflicht-Upstream. **Das war zum Watchdog-Zeitpunkt zutreffend.**
- Source Health [45238051798](https://github.com/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/actions/runs/45238051798) wurde um 18:26 UTC erfolgreich abgeschlossen und dispatchte Scanner [45238340007](https://github.com/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/actions/runs/45238340007).
- Scanner-Guard: `run=true, status=RUN, reason=Pflicht-Upstream frisch`; Scanner- und Guard-Job GitHub-`success`, final um 18:55:52 UTC.
- Ops-Report `outputs/monitoring/ops_status.md` wurde bereits um 18:51:40 UTC geschrieben; er zeigt im Abschnitt **Stand Watchdog** den älteren Zustand und im Abschnitt **Dieser Lauf** den Scanner-Step `success`. Das ist ein **Snapshot-/Zeitpunktkonflikt**, kein Nachweis für einen am Tagesende fehlenden Scan.
- Nachprüfung: Watchdog-Status nach Scanabschluss erneut refreshen oder Bericht um Live-Run-ID, Status-Zeitpunkt und `IN_PROGRESS/RECOVERED_PENDING_WATCHDOG` erweitern. **Keine Abschwächung von MISSED oder Recovery-Gates.**
- `main` hat `protected=false`, `rulesets=[]`; P0-Issue [#202](https://github.com/pcctradinginc-alt/Adaptive-Asymmetry-Scanner/issues/202) bleibt offen. Direkte Workflow-Output-Schreiber auf `main` vor Regeln inventarisieren: blinde Protection würde Archiv-/Quote-Erfassung blockieren. Owner/Admin muss geschützte Code-Merges und eng begrenzten Output-Writer-Zugriff getrennt lösen und LIVE testen.

## P0 – Safemode / Promotion

`outputs/state/system_state.json`: `safe_mode=true`, `safe_mode_scope=research_influence_only`; Performance Breakdown `t=-4.81` und severe feature drift (VIX, 21-Tage-VIX-Änderung, TNX). Der ursprüngliche Champion bleibt unverändert; Research-Boosts/Model-Promotion sind blockiert, nicht pauschal jeder Regel-Champion-Trade. `config/promotion_policy.yaml` erlaubt maximal `ABSTENTION_ONLY`, alle 22 Challenger derzeit `NONE` und `KEEP_SHADOW`. Keine automatische Freigabe, kein Risikoupgrade ohne eigenständigen prospektiven Beweis und Owner-Freigabe. Der separate globale Intelligence-Kill-Switch ist derzeit `active:false`; dessen Zweck und Scope dürfen nicht mit dem kanonischen Safe Mode verwechselt werden.

## P1 – Primäre Verlustattribution: ausführbare Quotes

Datengrundlage: `outputs/research/option_learner/status_5d.json` und `status_20d.json`, `options_native.components` (NBBO-crossed, keine nachgewiesenen Order-Fills). Der neue Prüfer `scripts/options_loss_attribution.py` liest dieselben committed Summaries, validiert Nenner und `exec_ret = mid_ret + cost` ohne Nachbesserungen und gibt ein strukturiertes JSON auf stdout aus.

| Horizont | Labels | Signaltage | Unabhängige Halteperioden | Mid | Quote-/Ausführungseffekt | Ausführbar |
|---|---:|---:|---:|---:|---:|---:|
| 5 Tage | 1.479 | 23 | 7 | +0,25 % | −16,19 Pp. | −15,94 % |
| 20 Tage | 492 | 13 | 1 | +6,81 % | −14,62 Pp. | −7,81 % |

Die 5-Tage-Basis ist vor Kosten ungefähr null, nach Kosten stark negativ. 20 Tage haben **nur eine** unabhängige Halteperiode, daher keine tragfähige Folgerung über den Langfristeffekt. Die Zerlegung der nicht-kostenbezogenen Komponenten (Delta, Gamma, Theta, Vega) stammt aus einer Black-Scholes-Approximation; sie beweist keine kausale einzelne Verlustquelle.

**Spread-Sensitivität, 5 Tage:** Einstiegsspread >=10 %: n=373, mittlere ausführbare Rendite −34,41 %, Kostenanteil −38,39 Pp. Bei 0–2 %: n=258, ausführbar −8,03 %. Der schlechte Befund bleibt also auch bei günstigen Quotes; Spread-Screening allein macht aus der Strategie keinen positiven Edge.

**Limit-Simulation** (nicht durch Fills validiert): 5 Tage im Mittel −8,73 %. Mehrfaches „Mid statt Ask/Bid“ wäre kein zulässiger Fix und keine handelbare Outperformance.

**Strukturvergleich:** `outputs/research/spread_execution_analysis.json`, geclusterter Shadow-Vergleich, 5 Tage: Produktionsstruktur minus Long-Only bei ausführbaren Quotes durchschnittlich −8,43 Pp./Tag über 31 Vergleichstage, bei Mid −0,10 Pp./Tag. 20 Tage: ausführbar −5,29 Pp./Tag über 21 Tage, Mid −0,04 Pp./Tag. Interpretation: Ausführbarkeit und Struktur interagieren; nicht den gesamten Unterschied ungeprüft als reine Gebühren ausweisen. Cohort-Selektion, payoff-normalisierter Risiko-Exposure-Vergleich, gleichzeitige Signale und Live-Fill-Daten müssen separat kontrolliert werden.

## P1 – Champion-Stichprobe, Stops und Kalibrierung

`outputs/research/paper_performance_analysis.json`: 118 geschlossene Paper-Trades, davon nur 8 `RELIABLE`; 70 `UNKNOWN` und 40 `RECONSTRUCTED`. In den 8: 5 Bull-Call-Spreads mit 0 Gewinnen, durchschnittlich −46,7 %; 3 Long Calls mit 1 Gewinn und durchschnittlich −0,16 %. Stop-Loss: 5 Trades, alle verloren (Ø −49,75 %). Earnings: 5, alle verloren (Ø −38,86 %). **Diese Untergruppen überschneiden sich**, n=8 ermöglicht keine unabhängige Statistik und keinen Nachweis einer falschen Stop-Regel.

MC-Band 65–75 %: 6/6 verloren (deskriptiv); Out-of-sample Prequential-Kalibrierung aktuell **0** auswertbare Bewertungen. Keine adaptive Schwellenänderung aus diesen acht Trades.

`outputs/research/failure_analysis.json` zählt 38 als `RELIABLE` klassifizierte Verlustfälle aus einem **anderen, weiteren Fehlerspeicher**. Nicht mit den 8 reliablen geschlossenen Champion-Trades vermischen.

## P1 – Selektions- und Lern-Bias

- Tagesreport 23.11.2026: 518 im Universum, 258 Kandidaten, 218 `sector_ok`, davon nur **8** quote-probed; 5 Haiku-Inputs, 2 Ja, 2 Sonnet, 1 Deep-Analysis-Überlebender, 0 nach Mismatch, 0 Champion-Trades. Die Kennzahl `analyzed=1` beschreibt die tiefe Analyse, **nicht** Gesamtuniversum oder gesamte Datenprüfung.
- `outputs/research/roi_learning/gate_economics.md`: Die 213 budgetbedingt unbepreisten Kandidaten (23.11.) haben **kein** verifiziertes Options-P&L. Sie sind keine „ökonomisch schlechten“ Trades. Shadow-Throughput-Probes dürfen nicht still die Produktionspopulation ersetzen.
- 5-Tage-Walk-Forward: Referenzvergleich nur 2 unabhängige Perioden, Modell −3,87 %/Tag vs. eingefrorene Referenz −1,39 %/Tag und „nie handeln“ 0 %. `NEED_MORE_DATA` korrekt; die Kandidaten stammen überwiegend aus dem Research-Census, nicht dem produktiven Richtungssignal.
- Das Featureset bei 5 Tagen: 980 Research-Census-Labels von 1.487 (66 %) ohne `impact/surprise`; gegenüber 507 Directional-Labels. Keine gepoolte Modellvalidierung als Nachweis für Richtungssignale.

## P1 – Quote- und Legacy-Lücken

`ops_status`: 20d real_opt und real_strat je 2 offene Marks, ältester 14 Tage; 45d real_opt 2 und real_strat 3, ältester 9 Tage. Weitere 50 markierte Verspätungen je Horizon-Metrik sind nicht mit **aktuell offen** identisch. Fälle mit Vertrags-ID, Mark-Datum, Quote-Status und Providerfehlern erfassen; zweimal vorhandene Metrik desselben Kontrakts nicht als unabhängige Ausfälle zählen. Verifikation: PIT-/Bid/Ask-Artefakte oder explizit `UNAVAILABLE`. Keine historische Imputation.

107 `OUTCOME_UNUSABLE` plus 3 `DATA_BAD` betreffen das Champion-Lifecycle-Audit; 117 fehlende historische Entry-Captures und 113 Legacy-Evaluations sind Altbestand. Nicht durch neue „synthetische“ Trades/Quotes reparieren. Separate Zähler für unvermeidbaren Altbestand und neue Post-Archiv-Verstöße; neue Fehler müssen gegen 0 getestet werden.

PortWatch: PR #211 22.11. merged, jüngster Recovery-Workflow am 23.11. erfolgreich; Source Health hält `imf_portwatch_ports` wegen veralteter Beobachtungsdaten `DEGRADED`. Der Pipeline-`success` beweist keine aktuelle Veröffentlichung. Kein stale-as-fresh-Fallback.

## P2 – Prospektives, vorregistriertes Forschungsexperiment

1. Eingefrorene Kohorten nach Signal-Tag/Ticker/Vertrags-ID und `DIRECTIONAL` vs. `RESEARCH_CENSUS`, mit tatsächlichem NBBO-Zeitstempel, Bid/Ask, OI, DTE, IV, Basispreis und Signaldatum. `PIT`- und Fill-Qualität separat auditieren.
2. 5d/20d und Ausstiegsvarianten **paarweise identische Signale/Kontrakte** vergleichen: `long_call`, `bull_call_spread`, `no_trade`, Mid nur Diagnose, tatsächlich ausführbares Quote-/Fill-Modell als bindendes Ergebnis. Stops und Earnings-Zeitpunkte strikt getrennt untersuchen.
3. Zeitlich vorab fixierte Train/Test-Splits, haltespannenweise Block-Bootstrap, Multiple-Testing-Korrektur, gleiches Erfassungsbudget, fehlende/fallengelassene Quotes explizit zählen. Ein fortgesetztes negatives Ergebnis bedeutet `REJECT`, nicht Threshold-Optimierung.
4. Label-Zulässigkeit, Model-Manifest, registrierte experimentelle Versionen und LIVE-Shadow-Ergebnisse gegen ein unverändertes Regel-Benchmark, no-trade und Kostenbaseline prüfen. **Kein Modell-/Optionsstrategiewechsel oder Promotion**, bevor die bindenden Prospektiv-Gates erfüllt sind.

### Maschinenlesbare Reproduktion

```bash
python scripts/options_loss_attribution.py --root .
python -m pytest -q tests/test_options_loss_attribution.py
```

Beide sind read-only. Tests müssen auf PR-HEAD und GitHub CI grün sein; kein Mergen ohne menschliches Review, kein Bypass des P0-Governance-Issues. Die neuen Dateien dürfen ausschließlich Beobachtbarkeit/Analyse verbessern.
