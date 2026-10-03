# Kostenoptimierung: Telemetrie, Baseline, Maßnahmen

Stand 2026-10-03. Module: `modules/cost_telemetry.py`, `modules/analysis_cache.py`; Konfiguration: `config/cost_policy.yaml`; Baseline-Schätzung: `scripts/cost_baseline.py`; Monatsbericht: Abschnitt **KOSTEN & EFFIZIENZ** in `monthly_report.py`. Es gibt keine neue Mail und keine neue Report-Pipeline.

## 1. Audit: LLM- und API-Nutzung

| Stelle | Workflow / Bereich | Modell | Häufigkeit | max_tokens |
|---|---|---|---|---|
| `modules/prescreener.py` | prescreening / production | Haiku 4.5 | 1 Call je 20 Kandidaten und Scan | 4096 |
| `modules/deep_analysis.py` | deep_analysis / production | Sonnet 4.6 | 1 Call je Kandidat nach ROI-Precheck und Pre-MC, inkl. Red Team im selben Call | 1600 |
| `modules/external/shadow_analysis.py` | shadow_relation / shadow | Haiku 4.5 | höchstens 25 je Scan | 500 |
| `modules/hypothesis_factory.py` (`--llm`) | hypothesis_factory / research | Opus 5.5 | nur mit `--llm`, je ausgewählter Hypothese | 1024 |

Kostenrelevante Fremd-APIs: Tradier, Finnhub, NewsAPI, Alpha Vantage, FlashAlpha, Eulerpool und FRED. Sie werden auf Transport-Ebene gezählt, ohne dass Keys oder URLs gespeichert werden. Geld kostet nur ein bezahlter Plan. Die Planbeträge sind in `paid_apis` konfiguriert, ein unbekannter Betrag erscheint als „nicht verfügbar“.

**Teuerste Stufe:** die Deep Analysis mit Sonnet. Etwa 70 % der Kosten eines Calls entfallen auf den Output, also das ausführliche JSON mit drei Red-Team-Argumenten. Seit dem 29.09.2026 ist das Universum auf 350–400 Ticker gewachsen. Seitdem gehen bis zu 110 Kandidaten pro Tag in die Deep Analysis; das Laufzeitbudget begrenzt sie auf etwa 70.

Quantitative Filter vor dem LLM gibt es bereits: Hard-Filter, Sector-Momentum, Earnings-Gate, ROI-Precheck und Pre-MC-Sigma-Gate. Prio 1, „sichere Filter vor das LLM“, ist damit weitgehend umgesetzt. Ein weiterer Filter (strengerer ROI-Precheck) wäre nicht replay-fähig, siehe unten.

## 2. Baseline (SCHÄTZUNG bis zur ersten Telemetrie-Monatsabdeckung)

`python scripts/cost_baseline.py` rechnet die Funnel-Zähler der Tagesreports mit Token-Annahmen und den offiziellen Preisen hoch. Annahmen pro Call:
- Sonnet: 2.000 Input- und 900 Output-Tokens.
- Haiku-Prescreen: 2.700 Input- und 800 Output-Tokens je 20er-Batch.
- Shadow: 900 Input- und 250 Output-Tokens.

| Monat | Läufe | Sonnet-Calls | Haiku-Prescreen | Haiku-Shadow | Sonnet $ | Haiku $ | Gesamt $ | $/Lauf |
|---|---|---|---|---|---|---|---|---|
| 2026-07 | 23 | 156 | 57 | 156 | 3.04 | 0.72 | 3.76 | 0.16 |
| 2026-08 | 20 | 103 | 43 | 103 | 2.01 | 0.51 | 2.52 | 0.13 |
| 2026-09 | 22 | 391 | 131 | 381 | 7.62 | 1.70 | 9.32 | 0.42 |
| 2026-10 (2 Läufe) | 2 | 81 | 37 | 25 | 1.58 | 0.30 | 1.88 | 0.94 |

**Laufrate seit dem größeren Universum:** etwa 0,9–1,0 $ pro Lauf, also rund 20 $ pro Monat. Davon entfallen etwa 84 % auf Sonnet.

**Kosten pro finalem Trade:** im September 9,32 $ / 4 Trades ≈ 2,3 $.

Diese Werte sind keine Messung. Ab jetzt schreibt jeder Call eine Zeile mit den echten `usage`-Werten in `outputs/costs/ledger-YYYY-MM.jsonl`. Der Monatsbericht nutzt ausschließlich diese Messwerte.

## 3. Telemetrie

Jeder Anthropic-Call läuft über `cost_telemetry.tracked_create(...)`. Der Call und sein Ergebnis bleiben unverändert; Exceptions werden nach dem Protokollieren weitergereicht. Erfasst werden:
- ts, run_id, workflow, stage, ticker, scope, provider, model, family;
- input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens;
- cost_usd, cache_savings_usd, runtime_s, success/error, stop_reason.

Fehlende usage-Werte bleiben `null`, nie 0. Ist kein Preis bekannt, steht `cost_usd = null`. Solche Zeilen gelten im Bericht als „unvollständig“. Die Aggregation läuft nach Tag, Woche, Monat, Modell, Familie, Workflow, Stufe und Bereich (production/shadow/research).

## 4. Maßnahmen und Bewertung (Baseline vs. optimiert)

| # | Maßnahme | Status | Begründung / Validierung |
|---|---|---|---|
| T | Kosten-Telemetrie + Ledger | **KEEP** | Keine Verhaltensänderung, die Calls sind identisch (Tests). |
| G | Budget-Guards: Woche 6 $, Monat 25 $, Warnung ab 80 % | **KEEP** | Ab 80 % pausiert Research, ab 100 % zusätzlich Shadow. Produktion (Prescreen, Deep Analysis) läuft immer weiter, nur mit Warnung in Log und `stats.cost_budget` (Tests). |
| 5a | Prompt-Caching des Deep-Analysis-Systemprompts (`cache_control`) | **KEEP** | Der Inhalt ist unverändert, daher ist kein Qualitätsrisiko möglich. Sonnet 4.6 cacht ab 1.024 Tokens; darunter verarbeitet die API ohne Cache und ohne Aufpreis. Ob gecacht wird, zeigt die Telemetrie (`cache_read_input_tokens`). Erwartete Ersparnis bei Wirksamkeit: etwa 0,0027 $ von ~0,02 $ je Call, also bis zu ~13 % der Deep-Analysis-Kosten. Prescreener und Shadow liegen unter dem Haiku-Minimum von 4.096 Tokens, Caching wäre dort wirkungslos. |
| 2 | Analyse-Hash-Cache (News-Cluster, 48h-Move-Bucket, MC-Hit-Rate, EPS, Earnings-Datum, Makro, Prompt-Version, Modell; TTL 26 h) | **MODIFY → observe** | Im Ledger wurden 27 von 144 Deep Analyses für denselben Ticker wiederholt. Ob die Eingaben dabei identisch waren, ist rückwirkend nicht prüfbar, weil die News nicht gespeichert sind. Deshalb läuft der Cache zunächst im Beobachtungsmodus: Ein Treffer wird protokolliert, Sonnet läuft trotzdem, und das gespeicherte Ergebnis wird gegen das frische verglichen (`kind=cache_check`). **Freigabe für `mode: active` durch einen Menschen**, sobald `activation_report` KEEP zeigt: mindestens 30 Vergleiche, Richtung und Red-Team-Verdikt jeweils ≥ 95 % gleich, mittlere Impact-Abweichung ≤ 1. Der Stand steht im Monatsbericht. |
| 3 | News-Dedup/Clustering | **KEEP (nur im Cache-Schlüssel) / REJECT (im Prompt)** | Syndizierte Doppelmeldungen sollen den Cache-Schlüssel nicht brechen. Im Prompt selbst (max. 3–8 Schlagzeilen, ~20 Tokens je Stück) läge die Ersparnis unter 1 % und würde Eingaben verändern. |
| 1 | Zusätzlicher quantitativer Filter vor Sonnet (strengerer ROI-Precheck) | **REJECT (vorerst)** | Am 02.10. scheiterten alle 22 finalen Kandidaten am ROI-Gate, Sonnet-Calls wären also vermeidbar gewesen. Die Precheck-ROI-Werte je Kandidat werden aber nicht gespeichert, ein Replay ist daher nicht möglich. Folgeschritt: Wert im Ledger mitschreiben, dann die Schwelle per Replay prüfen. |
| 4 | Modell-Routing HAIKU/SONNET | **MODIFY → automatischer gepaarter A/B** | Siehe §4b, Stufe 3. |
| 5b | Kompakteres JSON-Output (Red-Team-Argumente kürzen) | **MODIFY (Challenger nötig)** | Der größte Hebel pro Call: Der Output macht ~70 % der Kosten aus. Das verändert aber Inhalte, die der Auto-Veto (Narrativ-Mismatch in argument_1) liest. Deshalb nur als Challenger mit Replay auf Recall, Precision und Verdikt. |
| 6 | Research-Frequenz / Batch-API | **KEEP** | Die Batch-API ist für Prescreen, Deep Analysis und Shadow aktiv, mit synchronem Fallback (§4b). Die Fabrik ruft das LLM nur mit `--llm` auf und wird per Budget zuerst gedrosselt. |
| 7 | Fremd-APIs: Zählung, Quota, kein stiller Paid-Fallback | **KEEP** | `paid_tier_approved: false` für alle. Ein Wechsel in einen Paid-Tarif braucht eine menschliche Freigabe in `config/cost_policy.yaml`. |

## 4b. Zielbild „unter 5 $ pro Monat“ (Stand 2026-10-03)

### Nachfrage
- Seit dem Universum mit ~370 Titeln gehen pro Lauf ~90 Kandidaten in die Deep Analysis.
- Das Laufzeitbudget schnitt bisher willkürlich in Listenreihenfolge ab: am 01.10. 110 von 110, am 02.10. 13.

### Kostenprojektion
Erzeugt mit `python scripts/cost_baseline.py --project`; aktuelle Laufrate, Token-Annahmen siehe §2.

| Stufe | Gesamt $/Monat | Status / Qualitätssicherung |
|---|---|---|
| 0 Ist (synchron, Sonnet, Shadow 25) | 37,74 | – |
| 1 + Shadow-Deckel 8 (Research) | 36,94 | **aktiv**; betrifft nur Shadow, keine Produktion |
| 2 + Message Batches API (−50 %) für Prescreen, Deep Analysis, Shadow | 18,90 | **aktiv** |
| 3 + Haiku 4.5 für die Deep Analysis | 7,43 | **gepaarter A/B läuft automatisch** |
| 4 + Bearish-Vorfilter (Prescreen-Richtung) | 5,48 | **gepaarte Messung läuft** |
| 5 + Analyse-Cache aktiv | 5,10 | **Beobachtung läuft** |
| 6 + kompaktes Prescreen-Format | **4,89** | **gepaarter Vergleich läuft** |

### Stufe 2: Message Batches API (aktiv)
- Gleiches Modell und gleicher Prompt, deshalb identische Qualität.
- Die Antworten kommen gesammelt; Kandidaten werden dadurch parallel statt nacheinander bewertet. Die Kappung durch das Laufzeitbudget entfällt weitgehend.
- Nicht rechtzeitig beantwortete Requests werden synchron nachgeholt; kein Kandidat geht verloren.
- Maximale Wartezeit: Deep Analysis 20 min, Prescreen und Shadow je 10 min, immer innerhalb des Laufzeitbudgets.

### Stufe 3: Haiku für die Deep Analysis
- Pro Lauf werden 6 Kandidaten zusätzlich mit identischem Prompt von Haiku analysiert, im selben Batch.
- Verglichen werden die abgeleiteten Pipeline-Entscheidungen: Red-Team-Verdikt, Richtung, Impact- und Surprise-Floor.
- Umgeschaltet wird nur bei allen folgenden Kriterien (vorab festgelegt):
  - ≥ 80 Paare an ≥ 10 Tagen;
  - Gate-Übereinstimmung ≥ 92 %;
  - Recall der von Sonnet bestandenen Kandidaten ≥ 90 %;
  - Richtungs-Übereinstimmung ≥ 90 %;
  - ≥ 95 % lesbare Antworten.
- Danach überwacht eine Sonnet-Stichprobe von 2 pro Lauf weiter. Fallen die letzten 40 Paare unter die Schwellen, wird automatisch auf Sonnet zurückgeschaltet (DEMOTED).

### Stufe 4: Bearish-Vorfilter
- Bisher landeten 34 % der Sonnet-Analysen direkt in `bearish_disabled`.
- Der Prescreen gibt deshalb zusätzlich eine Richtung aus, als letztes JSON-Feld. Die YES/NO-Entscheidung ist dann bereits generiert.
- Aktiv nur, solange Bearish-Trades abgeschaltet sind, und erst, wenn von ≥ 40 Gate-bestandenen Kandidaten (≥ 10 Tage) höchstens 5 % vom Prescreen als BEARISH markiert waren.
- 2 Prescreen-BEARISH-Kandidaten pro Lauf werden weiter analysiert (Monitoring).

### Stufe 5: Analyse-Cache
- Modus `auto`: Der Cache wird aktiv, sobald `activation_report` KEEP meldet.
- Kriterien: ≥ 30 Vergleiche, Richtung und Verdikt je ≥ 95 % gleich, mittlere Impact-Abweichung ≤ 1.

### Stufe 6: kompaktes Prescreen-Format
- Begründung nur noch bei [YES]; die Begründungen für [NO] werden nirgends verwendet.
- Pro Lauf wird ein Batch in beiden Formaten bewertet.
- Aktiv erst bei: ≥ 200 Entscheidungen an ≥ 5 Tagen, ≥ 95 % gleiche YES/NO-Entscheidungen, YES-Anteil ≥ 90 % des bisherigen.

### Ehrliche Einordnung
- Die 4,89 $ sind eine Projektion. Sie setzt voraus, dass alle Prüfungen bestehen.
- Besteht eine Stufe ihre Prüfung nicht, bleibt sie aus. Die Kosten liegen dann entsprechend höher, die Qualität aber unverändert.
- Die gemessenen Werte und der Stand jeder Prüfung stehen monatlich im Abschnitt „KOSTEN & EFFIZIENZ“.
- Ein gerankter Top-K-Deckel vor Sonnet wurde per Replay verworfen: Bei K=10 blieben nur 18 von 47 Kandidaten erhalten, die das Options-Design erreichten.

## 5. Monatsbericht

Der Abschnitt „KOSTEN & EFFIZIENZ“ in der bestehenden Monatsmail enthält:
- Gesamtkosten, Vormonat und Δ in $ und %;
- Aufteilung Sonnet/Haiku/andere sowie Fremd-APIs laut Plan;
- Calls, Tokens und Fehler;
- Prompt-Cache-Ersparnis und den Analyse-Cache inklusive A/B-Stand;
- Kosten pro Scan-Lauf, pro Sonnet-Analyse und pro finalem Trade (Läufe und Trades aus den vorhandenen Tagesreports);
- die Aufteilung Produktion/Shadow/Research;
- eine Kurzerklärung, die nur aus Telemetriewerten erzeugt wird.

Nicht messbare Werte erscheinen als „nicht verfügbar“. Fehlende oder teilweise fehlende Telemetrie markiert den Abschnitt als „unvollständig“. Eine Ausnahme im Abschnitt bricht die Mail nie.

CLI: `python -m modules.cost_telemetry budget`, `python -m modules.cost_telemetry month 2026-10`.
