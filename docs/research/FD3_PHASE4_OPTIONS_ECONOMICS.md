# Phase 4 – FD3 Optionsökonomie auf echten RTH-Quotes

**Research/Paper only**. Kein produktiver Trading-Gate, Broker-Auftrag,
automatischer Merge einer neuen Modellversion oder Alpha-Freigabe.

## Prospektive Kontrakte statt nachträglichem Optionsvergleich

Die in Phase 1 eingefrorenen Aktienvorhersagen werden **erst ab
14.11.2026, 00:00 UTC**, über den bestehenden Alpha→Options-Link
registriert. Vorherige Vorhersagen werden nicht rückwirkend zu Trades
umgedeutet. Der Start liegt bewusst in der Zukunft.

- Von vier festen FD3-Familien werden je Woche zwei nach einem
  kalendarisch deterministischen Round-Robin ausgewählt, unabhängig von
  historischen Gewinnen, neuen Prognosefehlern oder späteren Outcomes.
- Je Familie: exakt ein Top-Rank-Aktientitel aus BASE sowie ein Top-Rank-
  Titel aus CHALLENGER. Identische Titel sind zulässig und geben
  für diese Beobachtung keinen Unterschied.
- Beide verwenden unveränderte `alpha_option_link.yaml`-Kontraktregeln:
  LONG_CALL und BULL_CALL_SPREAD, identische DTE-/Liquiditätskriterien
  und dieselbe Quote- und RTH-Capture-Infrastruktur.
- Das Gesamtlimit an Research-Signalen, Underlyings und Capture-Calls
  bleibt unverändert. Falls die Quote-Kapazität nicht reicht, erscheinen
  die betroffenen Daten ausdrücklich als Lücke. Diese Quelle kann daher
  tatsächlich zusätzliche Quotennutzung **innerhalb** der vorhandenen
  Obergrenzen auslösen, aber keinen separaten Provider kontaktieren.
- `fd3_shadow` darf keine historische Prognose vor ihrer
  Protokollregistrierung als Paper-Signal behandeln.

## Exekutierbare statt Mid-to-Mid-Ergebnisse

Die Auswertung liest nur hash-geprüfte, korrigierte
`modules.autonomous_alpha.option_link.read_all()`-Journale.
Berücksichtigt werden ausschließlich im Voraus registrierte FD3-
Signale mit verbindlicher Quote-Entscheidung und
`QUOTE_VERIFIED_PAPER` an Einstieg UND Ausstieg.

- Einstieg Long Ask, Short Bid.
- Ausstieg Long Bid, Short Ask.
- Gebühren je Contract und je Seite, auch beim Schließen.
- Separater adverser Ausführungstest aus den beobachteten Spreads.
- Alle Leg-IDs, RTH-Quote-Referenzen und Zeitpunkte müssen passen.
- Fehlende, einseitige oder inkonsistente Quote: **DATA_GAP**, nicht
  0 % Rendite und kein künstlich rekonstruiertes Ergebnis.

Ein Paar wird nur gewertet, wenn beide Varianten am gleichen
Signalstichtag und für dieselbe Struktur tatsächlich verifizierte
Optionsquotes besitzen. Kontraktergebnisse werden aus demselben
vorregistrierten Entscheidungsregelwerk gebildet.

Auswertungen überlappender Haltefenster werden nur einmal gezählt.
Unter acht unabhängigen Zeitblöcken steht `NEED_MORE_PAIRED_RTH_QUOTES`.

`outputs/research/feature_discovery_v3/options_economics.json`
beinhaltet Netto-ROI-Deltas, adversen Ausführungstest,
Bootstrap-Intervall, Quote-Referenzen und Missingness-Gründe.

## Strenge Evidenzgrenze

Auch nach acht Zeitblöcken ist das Ergebnis lediglich
`FORWARD_ECONOMIC_DIAGNOSTIC_REVIEW_ONLY`. Das System
bucht **keinen globalen Alpha-Ledger-Slot** und darf den Status
`FORWARD_CONFIRMED` nicht vergeben. Die nächste, menschlich
freigegebene Studie benötigt eine unabhängige Registrierung mit
globalem Fehlerbudget, ausreichender Teststärke und gegebenenfalls
weiteren Marktdatenperioden.

## Betrieb und Regression

```bash
python -m modules.research_feature_option_economics
python -m pytest -q tests/test_research_feature_option_economics.py
```

Der wöchentliche ML-Workflow startet den Auswerter **nach**
`autonomous_alpha option-link`. Ein weiterer Markt-Download ist für
die Auswertung selbst nicht notwendig. Der Existing Research Memory
erfasst lediglich unbestätigte Ergebnisse und Datenlücken.
