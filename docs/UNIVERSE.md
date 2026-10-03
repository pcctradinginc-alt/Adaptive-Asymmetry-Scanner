# Universum: UNIVERSE_V1 (Produktion) und UNIVERSE_V2 (US-Optionsuniversum, Shadow)

Kernregel: Das Research-Universum darf maximal breit sein. Das produktive Universum wird nur so breit,
wie prospektive Netto-Performance, Kalibrierung und Execution-Qualität es rechtfertigen.

## UNIVERSE_V1 – eingefroren
`outputs/universe/universe_v1_frozen.json` (Definitions-Hash `c9ac84a4…`, `universe_v2.v1_unchanged()` im Test).

- S&P 500 + Nasdaq-100 (`modules/universe.py`).
- Hard-Filter: Cap ≥ 2 Mrd., Ø-Volumen ≥ 1 Mio., Dollar-Volumen ≥ 10 Mio., RV-Filter (`modules/data_ingestion.py`).
- Alle Verträge, Hypothesen, Challenger, Outcomes und Promotion-Evidenz **ohne** `universe_version` gehören zu V1.
- Der V1-Candidate-Ledger trägt ab jetzt `universe_version=V1`, Market Cap und Liquidität.

## UNIVERSE_V2 – vollständiges US-Optionsuniversum
Konfiguration: `config/universe_v2.yaml`. Dort stehen alle Schwellen (Defaults dokumentiert, getestet).

| Schritt | Quelle | Regel |
|---|---|---|
| Listing | SEC `company_tickers_exchange.json` | reguläre US-Börsen (kein OTC), gültiges Equity-Mapping (keine Warrants/Units/Rights/Preferreds) |
| Status | Tradier-Quotes | letzter Handel > 5 Handelstage → STALE (delisted/suspendiert) → ausgeschlossen |
| Optionability | Tradier, Fallback yfinance | ≥ 1 gelisteter Verfall = **optionierbar** (nie automatisch tradeable) |
| Market Cap | yfinance `fast_info` (Quelle + Datum je Snapshot) | Bucket ULTRA_MICRO < 50 Mio. ≤ MICRO < 300 Mio. ≤ SMALL < 2 Mrd. ≤ MID < 10 Mrd. ≤ LARGE < 200 Mrd. ≤ MEGA; fehlend → `UNKNOWN` (nie 0) |
| Buckets | Aktie / Optionen | `liquidity_bucket` (Ø-Dollar-Volumen), `options_liquidity_bucket` (OI nahe am Geld), `execution_quality` (geschätzte Roundtrip-Kosten) |

**RESEARCH_UNIVERSE_V2:** `research_gates` – großzügig, keine Cap-Untergrenze. Ultra-/Micro-Caps ausdrücklich zugelassen.

**TRADEABLE_UNIVERSE_V2:** `production_gates`. Zusätzlich blockieren Risikoflags PENNY_STOCK und MANIPULATION_RISK.

Geprüft werden:
- Kurs, Ø-Volumen, Dollar-Volumen, Aktien-Spread,
- geeignete Verfälle/DTE, Open Interest, Optionsvolumen,
- Options-Spread absolut und relativ, Strike-Dichte,
- geschätzte Ausführungskosten.

**Snapshots** (`outputs/universe/v2_snapshots/*.json.gz`, wöchentlich, `universe_v2.yml`):
- point-in-time, append-only, rollierende Prüfung (älteste zuerst).
- Titel, die zuvor RESEARCH waren und jetzt fehlen oder STALE sind → `v2_delisting_events.jsonl`. Kein stilles Entfernen.

## V2-Shadow-Scan (täglich, `modules/universe_v2_scan.py`)
LLM-frei: die nicht-LLM Stufen des V1-Champions unverändert.
- Relative Volume ≥ 0,6 × clip(VIX/20), unter 0,25 nie; dazwischen nur mit ≥ 3 News.
- News-Präsenz.
- Pre-MC sigma-only ≥ 0,40.

Danach folgen eine frische Options-/Liquiditätsbewertung und der V2-Ledger (`outputs/universe/v2_ledger/`).

Die LLM-Stufen laufen für V2 bewusst nicht (Kostenbudget). Die V2-Evidenz betrifft daher die Quant-Stufen; ein LLM-V2-Test wäre ein eigener Vertrag.

**Outcomes** (`feedback.py`, `v2_outcomes.jsonl`, 20/45/60 T):
- `raw_underlying_return`, MFE/MAE,
- `option_theoretical_return` (Mid→Mid),
- `estimated_spread_cost`, `estimated_slippage`, `estimated_execution_cost`,
- `net_realizable_return`: Kauf Ask + Slippage, Verkauf Bid − Slippage, Kommission; ohne Exit-Quote mit dem relativen Einstiegs-Spread.

Sonderfälle:
- Delisting → `DELISTED_WORST_CASE` (−100 %, gezählt).
- Split im Fenster → `CORPORATE_ACTION`: nicht als Evidenz, gezählt.

## Universe Expansion als Hypothese
Verträge `UNIV-V2-SEG-{ULTRA_MICRO,MICRO,SMALL,MID}@v1`:
- `universe_version=V2`, Population `V2_CANDIDATE`, Klasse `universe_segment`, Forward ab 2026-10-12.
- Segment = Bucket ∩ nicht in V1.
- Baseline = V1-Referenz derselben Population (`in_universe_v1=1`).

**Promotion nur auf `net_realizable_return`.** Vorab im `spec_hash` fixiert:
- N ≥ 100, ≥ 60 Ereignis-Cluster, ≥ 30 Signaltage, ≥ 90 Tage Spanne.
- Netto-Expectancy > 0 mit CI-Untergrenze > 0 (Bootstrap über Tage UND Cluster, Bonferroni-Familie `universe_segment@V2_CANDIDATE`).
- Ausreißer-robust, ≥ 2/3 Zeitfenster positiv.
- Gegenüber der V1-Referenz: Precision@3 nicht schlechter (−0,05), Brier nicht schlechter (+0,02), MaxDD nicht schlechter (−0,10), Tail nicht schlechter (−0,10).
- Execution-Qualität GOOD/FAIR ≥ 70 %.

**Leiter je Segment:** SHADOW → FORWARD_VALIDATED (automatisch) → RERANK_ONLY → WEIGHT_10 (LIMITED_WEIGHT) → TRADE_RECOMMENDATION_ENABLED.
- Jede Stufe über FORWARD_VALIDATED nur per menschlicher Freigabe (`config/promotion_approvals.yaml`, `approved_level`).
- Je Look (28 T) höchstens eine Stufe, nur bei aktuell erfüllter Evidenz.
- Entscheidungen je Segment unabhängig (z. B. SMALL validiert, MICRO REJECTED).

**Demotion:** eine Stufe je Verstoß bis SHADOW, bei signifikant negativem Effekt REJECTED. Auslöser:
- Netto-Expectancy < 0,
- Slippage > 0,15,
- Execution-Qualität < 60 %,
- Precision@3 < 0,30,
- ECE > 0,15.

Der V1-Champion bleibt immer Fallback: V2-Verträge werden auf V1-Trades nie ausgewertet (Adapter filtert die Population).

**Trade-Mail:**
- V2-Kandidaten erscheinen nur, wenn ihr Segment-Vertrag TRADE_RECOMMENDATION_ENABLED hat (beim Senden erneut verifiziert), der Titel tradeable ist und kein Risikoflag trägt.
- Getrennter Block.
- Standard: nichts.

## Grenzen (ehrlich)
- **Keine historische V2-Studie.** Ein point-in-time-Universum aller optionierbaren US-Aktien inklusive Delistings und Kursen gibt es mit offiziellen freien Quellen nicht. V2 sammelt ausschließlich prospektive Evidenz. Survivorship wird über Snapshot-Diffs gemessen, nicht rekonstruiert.
- **Market Cap** ist ab dem ersten Snapshot point-in-time (Quelle + Datum); davor unbekannt.
- **Meta-Learning, Modelle und Datenquellen je Bucket** werden in der V2-Ledger-Auswertung je Segment gemessen. Das ML-Research-Panel bleibt V1 (S&P 500), bis ein V2-Panel eigene Verträge hat.
