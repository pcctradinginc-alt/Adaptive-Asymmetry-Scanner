# Scientific Hypothesis Factory

Stand: 2026-10-03 · Status: **SHADOW/RESEARCH**. Die Fabrik hat keinen Produktionseinfluss.

> Ziel ist nicht, möglichst viele Zusammenhänge zu finden. Das System soll lernen, welche
> Forschungsfragen neue, robuste und handelbare Information liefern, und welche nur
> Kapazität verbrauchen.

## 1. Architektur: Erweiterung statt Parallelwelt

| Schritt | Komponente | neu / bestehend |
|---|---|---|
| Ideen | `modules/hypothesis_factory.py`: Cross-Domain-Familien (`config/research_domains.yaml`), Cross-Source-Divergenzen, Drift aus `machine_state.json` (Meta-Cognition) | neu, liest Bestehendes |
| Mechanismus | Vorlage aus `research_domains.yaml`; optional LLM (`--llm`), **nur Text**. Zahlen, Refusal oder Fehler führen zur Vorlage | neu |
| Falsifizierbarer Vertrag | Population, Exposure, Signal (DSL-Whitelist), Richtung, Lag, Horizont, Kontrollgruppe, Primärmetrik, Mechanismus, Failure Condition, H0, H_alt, `spec_hash` | neu |
| Datenbereitschaft | PIT-Feature-Store? Mapping belastbar? Abdeckung ab 2016 mindestens 30 %? Sonst **DATA_GAP** mit kostenlosen Quellen, nie simuliert | neu |
| Research Memory | `modules/research_memory.py` synchronisiert append-only aus hypothesis_db, alt_data_validation, contract_proposals, promotion_state, factory_results und factory_plan (DATA_GAP) | neu, keine neue Datenhaltung |
| Ähnlichkeit / Varianten | Sperre ab Jaccard 0,75 gegen jede getestete Hypothese; `ALREADY_TESTED` (außer RETEST_LATER); höchstens 3 Varianten je Familie | neu |
| Budget | 6 Tests je Lauf, mindestens 20 % explorativ; Priorität siehe unten | neu |
| Historischer Test | **Research-Lab** (unverändert): Leakage-Whitelist, Duplikat-Fingerprint, Coverage, Walk-Forward 2019 bis Locked, Kosten, Hälften, Regime, BH-FDR über **alle** je getesteten Hypothesen, einmaliger Locked-Holdout, adversariale Rollen | bestehend |
| Robustheit | Batterie nur für Lab-ACCEPTED: Placebo (Permutation je Stichtag, p ≤ 0,05), Lag-Profil (0/4/8 Wochen, Lag 0 > 0), Replikation groß/klein (log_dollar_vol), Sektor-Replikation (≥ 60 % gleiches Vorzeichen), Ablation (Residual-IC nach allen 23 bestehenden Merkmalen, t ≥ 1,5) | neu |
| Prospektiv | ROBUST führt zu `factory_challengers.jsonl` (eingefroren, append-only), forward_start = nächster Montag; `factory_forward_ledger.jsonl` zählt nur neue Kohorten | neu, nutzt `alt_data.contracts.record_forward` |
| Produktion | **nie direkt.** Einziger Weg: menschlicher PR mit Champion-Population-Vertrag in `config/promotion_hypotheses.yaml`, dann PromotionController (SHADOW → ABSTENTION → RERANK → LIMITED_WEIGHT), dann Adapter. FULL bleibt menschlich | bestehend |
| Reviewer | `research_memory.review_candidate` läuft in `pipeline.py` als Stufe 4b **nach** der unabhängigen Deep Analysis. Er sieht nur FORWARD_VALIDATED oder Produktionszustände, prüft den Scope Sektor/Regime und schreibt nur `memory_review` in den Kandidaten-Ledger | neu |

Erweiterung des Research-Labs:

- Sektor-Exposures `exp_<sektor>` (0/1; unbekannter Sektor = NaN) sind jetzt DSL-Merkmale.
- `load_factory()` lädt `outputs/research/factory_hypotheses.json`.
- Die Lab-Datensätze tragen family, domain, exposure_sector und spec_hash.
- Die Sektorzuordnung ist der **heutige** yfinance-Sektor (Audit P2-3, nicht PIT). Jede Hypothese, die sie nutzt, trägt das Flag `non_pit_mapping`, und ihre Datenqualität sinkt auf 0,6.

CI (`ml_research.yml`, nur `full`):

1. Director
2. `hypothesis_factory plan --llm`
3. Research-Lab
4. `hypothesis_factory evaluate`
5. Alt-Data

Alle Schritte laufen mit `continue-on-error`.

Gepinnt und unter CODEOWNERS stehen `config/factory_protocol.yaml` (sha256 in `tests/test_hypothesis_factory.py`), `research_domains.yaml`, beide Module und der Test.

### Priorität (Research-Budget)

`Priorität = (Posterior + Posterior-SD) × Relevanz × Neuheit × Datenqualität ÷ Kosten`

- **Posterior:** Beta(1,4) über die Erfolgsrate der Forschungsrichtung. Prospektive Bestätigung zählt doppelt. UCB steht für den erwarteten Informationsgewinn.
- **Neuheit:** 1 minus die maximale Ähnlichkeit im Research Memory.
- **Kosten:** PIT-Panel 1,0; Alt-Feature 1,5.

### Meta-Learning über Forschungsrichtungen

`outputs/research/research_directions.json` erfasst je Richtung (Familie, Quelle oder Art):

- getestet, Erfolg, prospektiv, verworfen, Datenlücken
- Posterior und Bewertung: „liefert Forward-Mehrwert“, „historisch vielversprechend“, „verschwendet Kapazität“ oder „zu wenig getestet“

Diese Werte fließen über den Posterior direkt in die nächste Priorisierung.

Registrierte, aber nicht forward-validierte Promotion-Verträge zählen als `PENDING_FORWARD`. Sie gelten weder als Erfolg noch als Test.

## 2. Automatisch erzeugte Hypothesen (Offline-Plan 2026-10-03)

Der Plan entstand ohne Panel gegen das aktuelle Research Memory (45 Einträge). Im CI wird zusätzlich die Abdeckung im Panel geprüft.

| ID | Familie | Signal | Richtung | explorativ | Priorität | Plan |
|---|---|---|---|---|---|---|
| FAC-1A196936 | rates_x_financials | `exp_financials * sign(curve_10y_3m - 0)` | +1 | nein | 0.149 | SELECTED |
| FAC-D5870E93 | rates_x_realestate | `exp_real_estate * sign(tnx - 3.0)` | −1 | nein | 0.149 | SELECTED |
| FAC-706E9DE2 | oil_x_energy | `exp_energy * sign(wti_63d_chg - 0)` | +1 | nein | 0.131 | SELECTED |
| FAC-610889C8 | oil_x_airlines_industrials | `exp_industrials * sign(wti_63d_chg - 0)` | −1 | nein | 0.112 | SELECTED |
| FAC-989ACD2E | liquidity_x_technology | `exp_technology * sign(fed_assets_13w_chg - 0)` | +1 | ja | 0.093 | SELECTED |
| FAC-27DA4622 | div_procurement_vs_price | `rank(ted_awards_z) * step(-mom_3m)` | +1 | ja | 0.081 | SELECTED |
| FAC-5F1C6869 | dollar_x_technology | `exp_technology * sign(usd_63d_chg - 0)` | −1 | nein | 0.112 | Budget |
| FAC-44EA5A4F | inflation_x_consumer_defensive | `exp_consumer_defensive * sign(cpi_yoy - 3.0)` | +1 | nein | 0.093 | Budget |
| FAC-E17F4909 | volatility_x_utilities | `exp_utilities * sign(vix_chg_21 - 0)` | +1 | nein | 0.093 | Budget |
| FAC-A11F218F | drift_vol_20 | `rank(vol_20) * step(20 - vix)` | +1 | nein | 0.097 | Budget |
| FAC-AD1E02FC | drift_vol_60 | `rank(vol_60) * step(20 - vix)` | +1 | nein | 0.087 | Budget |
| FAC-9C8B3CD1 | trend_x_basic_materials | `exp_basic_materials * sign(spy_mom_63 - 0)` | +1 | ja | 0.075 | Budget |
| FAC-5E08E752 | procurement_x_industrials | `exp_industrials * sign(ted_awards_z - 0)` | +1 | ja | 0.075 | Budget |
| FAC-E7672D3D | div_insider_vs_price | `rank(sec_insider_net_value_90d) * step(-mom_3m)` | +1 | ja | 0.081 | Budget |
| FAC-671F7C67 | insider_x_healthcare | `exp_healthcare * sign(sec_insider_net_value_90d - 0)` | +1 | ja | 0.062 | Budget |

Die Drift-Ideen stammen aus gemessenen Strukturbrüchen von vol_20 und vol_60 (CUSUM, Meta-Cognition).

Die TED-Divergenz liefert erst Daten, wenn der TED-Backfill Abdeckung belegt. Vorher wird sie im CI zu DATA_GAP (Abdeckung < 30 %).

## 3. Verworfene Ideen

**Erster CI-Lauf (2026-10-03, ml_research full, Run 37112343765):** Alle 6 ausgewählten
Hypothesen wurden schon im Walk-Forward des Research-Labs verworfen. Die
Robustheitsprüfung war daher nicht nötig, und es gibt keinen Prospective Challenger.

| Hypothese | Walk-Forward netto | t | Jahre positiv | Status |
|---|---|---|---|---|
| rates_x_financials | −0.00233 | −1.62 | 14 % | REJECTED |
| rates_x_realestate | −0.00348 | −2.09 | 14 % | REJECTED |
| oil_x_energy | −0.00139 | −0.89 | 29 % | REJECTED |
| oil_x_airlines_industrials | −0.00035 | −0.46 | 57 % | REJECTED |
| liquidity_x_technology | −0.00126 | −0.44 | 29 % | REJECTED |
| div_insider_vs_price | +0.00035 | 0.02 | – | REJECTED (bei Stresskosten nicht positiv) |

- Im CI waren die TED-Ideen (`procurement_x_industrials`, `div_procurement_vs_price`) korrekt `DATA_GAP`, weil die Abdeckung ab 2016 nur 6 % beträgt. An ihre Stelle trat die Insider-Divergenz.
- Der Mechanismus-Text stammt überall aus der Vorlage. Das optionale LLM lieferte keinen Text, entweder weil das Secret fehlt oder weil Antworten verworfen wurden.
- Die Familien gelten jetzt als `ALREADY_TESTED`. Das Budget des nächsten Laufs geht an die übrigen Ideen und an aus Daten generierte Quelle × Sektor/Regime-Hypothesen (`docs/DATA_SOURCES_PROGRAM.md`).

**Bereits verworfen (Memory, prägt den Posterior):**

- 14 Director-Hypothesen: REJECTED, „verschwendet Kapazität“.
- 5 Literatur-Hypothesen, 4 Prior-Study-Hypothesen.
- 4 SEC-Insider-Verträge (ALT-SEC-001..004, OOS REJECT).
- 6 Abstinenz-Regeln (Walk-Forward).

Neue Ideen, die diesen ähneln (Jaccard ≥ 0,75), werden nicht getestet.

## 4. Datenlücken (nie simuliert)

| Idee | fehlt | kostenlose Quellen |
|---|---|---|
| weather_x_consumer | Wetter als PIT-Querschnittsfeature (Archiv vorhanden, nicht im Panel) | NOAA GHCN/NCEI CDO, Open-Meteo Archive, DWD CDC |
| river_x_chemicals | Flusspegel | USGS NWIS, PEGELONLINE (Rhein Kaub) |
| power_x_industrials | Stromverbrauch als Feature | EIA-930, ENTSO-E (Archiv vorhanden) |
| downloads_x_technology | Software-Downloads plus Mapping Paket → Emittent | pypistats, npm downloads API, GitHub API |
| (Domäne) port_activity | Hafenaktivität | IMF PortWatch (Historie kurz) |

Eine Lücke schließt sich nur durch einen PIT-Ingest mit `available_at` und Feature-Store-Eintrag (wie SEC/TED). Erst danach wird die Idee READY.

## 5. Laufende Prospective Challenger

Noch keine Challenger der Fabrik.

Unabhängig davon laufen seit 2026-10-05 forward:

- PROM-ABST-001..005 (Promotion, Einfluss NONE)
- ALT-SEC/ALT-TED-Verträge

## 6. Nachgewiesener inkrementeller Mehrwert

**Keiner.** Keine Komponente zeigt bisher belegten OOS- oder Forward-Mehrwert gegenüber dem Champion (siehe docs/SYSTEM_AUDIT_2026-10-03.md). Die Fabrik ändert daran nichts, bis ein Challenger forward bestätigt ist. Danach entscheidet ein Mensch per PR über einen Champion-Vertrag.

## 7. Grenzen

- Die Sektorzuordnung ist nicht PIT. Signale, die sie nutzen, bewerten frühere Jahre mit dem heutigen Sektor; das Flag `non_pit_mapping` wird gesetzt.
- Die Placebo-Permutation zerstört auch Sektor-Clusterung. Bei sektor-gescopten Signalen ist der Placebo daher eher streng.
- Die Batterie testet nur Lab-bestandene Hypothesen. Ihre Schwellen sind gepinnt und dürfen nur strenger werden.
- Das LLM sieht nur Titel, Population, Exposure, Signal, Richtung und Horizont, nie Ergebnisse. Text mit Ziffern wird verworfen.
