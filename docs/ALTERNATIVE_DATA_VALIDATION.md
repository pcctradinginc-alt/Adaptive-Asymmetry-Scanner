# Alternative Data – Validierung (Protokoll alt-v1)

Protokoll: `config/alt_data_protocol.yaml` – vor jeder Auswertung echter Daten registriert (2026-10-02),
Hash in `tests/test_alt_data.py` gepinnt, CODEOWNERS. Nur strengere Änderungen zulässig.

## Ablauf je Quelle (`modules/alt_data/evaluate.py`)

1. **Screen** (nur Auswahljahre 2016–2018): Abdeckung je Feature, max. |mittlere Querschnitts-Rangkorrelation| zu
   allen bestehenden Features (ähnlichstes Feature benannt), `incremental_information_score` (1 − max|ρ|), Auswahl-IC.
2. **Selektion**: Abdeckung ≥ 0,20, Redundanz ≤ 0,80, höchstens 5 Features, nach Auswahl-IC.
3. **Walk-Forward** (Dev-Jahre 2019–2025) für jede Baseline (`enet_xs20_v1`, `hgb_xs20_v1`): gleiche Spec mit und ohne
   `extra_features`, identische Folds.
4. **Metriken** gepaart: Δ IC, Δ Brier, Δ ECE, Δ LogLoss, Δ Precision@K, Δ Netto-Monatsrendite Top-Dezil,
   Δ Sharpe, Δ MaxDD; Δ je Jahr; Aufschlüsselung nach Sektor und VIX-Regime.
5. **Block-Bootstrap** (2000, Seed 29) der monatlichen Δ-Rendite; einseitiges α = 0,05 Bonferroni über Baselines × Quellen.
6. **Entscheidung**:
   * KEEP – Untergrenze Δ Monatsrendite > 0 bei mind. einer Baseline UND Δ Brier ≤ 0,0005 UND Abdeckung ≥ 0,30.
   * MODIFY – Δ > 0 bei beiden Baselines, aber CI schließt 0 ein; oder nur in einzelnen Sektoren/Regimen positiv.
   * REJECT – sonst.

KEEP bedeutet nur: Quelle bleibt SHADOW-Input und darf Prospective-Challenger-Hypothesen speisen.

## Promotion-Pfad

SHADOW → HISTORICALLY_VALIDATED → PROSPECTIVE_CHALLENGER → FORWARD_VALIDATED → ABSTENTION → RERANK → LIMITED_WEIGHT.
Forward: ab 2026-10-05, mindestens 26 Kohorten. Jeder Schritt ab FORWARD_VALIDATED nur über den
PromotionController und menschliche Freigabe.

## Ausgaben

`outputs/research/alt_data_validation.{json,md}`, `outputs/research/source_scoreboard.json`,
`outputs/research/hypothesis_contracts.jsonl`, `outputs/research/alt_forward_ledger.jsonl`.

## Tests (synthetisch, `tests/test_alt_data.py`)

* Orthogonales informatives Feature → ausgewählt, Δ IC > 0, Δ Monatsrendite > 0.
* Redundantes Feature (≈ mom_3m) → erkannt (ähnlichstes = mom_3m), nicht ausgewählt.
* Reines Rauschen → nicht KEEP. Abdeckung 0 → REJECT mit Begründung.

## Ergebnisse auf echten Daten

Lauf 37105528299 (2026-10-03, Daten aus `alt_data.yml` 37104561116: Form 345 2014Q1–2026Q2, Filings 609/616 CIKs):

* Selektion (2016–2018): sec_late_filing_365d, sec_insider_net_value_90d, sec_exec_change_90d,
  sec_insider_cluster_30d, sec_filing_delay_z; alle Features wenig redundant (max |ρ| ≤ 0,35).
* ENet: Δ Monatsrendite −0,00023 (CI [−0,00097; +0,00051]), Δ IC −0,0017, Δ Sharpe −0,016.
* HGB: Δ Monatsrendite +0,00039 (CI [−0,0015; +0,0023]), Δ Max DD −0,046, Δ IC −0,0032.
* **Verdikt: REJECT** (Source Value Score 0,727 wird von Abdeckung/Frische getragen, nicht von Nutzen).
* Verträge ALT-SEC-001…004 im Research-Lab REJECTED; 001 und 003 signifikant in Gegenrichtung –
  kein Vorzeichenwechsel; eine Gegenhypothese wäre neu zu registrieren und prospektiv zu testen.
* Phase 3 (TED) beginnt laut Auftrag erst nach dieser Bewertung; der SEC-Befund rechtfertigt sie nicht automatisch.
