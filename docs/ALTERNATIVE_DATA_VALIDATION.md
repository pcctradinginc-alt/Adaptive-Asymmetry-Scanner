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

Stehen erst nach dem ersten CI-Lauf von `alt_data.yml` + `ml_research.yml` (full) fest und werden hier mit Lauf-ID
eingetragen. Bis dahin: **keine Aussage zum inkrementellen Nutzen**.
