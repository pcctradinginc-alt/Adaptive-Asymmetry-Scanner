# Research-System: beobachten → Hypothese → testen → kritisieren → erinnern

Stand: 2026-09-29. Alles läuft im **Schattenbetrieb**. Keine Komponente ändert
Scoring, Gates, PPO oder Trades. Promotion ist immer ein menschlicher PR.

```
          config/research_protocol.yaml  (GESCHÜTZT: Kosten, Zeiträume, Locked, FDR, Kriterien)
                         │ nur lesen
 Hypothesen ─────────────┼──────────────────────────────┐
 (config/hypotheses.yaml │ + Discovery-Engine)          │
        ↓                ↓                              ↓
 Research-Lab ── Leakage → Duplikat → Walk-Forward → Kosten/Stabilität/Regime → BH-FDR → Locked (1×)
        ↓                                                                    ↓
 hypothesis_db (Gedächtnis: nichts wird "neu entdeckt")          angenommen → neuer Challenger
                                                                              ↓
 ML-Kreislauf: Feature-Store → Labels → gepurgter Walk-Forward → Locked → Forward-Shadow
        ↓                                                  ↓
 Unsicherheitskarten (Intervall, Drawdown, P(>10 %),   Champion/Challenger-Empfehlung
 Uneinigkeit, Datenqualität, Regime, Analogien,         (config/model_registry.yaml)
 Kontrafaktisches)                                            ↓
        ↓                                              menschlicher PR
 Candidate-Ledger (ml_*-Merkmale) → factor_monitor (prospektiv)
        ↓
 Trade-Gedächtnis + Failure Analyzer → Ursachen-Statistik → nächste Hypothesen
```

| Baustein | Modul | Ausgabe |
|---|---|---|
| Bewertungsprotokoll (geschützt) | `config/research_protocol.yaml`, `tests/test_research_protocol.py` (Hash gepinnt), CODEOWNERS | – |
| Feature-Store, Labels, Walk-Forward, Registry, Forward-Ledger, Attribution | `modules/ml_research.py` | `outputs/research/ml_research.{json,md}`, `ml_predictions/` |
| Unsicherheit + Kalibrierungsprüfung, Analogien, Kontrafaktisches | `modules/ml_research.py` (`calibration_wf`, `build_cards`) | `outputs/research/ml_cards.json` |
| Spezialisten-Modelle | `config/model_registry.yaml` (`group:`-Merkmale) | wie oben |
| Hypothesen-Datenbank, Experiment-Lab, Discovery-Engine | `modules/research_lab.py`, `config/hypotheses.yaml` | `outputs/research/hypothesis_db.{json,md}` |
| Trade-Gedächtnis, Failure Analyzer, ähnliche Fälle | `modules/trade_memory.py`, `docs/research/FAILURE_TAXONOMY.md` | `outputs/research/trade_memory.jsonl`, `failure_analysis.{json,md}` |
| Regime | `modules/external/regime.py`, `modules/factor_monitor.py` | Regime-Attribution in allen Berichten |

## Harte Regeln
1. Das Bewertungsprotokoll (Kosten, Locked-Holdout, Leakage-Sprache,
   Mehrfachtest, Kriterien) liegt außerhalb der Kontrolle jedes Research-Agenten.
   Eine Änderung braucht einen Hash-Wechsel im Test und ein CODEOWNERS-Review.
   Durchgesetzt wird das nur mit Branch-Protection „Require review from Code
   Owners“ auf `main`.
2. Signale werden in einer eingeschränkten Sprache geschrieben, die nur
   PIT-Merkmale kennt. Labels und beliebiger Code sind darin nicht möglich.
3. Die Discovery-Engine sieht nur Daten bis `discovery_end` (2019). Geprüft wird
   ausschließlich auf den Jahren danach.
4. Benjamini-Hochberg läuft über **alle** jemals getesteten Hypothesen. Jeder
   weitere Versuch macht die nächsten Tests strenger.
5. Der Locked-Holdout wird je Hypothese höchstens einmal ausgewertet. Bei
   ML-Modellen wird jede Auswertung gezählt.
6. Eine getestete Hypothese oder ein Modell wird nie umformuliert. Jede Änderung
   braucht eine neue id; sonst gilt `invalid_modified`.

## Bewusst (noch) nicht umgesetzt
- **Meta-Modell / dynamische Modellgewichte:** Das ist erst sinnvoll, wenn
  mindestens zwei Spezialisten einzeln belegt sind. Sonst lernt es nur, welches
  Rauschen zuletzt gut aussah.
- **Bull-/Bear-/Risk-Debatte mehrerer LLM-Agenten:** Die Tiefenanalyse hat
  bereits `red_team`, `bear_case` und `bear_case_severity`. Ob die
  LLM-Beurteilung überhaupt einen Mehrwert hat, misst `llm_value_add` (Urteil
  frühestens Mitte November 2026). Zusätzliche LLM-Stimmen erst danach.
- **Automatische LLM-Hypothesengenerierung:** Hypothesen formuliert vorerst ein
  Mensch oder eine Research-Sitzung in `config/hypotheses.yaml`. Die
  Discovery-Engine sucht in einem begrenzten, vorab festgelegten Suchraum.
  Unbegrenzte LLM-Ideen würden den Mehrfachtest-Nenner ohne Kontrolle aufblähen.
- **Alternative Daten (Maut, PortWatch, Wetter, Strom) als Signale:** Die
  PIT-Historie ist zu kurz für jahresweisen Walk-Forward, und es gibt keine
  belastbare Ticker-Zuordnung. Diese Ideen stehen als `blocked_data` in der
  Datenbank.
