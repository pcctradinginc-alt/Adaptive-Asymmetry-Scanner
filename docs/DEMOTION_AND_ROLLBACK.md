# Demotion und Rollback

## Vorab festgelegte Kriterien

Jeder Vertrag enthält `demotion_criteria` (Pflicht) – ergänzt um `policy.demotion_defaults`:
rollierendes Fenster (30 Beobachtungen seit Promotion), `rolling_expectancy_min`, `ece_max`,
`max_drawdown_worsening`, für Abstinenz `abstention_net_value_min` (verhinderte Trades müssen im Mittel
schlechter sein als durchgelassene).

## Monitoring (aktive Hypothesen)

Gemessen ausschließlich auf Beobachtungen **seit dem Promotionszeitpunkt** (`post_promotion_evidence`):
rollierende Expectancy/Δ, Win Rate, Brier/ECE, Drawdown, Abstinenz-Nettowert, blockiert-vs-durchgelassen.
Effektumkehr (H_alt) → sofort DEMOTED.

## Abstufung (je Lauf genau eine Stufe)

Leiter je Produktionsklasse (`promotion_controller.LADDER`):

```
weight:      WEIGHT_25 → WEIGHT_10 → RERANK_ONLY → SHADOW
score:       SCORE_LIMITED → RERANK_ONLY → SHADOW
rerank:      RERANK_ONLY → SHADOW
abstention:  ABSTENTION_ONLY → SHADOW
SHADOW = PROSPECTIVE_CHALLENGER (Einfluss NONE); Effektumkehr oder Integritätsfehler → DEMOTED
```

Abstinenz-Hypothese, deren blockierte Trades nicht schlechter sind → automatisch zurück auf SHADOW
(`PROSPECTIVE_CHALLENGER`, Einfluss NONE, sammelt weiter). Test: `test_acceptance_promote_abstention_then_decay_demotes`.

## Rollback

* **Integrität:** Vertrag nach Registrierung verändert, Registry-/Transition-Kette oder State-Datei
  manipuliert → Adapter ignoriert die Hypothese sofort (Einfluss NONE); Controller setzt aktive
  Hypothesen auf DEMOTED (`ROLLBACK`).
* **Mensch:** `config/promotion_approvals.yaml` → `rollback: true` (CODEOWNERS, PR) → DEMOTED, NONE.
* **Fehler im Adapter:** `pipeline.py` fängt jede Ausnahme ab → Champion-Entscheidung unverändert.
