# Meta-Cognition, Alpha Decay, Research Value, Safe Mode

Modul: `modules/meta_cognition.py`. Ausgaben: `outputs/research/machine_state.{json,md}`
und `safe_mode.json`.

## Machine Intelligence State
Jede Aussage trägt ihre Metrik. Ohne Messung gibt es keine Aussage; im Bericht
steht dann „(keine gemessene Aussage)“.

| Frage | Quelle |
|---|---|
| Was wissen wir? | Champion-Kennzahlen, Regime-Abhängigkeit, Abstinenz-Bestätigung, akzeptierte Hypothesen |
| Worüber sind wir unsicher? | Intervall-Abdeckung, P(>10 %)-Skill, World-Model-Unsicherheit, fehlende Dimensionen |
| Wo irren wir systematisch? | Blind-Spot-Cluster, überkonfidente Kalibrierungs-Buckets |
| Welche Annahmen sind veraltet? | Regime-Merkmale außerhalb des Trainingsbandes |
| Welche Modelle sind redundant? | Leave-one-out-Beitrag ≤ 0, Duplikat `rs_63` ≡ `mom_3m` |
| Welche Merkmale verlieren Kraft? | Alpha-Decay und CUSUM-Brüche je Merkmal, Modell-Trend |
| Wo ist die Datenqualität niedrig? | Source-Health (FAIL/WARN/STALE) |
| Wo ist die Uneinigkeit hoch? | aktuelles Disagreement-Level |
| Welche Research-Tracks bringen echten OOS-Wert, welche verschwenden Ressourcen? | Research Value Attribution |

## Alpha Decay
Je Merkmal wird der Rank-IC pro Stichtag berechnet, nur mit fertigen Labels.
Verglichen werden die letzten 52 Wochen mit den 52 Wochen davor (t-Test).

- **decaying:** t ≤ −2 bei zuvor positivem IC.
- **break:** CUSUM-Strukturbruch (Statistik > 1,36).

Ein Merkmal bleibt so nicht dauerhaft hoch gewichtet, nur weil es früher
funktioniert hat.

## Research Value Attribution
Je Track (Literatur, Mensch, Discovery, Director, ML-Modelle, Meta-Varianten)
werden erfasst: Experimente, Annahmen, Ablehnungen und OOS-Beitrag.

Die Effizienz ist:
- **HIGH:** Annahmequote ≥ 10 % bei positivem OOS-Beitrag;
- **MEDIUM:** es gibt Annahmen;
- **LOW:** sonst.

## Safe Mode
Die Auslöser sind in `config/next_protocol.yaml` festgelegt:
- Feature-Drift;
- mindestens 50 % der Modelle `deteriorating`;
- Kalibrierungsfehler (Intervalle nicht kalibriert oder ECE > 0,05);
- mindestens 25 % der Quellen FAIL;
- Disagreement-Level HIGH;
- World-Model-Unsicherheit ≥ 0,6;
- Forward-Expectancy über 12 Kohorten mit t ≤ −2.

**Im Safe Mode:**
- keine High-Confidence-Alerts;
- Rückfall auf den stabilen Champion (statisches Ensemble);
- Warnung im Weekly Report;
- Gründe stehen in `safe_mode.json`.


## Forensische Korrektur 2026-11-23: Forward-Safe-Mode

Das alte Signal `Forward-Expectancy t` berechnete die t-Statistik aus den
letzten 12 Zeilen des Prediction-Memory. Diese Zeilen sind *keine*
12 unabhängigen Halteperioden: mehrere Ticker können denselben Signaltag und
überlappende 20-Handelstage-Labels besitzen. Ein solcher Zeilen-t-Wert ist
**kein statistisch gültiger Test** und belegt weder Netto-Options-Alpha noch
Signifikanz eines unabhängigen Forward-Effekts.

- Das bestehende defensive Breakdown-Triggerverhalten bleibt bis zur
  eigenständig genehmigten Ablösung erhalten; es wird **nicht** gelockert.
- Das Signal heißt im Bericht künftig `PROVISIONAL, correlated raw rows`;
  `forward_diagnostic.independent_significance=false` und
  `production_promotion_eligible=false` sind explizit.
- `independent_forward_audit` zeigt getrennt die Aussage aus
  `outputs/research/forward_cohort_audit.json`, inklusive
  `independent_blocks_upper_bound` und `alpha_after_costs`.
  Unvollständige oder fehlende Evidenz wird als solche angezeigt und kann
  niemals den Safe Mode automatisch aufheben.
- Historische Statusmeldungen und vergebene Experiment-Alpha-Budgets werden
  nicht rückwirkend umgeschrieben.
- Keine neuen Marktdaten-, LLM-, Options- oder GitHub-Workflows erforderlich.

Eine spätere statistisch gültige Ablösung des konservativen Triggers erfordert
vorregistrierte, nicht überlappende Signal-/Outcome-Blöcke und separate
gepaarte post-cost Optionsvergleichsdaten. Dafür reicht diese reine
Metadatenkorrektur noch nicht aus.
