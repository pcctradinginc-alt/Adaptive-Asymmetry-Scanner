# Research Director und Research Memory

Module: `modules/research_director.py`, `modules/research_lab.py`.

## Research Director
Er entscheidet, welche Forschungsfragen als Nächstes getestet werden. Die
Grundlage sind nur **gemessene** Befunde:

| Eingang | Kandidat |
|---|---|
| Blind-Spot-Cluster | „Segment meiden“, als PIT-Ausdruck in der Signal-Sprache (`step(...)`) |
| Regime-Befund der Meta-Validierung | regime-gegatete Retests knapp gescheiterter Hypothesen (neue id) |
| Modell-Drift (`deteriorating`) | Diagnose-Kandidat |
| Nicht in der Signal-Sprache testbar (z.B. Sektor) | `untestable` → Infrastruktur-/Datenlücke |

Felder je Kandidat: `research_id`, `question`, `hypothesis`,
`economic_rationale`, `required_data`, `available_data`,
`expected_information_gain`, `novelty`, `risk_of_overfitting`,
`research_cost`, `priority`.

**Priorität** = EIG × Wert × Neuheit × (1 − Overfitting-Risiko) ÷ (Beschaffungs- × Forschungskosten).

**Begrenzung:** höchstens 5 neue Hypothesen je Lauf. Sie landen in
`director_hypotheses.json` und durchlaufen im Lab dieselbe Prüfkette. Dazu
gehört die BH-Korrektur über **alle** jemals getesteten Hypothesen; jeder
weitere Versuch macht künftige Tests strenger.

**Neuheit** = 1 − maximale Ähnlichkeit zu bereits Getestetem. Ein identisches
Signal hat die Neuheit 0 und wird nicht ausgewählt.

## Research Memory (Hypothesen-Datenbank)
Je Hypothese werden gespeichert:
- Frage, Grund des Tests, verwendete Daten, Feature-Definitionen, Parameter
  (keine), Baseline, Validierungsdesign;
- OOS-Ergebnis, Regime-Urteile, Gründe, Entscheidung, Datum, Code-Version.

Status-Vokabular:
- **ACCEPTED**
- **REJECTED**
- **INCONCLUSIVE**
- **RETEST_LATER** (mit `retest_after`)

**Ähnlichkeitssperre:** Eine neue Hypothese mit Wort-Jaccard ≥ 0,6 zu einer
verworfenen wird nicht getestet (`blocked_similar_to_rejected`). Ist das Signal
rangähnlich (|ρ| ≥ 0,9), wird sie als Duplikat erkannt. Das verhindert, dass
eine widerlegte Idee leicht verändert immer wieder getestet wird.
