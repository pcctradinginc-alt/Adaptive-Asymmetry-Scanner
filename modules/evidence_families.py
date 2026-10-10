"""modules/evidence_families.py – familienbewusste Evidenz-Aggregation (generisch, deterministisch).

Korrelierte oder inhaltlich redundante Evidenz darf nicht mehrfach voll als unabhängige Bestätigung zählen.
Beispiele: XLF, KRE, JPM und BAC sind kein vierfaches Banken-Signal, und 10J-Rendite und TLT sind
dieselbe Zinsinformation.

Regel (reihenfolge-unabhängig): Eine Familie mit k verfügbaren Evidenzen hat das effektive Gewicht
    W_f = Σ_{i<k} d^i      (d = Dämpfungsfaktor aus der Config; d=0 -> je Familie höchstens 1, d=1 -> keine Dämpfung)
und verteilt es anteilig auf ihre Zustände:
    effektiv bestätigend = W_f · n_bestätigend / k,  effektiv widersprechend = W_f · n_widersprechend / k.
Die erste Evidenz einer Familie zählt damit voll, jede weitere gedämpft. Die Rohzählungen bleiben erhalten.

Fehlende Evidenz (state None) zählt weder im Zähler noch im Nenner. Sie ist nie eine negative Bestätigung.
"""
from __future__ import annotations

UNMAPPED = "unmapped"


def family_weight(k: int, dampening: float) -> float:
    if not 0.0 <= dampening <= 1.0:
        raise ValueError(f"Dämpfungsfaktor muss in [0, 1] liegen, nicht {dampening}")
    return float(sum(dampening ** i for i in range(max(0, int(k)))))


def aggregate(items: list[dict], dampening: float) -> dict:
    """items: [{"id", "family", "state": +1 | 0 | -1 | None}] (None = fehlend).
    -> Rohzählungen, effektive Werte, family_count, family_breakdown."""
    fams: dict[str, dict] = {}
    for it in items:
        fam = it.get("family") or UNMAPPED
        f = fams.setdefault(fam, {"n_expected": 0, "n_available": 0, "n_confirming": 0, "n_conflicting": 0,
                                  "n_neutral": 0, "n_missing": 0, "signals": []})
        f["n_expected"] += 1
        f["signals"].append(it.get("id"))
        st = it.get("state")
        if st is None:
            f["n_missing"] += 1
            continue
        f["n_available"] += 1
        if st > 0:
            f["n_confirming"] += 1
        elif st < 0:
            f["n_conflicting"] += 1
        else:
            f["n_neutral"] += 1
    eff_c = eff_x = eff_a = 0.0
    for f in fams.values():
        k = f["n_available"]
        w = family_weight(k, dampening)
        f["weight"] = round(w, 6)
        f["effective_confirming"] = round(w * f["n_confirming"] / k, 6) if k else 0.0
        f["effective_conflicting"] = round(w * f["n_conflicting"] / k, 6) if k else 0.0
        f["status"] = "AVAILABLE" if k else "MISSING"
        eff_c += f["effective_confirming"]
        eff_x += f["effective_conflicting"]
        eff_a += w
    raw_c = sum(f["n_confirming"] for f in fams.values())
    raw_x = sum(f["n_conflicting"] for f in fams.values())
    raw_a = sum(f["n_available"] for f in fams.values())
    return {
        "raw_confirmation_count": raw_c, "raw_conflict_count": raw_x, "raw_available": raw_a,
        "effective_confirmation": round(eff_c, 6), "effective_conflict": round(eff_x, 6),
        "effective_available": round(eff_a, 6),
        "effective_confirmation_ratio": round(eff_c / eff_a, 4) if eff_a else None,
        "effective_conflict_share": round(eff_x / eff_a, 4) if eff_a else None,
        "family_count": sum(1 for f in fams.values() if f["n_available"]),
        "families_missing": sorted(k for k, f in fams.items() if not f["n_available"]),
        "family_breakdown": dict(sorted(fams.items())),
        "family_dampening": dampening,
    }
