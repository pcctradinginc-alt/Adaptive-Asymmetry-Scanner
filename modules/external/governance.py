"""
modules/external/governance.py – Lern-Governance für externe Features.

Zweck: EXPLIZIT und versioniert festlegen, welche Features in
Produktions-Lern-Pfaden (Pearson-Gewichte, QuasiML, RL-Observation) landen
dürfen. Alles, was nicht in dieser Liste steht, ist "unpromoted" und darf
NIE in einen Produktions-Lernpfad einfließen — auch nicht versehentlich
über **kwargs/dict-Merges.

Quelle der Wahrheit: config.yaml
  learning_features.production            – Kern-Features (impact, mismatch, eps_drift)
  external_context.learning.promoted_external_features – externe Features,
      NUR nach Challenger-Validierung + gemergtem PR befüllt (Default: []).
"""

from __future__ import annotations


def _cfg():
    from modules.config import cfg
    return cfg


def production_learning_features() -> list[str]:
    """Vollständige, versionierte Liste der für Produktions-Lernpfade
    zugelassenen Features: Kern-Features + explizit promotete externe
    Features. Fehler beim Config-Lesen → nur die Code-Defaults (nie werfen)."""
    defaults = ["impact", "mismatch", "eps_drift"]
    try:
        cfg = _cfg()
        core = list(getattr(getattr(cfg, "learning_features", None),
                             "production", defaults) or defaults)
    except Exception:
        core = list(defaults)

    promoted: list[str] = []
    try:
        cfg = _cfg()
        learning = getattr(getattr(cfg, "external_context", None), "learning", None)
        promoted = list(getattr(learning, "promoted_external_features", []) or [])
    except Exception:
        promoted = []

    out = list(core)
    for f in promoted:
        if f not in out:
            out.append(f)
    return out


def is_production_eligible(feature: str) -> bool:
    """True nur, wenn `feature` in production_learning_features() steht."""
    return feature in production_learning_features()


class UnpromotedFeatureError(ValueError):
    """Ein nicht-promotetes (externes) Feature wurde in einem
    Produktions-Lernpfad gefunden."""


def assert_no_unpromoted_in(features: "dict | list | set", context: str = "") -> None:
    """Wirft UnpromotedFeatureError, wenn `features` (Keys eines dict oder
    Elemente einer Liste/Set) irgendein Feature enthält, das NICHT in
    production_learning_features() steht.

    `context` ist nur für die Fehlermeldung (z.B. "QuasiML.weights",
    "rl_environment.OBS_DIM", "feedback.compute_pearson_weights")."""
    allowed = set(production_learning_features())
    names = set(features.keys()) if isinstance(features, dict) else set(features)
    unpromoted = names - allowed
    if unpromoted:
        raise UnpromotedFeatureError(
            f"{context or 'Lernpfad'}: nicht-promotete Feature(s) {sorted(unpromoted)} "
            f"— erlaubt sind nur {sorted(allowed)}. "
            f"Externe Features benötigen einen gemergten PR in "
            f"external_context.learning.promoted_external_features."
        )
