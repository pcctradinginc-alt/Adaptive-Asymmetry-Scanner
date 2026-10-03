"""Registrierte Alternative-Data-Features (Name -> Quelle, Version, Beschreibung).
Nur hier registrierte Namen dürfen in Hypothesen (Signal-Sprache) und als
extra_features in Modell-Specs vorkommen."""
from __future__ import annotations

from modules.external.sources import sec_features as _sec

SOURCES = {
    "sec_deep_events": {"features": list(_sec.FEATURES), "feature_version": _sec.FEATURE_VERSION,
                        "availability_col": "alt_sec_available", "path": "outputs/research/feature_store/alt_sec.csv.gz",
                        "contracts": ["sec_form345", "sec_submissions"]},
}
ALT_FEATURES: dict[str, dict] = {f: {"source": s, "feature_version": v["feature_version"],
                                     "description": (_sec.FEATURES.get(f) if s == "sec_deep_events" else "")}
                                 for s, v in SOURCES.items() for f in v["features"]}
AVAILABILITY_COLS = [v["availability_col"] for v in SOURCES.values()]
