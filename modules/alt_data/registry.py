"""Registrierte Alternative-Data-Features (Name -> Quelle, Version, Beschreibung).
Nur hier registrierte Namen dürfen in Hypothesen (Signal-Sprache) und als
extra_features in Modell-Specs vorkommen."""
from __future__ import annotations

from modules.external.sources import sec_features as _sec
from modules.external.sources import ted_features as _ted

SOURCES = {
    "sec_deep_events": {"features": list(_sec.FEATURES), "feature_version": _sec.FEATURE_VERSION,
                        "availability_col": "alt_sec_available", "path": "outputs/research/feature_store/alt_sec.csv.gz",
                        "health": "outputs/external_data/sec/health.json",
                        "contracts": ["sec_form345", "sec_submissions"]},
    "ted_procurement": {"features": list(_ted.FEATURES), "feature_version": _ted.FEATURE_VERSION,
                        "availability_col": "alt_ted_available", "path": "outputs/research/feature_store/alt_ted.csv.gz",
                        "health": "outputs/external_data/ted/health.json",
                        "contracts": ["ted_awards"]},
}
_DESC = {"sec_deep_events": _sec.FEATURES, "ted_procurement": _ted.FEATURES}
ALT_FEATURES: dict[str, dict] = {f: {"source": s, "feature_version": v["feature_version"],
                                     "description": _DESC[s].get(f, "")}
                                 for s, v in SOURCES.items() for f in v["features"]}
AVAILABILITY_COLS = [v["availability_col"] for v in SOURCES.values()]
