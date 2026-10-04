"""Registrierte Alternative-Data-Features (Name -> Quelle, Version, Beschreibung).
Nur hier registrierte Namen dürfen in Hypothesen (Signal-Sprache) und als
extra_features in Modell-Specs vorkommen."""
from __future__ import annotations

from modules.external.sources import sec_features as _sec
from modules.external.sources import sec_xbrl as _xbrl
from modules.external.sources import ted_features as _ted
from modules import commodity_intelligence as _cmd

SOURCES = {
    "sec_deep_events": {"features": list(_sec.FEATURES), "feature_version": _sec.FEATURE_VERSION,
                        "availability_col": "alt_sec_available", "path": "outputs/research/feature_store/alt_sec.csv.gz",
                        "health": "outputs/external_data/sec/health.json",
                        "contracts": ["sec_form345", "sec_submissions"]},
    "ted_procurement": {"features": list(_ted.FEATURES), "feature_version": _ted.FEATURE_VERSION,
                        "availability_col": "alt_ted_available", "path": "outputs/research/feature_store/alt_ted.csv.gz",
                        "health": "outputs/external_data/ted/health.json",
                        "contracts": ["ted_awards"]},
    "sec_xbrl_fundamentals": {"features": list(_xbrl.FEATURES), "feature_version": _xbrl.FEATURE_VERSION,
                              "availability_col": "alt_xbrl_available", "path": str(_xbrl.FEATURE_PATH),
                              "health": str(_xbrl.HEALTH), "contracts": ["sec_companyfacts"]},
}
# Commodity Intelligence (RESEARCH/SHADOW): Kreuzfeatures Exposure-Richtung (NON_PIT-Mapping) × Datums-Feature,
# je Gruppe eigene Quellen-Verträge -> Source Health setzt nur die abhängige Gruppe UNAVAILABLE.
# attach "date_level": ein Datums-Feature-Store, Ticker-Ebene erst beim Anbinden (modules/commodity_intelligence).
_CMD_DESC = _cmd.feature_descriptions()
for _g in ("price", "fundamental", "positioning", "divergence"):
    SOURCES[f"commodity_{_g}"] = {"features": _cmd.cross_features(_g), "feature_version": _cmd.FEATURE_VERSION,
                                  "availability_col": f"alt_cmd_{_g}_available", "path": str(_cmd.STORE_PATH),
                                  "health": str(_cmd.STATUS_PATH), "contracts": list(_cmd.GROUP_SOURCES[_g]),
                                  "attach": "date_level", "group": _g, "non_pit_mapping": True,
                                  "feature_contracts": _cmd.feature_contracts(_g)}
_DESC = {"sec_deep_events": _sec.FEATURES, "ted_procurement": _ted.FEATURES, "sec_xbrl_fundamentals": _xbrl.FEATURES,
         **{f"commodity_{g}": _CMD_DESC[g] for g in ("price", "fundamental", "positioning", "divergence")}}
ALT_FEATURES: dict[str, dict] = {f: {"source": s, "feature_version": v["feature_version"],
                                     "description": _DESC[s].get(f, "")}
                                 for s, v in SOURCES.items() for f in v["features"]}
AVAILABILITY_COLS = [v["availability_col"] for v in SOURCES.values()]

# ── SourceContracts (config/external_sources/*.yaml) ────────────────────────
CONTRACT_DIR = "config/external_sources"
ALT_CONTRACT_FIELDS = ("retrieval_frequency", "expected_latency", "revision_policy", "historical_depth",
                       "entity_level", "coverage", "rate_limits", "cost", "failure_behavior",
                       "point_in_time_capable", "revision_risk", "maintenance_cost", "semantics")
LEVEL = {"low": 0.0, "medium": 0.5, "high": 1.0}


def load_contracts(directory: str | None = None) -> dict[str, dict]:
    """Alle SourceContracts {source_id: Vertrag} (Pflichtfelder prüft modules.external.registry)."""
    from pathlib import Path

    import yaml
    out = {}
    for p in sorted(Path(directory or CONTRACT_DIR).glob("*.yaml")):
        for s in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("sources") or []:
            out[s["source_id"]] = s
    return out


def contract_gaps(contracts: dict[str, dict] | None = None) -> dict[str, list[str]]:
    """Fehlende Alt-Data-Vertragsfelder je in SOURCES referenziertem Vertrag."""
    contracts = contracts if contracts is not None else load_contracts()
    gaps = {}
    for s in SOURCES.values():
        for cid in s["contracts"]:
            c = contracts.get(cid)
            missing = ["<Vertrag fehlt>"] if c is None else [f for f in ALT_CONTRACT_FIELDS if c.get(f) in (None, "", [])]
            if missing:
                gaps[cid] = missing
    return gaps
