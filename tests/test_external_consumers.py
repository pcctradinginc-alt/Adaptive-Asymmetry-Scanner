"""Externe Aufnahme nur mit registriertem Nutzer (Aktionsplan 2026-11-09, Punkt 8).

Jede Quelle, die der Orchestrator abrufen würde, braucht in config/external_consumers.yaml einen Nutzer, der
tatsächlich existiert und die Quelle (oder ihre Familie) liest, oder eine begründete PIT-Archiv-Pflicht (nur ohne
nachholbare Historie). Quellen ohne Nutzer stehen auf DEFERRED: kein Abruf, Archiv unverändert.
"""
from __future__ import annotations

from modules.external.registry import (
    SourceRegistry,
    consumer_problems,
    gate_source,
    load_consumers,
)

ORPHANS = ("clinicaltrials_sponsors", "usaspending_contracts", "sec_13f_holdings", "noaa_storm_events")


def test_every_active_source_has_a_verified_registered_consumer():
    problems = consumer_problems(SourceRegistry().sources, load_consumers())
    assert problems == [], "\n".join(problems)


def test_sources_without_consumer_are_deferred_not_deleted():
    reg, cons = SourceRegistry(), load_consumers()
    for sid in ORPHANS:
        assert sid in reg.sources and sid not in cons                       # Konfiguration bleibt erhalten
        assert gate_source(reg.sources[sid]) == (False, "status_override=DEFERRED")


def test_rule_rejects_unregistered_fake_and_misused_pit_entries(tmp_path):
    (tmp_path / "m.py").write_text("x = 'src_a'\n")
    base = {"enabled": True, "license_status": "OK", "family": "fam", "supports_history": True}
    sources = {"src_a": {**base}, "src_b": {**base}, "src_c": {**base}, "src_d": {**base, "status_override": "DEFERRED"},
               "src_e": {**base, "family": "other"}}
    cons = {"src_a": {"consumers": ["m.py"]}, "src_c": {"consumers": ["missing.py"], "pit_archive": "x"},
            "src_e": {"consumers": ["m.py"]}}
    got = consumer_problems(sources, cons, root=tmp_path)
    assert got == ["src_b: aktiv ohne registrierten Nutzer (config/external_consumers.yaml)",
                   "src_c: pit_archive nur ohne nachholbare Historie (supports_history: true)",
                   "src_c: Nutzer missing.py existiert nicht",
                   "src_e: Nutzer m.py nennt weder die Quelle noch ihre Familie"]
    assert load_consumers(tmp_path / "none.yaml") == {}
