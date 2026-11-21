"""The four historically missing ports require live name+country+ID evidence."""
from pathlib import Path

import yaml

from modules.external.sources import maritime as m


ROOT = Path(__file__).resolve().parents[1]
IDS = {"Houston": "port481", "Los Angeles": "port664",
       "Qingdao": "port1069", "Shenzhen": "port1189"}
OFFICIAL_NAMES = {"Houston": "Houston (US-TX)",
                  "Los Angeles": "Los Angeles-Long Beach",
                  "Qingdao": "Qingdao Port",
                  "Shenzhen": "Shekou (Shenzhen)"}
COUNTRIES = {"Houston": "United States", "Los Angeles": "United States",
             "Qingdao": "China", "Shenzhen": "China"}


def test_verified_hints_are_exact_and_aliases_preserve_scope():
    cfg = yaml.safe_load((ROOT / "config/port_universe.yaml").read_text())
    assert cfg["verified_port_id_hints"] == IDS
    aliases = m.load_port_aliases(cfg)
    for name, official in OFFICIAL_NAMES.items():
        assert official in aliases[name]
    # Shenzhen's configured series covers Shekou only, not every terminal.
    assert "Yantian" not in aliases["Shenzhen"]


def test_historical_ids_resolve_only_from_live_reference_rows():
    cfg = yaml.safe_load((ROOT / "config/port_universe.yaml").read_text())
    rows = [{"attributes": {"portid": pid, "portname": OFFICIAL_NAMES[name],
                             "country": COUNTRIES[name]}} for name, pid in IDS.items()]
    wanted = [{"name": name, "country": COUNTRIES[name]} for name in IDS]
    got = m.resolve_entities_by_name(
        wanted, rows, "portid", "portname", "country",
        alias_map=cfg["port_aliases"], verified_ids=cfg["verified_port_id_hints"])
    assert {x.name: x.entity_id for x in got} == IDS
    assert all(x.status == "resolved" for x in got)


def test_qingdao_similarly_named_other_port_and_yantian_cannot_be_substituted():
    cfg = yaml.safe_load((ROOT / "config/port_universe.yaml").read_text())
    rows = [{"attributes": {"portid": "port1070", "portname": "Qingdao Port",
                             "country": "China"}},
            {"attributes": {"portid": "port1190", "portname": "Shekou (Shenzhen)",
                             "country": "China"}},
            {"attributes": {"portid": "port1189", "portname": "Yantian",
                             "country": "China"}}]
    wanted = [{"name": "Qingdao", "country": "China"},
              {"name": "Shenzhen", "country": "China"}]
    got = m.resolve_entities_by_name(
        wanted, rows, "portid", "portname", "country",
        alias_map=cfg["port_aliases"], verified_ids=IDS)
    assert all(x.status == "no_match" and x.entity_id is None for x in got)


def test_id_filter_checks_each_alias_before_declaring_missing():
    cfg = yaml.safe_load((ROOT / "config/port_universe.yaml").read_text())
    rows = [{"attributes": {"portid": "port1070", "portname": "Qingdao Gang",
                             "country": "China"}},
            {"attributes": {"portid": "port1069", "portname": "Qingdao Port",
                             "country": "China"}}]
    got = m.resolve_entities_by_name(
        [{"name": "Qingdao", "country": "China"}], rows,
        "portid", "portname", "country",
        alias_map=cfg["port_aliases"], verified_ids=IDS)
    assert len(got) == 1 and got[0].entity_id == "port1069"


def test_mismatched_country_never_resolves_even_with_matching_id_and_name():
    cfg = yaml.safe_load((ROOT / "config/port_universe.yaml").read_text())
    rows = [{"attributes": {"portid": "port481", "portname": "Houston (US-TX)",
                             "country": "Canada"}},
            {"attributes": {"portid": "port481", "portname": "Houston (US-TX)",
                             "country": "United States"}}]
    got = m.resolve_entities_by_name(
        [{"name": "Houston", "country": "United States"}],
        rows, "portid", "portname", "country",
        alias_map=cfg["port_aliases"], verified_ids=IDS)
    assert got[0].status == "resolved" and got[0].entity_id == "port481"


def test_daily_window_reference_does_not_accept_wrong_official_port_id():
    cfg = yaml.safe_load((ROOT / "config/port_universe.yaml").read_text())
    rows = [{"attributes": {"portid": "port1070", "portname": "Qingdao Port",
                             "country": "China"}}]
    refs = m.verified_daily_reference(
        rows, [{"name": "Qingdao", "country": "China"}],
        "portid", "portname", "country",
        alias_map=cfg["port_aliases"], verified_ids=IDS)
    assert refs is None
