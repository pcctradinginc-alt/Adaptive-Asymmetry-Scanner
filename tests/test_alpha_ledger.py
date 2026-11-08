"""Globales Fehlerbudget (config/alpha_ledger.yaml) und Power-Gate (research_lab.design_power).

Geprüft wird die statistische Garantie, nicht nur die Mechanik:
  * familienweise Fehlerrate <= total_alpha über Altlast + alle je möglichen Zuteilungen beider Klassen;
  * feste, ergebnisunabhängige Folge, idempotent, append-only mit Hash-Kette, fail closed;
  * Umgehung (Slot in einem alten Register ohne Zuteilung) ist ein Befund;
  * Power-Gate: unter H0 bestätigt der Plan höchstens mit seinem Alpha; unterpowerte Hypothesen verbrauchen nichts.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import yaml

from modules import option_learner as ol
from modules import research_director as rd
from modules import research_lab as rl
from modules.atomic_io import append_jsonl

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA = "6846fb6ae924d4af07c0cf217f795a431d9f32fd174d5e3d4e2762a5828d7fd2"
NOW = datetime(2026, 11, 9, tzinfo=timezone.utc)


def _cfg(**over):
    cfg = ol.alpha_ledger_config()
    for k, v in over.items():
        cfg[k] = v
    return cfg


def _research(tmp: Path, legacy=()):
    """outputs/research-Struktur mit Altlast-Registern."""
    rdir = tmp / "outputs" / "research"
    for sub, tid, a in legacy:
        if sub == "option_learner_variant":
            append_jsonl(rdir / "option_learner" / ol.VARIANTS, [{"event": "registered", "variant_id": tid,
                                                                    "slot_alpha": a}])
        elif sub == "error_research":
            append_jsonl(rdir / "error_forward_journal.jsonl", [{"event": "slot_assigned", "hypothesis_id": tid,
                                                                  "slot_alpha": a}])
        else:
            append_jsonl(rdir / "autonomous_alpha" / "option_validation" / ol.VARIANTS,
                         [{"event": "registered", "plan_id": tid, "alpha_slot": a}])
    return rdir


def test_config_is_pinned_and_invariants_fail_closed(tmp_path):
    raw = (ROOT / "config" / "alpha_ledger.yaml").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA, "Ledger geändert – nur strenger oder bei belegtem Fehler"
    cfg = yaml.safe_load(raw)
    assert cfg["total_alpha"] == 0.05 and cfg["gamma_exponent"] == 1.5
    assert sum(c["share"] for c in cfg["classes"].values()) == pytest.approx(1.0)
    for bad in ({"total_alpha": 0.1}, {"gamma_exponent": 1.0}, {"version": "x"}):
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.safe_dump({**cfg, **bad}))
        with pytest.raises(ValueError):
            ol.alpha_ledger_config(f)
    dup = json.loads(json.dumps(cfg))
    dup["classes"]["model_variant"]["subsystems"].append("error_research")
    (tmp_path / "dup.yaml").write_text(yaml.safe_dump(dup))
    with pytest.raises(ValueError):
        ol.alpha_ledger_config(tmp_path / "dup.yaml")


def test_legacy_is_frozen_and_the_family_wise_bound_holds_for_every_possible_allocation(tmp_path):
    rdir = _research(tmp_path, [("option_learner_variant", "OLV-a", 0.0025), ("option_learner_variant", "OLV-b", 0.0025),
                                ("error_research", "ERR-x", 0.05 / 24)])
    led = rdir / "alpha_ledger.jsonl"
    st = ol.alpha_ledger_state(led, now=NOW)
    legacy = 0.0025 * 2 + 0.05 / 24
    assert st["init"]["legacy_alpha"] == pytest.approx(legacy) and st["init"]["w0"] == pytest.approx(0.05 - legacy)
    total = legacy
    for cls, sub in (("alpha_hypothesis", "error_research"), ("model_variant", "option_learner_variant")):
        n = st["init"]["classes"][cls]["max_registrations"]
        allocs = [ol.alpha_allocate(f"{sub}-{k}", sub, "h", NOW, path=led) for k in range(1, n + 1)]
        assert [a["class_index"] for a in allocs] == list(range(1, n + 1))
        assert all(a["alpha"] > b["alpha"] for a, b in itertools.pairwise(allocs))      # feste, fallende Folge
        assert allocs[0]["alpha"] == pytest.approx(st["init"]["classes"][cls]["wealth"] * ol.ledger_gamma(1, 1.5))
        assert ol.alpha_allocate(f"{sub}-over", sub, "h", NOW, path=led) is None          # Grenze erreicht
        total += sum(a["alpha"] for a in allocs)
    assert total <= 0.05 + 1e-12                                                          # Union-Bound, inkl. Altlast
    # Garantie auch ohne Registrierungsgrenze: Summe der unendlichen Folge = Klassenbudget
    assert sum(ol.ledger_gamma(k, 1.5) for k in range(1, 200_000)) < 1.0


def test_allocation_is_idempotent_ordered_and_rejects_unknown_subsystems(tmp_path):
    led = _research(tmp_path) / "alpha_ledger.jsonl"
    a = ol.alpha_allocate("T1", "error_research", "h", NOW, path=led)
    assert ol.alpha_allocate("T1", "error_research", "h", NOW, path=led) == a           # endgültig, kein zweites Alpha
    assert ol.alpha_allocate("T1", "alpha_option_validation", "h", NOW, path=led) is None   # anderes Teilsystem
    b = ol.alpha_allocate("T2", "alpha_option_validation", "h", NOW, path=led)           # gleiche Klasse
    assert (a["class_index"], b["class_index"]) == (1, 2) and b["alpha"] < a["alpha"]
    assert ol.alpha_allocate("T3", "unknown", "h", NOW, path=led) is None
    preview = ol.alpha_allocate("T4", "error_research", "h", NOW, path=led, preview=True)
    assert preview["class_index"] == 3 and "T4" not in ol.alpha_ledger_state(led, persist=False)["allocations"]


def test_tampering_and_bypass_are_critical_and_block_new_allocations(tmp_path):
    rdir = _research(tmp_path)
    led = rdir / "alpha_ledger.jsonl"
    ol.alpha_allocate("T1", "error_research", "h", NOW, path=led)
    lines = led.read_text().splitlines()
    ev = json.loads(lines[1])
    ev["alpha"] = 0.05                                                                    # nachträglich erhöht
    led.write_text("\n".join([lines[0], json.dumps(ev)]) + "\n")
    st = ol.alpha_ledger_state(led, persist=False)
    assert st["summary"]["status"] == "CRITICAL" and "Hash-Kette" in st["problems"][0]
    assert ol.alpha_allocate("T2", "error_research", "h", NOW, path=led) is None
    rdir2 = _research(tmp_path / "b")
    led2 = rdir2 / "alpha_ledger.jsonl"
    ol.alpha_allocate("T1", "error_research", "h", NOW, path=led2)
    _research(tmp_path / "b", [("option_learner_variant", "OLV-bypass", 0.0025)])         # alter Pfad nach dem Start
    st2 = ol.alpha_ledger_state(led2, persist=False)
    assert st2["summary"]["status"] == "CRITICAL" and "ohne Ledger-Zuteilung" in st2["problems"][0]


def test_subsystems_share_one_ledger(tmp_path, monkeypatch):
    """Varianten (Klasse model_variant) und Prüfpläne/Fehlerforschung (Klasse alpha_hypothesis) im selben Ledger."""
    rdir = _research(tmp_path)
    out = rdir / "option_learner"
    cfg = ol.variant_config()
    assert cfg["test_budget"]["alpha_ledger"] is True
    prop = {"proposal_id": "P1", "horizon": 5, "change": {"ridge_alpha": 30.0}, "auto_experiment": True,
            "falsification": {"max_days": 90}}
    v = ol.register_variants(out, [prop], NOW, cfg)[0]
    assert ol.ledger_path(out) == rdir / "alpha_ledger.jsonl"
    st = ol.alpha_ledger_state(rdir / "alpha_ledger.jsonl", persist=False)
    alloc = st["allocations"][v["variant_id"]]
    assert alloc["class"] == "model_variant" and v["slot_alpha"] == pytest.approx(alloc["alpha"])
    assert v["test_budget"]["ledger"] == ol.LEDGER_VERSION and v["alpha_slot"] == 1
    assert ol.alpha_allocate("ERR-1", "error_research", "h", NOW, path=rdir / "alpha_ledger.jsonl")["class_index"] == 1
    tb = ol.test_budget(out, cfg)
    assert tb["source"] == ol.LEDGER_VERSION and tb["ledger_registered"] == 1 and tb["used"] == 0
    assert tb["next_alpha"] == pytest.approx(st["init"]["classes"]["model_variant"]["wealth"] * ol.ledger_gamma(2, 1.5))


def test_real_repository_ledger_counts_all_existing_slots_as_legacy():
    st = ol.alpha_ledger_state(persist=False)
    subs = {x["subsystem"] for x in st["init"]["legacy"]}
    assert "error_research" in subs and "option_learner_variant" in subs
    assert any(x["test_id"] == "ERR-d6695229" for x in st["init"]["legacy"])          # unverändert, Slot 1
    assert 0 < st["init"]["w0"] < 0.05 and st["summary"]["status"] == "OK"


# ── Power-Gate ───────────────────────────────────────────────────────────────

def test_design_power_holds_size_under_h0_and_grows_with_effect():
    looks, la = [4, 8, 12], ol.alpha_spending(0.01)
    assert rl.design_power(0.0, 0.06, looks, la, 20_000) <= 0.01 + 0.004                 # Testgröße ~ Alpha
    p = [rl.design_power(e, 0.06, looks, la) for e in (0.0, 0.03, 0.06, 0.12)]
    assert p == sorted(p) and p[-1] > 0.9
    mde = rl.min_detectable_effect(0.06, looks, la, 0.5)
    assert rl.design_power(mde, 0.06, looks, la) >= 0.5 > rl.design_power(mde * 0.8, 0.06, looks, la)
    assert rl.design_power(0.05, 0.0, looks, la) == 0.0 and rl.min_detectable_effect(0.0, looks, la, 0.5) is None


def _v3_proto(**gate):
    from test_error_driven_research import _proto
    p = _proto(ledger=True, budget={"max_new_per_run": 3, "max_lifetime": 24})
    p["forward"]["power_gate"] = {**p["forward"]["power_gate"], **gate}
    return p


def test_v3_registration_allocates_from_the_ledger_only_for_decidable_hypotheses(tmp_path):
    from test_error_driven_research import _run, _world
    world = _world()
    r = _run(world, date(2026, 11, 7), tmp_path, proto=_v3_proto())
    led = tmp_path / "alpha_ledger.jsonl"
    allocs = ol.alpha_ledger_state(led, persist=False)["allocations"]
    by = {h["spec"]["feature"]: h for h in r["hypotheses"]}
    assert by["rel_volume"]["status"] == "REJECTED_WF" and by["rel_volume"]["forward_plan"] is None
    assert by["rel_volume"]["id"] not in allocs                                            # verworfen: kein Alpha
    for h in by.values():
        if h["id"] in allocs:
            a = allocs[h["id"]]
            assert h["forward_plan"]["slot_alpha"] == pytest.approx(a["alpha"]) and a["class"] == "alpha_hypothesis"
            assert sum(h["forward_plan"]["look_alpha"]) == pytest.approx(a["alpha"])
            assert h["spec"]["forward"]["alpha_source"] == "alpha-ledger-v1"
    gated = [d for d in r["decisions"] if d["decision"] in ("register", "underpowered")]
    assert gated and r["alpha_ledger"]["status"] == "OK"
    n = len(led.read_text().splitlines())
    _run(world, date(2026, 11, 7), tmp_path, proto=_v3_proto())                           # wiederholter Lauf
    assert len(led.read_text().splitlines()) == n                                          # keine zweite Zuteilung


def test_underpowered_hypotheses_never_spend_alpha_and_become_data_gaps(tmp_path):
    from test_error_driven_research import _run, _world
    r = _run(_world(), date(2026, 11, 7), tmp_path, proto=_v3_proto(effect_shrinkage=1e-6))      # Effekt ~ 0: Power ~ Alpha
    allocs = ol.alpha_ledger_state(tmp_path / "alpha_ledger.jsonl", persist=False)["allocations"]
    assert not allocs
    under = [d for d in r["decisions"] if d["decision"] == "underpowered"]
    gaps = [g for g in r["data_gaps"] if g["kind"] == "underpowered"]
    assert under and len(gaps) == len(under) and all(g["power"] < 0.5 for g in gaps)
    assert not any(h.get("forward_plan") for h in r["hypotheses"])


def test_legacy_registration_keeps_its_frozen_slot_plan():
    """ERR-d6695229: v1-Registrierung, Slot 1 aus dem alten Budget – vom Ledger unberührt."""
    events, problems = rd.read_journal(rd.ERROR_JOURNAL)
    slot = next(e for e in events if e.get("event") == "slot_assigned" and e["hypothesis_id"] == "ERR-d6695229")
    assert not problems and slot["slot"] == 1 and slot["slot_alpha"] == pytest.approx(0.05 / 24)
    reg = [json.loads(x) for x in rd.ERROR_REGISTRY.read_text().splitlines() if x.strip()]
    assert hashlib.sha256(rd.ERROR_REGISTRY.read_bytes()).hexdigest().startswith("491b46e1")
    assert reg[0]["spec"]["protocol"] == "error-research-v1" and "alpha_source" not in reg[0]["spec"]["forward"]
