"""Alpha-to-Options prospective validation (Phase C): zero-network tests.

All rows are synthetic fixtures. They test preregistration, sequential looks,
the eight criteria, learning and the mail path. They are NEVER evidence of an
economic edge; real data on main currently yield NEED_MORE_DATA.
"""
import json
import shutil
from datetime import date, datetime, timedelta, timezone

import pytest
import yaml

from modules import meta_cognition as mc
from modules import research_director as rd
from modules import research_memory as rm
from modules.autonomous_alpha import option_economics as oe
from modules.autonomous_alpha import option_link as al
from modules.autonomous_alpha import option_validation as ov
from modules.autonomous_alpha.registry import EvidenceIntegrityError
from tests.test_alpha_option_economics import _row
from tests.test_alpha_option_link import REPO
from tests.test_improvement_notifications import Outbox, _inputs, _notify

REG = datetime(2026, 11, 7, 15, 0, tzinfo=timezone.utc)
HID = "hgb_xs20_v1"


def _setup(root, **over):
    (root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / al.CONFIG, root / al.CONFIG)
    cfg = yaml.safe_load((REPO / ov.CONFIG).read_text())
    for k, v in over.items():
        cfg[k] = {**cfg[k], **v} if isinstance(v, dict) else v
    (root / ov.CONFIG).write_text(yaml.safe_dump(cfg))


def _signal(hid, d, i, version="v1", group="TREATMENT"):
    return {"signal_id": f"s{d}{hid}{i}", "hypothesis_id": hid, "hypothesis_version": version,
            "hypothesis_hash": "h", "hypothesis_source": "ml_registry", "comparison_group": group,
            "hypothesis_registered_at": "2026-09-29T16:00:00Z", "experiment_version": "x",
            "horizon_trading_days": 20, "pit_data_date": d, "registered_at": f"{d}T15:00:00+00:00",
            "tradable": True, "features": {}, "features_status": "OK"}


REF = "momentum_12_1"


def _world(monkeypatch, h_ret, c_ret=-0.05, weeks=40, per_date=2, ticker_fn=None, f_ret=0.0, hid=HID):
    """Synthetic journals: signals/decisions + economics rows for H, the RANDOM_CONTROL and the
    FROZEN_ML_BENCHMARK (REF) after plan start. c_ret/f_ret=None: that benchmark was never observed."""
    signals, rows = [], []
    for w in range(weeks):
        d = (date(2026, 11, 13) + timedelta(days=7 * w)).isoformat()
        for i in range(per_date):
            t = ticker_fn(w, i) if ticker_fn else f"T{w}_{i}"
            group = "REFERENCE" if hid == REF else "TREATMENT"
            r = _row(hid, d, h_ret(w, i), ticker=t, group=group)
            r["signal_id"] = f"s{d}{hid}{i}"
            rows.append(r)
            signals.append(_signal(hid, d, i, group=group))
            if c_ret is not None:
                rows.append(_row(al.CONTROL_ID, d, c_ret, ticker="C" + t, group="CONTROL"))
            if f_ret is not None and hid != REF:
                fr = _row(REF, d, f_ret, ticker="F" + t, group="REFERENCE")
                fr["signal_id"] = f"s{d}{REF}{i}"
                rows.append(fr)
                signals.append(_signal(REF, d, i, group="REFERENCE"))
    data = {"signals": signals, "decisions": [{"signal_id": s["signal_id"]} for s in signals],
            "outcomes": [], "corrections": []}
    monkeypatch.setattr(al, "read_all", lambda root: data)
    monkeypatch.setattr(oe, "trades", lambda root, d, cfg: (rows, {}))
    return rows


ECON = {"proposals": {HID: {"structure": "LONG_CALL"}, "momentum_12_1": {"structure": None}}}


# ── config and preregistration ───────────────────────────────────────────────

@pytest.mark.parametrize("over", [{"min_quote_coverage": 0.5}, {"min_verified_outcomes": 10},
                                  {"test_budget": {"total_alpha": 0.2}}, {"comparisons": ["NO_TRADE"]},
                                  {"comparisons": ["NO_TRADE", "BASE_STRATEGY"]},
                                  {"production_influence": "SHADOW"}])
def test_config_cannot_be_relaxed(tmp_path, over):
    _setup(tmp_path, **over)
    with pytest.raises(EvidenceIntegrityError):
        ov.config(tmp_path)


def test_repo_config_valid():
    assert ov.config(REPO)["comparisons"] == ["NO_TRADE", "RANDOM_CONTROL", "FROZEN_ML_BENCHMARK"]


def test_plan_is_preregistered_create_only_with_budget_slot(tmp_path, monkeypatch):
    _setup(tmp_path)
    _world(monkeypatch, lambda w, i: 0.1)
    created = ov.register_plans(tmp_path, ECON, now=REG)
    assert len(created) == 1                                           # proposal None -> no plan
    plan = ov.plans(tmp_path)[0]
    assert not plan.get("invalid") and plan["hypothesis_id"] == HID and plan["structure"] == "LONG_CALL"
    assert plan["start_date"] == "2026-11-08"                          # only data AFTER registration
    assert plan["alpha_slot"] == pytest.approx(0.05 / 20)
    assert sum(plan["look_alpha"]) == pytest.approx(plan["alpha_slot"])
    assert plan["look_cutoffs"][-1] == (date(2026, 11, 8) + timedelta(days=364)).isoformat()
    for key in ("signal_definition", "contract_rule", "target", "comparisons", "population",
                "min_decision_coverage", "min_independent_periods", "stop_rules", "test_budget"):
        assert plan[key]
    assert ov.register_plans(tmp_path, ECON, now=REG + timedelta(days=7)) == []     # never re-registered
    assert ov._budget(tmp_path, ov.config(tmp_path))["used"] == 1


def test_tampered_plan_is_invalid(tmp_path, monkeypatch):
    _setup(tmp_path)
    _world(monkeypatch, lambda w, i: 0.1)
    ov.register_plans(tmp_path, ECON, now=REG)
    f = next((tmp_path / ov.OUT / ov.PLAN_DIR).glob("*.json"))
    plan = json.loads(f.read_text())
    plan["min_verified_outcomes"] = 1
    f.write_text(json.dumps(plan))
    res = ov.evaluate_plans(tmp_path, now=datetime(2027, 8, 1, tzinfo=timezone.utc))
    assert res[0]["status"] == "PLAN_INVALID" and res[0]["classification"] == "RESEARCH"


def test_exhausted_budget_never_qualifies(tmp_path, monkeypatch):
    _setup(tmp_path, test_budget={"max_tests": 1})
    _world(monkeypatch, lambda w, i: 0.3)
    ov.register_plans(tmp_path, {"proposals": {HID: {"structure": "LONG_CALL"}}}, now=REG)
    data = al.read_all(tmp_path)
    data["signals"].append(_signal("other", "2026-11-13", 0, version="v9"))
    ov.register_plans(tmp_path, {"proposals": {"other": {"structure": "LONG_CALL"}}}, now=REG)
    res = {r["hypothesis_id"]: r for r in ov.evaluate_plans(tmp_path, now=datetime(2027, 8, 1, tzinfo=timezone.utc))}
    assert res["other"]["status"] == "NO_BUDGET" and res["other"]["classification"] == "RESEARCH"


# ── sequential looks ─────────────────────────────────────────────────────────

def _plan(tmp_path, monkeypatch, h_ret, **kw):
    _setup(tmp_path)
    rows = _world(monkeypatch, h_ret, **kw)
    hid = kw.get("hid", HID)
    ov.register_plans(tmp_path, {"proposals": {hid: {"structure": "LONG_CALL"}}}, now=REG)
    return rows


def test_no_look_before_cutoff_and_need_more_data_early(tmp_path, monkeypatch):
    _plan(tmp_path, monkeypatch, lambda w, i: 0.3 + 0.01 * (w % 3))
    early = ov.evaluate_plans(tmp_path, now=datetime(2027, 1, 15, tzinfo=timezone.utc))[0]
    assert early["status"] == "AWAITING_LOOK" and early["looks"] == []
    first = ov.evaluate_plans(tmp_path, now=datetime(2027, 3, 20, tzinfo=timezone.utc))[0]
    assert [x["verdict"] for x in first["looks"]] == ["NEED_MORE_DATA"] and first["status"] == "RUNNING"
    assert first["classification"] == "RESEARCH"
    again = ov.evaluate_plans(tmp_path, now=datetime(2027, 3, 27, tzinfo=timezone.utc))[0]
    assert again["looks"][0]["report_id"] == first["looks"][0]["report_id"]           # frozen, not re-tested
    assert len(list((tmp_path / ov.OUT / ov.LOOK_DIR).glob("*.json"))) == 1


def test_strong_prospective_edge_qualifies_only_with_all_eight_criteria(tmp_path, monkeypatch):
    _plan(tmp_path, monkeypatch, lambda w, i: 0.3 + 0.01 * ((w + i) % 3))
    res = ov.evaluate_plans(tmp_path, now=datetime(2027, 8, 1, tzinfo=timezone.utc))[0]
    assert [x["verdict"] for x in res["looks"]] == ["NEED_MORE_DATA", "CONFIRMED"]
    q = res["qualification"]
    assert q["qualified"] and len(q["criteria"]) == 8 and all(q["criteria"].values())
    assert res["classification"] == "QUALIFIED_FOR_HUMAN_REVIEW"
    assert q["concentration"]["top_ticker_share_abs_pnl"] <= 0.25


def test_concentrated_winners_are_disclosed_and_block_qualification(tmp_path, monkeypatch):
    _plan(tmp_path, monkeypatch, lambda w, i: 3.0 if i == 0 else 0.02, ticker_fn=lambda w, i: "MEGA" if i == 0 else f"T{w}")
    res = ov.evaluate_plans(tmp_path, now=datetime(2027, 8, 1, tzinfo=timezone.utc))[0]
    assert res["status"] == "CONFIRMED"                                   # statistically positive ...
    assert res["qualification"]["failed"] == ["5_concentration_disclosed_and_limited"]   # ... but one name
    assert res["qualification"]["concentration"]["top_ticker"] == "MEGA"                # disclosed
    assert res["classification"] == "NOT_CONFIRMED"


def test_negative_edge_is_falsified_and_plan_stops(tmp_path, monkeypatch):
    _plan(tmp_path, monkeypatch, lambda w, i: -0.3 - 0.01 * (w % 3))
    res = ov.evaluate_plans(tmp_path, now=datetime(2027, 12, 30, tzinfo=timezone.utc))[0]
    assert res["status"] == "FALSIFIED" and len(res["looks"]) == 2 and res["classification"] == "NOT_CONFIRMED"


def test_profit_factor_and_winrate_alone_do_not_confirm(tmp_path, monkeypatch):
    # 60 % winners, PF > 1, but noisy and no advantage over the base strategy -> never CONFIRMED
    _plan(tmp_path, monkeypatch, lambda w, i: [0.5, -0.6, 0.4, 0.3, -0.5][(w + i) % 5], c_ret=0.05)
    res = ov.evaluate_plans(tmp_path, now=datetime(2027, 12, 30, tzinfo=timezone.utc))[0]
    assert res["status"] == "NOT_CONFIRMED" and res["classification"] == "NOT_CONFIRMED"
    m = res["binding"]["metrics"]
    assert m["winrate"] >= 0.5 and m["profit_factor"] > 1


def test_corrected_data_create_versioned_look_report(tmp_path, monkeypatch):
    rows = _plan(tmp_path, monkeypatch, lambda w, i: 0.3 + 0.01 * (w % 3))
    ov.evaluate_plans(tmp_path, now=datetime(2027, 3, 20, tzinfo=timezone.utc))
    rows[0]["economics"]["ADVERSE"]["return"] = -0.9                      # corrected input data
    res = ov.evaluate_plans(tmp_path, now=datetime(2027, 3, 21, tzinfo=timezone.utc))[0]
    assert res["looks"][0]["report_version"] == 2
    v2 = json.loads(next((tmp_path / ov.OUT / ov.LOOK_DIR).glob("*_v2.json")).read_text())
    assert v2["supersedes"].endswith("_v1") and (tmp_path / ov.OUT / ov.LOOK_DIR / (v2["supersedes"] + ".json")).exists()


# ── learning, memory, meta cognition ─────────────────────────────────────────

def test_learning_identifies_failures_regimes_structures_and_gaps(tmp_path, monkeypatch):
    _setup(tmp_path)
    _world(monkeypatch, lambda w, i: -0.2, weeks=12)
    neg = {"research_decision": {"label": "NO_ECONOMIC_EDGE_AFTER_COSTS"},
           "coverage": {"failure_reasons": {"EXIT_QUOTE_NOT_IN_ARCHIVE": 2}, "abstain_reasons": {"IV_UNKNOWN": 3}},
           "by_regime": {"vix_ge_20|spy_uptrend": {"n": 12, "net_ev_return": 0.1}}}
    econ = {"hypotheses": {HID: {"LONG_CALL": neg, "BULL_CALL_SPREAD": neg}}}
    cfg = ov.config(tmp_path)
    learn = ov.learning(tmp_path, econ, [{"hypothesis_id": HID, "status": "FALSIFIED"}],
                        {"pre_link": {"ml_prediction_rows": 48}, "executed": {}, "hypotheses": {}}, cfg)
    assert sum(learn["failing_feature_combinations"].values()) == 3 and learn["repeatedly_failing_feature_combinations"]
    assert "vix_lt_20|spy_uptrend" in learn["problematic_regimes"]
    assert set(learn["economically_unsuitable_structures"]) == {"LONG_CALL", "BULL_CALL_SPREAD"}
    assert learn["missing_data"]["ABSTAIN:IV_UNKNOWN"] == 6 and learn["missing_data"]["PRE_LINK_PREDICTIONS_WITHOUT_OPTION_DECISION"] == 48
    assert 0 < len(learn["suggested_hypotheses"]) <= cfg["learning"]["max_suggested_hypotheses"]
    assert all(s["status"] == "SUGGESTED_NOT_REGISTERED" for s in learn["suggested_hypotheses"])


def test_research_memory_and_meta_cognition_entries():
    econ = {"hypotheses": {HID: {"LONG_CALL": {"research_decision": {"label": "NEED_MORE_DATA", "reasons": ["x"]},
                                               "metrics": {"ADVERSE": {"n": 3}}}}}}
    val = {"plans": [{"plan_id": "AOV-1", "hypothesis_id": HID, "structure": "LONG_CALL", "status": "CONFIRMED",
                      "classification": "QUALIFIED_FOR_HUMAN_REVIEW", "binding": {"look": 2}},
                     {"plan_id": "AOV-2", "hypothesis_id": HID, "structure": "BULL_CALL_SPREAD",
                      "status": "FALSIFIED", "classification": "NOT_CONFIRMED"},
                     {"plan_id": "AOV-3", "hypothesis_id": HID, "structure": "LONG_CALL", "status": "RUNNING",
                      "classification": "RESEARCH"}]}
    st = {e["hypothesis_id"]: e["status"] for e in rm.alpha_option_entries(econ, val)}
    assert st == {f"{HID}:LONG_CALL": "DATA_GAP", "AOV-1": "ACCEPTED", "AOV-2": "REJECTED", "AOV-3": "PENDING_FORWARD"}
    assert "FORWARD_VALIDATED" not in st.values()                       # never a production state
    lines = mc.alpha_lines({"status": "EVALUATED", "n_plans": 3, **val,
                            "learning": {"missing_data": {"IV_UNKNOWN": 2}}})
    assert any("AOV-1" in x for x in lines) and any("Fehlende Daten" in x for x in lines)


# ── result e-mail via the existing notification path ─────────────────────────

def _qualified_report(tmp_path, monkeypatch, now):
    _plan(tmp_path, monkeypatch, lambda w, i: 0.3 + 0.01 * ((w + i) % 3))
    rep = ov.run(tmp_path, econ=ECON | {"hypotheses": {}}, link_cov={"status": "OK"}, now=now)
    assert rep["status"] == "QUALIFIED_FOR_HUMAN_REVIEW"
    return rep


def test_qualified_strategy_mail_has_exactly_ten_sections(tmp_path, monkeypatch):
    now = datetime(2027, 8, 1, 12, tzinfo=timezone.utc)
    rep = _qualified_report(tmp_path, monkeypatch, now)
    inputs = _inputs(at=now)
    inputs["alpha_validation"] = rep
    tasks, _ = rd.improvement_tasks(inputs, now)
    alpha = [t for t in tasks if t["kind"] == "alpha_option_qualified"]
    assert len(alpha) == 1 and alpha[0]["priority"] == "P1"
    out_dir = tmp_path / "mail"
    out_dir.mkdir()
    _, box = _notify(out_dir, alpha, now=now, inputs=inputs, send=Outbox())
    assert len(box.mails) == 1
    mail = box.mails[0]
    assert mail["subject"].startswith("[Scanner] Alpha-to-Options Forschungsergebnis")
    text = mail["text"]
    pos = [text.index(f"\n{i}. {title}") for i, title in enumerate(rd.ALPHA_SECTIONS, 1)]
    assert pos == sorted(pos) and "\n11. " not in text
    assert "Keine Renditeversprechen" in text and "Keine automatisch freigegebenen Trades" in text
    assert "für menschliche Prüfung qualifiziert (keine Freigabe)" in text
    assert "BEGIN CLAUDE CODE TASK" in text and "Kein automatischer" in text
    for forbidden in ("garantiert", "sicherer Gewinn", "Order ausgeführt"):
        assert forbidden not in text


def test_not_confirmed_result_mail_without_claude_task_and_no_mail_while_running(tmp_path, monkeypatch):
    now = datetime(2027, 12, 30, 12, tzinfo=timezone.utc)
    _plan(tmp_path, monkeypatch, lambda w, i: -0.3 - 0.01 * (w % 3))
    rep = ov.run(tmp_path, econ=ECON | {"hypotheses": {}}, link_cov={"status": "OK"}, now=now)
    inputs = _inputs(at=now)
    inputs["alpha_validation"] = rep
    tasks = [t for t in rd.improvement_tasks(inputs, now)[0] if t["kind"].startswith("alpha_option")]
    assert [t["kind"] for t in tasks] == ["alpha_option_result"] and tasks[0]["priority"] == "P3"
    text = rd.task_text({**tasks[0], "status": "OPEN"}, "neu erkannt")
    assert "Keine technische Änderung notwendig." in text and "BEGIN CLAUDE CODE TASK" not in text
    running = {**rep, "plans": [{**rep["plans"][0], "status": "RUNNING", "classification": "RESEARCH"}]}
    inputs["alpha_validation"] = running
    assert not [t for t in rd.improvement_tasks(inputs, now)[0] if t["kind"].startswith("alpha_option")]
    assert rd.resolution(tasks[0], inputs, now)[0] == "RESOLVED"
    stale = _inputs(at=now)
    stale["alpha_validation"] = {**rep, "generated_at": "2027-01-01T00:00:00+00:00"}
    assert rd.resolution(tasks[0], stale, now)[0] == "UNKNOWN"


def test_real_repository_data_yield_need_more_data_never_success():
    rep = ov.run(REPO, persist=False)
    assert rep["status"] == "NEED_MORE_DATA" and rep["n_plans"] == 0
    assert rep["order_placed"] is False and rep["production_influence"] == "NONE"


def test_cli_runs_validation_after_economics(tmp_path, monkeypatch):
    from modules.autonomous_alpha import __main__ as cli
    from tests.test_alpha_option_economics import _settled
    from tests.test_alpha_option_link import install_market
    _setup(tmp_path)
    _settled(tmp_path, install_market(monkeypatch))
    assert cli.main(["option-link", "--root", str(tmp_path)]) == 0
    rep = json.loads((tmp_path / ov.REPORT).read_text())
    assert rep["status"] == "NEED_MORE_DATA" and rep["n_plans"] == 0           # fixture: no proposal, no success
    assert (tmp_path / ov.LEARNING).exists()


# ── audit: benchmark definitions ─────────────────────────────────────────────


def test_benchmarks_are_separate_and_existing_base_is_never_used(tmp_path, monkeypatch):
    _plan(tmp_path, monkeypatch, lambda w, i: 0.3 + 0.01 * ((w + i) % 3))
    res = ov.evaluate_plans(tmp_path, now=datetime(2027, 8, 1, tzinfo=timezone.utc))[0]
    b, q = res["binding"], res["qualification"]
    assert b["vs_random_control"]["status"] == b["vs_frozen_ml_benchmark"]["status"] == "DESCRIPTIVE_NOT_A_TEST"
    assert b["existing_base_strategy"]["used_for_superiority"] is False
    assert "2_advantage_vs_comparable_benchmarks" in q["criteria"]
    assert not any("base_strategy" in k for k in q["criteria"])
    assert q["benchmarks"]["existing_base_strategy_used_for_superiority"] is False
    plan = ov.plans(tmp_path)[0]
    assert "not the existing base strategy" in plan["comparisons"]["RANDOM_CONTROL"]
    assert "never for superiority" in plan["comparisons"]["EXISTING_BASE_STRATEGY"]


@pytest.mark.parametrize("missing", [{"c_ret": None}, {"f_ret": None}])
def test_missing_benchmark_data_never_confirms_or_qualifies(tmp_path, monkeypatch, missing):
    _plan(tmp_path, monkeypatch, lambda w, i: 0.3 + 0.01 * ((w + i) % 3), **missing)
    res = ov.evaluate_plans(tmp_path, now=datetime(2027, 12, 30, tzinfo=timezone.utc))[0]
    assert res["status"] == "NOT_CONFIRMED" and res["classification"] == "NOT_CONFIRMED"
    assert "CONFIRMED" not in [x["verdict"] for x in res["looks"]]


def test_worse_than_frozen_ml_benchmark_never_confirms(tmp_path, monkeypatch):
    _plan(tmp_path, monkeypatch, lambda w, i: 0.3 + 0.01 * ((w + i) % 3), f_ret=0.6)
    res = ov.evaluate_plans(tmp_path, now=datetime(2027, 12, 30, tzinfo=timezone.utc))[0]
    assert res["status"] == "NOT_CONFIRMED"
    assert res["binding"]["vs_frozen_ml_benchmark"]["mean_difference"] < 0


def test_frozen_benchmark_itself_needs_no_self_comparison(tmp_path, monkeypatch):
    _plan(tmp_path, monkeypatch, lambda w, i: 0.3 + 0.01 * ((w + i) % 3), hid=REF)
    res = ov.evaluate_plans(tmp_path, now=datetime(2027, 8, 1, tzinfo=timezone.utc))[0]
    assert res["binding"]["vs_frozen_ml_benchmark"] == {"status": "SELF"}
    assert res["status"] == "CONFIRMED" and res["qualification"]["criteria"]["2_advantage_vs_comparable_benchmarks"]


def test_mail_names_benchmarks_explicitly(tmp_path, monkeypatch):
    now = datetime(2027, 8, 1, 12, tzinfo=timezone.utc)
    rep = _qualified_report(tmp_path, monkeypatch, now)
    inputs = _inputs(at=now)
    inputs["alpha_validation"] = rep
    task = next(t for t in rd.improvement_tasks(inputs, now)[0] if t["kind"] == "alpha_option_qualified")
    text = rd.task_text({**task, "status": "OPEN"}, "neu erkannt")
    sec6 = text.split("\n6. ")[1].split("\n7. ")[0]
    assert "Zufallskontrolle" in sec6 and "NICHT die bestehende Basisstrategie" in sec6
    assert "eingefrorenen ML-Benchmark" in sec6 and "KEIN Überlegenheitsnachweis" in sec6
