"""Auto-Registrierung eingefrorener Challenger-Vorschläge (nach 10 Wochen
Ledger) + zusammengesetzte Regeln + robuste PPO-Belohnung."""
from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from modules import challenger, challenger_registrar as reg

REPO = Path(__file__).resolve().parent.parent


def _setup(tmp_path, first_ledger_day: date, n_rows: int = 150):
    for name in ("config/challenger_proposals.yaml", "config/challenger_proposals.lock.json", "challengers.yaml"):
        dst = tmp_path / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, dst)
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    rows = [{"date": (first_ledger_day + timedelta(days=i % 60)).isoformat(), "ticker": f"T{i}"}
            for i in range(n_rows)]
    (ledger / "2026-09.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return dict(proposals_path=tmp_path / "config/challenger_proposals.yaml",
                lock_path=tmp_path / "config/challenger_proposals.lock.json",
                registry_path=tmp_path / "challengers.yaml", ledger_dir=ledger)


def test_lock_matches_committed_proposals():
    cfg = yaml.safe_load((REPO / "config/challenger_proposals.yaml").read_text())
    lock = json.loads((REPO / "config/challenger_proposals.lock.json").read_text())
    assert {p["id"]: reg.spec_sha256(p) for p in cfg["proposals"]} == lock
    assert len(cfg["proposals"]) == 3 and cfg["activation"]["min_ledger_days"] == 70


def test_not_registered_before_ten_weeks(tmp_path):
    paths = _setup(tmp_path, date(2026, 9, 28))
    out = reg.run(today=date(2026, 11, 30), **paths)          # 63 Tage
    assert out["registered"] == [] and "nicht reif" in out["reason"]
    ids = [c["id"] for c in yaml.safe_load(paths["registry_path"].read_text())["challengers"]]
    assert "ext_joint_freight_contraction" not in ids


def test_registered_after_ten_weeks_with_future_start(tmp_path):
    paths = _setup(tmp_path, date(2026, 9, 28))
    today = date(2026, 12, 7)                                   # 70 Tage
    now = datetime(2026, 12, 7, 6, 0, tzinfo=timezone.utc)
    out = reg.run(today=today, now=now, **paths)
    assert sorted(out["registered"]) == ["ext_joint_freight_contraction", "ext_relation_contradict",
                                         "ext_weather_disruption_exposed"]
    entries = challenger.load_registry(paths["registry_path"])
    new = {c["id"]: c for c in entries if c["id"].startswith("ext_")}
    assert len(new) == 3                                        # load_registry validiert
    for c in new.values():
        assert c["registered_on"] == "2026-12-07" and c["start_date"] == "2026-12-08"
        assert c["status"] == "active" and c["proposal_sha256"]
    # idempotent: zweiter Lauf registriert nichts doppelt
    again = reg.run(today=today + timedelta(days=1), now=now, **paths)
    assert again["registered"] == []


def test_changed_spec_is_refused(tmp_path):
    paths = _setup(tmp_path, date(2026, 9, 28))
    cfg = yaml.safe_load(paths["proposals_path"].read_text())
    cfg["proposals"][0]["min_n"] = 5                            # nachträgliche Änderung
    paths["proposals_path"].write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    out = reg.run(today=date(2026, 12, 7), **paths)
    assert "ext_joint_freight_contraction" not in out["registered"]
    assert "Hash" in out["skipped"]["ext_joint_freight_contraction"]


def test_only_post_registration_rows_enter_confirmation(tmp_path):
    """Zeilen vor start_date (auch die 10 Reife-Wochen) werden nie ausgewertet."""
    paths = _setup(tmp_path, date(2026, 9, 28))
    reg.run(today=date(2026, 12, 7), now=datetime(2026, 12, 7, 6, tzinfo=timezone.utc), **paths)
    c = next(x for x in challenger.load_registry(paths["registry_path"])
             if x["id"] == "ext_joint_freight_contraction")
    pre = {"date": "2026-12-01", "status": "proposed", "outcomes": {"real_strat_ret_45d": 5.0}}
    post = {"date": "2026-12-09", "status": "proposed", "outcomes": {"real_strat_ret_45d": -0.5}}
    res = challenger.evaluate(c, [pre, post], date(2027, 3, 1), n_active=6)
    assert res["n_baseline"] == 1 and res["mean_baseline"] == -0.5


def test_joint_freight_rule_excludes_only_triggered_candidates():
    cfg = yaml.safe_load((REPO / "config/challenger_proposals.yaml").read_text())
    rule = next(p for p in cfg["proposals"] if p["id"] == "ext_joint_freight_contraction")["rule"]

    def row(direction, road, state_f, state_m):
        return {"status": "proposed", "direction": direction,
                "external": {"ticker_exposure": {"road_freight_relevance": road, "maritime_relevance": "NONE"},
                             "states": {"global_freight_state": state_f, "global_maritime_state": state_m}}}
    triggered = row("BULLISH", "HIGH", "CONTRACTION", "STRONG_CONTRACTION")
    assert challenger.select([triggered], rule) == []
    for r in (row("BEARISH", "HIGH", "CONTRACTION", "CONTRACTION"),      # falsche Richtung
              row("BULLISH", "LOW", "CONTRACTION", "CONTRACTION"),       # nicht exponiert
              row("BULLISH", "HIGH", "CONTRACTION", "NEUTRAL"),          # nur Straße
              {"status": "proposed", "direction": "BULLISH"}):           # kein Kontext
        assert challenger.select([r], rule) == [r]


def test_min_clusters_tightens_but_never_loosens():
    c = {"id": "x", "registered_on": "2026-01-01", "start_date": "2026-01-02", "rule": [],
         "metric": "m", "min_n": 1, "min_clusters": 15, "max_duration_days": 90}
    rows = [{"date": f"2026-01-{d:02d}", "status": "proposed", "m": 0.1 * d} for d in range(3, 15)]
    res = challenger.evaluate(c, rows, date(2026, 2, 1), n_active=1)
    assert res["verdict"] == "running" and res["min_clusters"] == 15


def test_robust_reward_is_concave_and_uses_position_fraction():
    from modules.rl_environment import ACTION_BOOST, ACTION_NORMAL, ACTION_SKIP, robust_reward
    assert robust_reward(-1.0, ACTION_SKIP) == 0.0
    f = 0.10
    # Totalverlust kostet genau den Positionsanteil (log(0.9)/0.1)
    assert abs(robust_reward(-1.0, ACTION_NORMAL, f) - (-1.0536)) < 1e-3
    # Konkav: Gewinn wird gedämpft, Verlust verstärkt gegenüber linear
    assert robust_reward(5.0, ACTION_NORMAL, f) < 5.0
    assert robust_reward(-0.5, ACTION_NORMAL, f) < -0.5
    # BOOST lohnt nur, wenn der Erwartungswert die höhere Varianz trägt
    assert robust_reward(-1.0, ACTION_BOOST, f) < robust_reward(-1.0, ACTION_NORMAL, f)
