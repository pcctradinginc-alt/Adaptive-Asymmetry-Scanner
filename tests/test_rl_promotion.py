"""RL PROMOTION CANDIDATE: nur Meldung; Kollaps, Stabilität und Forward-Mehrwert sind Pflicht."""
from __future__ import annotations

import json

from modules import rl_promotion as rp


def _meta(wf=(0, 10, 10), ins=(5, 10, 5), at="2026-10-01T00:00:00+00:00"):
    c = lambda t: {"SKIP": t[0], "NORMAL": t[1], "BOOST": t[2]}
    share = lambda t: max(t) / sum(t)
    return {"trained_at": at, "n_trades": 100,
            "walk_forward": {"test": {"action_counts": c(wf), "collapsed": share(wf) >= 0.95}},
            "in_sample": {"action_counts": c(ins), "collapsed": share(ins) >= 0.95}}


def _hist(n, collapsed=False, shares=None):
    sh = shares or {"SKIP": 0.0, "NORMAL": 0.5, "BOOST": 0.5}
    return [{"trained_at": f"2026-09-{i + 1:02d}", "wf_collapsed": collapsed, "in_sample_collapsed": False,
             "wf_shares": sh} for i in range(n)]


GOOD_CH = {"id": "ppo_robust_shadow", "verdict": "promote_recommended", "n_challenger": 40, "ci_lower": 0.01, "ci_upper": 0.2}


def test_collapsed_policy_is_not_ready():
    r = rp.assess(_meta(wf=(0, 0, 24)), _hist(3), GOOD_CH, [])
    assert r["status"] == "NOT_READY" and r["veto_enabled"] is False
    assert any("Walk-Forward kollabiert" in x for x in r["reasons"])


def test_live_action_collapse_blocks():
    rows = [{"features": {"rl_robust_action": "SKIP"}}] * 25
    r = rp.assess(_meta(), _hist(3), GOOD_CH, rows)
    assert r["status"] == "NOT_READY" and r["live_actions"]["collapsed"] is True


def test_accumulating_without_forward_evidence_or_stability():
    r = rp.assess(_meta(), _hist(3), {"verdict": "running", "n_challenger": 5}, [])
    assert r["status"] == "ACCUMULATING" and any("Forward-Mehrwert" in x for x in r["reasons"])
    r = rp.assess(_meta(), _hist(1), GOOD_CH, [])
    assert r["status"] == "ACCUMULATING" and any("Stabilität" in x for x in r["reasons"])
    shifting = _hist(2) + [{"trained_at": "2026-09-30", "wf_collapsed": False, "in_sample_collapsed": False,
                            "wf_shares": {"SKIP": 0.9, "NORMAL": 0.1, "BOOST": 0.0}}]
    assert rp.assess(_meta(), shifting, GOOD_CH, [])["status"] == "ACCUMULATING"


def test_candidate_only_reported_never_enabled():
    r = rp.assess(_meta(), _hist(3), GOOD_CH, [{"features": {"rl_robust_action": a}} for a in ("NORMAL", "BOOST") * 15])
    assert r["status"] == "RL PROMOTION CANDIDATE"
    assert r["veto_enabled"] is False and r["requires_human_approval"] is True


def test_run_appends_history_once_per_training(tmp_path, monkeypatch):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(_meta()))
    monkeypatch.setattr(rp, "_history", rp._history)
    out, hist = tmp_path / "out.json", tmp_path / "hist.jsonl"
    from modules import challenger as ch
    monkeypatch.setattr(ch, "load_ledger_rows", lambda *a, **k: [])
    monkeypatch.setattr(ch, "evaluate_all", lambda *a, **k: [])
    rp.run(meta, out, hist)
    rp.run(meta, out, hist)
    assert len(hist.read_text().splitlines()) == 1
    assert json.loads(out.read_text())["status"] == "ACCUMULATING"
    assert rp.run(tmp_path / "missing.json", out, hist)["status"] == "NOT_READY"


def test_config_keeps_rl_veto_disabled():
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    assert cfg["rl"]["veto_enabled"] is False
