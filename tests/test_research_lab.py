"""Research-Lab: Leakage-Schutz der Signal-Sprache, Duplikate, BH-Mehrfachtest,
Discovery nur im Discovery-Fenster, Locked höchstens einmal, Gedächtnis
(invalid_modified), Seeds. Synthetisch, kein Netz."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ml_research as T  # noqa: E402

from modules import ml_research as ml  # noqa: E402
from modules import research_lab as lab  # noqa: E402


@pytest.mark.parametrize("expr", ["fwd_xs_20", "mfe_20 * 2", "rank(label_end_20)", "__import__('os')",
                                  "mom_3m.shift(-1)", "close", "rank(mom_3m, 1)", "mom_3m ** 2", "'a'"])
def test_signal_language_blocks_labels_and_code(expr):
    with pytest.raises(lab.SignalError):
        lab.validate_expr(expr)


def test_signal_language_allows_pit_expressions():
    for e in ("rev_1m", "rank(mom_12_1) - rank(rev_1m)", "-vol_60 * sign(spy_trend_200)", "abs(ret_5d) / 2"):
        lab.validate_expr(e)
    assert lab.signal_key("rev_1m") == lab.signal_key(" rev_1m ")


def test_benjamini_hochberg():
    p = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5}
    assert lab.benjamini_hochberg(p, 0.10) == {"a", "b", "c"}
    assert lab.benjamini_hochberg(p, 0.01) == {"a"}
    assert lab.benjamini_hochberg({"x": 0.2}, 0.1) == set()


@pytest.fixture(scope="module")
def panel():
    p = T._panel(n_days=3200, n_stocks=60, signal=False)
    rnd = np.random.default_rng(11)
    # echter, stabiler Zusammenhang für vol_60 (negativ) über den gesamten Zeitraum
    p["fwd_xs_20"] = -p["vol_60"] * 0.08 + rnd.normal(0, 0.02, len(p))
    p.loc[p["label_end_20"].isna(), "fwd_xs_20"] = np.nan
    vix = pd.Series(rnd.uniform(12, 30, p["date"].nunique()), index=sorted(p["date"].unique()))
    p["vix"] = p["date"].map(vix)
    return p


def _hyps(tmp_path, hyps):
    path = tmp_path / "h.yaml"
    path.write_text(yaml.safe_dump({"hypotheses": hyps}))
    return path


def test_true_hypothesis_accepted_noise_rejected_and_duplicates(tmp_path, panel):
    hp = _hyps(tmp_path, [
        {"id": "H-TRUE", "signal": "vol_60", "direction": -1},
        {"id": "H-DUP", "signal": "rank(vol_60) * 2", "direction": -1},
        {"id": "H-NOISE", "signal": "relvol_5_60", "direction": 1},
        {"id": "H-LEAK", "signal": "fwd_xs_20", "direction": 1},
        {"id": "P-1", "title": "alt", "status": "prior_result", "evidence": "x"},
    ])
    db = lab.run(panel, hyp_path=hp, db_path=tmp_path / "db.json", with_discovery=False)
    h = db["hypotheses"]
    assert h["H-TRUE"]["status"] == "accepted" and h["H-TRUE"]["locked"]["base"]["mean"] > 0
    assert h["H-DUP"]["status"] == "duplicate" and h["H-DUP"]["duplicate_of"] == "H-TRUE"
    assert h["H-NOISE"]["status"] in ("rejected", "not_significant_after_fdr")
    assert h["H-LEAK"]["status"] == "rejected_leakage_or_invalid"
    assert h["P-1"]["status"] == "prior_result"
    assert (tmp_path / "hypothesis_db.md").exists()


def test_locked_evaluated_only_once_and_modification_detected(tmp_path, panel, monkeypatch):
    hp = _hyps(tmp_path, [{"id": "H-TRUE", "signal": "vol_60", "direction": -1}])
    dbp = tmp_path / "db.json"
    lab.run(panel, hyp_path=hp, db_path=dbp, with_discovery=False)
    calls = []
    monkeypatch.setattr(lab, "locked_check", lambda *a, **k: calls.append(1) or {"base": {"mean": 1}})
    lab.run(panel, hyp_path=hp, db_path=dbp, with_discovery=False)
    assert calls == []                                   # Locked nicht erneut
    hp2 = _hyps(tmp_path, [{"id": "H-TRUE", "signal": "vol_20", "direction": -1}])
    db = lab.run(panel, hyp_path=hp2, db_path=dbp, with_discovery=False)
    assert db["hypotheses"]["H-TRUE"]["status"] == "invalid_modified"


def test_discovery_uses_only_discovery_window(panel, monkeypatch):
    seen = []
    orig = ml.daily_ic

    def spy(df, score, target="fwd_xs_20"):
        seen.append(df["label_end_20"].max())
        return orig(df, score, target)
    monkeypatch.setattr(ml, "daily_ic", spy)
    d = lab.discover(panel, ml.PROTOCOL)
    end = pd.Timestamp(ml.PROTOCOL["periods"]["discovery_end"])
    assert seen and all(x < end for x in seen)
    assert d["n_tested"] == len(lab.discovery_candidates())
    assert any("vol_60" in s["signal"] for s in d["survivors"])
    for s in d["survivors"]:
        lab.validate_expr(s["signal"])


def test_seed_file_is_valid():
    hyps = lab.load_hypotheses()
    ids = [h["id"] for h in hyps]
    assert len(ids) == len(set(ids))
    for h in hyps:
        if h.get("status") in ("blocked_data", "prior_result"):
            assert h.get("evidence")
        else:
            lab.validate_expr(h["signal"])
            assert h["direction"] in (1, -1)
