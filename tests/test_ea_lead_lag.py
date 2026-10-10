"""Expectation Alpha – Lead-Lag-Diagnostik: nur Research, kein Lookahead, keine automatische Horizont-Auswahl."""
from __future__ import annotations

import json

import numpy as np

from modules.expectation_alpha import config as eacfg
from modules.expectation_alpha import lead_lag as ll

CFG = eacfg.load()
HS = CFG["outcomes"]["horizons"]


def _data(n=120, seed=3, predictive_h=60):
    rng = np.random.default_rng(seed)
    rows, outs = [], {}
    for i in range(n):
        d = f"2027-{1 + (i // 28) % 12:02d}-{1 + i % 28:02d}"
        x = float(rng.normal())
        rows.append({"observation_id": f"o{i}", "date": d, "status": "TRADE", "regime": ("vix_low", "vix_high")[i % 2],
                     "macro_alignment": x, "env": {"ea_gap_aligned": i % 2, "ea_effective_confirmation_ratio": None},
                     "news_edge": {"impact": 5, "surprise": 4}})
        for h in HS:
            y = (0.05 * x if h == predictive_h else 0.0) + float(rng.normal(0, 0.01))
            outs[(f"o{i}", "UNDERLYING", "immediate", h)] = {"outcome_net": y, "mae": -0.02, "mfe": 0.03,
                                                             "exit_date": "2028-12-31"}
    return rows, outs


def test_all_preregistered_horizons_reported_no_best_selection():
    rows, outs = _data()
    rep = ll.run(rows, outs, CFG)
    assert rep["horizons_preregistered"] == HS and rep["selection"].startswith("keine")
    txt = json.dumps(rep).lower()
    assert "best" not in txt and "argmax" not in txt and "selected_horizon" not in txt
    for f in rep["features"].values():
        assert set(f["horizons"]) == {str(h) for h in HS}
    m = rep["features"]["macro_alignment"]["horizons"]
    assert m["60"]["status"] == "OK" and m["60"]["ic"] > 0.5            # echter Zusammenhang wird gemessen ...
    assert all(m[str(h)]["status"] == "OK" for h in HS)                  # ... und trotzdem alle Horizonte berichtet
    assert m["60"]["forward_return_spread"] > 0 and m["60"]["spread_ci90"][0] is not None
    assert set(m["60"]["regime_breakdown"]) == {"vix_low", "vix_high"}
    assert {"n", "independent_signal_days", "ic", "ic_ci90", "hit_rate_top", "mae_top", "mfe_top"} <= set(m["60"])


def test_need_more_data_and_missing_features():
    rows, outs = _data(n=20)
    rep = ll.run(rows, outs, CFG)
    assert rep["lead_lag_status"] == "NEED_MORE_DATA"
    assert rep["features"]["macro_alignment"]["horizons"]["60"]["status"] == "NEED_MORE_DATA"
    assert "ic" not in rep["features"]["macro_alignment"]["horizons"]["60"]          # keine Effektaussage
    rep2 = ll.run(*_data(), CFG)
    assert rep2["features"]["confirmation_ratio_effective"]["horizons"]["60"]["n"] == 0   # fehlend != 0


def test_no_lookahead_outcome_must_mature_after_decision():
    rows, outs = _data()
    for k in list(outs)[:40]:
        outs[k] = {**outs[k], "exit_date": "2026-01-01"}                 # vor der Entscheidung -> unzulässig
    p = ll._pairs(rows, outs, "row:macro_alignment", 60)
    assert all(q["date"] < "2028-12-31" for q in p)
    full = ll._pairs(*_data(), "row:macro_alignment", 60)
    assert len(p) < len(full)


def test_feature_from_frozen_snapshot_and_errors_excluded():
    rows, outs = _data()
    rows[0]["status"] = "ERROR"
    p = ll._pairs(rows, outs, "row:macro_alignment", 60)
    assert len(p) == len(rows) - 1                                        # ERROR-Zeile (Datenfehler) zählt nie
    assert ll.feature_value({"env": {"ea_gap_aligned": True}}, "env:ea_gap_aligned") is None   # bool/str nie als Zahl


def test_binary_feature_split_and_determinism():
    rows, outs = _data()
    a = ll.run(rows, outs, CFG)["features"]["gap_aligned"]["horizons"]["60"]
    b = ll.run(rows, outs, CFG)["features"]["gap_aligned"]["horizons"]["60"]
    assert a == b and a["split"] == "binary"


def test_every_feature_has_family():
    for name, (fam, spec) in ll.FEATURES.items():
        assert fam and spec.split(":")[0] in ("row", "env", "gap", "news")
    fams = {f for f, _ in ll.FEATURES.values()}
    assert {"expectation_gap", "rate_of_change", "percentile", "relative", "cross_asset", "regime",
            "regime_transition"} <= fams


def test_evaluation_contains_lead_lag_but_no_production_fields(tmp_path):
    from modules.expectation_alpha import evaluation as ev
    rep = ev.run(root=tmp_path, cfg=CFG, promotion_state={}, write=True)
    assert rep["lead_lag"]["lead_lag_status"] == "NEED_MORE_DATA"
    assert "Lead-Lag" in (tmp_path / "evaluation.md").read_text()
    assert rep["proposals"]["auto_applied"] is False
