"""Erfassungs-Bandit (option_learner.acquisition_*, candidate_ledger._research_selection).

Geprüft werden die Eigenschaften, auf denen die Off-Policy-Bewertung beruht:
  * Einschlusswahrscheinlichkeiten: Σ π = Budget, 0 < ε·B/N <= π <= 1, deterministisch, ohne Modell gleichverteilt;
  * die systematische Stichprobe zieht genau B Kandidaten mit P(gezogen) = π (empirisch);
  * Horvitz-Thompson ist unter dem Bandit-Design erwartungstreu für den Poolmittelwert (empirisch);
  * fail safe: ohne gültiges Modell oder bei Fehlern die bisherige Hash-Stichprobe; nie über Budget.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from modules import candidate_ledger as cl
from modules import option_learner as ol


def _model(k=3, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    names = list(ol.LEDGER_FEATURES[:k])
    cov = np.eye(k) * 0.01
    return {"policy": ol.ACQ_POLICY, "horizon": 5, "label_cutoff": "2026-11-08", "n": 200, "names": names,
            "median": [0.0] * k, "mu": [0.0] * k, "sd": [1.0] * k, "intercept": 0.0,
            "coef": (rng.normal(0, scale, k)).tolist(), "cov": cov.tolist(), "sigma2": 0.04, "model_id": "m1"}


def _xs(n, k=3, seed=1):
    rng = np.random.default_rng(seed)
    names = list(ol.LEDGER_FEATURES[:k])
    return [{nm: float(v) for nm, v in zip(names, rng.normal(0, 1, k))} for _ in range(n)]


def test_propensities_sum_to_budget_are_bounded_and_deterministic():
    xs, b = _xs(200), 30
    pis = ol.acquisition_propensities(xs, b, _model(), "acq:2026-11-09")
    assert sum(pis) == pytest.approx(b) and max(pis) <= 1.0
    assert min(pis) >= ol.ACQ_EPSILON * b / 200 - 1e-12                       # Gewichte beschränkt: <= 2N/B
    assert pis == ol.acquisition_propensities(xs, b, _model(), "acq:2026-11-09")
    assert ol.acquisition_propensities(xs, b, None, "x") == [b / 200] * 200
    assert ol.acquisition_propensities(xs[:10], 30, _model(), "x") == pytest.approx([1.0] * 10)   # Budget > Pool


def test_thompson_prefers_high_predicted_value_but_keeps_exploring():
    m = _model(scale=0.0)
    m["coef"] = [1.0, 0.0, 0.0]                                                 # nur Merkmal 0 zählt
    m["cov"] = (np.eye(3) * 1e-6).tolist()
    xs = _xs(100)
    pis = ol.acquisition_propensities(xs, 10, m, "s")
    f0 = [x[ol.LEDGER_FEATURES[0]] for x in xs]
    hi, lo = np.argsort(f0)[-10:], np.argsort(f0)[:10]
    assert np.mean([pis[i] for i in hi]) > 0.5 > np.mean([pis[i] for i in lo]) > 0.0


def test_systematic_sample_has_exact_size_and_inclusion_probabilities():
    xs, b = _xs(60), 12
    pis = ol.acquisition_propensities(xs, b, _model(), "acq:fix")
    keys = [f"s{i}" for i in range(60)]
    freq = np.zeros(60)
    runs = 4000
    for r in range(runs):
        picks = ol.systematic_sample(keys, pis, f"seed{r}")
        assert len(picks) == b
        freq[list(picks)] += 1
    assert np.max(np.abs(freq / runs - np.array(pis))) < 0.04                  # P(gezogen) = π


def test_horvitz_thompson_is_unbiased_under_the_bandit_design():
    rng = np.random.default_rng(7)
    xs, b = _xs(80), 16
    y = np.array([0.3 * x[ol.LEDGER_FEATURES[0]] + rng.normal(0, 0.2) for x in xs])   # Ergebnis hängt an x
    m = _model(scale=0.0)
    m["coef"] = [1.0, 0.0, 0.0]
    pis = np.array(ol.acquisition_propensities(xs, b, m, "acq:ht"))
    keys = [f"s{i}" for i in range(80)]
    est, naive = [], []
    for r in range(3000):
        picks = list(ol.systematic_sample(keys, list(pis), f"ht{r}"))
        est.append(float(np.sum(y[picks] / pis[picks]) / 80))
        naive.append(float(np.mean(y[picks])))
    assert abs(np.mean(est) - y.mean()) < 0.01                                 # erwartungstreu
    assert np.mean(naive) - y.mean() > 0.05                                    # ungewichtet verzerrt (bevorzugt hohe x)


def test_research_selection_falls_back_to_hash_sample_and_never_exceeds_budget(tmp_path, monkeypatch):
    pool = [(f"T{i}", {"signal_id": f"sig{i}", "features": {"rel_volume": float(i)}}, 10.0) for i in range(50)]

    def hash_rank(items, salt=""):
        return sorted(items, key=lambda it: it[1]["signal_id"])

    ids, design = cl._research_selection(pool, 8, "2026-11-09", hash_rank, model_dir=tmp_path)   # kein Modell
    assert len(ids) == 8 and all(d["policy"] == "uniform-hash" and d["pi"] == pytest.approx(8 / 50)
                                 for d in design.values())
    (tmp_path / ol.ACQ_MODEL).write_text(json.dumps({**_model(), "names": ["rel_volume", "scan_day_ret",
                                                                           "short_mom"]}))
    ids2, design2 = cl._research_selection(pool, 8, "2026-11-09", hash_rank, model_dir=tmp_path)
    assert len(ids2) == 8 and {d["policy"] for d in design2.values()} == {ol.ACQ_POLICY}
    assert sum(d["pi"] for d in design2.values()) == pytest.approx(8, abs=1e-5) and len(design2) == 50
    monkeypatch.setattr(ol, "acquisition_propensities", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    ids3, design3 = cl._research_selection(pool, 8, "2026-11-09", hash_rank, model_dir=tmp_path)
    assert ids3 == ids and design3 == design                                    # Fehler -> Hash-Stichprobe
    monkeypatch.undo()
    monkeypatch.setattr(cl, "_acquisition_bandit_enabled", lambda: False)                 # Abschalter
    assert cl._research_selection(pool, 8, "2026-11-09", hash_rank, model_dir=tmp_path) == (ids, design)
    old = {**_model(), "label_cutoff": "2026-10-01"}
    (tmp_path / ol.ACQ_MODEL).write_text(json.dumps(old))                       # veraltet -> gleichverteilt
    assert cl._research_selection(pool, 8, "2026-11-09", hash_rank, model_dir=tmp_path)[0] == ids


def _cands(n_days=12, per_day=30, seed=3):
    rng = np.random.default_rng(seed)
    cands, rows = [], []
    d0 = date(2026, 9, 1)
    for k in range(n_days):
        d = (d0 + timedelta(days=7 * k)).isoformat()
        for j in range(per_day):
            x = {"rel_volume": float(rng.normal()), "scan_day_ret": float(rng.normal()),
                 "sentiment_score": float(rng.normal())}
            y = float(0.05 * x["rel_volume"] + rng.normal(0, 0.3))
            sid = f"{d}-{j}"
            cands.append({"date": d, "signal_id": sid, "ticker": f"T{j}",
                          "exprs": [{"expression": "single", "status": "VERIFIED", "x": x, "y": y,
                                     "label_day": date.fromisoformat(d) + timedelta(days=8), "contract": [sid]}]})
            rows.append({"date": d, "signal_id": sid, "option_sample": "research_census", "direction": None,
                         "snapshot_budget": per_day, "snapshot_eligible_n": per_day * 4})
    return rows, cands


def test_acquisition_model_fit_is_prospective_and_loads_only_when_fresh(tmp_path):
    _rows, cands = _cands()
    day = date(2026, 11, 20)
    m = ol.write_acquisition_model(tmp_path, cands, day, datetime(2026, 11, 20, tzinfo=timezone.utc))
    assert m and m["n"] == 330                            # letzter Tag (Mark 25.11.) liegt nach dem Stichtag
    assert m["label_cutoff"] == "2026-11-20" and set(m["names"]) <= set(ol.LEDGER_FEATURES)
    assert ol.acquisition_fit(cands, date(2026, 9, 2)) is None                 # keine Labels vor dem Stichtag
    assert ol.load_acquisition_model(day, tmp_path)["model_id"] == m["model_id"]
    assert ol.load_acquisition_model(day + timedelta(days=8), tmp_path) is None
    assert ol.load_acquisition_model(day - timedelta(days=1), tmp_path) is None   # Modell aus der Zukunft


def test_acquisition_report_uses_logged_propensities_including_legacy_uniform_rows():
    rows, cands = _cands()
    rep = ol.acquisition_report(rows, cands, date(2026, 12, 31), 5)
    assert rep["propensity_sources"] == {"uniform-hash": 360} and rep["coverage"] == 1.0
    assert rep["max_weight"] == pytest.approx(4.0)                              # π = 30/120
    ys = {c["signal_id"]: c["exprs"][0]["y"] for c in cands}
    # HT je Tag = (1/N) Σ y/π = Mittel der gezogenen Labels (gleichverteilt), Mittel über Tage
    per_day = [np.mean([ys[c["signal_id"]] for c in cands if c["date"] == d]) for d in sorted({c["date"] for c in cands})]
    assert rep["values"]["trade_all"]["ht"]["mean_diff"] == pytest.approx(float(np.mean(per_day)), abs=1e-6)
    assert rep["values"]["acq_model_positive"]["ht"] is not None and rep["never_trade"] == 0.0
    assert rep["production_influence"] == "NONE"
