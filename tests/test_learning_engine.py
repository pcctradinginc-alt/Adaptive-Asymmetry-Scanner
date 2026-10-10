"""Learning/Evidence Engine (Master-Auftrag 2026-10-10) – rein technische Tests mit synthetischen Daten.
Synthetische Daten beweisen nur die Mechanik (Leakage-Schutz, Lernkreislauf, Rollback), nie wirtschaftliches Alpha."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from modules import evidence_core as ec
from modules import evidence_ledger as el
from modules import meta_label as ml

D0 = date(2025, 1, 6)


def _recs(n=600, seed=1, signal=1.5, days=120, h=20):
    """Kandidaten über `days` Signaltage; Label hängt (bei signal>0) von 'impact' ab."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        d = D0 + timedelta(days=int(i * days / n))
        imp = float(rng.integers(1, 10))
        net = signal * (imp - 5) / 100 + float(rng.normal(0, 0.03))
        out.append({"schema": el.SCHEMA, "prediction_id": f"s{i}:{h}d", "population": "CANDIDATE", "entity": f"T{i % 50}",
                    "decision_time": f"{d.isoformat()}T15:00:00+00:00", "decision_date": d.isoformat(),
                    "direction": 1 if i % 3 else -1, "features": {"impact": imp, "surprise": float(rng.integers(1, 9)),
                                                                   "scan_day_ret": float(rng.normal(0, .02))},
                    "horizon": h, "horizon_unit": "calendar", "outcome": net + 0.002, "outcome_net": net,
                    "quality": "PROXY", "price_type": "UNDERLYING_PROXY",
                    "label_available_at": (d + timedelta(days=h + 1)).isoformat()})
    return out


# ── evidence_core ───────────────────────────────────────────────────────────
def test_purged_walk_forward_has_no_label_overlap_and_respects_embargo():
    recs = _recs()
    dates = [r["decision_date"] for r in recs]
    ends = [r["label_available_at"] for r in recs]
    folds = ec.purged_walk_forward(dates, ends, n_folds=4, embargo_days=5, min_train=10, min_test=10)
    assert folds
    for f in folds:
        ts = date.fromisoformat(f["test_start"])
        assert all(date.fromisoformat(ends[i]) < ts - timedelta(days=5) for i in f["train_idx"])   # Purge+Embargo
        assert not set(f["train_idx"]) & set(f["test_idx"])


def test_group_bootstrap_draws_whole_groups():
    lo, hi = ec.group_bootstrap_ci([1.0, 1.0, -1.0, -1.0], ["a", "a", "b", "b"], n=500)
    assert lo is not None and lo <= 0 <= hi
    assert ec.group_bootstrap_ci([1.0, 2.0], ["a", "a"]) == (None, None)             # < 2 Gruppen: keine Aussage
    assert ec.bonferroni_alpha(0.05, 2) == 0.025


# ── evidence_ledger ─────────────────────────────────────────────────────────
def test_labeled_never_returns_labels_unknown_at_training_time():
    recs = _recs(n=100)
    cut = D0 + timedelta(days=60)
    lab = el.labeled(recs, as_of=cut, population="CANDIDATE", horizon=20)
    assert lab and all(date.fromisoformat(r["label_available_at"]) <= cut for r in lab)
    assert len(lab) < len(recs)


def test_champion_trades_carry_price_type_and_quality():
    hist = {"closed_trades": [
        {"ticker": "A", "entry_date": "2026-10-01", "close_date": "2026-10-20", "outcome": 0.2,
         "outcome_method": "spread_quote", "outcome_reliable": True, "strategy": "BULL_CALL_SPREAD", "option": {}},
        {"ticker": "B", "entry_date": "2026-10-01", "close_date": "2026-10-20", "outcome": 0.5,
         "outcome_method": "delta_approx", "outcome_reliable": False, "strategy": "LONG_CALL", "option": {}}]}
    r = {x["entity"]: x for x in el.from_champion_trades(hist)}
    assert r["A"]["quality"] == "RELIABLE" and r["A"]["price_type"] == "OBSERVED_QUOTE"
    assert r["B"]["quality"] != "RELIABLE" and r["B"]["price_type"] == "MODELED"


# ── meta_label ──────────────────────────────────────────────────────────────
def test_need_more_data_without_model(tmp_path):
    st = ml.train(recs=_recs(n=50, days=10), root=tmp_path, as_of=D0 + timedelta(days=200))
    assert st["status"] == "NEED_MORE_DATA" and not (tmp_path / "registry.json").exists()


def test_learning_cycle_new_outcomes_enter_next_version(tmp_path):
    recs = _recs()
    t1 = datetime(2025, 4, 1, tzinfo=timezone.utc)
    s1 = ml.train(recs=recs, root=tmp_path, as_of=t1.date(), now=t1)
    assert s1["status"] == "TRAINED"
    ev = s1["evaluation"]
    assert ev["status"] == "OK" and set(ev["models"]) == {"logit_l2", "hgb"}
    assert ev["models"]["logit_l2"]["auc"] > 0.6                       # geplantes Signal wird erkannt (Mechanik)
    s_same = ml.train(recs=recs, root=tmp_path, as_of=t1.date(), now=t1 + timedelta(hours=1))
    assert s_same["status"] == "UNCHANGED"                              # keine neuen Labels -> keine neue Version
    t2 = datetime(2025, 5, 15, tzinfo=timezone.utc)
    s2 = ml.train(recs=recs, root=tmp_path, as_of=t2.date(), now=t2)
    assert s2["status"] == "TRAINED" and s2["new_labels_since_previous"] > 0      # neue Outcomes im nächsten Zyklus
    reg = ml._registry(tmp_path)
    assert [v["version"] for v in reg["versions"]] == [s1["version"], s2["version"]] and reg["active"] == s2["version"]
    m2 = json.loads((tmp_path / "versions" / s2["version"] / "manifest.json").read_text())
    assert m2["label_cutoff"] <= t2.date().isoformat()                  # nie ein Label nach der Trainingszeit
    assert m2["previous_version"] == s1["version"] and m2["n_train"] > s1["n_labels"]


def test_noise_challenger_is_not_better_and_baseline_stays_active(tmp_path):
    t = datetime(2025, 5, 15, tzinfo=timezone.utc)
    st = ml.train(recs=_recs(signal=0.0, seed=7), root=tmp_path, as_of=t.date(), now=t)
    assert st["status"] == "TRAINED"
    assert st["evaluation"]["models"]["hgb"]["verdict"] == "NOT_BETTER"
    assert st["active_model"] == "logit_l2"                             # Challenger ohne Evidenz wird nicht aktiv


def test_predict_is_point_in_time_and_rollback_and_integrity(tmp_path):
    recs = _recs()
    t1, t2 = datetime(2025, 4, 1, tzinfo=timezone.utc), datetime(2025, 5, 15, tzinfo=timezone.utc)
    s1 = ml.train(recs=recs, root=tmp_path, as_of=t1.date(), now=t1)
    s2 = ml.train(recs=recs, root=tmp_path, as_of=t2.date(), now=t2)
    f = {"impact": 9, "surprise": 5}
    assert ml.predict_one(f, "BULLISH", as_of=t2 - timedelta(days=1), root=tmp_path) is None   # Modell jünger als Entscheidung
    p = ml.predict_one(f, "BULLISH", as_of=t2 + timedelta(days=1), root=tmp_path)
    assert p and p["version"] == s2["version"] and 0 <= p["p"] <= 1
    ml.rollback(root=tmp_path, reason="Test")
    assert ml._registry(tmp_path)["active"] == s1["version"]
    assert ml._registry(tmp_path)["history"][-1]["action"] == "ROLLBACK"
    # Rollback hält: gleiche Labels -> kein Neutraining; neue Labels -> neue Version, aber NICHT aktiv
    assert ml.train(recs=recs, root=tmp_path, as_of=t2.date(), now=t2 + timedelta(days=8))["status"] == "UNCHANGED"
    t3 = datetime(2025, 6, 1, tzinfo=timezone.utc)
    s3 = ml.train(recs=recs, root=tmp_path, as_of=t3.date(), now=t3)
    assert s3["status"] == "TRAINED" and s3["activated"] is False and ml._registry(tmp_path)["active"] == s1["version"]
    assert ml._registry(tmp_path)["history"][-1]["action"] == "TRAINED_HELD"
    ml.release(root=tmp_path, reason="geprüft")
    assert ml._registry(tmp_path)["active"] == s3["version"] and "hold" not in ml._registry(tmp_path)
    ml.rollback(root=tmp_path, to_version=s1["version"], reason="Test")
    v, _ = ml.active_model(tmp_path)
    art = tmp_path / v["artifacts"][v["active_model"]]["path"]
    art.write_bytes(art.read_bytes() + b"x")                           # manipuliertes Artefakt
    ml._CACHE.clear()
    assert ml.active_model(tmp_path) == (None, None)                   # fail-safe: kein Modell, keine Prognose
    assert ml.predict_one(f, "BULLISH", as_of=t2 + timedelta(days=1), root=tmp_path) is None
    with pytest.raises(ValueError):
        ml.rollback(root=tmp_path, to_version="gibt-es-nicht")


def test_shadow_predictions_only_after_training_and_immutable(tmp_path):
    recs = _recs()
    t1 = datetime(2025, 4, 1, tzinfo=timezone.utc)
    ml.train(recs=recs, root=tmp_path, as_of=t1.date(), now=t1)
    n = ml.record_predictions(recs=recs, root=tmp_path, now=t1 + timedelta(days=1))
    later = [r for r in recs if r["decision_time"] > t1.isoformat()]
    assert n == len(later) and n > 0                                   # nur Entscheidungen NACH dem Training
    assert ml.record_predictions(recs=recs, root=tmp_path, now=t1 + timedelta(days=1)) == 0     # idempotent


def test_retrain_throttle_and_artifact_retention(tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "KEEP_ARTIFACT_VERSIONS", 2)
    recs = _recs(n=2000, days=200)                                                # ~10 Labels je Tag
    t = datetime(2025, 4, 1, tzinfo=timezone.utc)
    vs = [ml.train(recs=recs, root=tmp_path, as_of=t.date(), now=t)["version"]]
    early = ml.train(recs=recs, root=tmp_path, as_of=(t + timedelta(days=3)).date(), now=t + timedelta(days=3))
    assert early["status"] == "UNCHANGED" and "T alt" in early["reason"]          # höchstens wöchentlich
    for k in range(1, 4):
        tk = t + timedelta(days=14 * k)
        st = ml.train(recs=recs, root=tmp_path, as_of=tk.date(), now=tk)
        assert st["status"] == "TRAINED" and st["new_labels_since_previous"] >= ml.RETRAIN_MIN_NEW_LABELS
        vs.append(st["version"])
    reg = ml._registry(tmp_path)
    pruned = [v["version"] for v in reg["versions"] if v.get("artifacts_pruned")]
    assert pruned == vs[:2]                                                        # nur die zwei neuesten behalten
    for v in vs[:2]:
        m = json.loads((tmp_path / "versions" / v / "manifest.json").read_text())
        assert "train_ids" not in m and m["train_ids_hash"] and not list((tmp_path / "versions" / v).glob("*.pkl"))
    with pytest.raises(ValueError):
        ml.rollback(root=tmp_path, to_version=vs[0])                               # ohne Artefakt kein Ziel
    assert ml.active_model(tmp_path)[0]["version"] == vs[-1]


def test_shadow_predictions_dedupe_across_months(tmp_path):
    recs = _recs()
    t1 = datetime(2025, 4, 1, tzinfo=timezone.utc)
    ml.train(recs=recs, root=tmp_path, as_of=t1.date(), now=t1)
    n1 = ml.record_predictions(recs=recs, root=tmp_path, now=t1 + timedelta(days=1))
    assert n1 > 0 and ml.record_predictions(recs=recs, root=tmp_path, now=t1 + timedelta(days=40)) == 0
