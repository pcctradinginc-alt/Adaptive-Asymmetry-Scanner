"""Alpha-Entdeckung: findet echte Signale, schweigt bei Rauschen, fällt nicht
auf Kalender-Vermengung herein, nutzt keine Outcome-Felder, und überführt
Treffer nur in PROSPEKTIVE Challenger."""
from __future__ import annotations

import json
import random
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from modules import alpha_discovery as ad
from modules import challenger, challenger_registrar as reg

REPO = Path(__file__).resolve().parent.parent


def _ledger(tmp_path, n_dates=80, per_date=8, signal=0.0, seed=1, noise_features=5,
            macro_regimes=None, leak=False):
    rnd = random.Random(seed)
    rows = []
    start = date(2026, 1, 1)
    for d in range(n_dates):
        day = (start + timedelta(days=d)).isoformat()
        day_shock = rnd.gauss(0, 0.08)                      # großer Markt-/Tageseffekt
        macro = None
        if macro_regimes:
            macro = float(d * macro_regimes // n_dates)       # wenige Regime, zeitlich geblockt
        for i in range(per_date):
            x = rnd.gauss(0, 1)
            y = day_shock + signal * x + rnd.gauss(0, 0.10)
            feats = {"x_signal": x, **{f"noise_{k}": rnd.gauss(0, 1) for k in range(noise_features)}}
            if leak:
                feats["realized_ret_hint"] = y              # muss ignoriert werden
            ext = {"primitives": {"macro_z": macro}} if macro is not None else None
            rows.append({"date": day, "ticker": f"T{d}_{i}", "direction": "BULLISH", "status": "rejected",
                         "features": feats, "external": ext,
                         "outcomes": {"ret_20d": round(y, 5), "ret_45d": round(y * 1.3, 5)}})
    led = tmp_path / "ledger"
    led.mkdir(exist_ok=True)
    (led / "2026-01.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return led


def _accepted(res):
    return [f["feature"] for f in res["findings"] if f["accepted"]]


def test_planted_cross_sectional_signal_is_found_despite_large_day_shocks(tmp_path):
    led = _ledger(tmp_path, signal=0.04)
    res = ad.discover(ad.load_rows(led, 20), 20)
    assert res["status"] == "OK"
    assert "features.x_signal" in _accepted(res)
    assert not any(k.startswith("features.noise_") for k in _accepted(res))


def test_pure_noise_yields_no_accepted_findings(tmp_path):
    led = _ledger(tmp_path, signal=0.0, noise_features=30, seed=7)
    res = ad.discover(ad.load_rows(led, 20), 20)
    assert _accepted(res) == []


def test_macro_feature_with_few_regimes_is_not_accepted(tmp_path):
    """Makro konstant je Tag, nur 2 zeitlich geblockte Regime -> Zeitreihen-
    test über Tage; ohne echten Effekt kein Treffer (April/Mai-Falle)."""
    led = _ledger(tmp_path, signal=0.0, macro_regimes=2, seed=3)
    res = ad.discover(ad.load_rows(led, 20), 20)
    assert "external.primitives.macro_z" not in _accepted(res)


def test_outcome_like_features_are_never_tested(tmp_path):
    led = _ledger(tmp_path, signal=0.0, leak=True)
    rows = ad.load_rows(led, 20)
    assert all("features.realized_ret_hint" not in r["num"] for r in rows)


def test_insufficient_data_is_reported(tmp_path):
    led = _ledger(tmp_path, n_dates=10)
    assert ad.discover(ad.load_rows(led, 20), 20)["status"].startswith("INSUFFICIENT_DATA")


def test_benjamini_hochberg():
    assert ad.benjamini_hochberg([0.001, 0.2, 0.03, 0.9], q=0.10) == [True, False, True, False]


def test_finding_becomes_prospective_challenger(tmp_path):
    led = _ledger(tmp_path, signal=0.04)
    props = tmp_path / "auto.yaml"
    today = date(2026, 4, 1)
    report = ad.run(ledger_dir=led, out_dir=tmp_path / "out", proposals_path=props, today=today)
    assert report["new_proposals"] and len(report["new_proposals"]) <= ad.MAX_NEW_PER_RUN
    registry = tmp_path / "challengers.yaml"
    shutil.copy(REPO / "challengers.yaml", registry)
    out = reg.run_auto(today=today, proposals_path=props, registry_path=registry,
                       now=datetime(2026, 4, 1, 6, tzinfo=timezone.utc))
    assert out["registered"] == report["new_proposals"]
    new = [c for c in challenger.load_registry(registry) if c["id"] in out["registered"]]
    assert new and all(c["start_date"] == "2026-04-02" for c in new)
    # Entdeckungsdaten (vor start_date) zählen nie zur Bestätigung
    rows = [json.loads(l) for l in (led / "2026-01.jsonl").read_text().splitlines()]
    res = challenger.evaluate(new[0], rows, date(2026, 6, 1), n_active=5)
    assert res["n_baseline"] == 0
    # idempotent
    assert reg.run_auto(today=today, proposals_path=props, registry_path=registry)["registered"] == []


def test_tampered_auto_proposal_is_refused(tmp_path):
    led = _ledger(tmp_path, signal=0.04)
    props = tmp_path / "auto.yaml"
    ad.run(ledger_dir=led, out_dir=tmp_path / "out", proposals_path=props, today=date(2026, 4, 1))
    cfg = yaml.safe_load(props.read_text())
    cfg["proposals"][0]["min_n"] = 5
    props.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    registry = tmp_path / "challengers.yaml"
    shutil.copy(REPO / "challengers.yaml", registry)
    out = reg.run_auto(today=date(2026, 4, 1), proposals_path=props, registry_path=registry)
    assert cfg["proposals"][0]["id"] not in out["registered"]


def test_gate_efficacy_flags_gate_that_rejects_winners(tmp_path):
    rnd = random.Random(5)
    rows = []
    for d in range(40):
        day = (date(2026, 1, 1) + timedelta(days=d)).isoformat()
        shock = rnd.gauss(0, 0.05)
        for i in range(6):
            grp = ["proposed", "bad_gate", "prescreen_no"][i % 3]
            bonus = 0.05 if grp == "bad_gate" else 0.0      # Gate verwirft Gewinner
            r = {"date": day, "ticker": f"T{d}{i}", "outcomes": {"ret_20d": shock + bonus + rnd.gauss(0, 0.03)}}
            if grp == "proposed":
                r["status"] = "proposed"
            else:
                r.update(status="rejected", reject_reason=grp)
            rows.append(r)
    led = tmp_path / "led"
    led.mkdir()
    (led / "2026-01.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    g = ad.gate_efficacy(led, 20)["groups"]
    assert g["bad_gate"]["mean_adj"] > g["proposed"]["mean_adj"]
    assert g["bad_gate"]["t_vs_day_mean"] > 2
