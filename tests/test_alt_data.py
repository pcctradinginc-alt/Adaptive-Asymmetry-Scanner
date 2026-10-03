"""Alternative Data: Protokoll-Pin, Feature-Registry/-Store, Redundanzprüfung,
inkrementelle Bewertung (Baseline vs. Baseline + Quelle), Entscheidungsregel,
deterministischer Source Value Score."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ml_research as T  # noqa: E402

from modules import ml_research as ml  # noqa: E402
from modules.alt_data import evaluate as ev  # noqa: E402
from modules.alt_data import feature_store as fs  # noqa: E402
from modules.alt_data.registry import ALT_FEATURES, SOURCES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PINNED = "32c319716c0a677493d92017ab23dfaba83a28dda7878a3d4a45f0dad5992104"


def test_protocol_pinned():
    h = hashlib.sha256((ROOT / "config" / "alt_data_protocol.yaml").read_bytes()).hexdigest()
    assert h == PINNED, "config/alt_data_protocol.yaml geändert – nur strenger zulässig (CODEOWNERS)"


def test_registry_and_extra_features_guard():
    assert "sec_insider_buy_value_90d" in ALT_FEATURES and SOURCES["sec_deep_events"]["availability_col"]
    spec = {"id": "x", "model": "elastic_net", "target": "fwd_xs_20", "features": "all",
            "extra_features": ["sec_insider_buy_value_90d"]}
    assert ml.feature_list(spec)[-1] == "sec_insider_buy_value_90d"
    assert "sec_insider_buy_value_90d" not in ml.feature_list({**spec, "extra_features": []})   # Champion unberührt
    with pytest.raises(ValueError):
        ml.feature_list({**spec, "extra_features": ["fwd_xs_20"]})                               # nie Labels/Unbekanntes


def test_feature_store_attach_missing_source_is_nan_not_zero(tmp_path):
    panel = pd.DataFrame({"date": pd.to_datetime(["2024-01-05", "2024-01-05"]), "ticker": ["A", "B"]})
    src = {"s": {"features": ["sec_insider_buy_value_90d"], "availability_col": "alt_sec_available",
                 "path": str(tmp_path / "missing.csv.gz")}}
    out = fs.attach(panel, src)
    assert out["sec_insider_buy_value_90d"].isna().all() and (out["alt_sec_available"] == 0).all()
    pd.DataFrame({"date": ["2024-01-05"], "ticker": ["A"], "sec_insider_buy_value_90d": [3.2],
                  "alt_sec_available": [1.0]}).to_csv(tmp_path / "f.csv.gz", index=False, compression="gzip")
    src["s"]["path"] = str(tmp_path / "f.csv.gz")
    out = fs.attach(panel, src).set_index("ticker")
    assert out.loc["A", "sec_insider_buy_value_90d"] == 3.2 and np.isnan(out.loc["B", "sec_insider_buy_value_90d"])
    assert out.loc["B", "alt_sec_available"] == 0.0


@pytest.fixture(scope="module")
def alt_panel():
    p = T._panel(n_days=1900, n_stocks=60, signal=True)
    rnd = np.random.default_rng(11)
    p["sec_insider_buy_value_90d"] = rnd.normal(0, 1, len(p))                           # orthogonal + informativ
    p["fwd_xs_20"] = p["fwd_xs_20"] + 0.03 * p["sec_insider_buy_value_90d"]
    p.loc[p["label_end_20"].isna(), "fwd_xs_20"] = np.nan
    p["sec_insider_net_value_90d"] = p["mom_3m"] + rnd.normal(0, 0.01, len(p))           # redundant
    p["sec_8k_negative_90d"] = rnd.normal(0, 1, len(p))                                  # Rauschen
    for f in ALT_FEATURES:
        if f not in p:
            p[f] = np.nan
    p["alt_sec_available"] = 1.0
    p["vix"] = 15.0 + 10 * (p["date"].dt.month % 2)
    p["sector"] = np.where(p["ticker"].str[-1].astype(int) % 2 == 0, "Tech", "Energy")
    return p


@pytest.fixture
def small_protocol(monkeypatch):
    ap = {**ev.AP, "evaluation": {**ev.AP["evaluation"], "dev_years": [2019, 2020, 2021], "bootstrap_n": 300},
          "feature_selection": {**ev.AP["feature_selection"], "selection_years": [2016, 2017, 2018]}}
    monkeypatch.setattr(ev, "AP", ap)
    return ap


SPEC = {"enet": {"id": "enet", "model": "elastic_net", "target": "fwd_xs_20", "features": "all",
                 "params_grid": {"alpha": [0.0005], "l1_ratio": [0.5]}}}


def test_screen_flags_redundancy_and_selects_informative(alt_panel, small_protocol):
    feats = ["sec_insider_buy_value_90d", "sec_insider_net_value_90d", "sec_8k_negative_90d", "sec_late_filing_365d"]
    sc = ev.feature_screen(alt_panel, feats, [2016, 2017, 2018])
    assert sc["sec_insider_net_value_90d"]["max_abs_corr_existing"] > 0.9
    assert sc["sec_insider_net_value_90d"]["most_similar_existing"] == "mom_3m"
    assert sc["sec_late_filing_365d"]["coverage"] == 0.0
    chosen = ev.select_features(sc)
    assert chosen[0] == "sec_insider_buy_value_90d" and "sec_insider_net_value_90d" not in chosen
    assert "sec_late_filing_365d" not in chosen


def test_incremental_value_detected_and_noise_rejected(alt_panel, small_protocol):
    src = {"features": ["sec_insider_buy_value_90d", "sec_insider_net_value_90d"], "availability_col": "alt_sec_available"}
    r = ev.evaluate_source(alt_panel, "informative", src, SPEC)
    b = r["baselines"]["enet"]
    assert r["selected_features"] == ["sec_insider_buy_value_90d"]
    assert b["delta"]["ic"] > 0 and b["bootstrap"]["delta_monthly_mean"] > 0
    assert r["verdict"] in ("KEEP", "MODIFY") and set(b["breakdown"]["regime"]) and set(b["breakdown"]["sector"])
    noise = alt_panel.assign(fwd_xs_20=alt_panel["fwd_xs_20"] - 0.03 * alt_panel["sec_insider_buy_value_90d"])
    noise.loc[noise["label_end_20"].isna(), "fwd_xs_20"] = np.nan
    rn = ev.evaluate_source(noise, "noise", {"features": ["sec_8k_negative_90d"], "availability_col": "alt_sec_available"},
                            SPEC)
    assert rn["verdict"] != "KEEP"
    low = ev.evaluate_source(alt_panel.assign(alt_sec_available=0.0), "low", src, SPEC)
    assert low["verdict"] == "REJECT" and "Abdeckung" in low["verdict_reason"]


def test_decision_rule_and_value_score():
    mk = lambda lo, d, brier=0.0: {"bootstrap": {"ci_monthly_mean": [lo, lo + 0.01], "delta_monthly_mean": d},
                                   "delta": {"brier": brier}, "breakdown": {"sector": {}}, "delta_by_year": {"2020": d}}
    assert ev.decide({"baselines": {"a": mk(0.001, 0.003)}}, 0.9)[0] == "KEEP"
    assert ev.decide({"baselines": {"a": mk(0.001, 0.003, brier=0.01)}}, 0.9)[0] != "KEEP"     # Kalibrierung schlechter
    assert ev.decide({"baselines": {"a": mk(-0.002, 0.001)}}, 0.9)[0] == "MODIFY"
    assert ev.decide({"baselines": {"a": mk(-0.004, -0.001)}}, 0.9)[0] == "REJECT"
    s1 = ev.source_value_score({"baselines": {"a": mk(0.001, 0.004)}, "coverage_dev": 0.8},
                               {"last_observation": pd.Timestamp.now(tz="UTC").isoformat(), "error_rate": 0.0}, 0.95)
    s2 = ev.source_value_score({"baselines": {"a": mk(-0.004, -0.004)}, "coverage_dev": 0.2}, None, None)
    assert 0 <= s2["score"] < s1["score"] <= 1 and s1["components"]["api_reliability"] == 1.0


# ── Hypothesen-Verträge ─────────────────────────────────────────────────────

def test_contracts_valid_hash_frozen_and_modification_rejected(tmp_path):
    from modules.alt_data import contracts as ac
    cs = ac.load()
    assert {c["hypothesis_id"] for c in cs} >= {"ALT-SEC-001", "ALT-SEC-002", "ALT-SEC-003", "ALT-SEC-004"}
    reg = tmp_path / "reg.jsonl"
    st = ac.register(cs, reg, now="2026-10-02T00:00:00+00:00")
    assert all(v["status"] == "VALID" for v in st.values()), st
    assert ac.register(cs, reg)["ALT-SEC-001"]["status"] == "VALID"                     # unverändert
    mod = [dict(c, threshold="anders") if c["hypothesis_id"] == "ALT-SEC-001" else c for c in cs]
    assert ac.register(mod, reg)["ALT-SEC-001"]["status"] == "INVALID_MODIFIED"
    bad = [{**cs[0], "hypothesis_id": "X1", "signal": "fwd_xs_20"}, {**cs[0], "hypothesis_id": "X2", "horizon": 60},
           {**cs[0], "hypothesis_id": "X3", "source_features": ["mom_3m"]}, {"hypothesis_id": "X4"}]
    st2 = ac.register(bad, reg)
    assert all(v["status"] == "INVALID" for v in st2.values())
    lab = ac.lab_hypotheses(cs, st)
    assert lab[0]["signal"] == cs[0]["signal"] and lab[0]["contract_hash"] and lab[0]["source"] == "alt_data:sec_deep_events"


def test_forward_ledger_append_only(tmp_path, alt_panel):
    from modules.alt_data import contracts as ac
    c = [{**ac.load()[0], "forward_start": "2020-06-01", "signal": "sec_insider_buy_value_90d",
          "source_features": ["sec_insider_buy_value_90d"]}]
    st = {c[0]["hypothesis_id"]: {"status": "VALID", "spec_hash": "h"}}
    led = tmp_path / "f.jsonl"
    n1 = ac.record_forward(alt_panel, c, st, led)
    assert n1 > 0 and ac.record_forward(alt_panel, c, st, led) == 0
    rows = [json.loads(x) for x in led.read_text().splitlines()]
    assert all(r["date"] >= "2020-06-01" and len(r["top_decile"]) >= 1 for r in rows)


def test_lab_blocks_contracts_without_features(tmp_path, monkeypatch):
    from modules import research_lab as rl
    p = T._panel(n_days=700, n_stocks=55, signal=False)
    monkeypatch.setattr(rl, "DIRECTOR_PATH", tmp_path / "none.json")
    import json as _json
    (tmp_path / "fac.json").write_text(_json.dumps({"hypotheses": [
        {"id": "FAC-X", "title": "t", "signal": "rank(sec_insider_net_value_90d) * step(-mom_3m)", "direction": 1}]}))
    monkeypatch.setattr(rl, "FACTORY_PATH", tmp_path / "fac.json")
    from modules.alt_data import contracts as ac
    monkeypatch.setattr(ac, "REGISTRY", tmp_path / "reg.jsonl")
    hyp = tmp_path / "h.yaml"
    hyp.write_text("hypotheses: []\n")
    monkeypatch.setattr(rl, "HYP_CONFIG", hyp)
    db = rl.run(p, hyp_path=hyp, db_path=tmp_path / "db.json", with_discovery=False)
    recs = db["hypotheses"]
    assert recs["ALT-SEC-001"]["status"] in ("blocked_data",) and "fehlen" in str(recs["ALT-SEC-001"])
    assert recs["FAC-X"]["status"] == "blocked_data"                               # Fabrik: nie Absturz, nie 0


def test_feature_store_attach_holiday_week_uses_earlier_cutoff_only(tmp_path):
    pd.DataFrame({"date": ["2024-03-22", "2024-03-29"], "ticker": ["A", "A"],
                  "sec_insider_buy_value_90d": [1.0, 2.0], "alt_sec_available": [1.0, 1.0]}
                 ).to_csv(tmp_path / "f.csv.gz", index=False, compression="gzip")
    src = {"s": {"features": ["sec_insider_buy_value_90d"], "availability_col": "alt_sec_available",
                 "path": str(tmp_path / "f.csv.gz")}}
    panel = pd.DataFrame({"date": pd.to_datetime(["2024-03-28", "2024-03-29", "2024-04-25"]), "ticker": ["A"] * 3})
    out = fs.attach(panel, src)
    assert out["sec_insider_buy_value_90d"].iloc[0] == 1.0          # Gründonnerstag: Vorwoche, nie 29.03.
    assert out["sec_insider_buy_value_90d"].iloc[1] == 2.0
    assert np.isnan(out["sec_insider_buy_value_90d"].iloc[2]) and out["alt_sec_available"].iloc[2] == 0.0


def test_attach_handles_datetime_unit_mismatch_and_never_breaks(tmp_path, monkeypatch):
    """CI 2026-10-03: Panel M8[s] vs. Feature-Datei M8[us] -> MergeError brach den ML-Lauf ab."""
    pd.DataFrame({"date": ["2024-03-22"], "ticker": ["A"], "sec_insider_buy_value_90d": [1.5],
                  "alt_sec_available": [1.0]}).to_csv(tmp_path / "f.csv.gz", index=False, compression="gzip")
    src = {"s": {"features": ["sec_insider_buy_value_90d"], "availability_col": "alt_sec_available",
                 "path": str(tmp_path / "f.csv.gz")}}
    panel = pd.DataFrame({"date": pd.to_datetime(["2024-03-22", "2024-03-29"]).astype("datetime64[s]"),
                          "ticker": ["A", "A"], "x": [1, 2]})
    out = fs.attach(panel, src)
    assert out["sec_insider_buy_value_90d"].iloc[0] == 1.5 and out["date"].dtype == panel["date"].dtype
    assert list(out["x"]) == [1, 2]
    monkeypatch.setattr(fs, "_attach_one", lambda *a, **k: (_ for _ in ()).throw(ValueError("kaputt")))
    safe = fs.attach(panel, src)                                       # Fehler -> NaN, nie Abbruch
    assert safe["sec_insider_buy_value_90d"].isna().all() and (safe["alt_sec_available"] == 0).all()


def test_breakdown_when_positions_already_carry_sector_and_vix():
    panel = pd.DataFrame({"date": pd.to_datetime(["2024-01-05"] * 2), "ticker": ["A", "B"],
                          "sector": ["Tech", "Energy"], "vix": [25.0, 25.0]})
    pos = panel.assign(ret=[0.01, -0.02])                       # Positionen mit eigenen sector/vix-Spalten
    out = ev.breakdown(pos, pos, panel)
    assert set(out["regime"]) == {"vix_ge_20"} and set(out["sector"]) == {"Tech", "Energy"}
