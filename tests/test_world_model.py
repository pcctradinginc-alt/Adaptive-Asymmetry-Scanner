"""World Model: PIT-Makro (Vintages, Veröffentlichung), Normierung ohne
Look-ahead, Zustände/Unsicherheit, Validierungslogik (informativ -> KEEP,
Rauschen -> REJECT), Protokoll-Pin. Synthetisch, kein Netz."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules import world_model as wm
from modules.external.pit import AvailabilityPrecision, Observation

ROOT = Path(__file__).resolve().parent.parent
UTC = timezone.utc
EXPECTED_SHA = "d3f7ed7bae3586a12b82cc36218f545e95da6e815ef5ba17e41f42a742307709"


def test_intelligence_protocol_hash_pinned():
    h = hashlib.sha256((ROOT / "config" / "intelligence_protocol.yaml").read_bytes()).hexdigest()
    assert h == EXPECTED_SHA, "config/intelligence_protocol.yaml geändert – nur neue Abschnitte/strenger zulässig"


def _o(metric, period, value, avail, src="fred_us_macro"):
    return Observation(source_id=src, dataset="d", series_id=metric, entity_id="US", metric=metric, value=value,
                       unit="u", observation_time=period, available_at=avail, retrieved_at=avail,
                       vintage_time=avail, availability_precision=AvailabilityPrecision.EXACT_DATE, parser_version="1")


def test_pit_snapshots_respect_publication_and_revisions():
    p = datetime(2020, 1, 1, tzinfo=UTC)
    obs = [_o("us_indpro", p, 100.0, datetime(2020, 2, 15, tzinfo=UTC)),
           _o("us_indpro", p, 105.0, datetime(2020, 3, 15, tzinfo=UTC))]          # Revision
    d = [pd.Timestamp("2020-02-10"), pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-20")]
    s = wm.pit_snapshots(obs, "us_indpro", d)
    assert s[d[0]].empty                                                            # noch nicht veröffentlicht
    assert s[d[1]].iloc[-1] == 100.0 and s[d[2]].iloc[-1] == 105.0                  # Revision erst ab Veröffentlichung


def test_macro_indicator_stale_is_missing_not_default():
    snap = pd.Series({pd.Timestamp("2020-01-01", tz="UTC"): 100.0, pd.Timestamp("2019-01-01", tz="UTC"): 95.0})
    snap = snap.sort_index()
    assert wm.macro_indicator(snap, ("chg_days", 365), 75, pd.Timestamp("2020-02-20")) == pytest.approx(100 / 95 - 1)
    assert np.isnan(wm.macro_indicator(snap, ("chg_days", 365), 75, pd.Timestamp("2020-09-01")))


def test_expanding_z_uses_only_past():
    ind = pd.DataFrame({"a": np.arange(300, dtype=float)}, index=pd.date_range("2015-01-02", periods=300, freq="W-FRI"))
    z1 = wm.expanding_z(ind)
    ind2 = ind.copy()
    ind2.iloc[200:] = 1e6                                                           # Zukunft manipulieren
    z2 = wm.expanding_z(ind2)
    assert z1.iloc[:200].equals(z2.iloc[:200])
    assert z1.iloc[:104].isna().all().all()                                         # erst nach 2 Jahren Historie


def test_world_states_labels_and_uncertainty():
    z = pd.DataFrame({"indpro_yoy": [1.5, 0.0, -1.2], "copper_gold_63d": [1.0, np.nan, -1.0], "cyc_def_63d": [2.0, 0.1, -0.8]},
                     index=pd.date_range("2020-01-03", periods=3, freq="W-FRI"))
    st = wm.world_states(z)
    assert list(st["growth_state"]) == ["high", "neutral", "low"]
    assert st["earnings_momentum_state"].iloc[0] == "unavailable"
    assert st["labour_market_state"].iloc[0] == "no_data"
    assert st["growth_uncertainty"].iloc[0] < st["growth_uncertainty"].iloc[1]     # weit weg von der Schwelle + volle Abdeckung


def _synthetic_market(n=2600, seed=0):
    rnd = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    # latenter Stressfaktor: persistente AR(1); steuert künftige Vola und ist im HYG/LQD-Verhältnis sichtbar
    f = np.zeros(n)
    for i in range(1, n):
        f[i] = 0.995 * f[i - 1] + rnd.normal(0, 0.05)
    vol = 0.008 * np.exp(0.8 * np.roll(f, -10))                                     # Vola folgt dem Faktor mit Vorlauf
    spy = 100 * np.cumprod(1 + rnd.normal(0.0003, 1, n) * vol)
    px = pd.DataFrame({"SPY": spy, "^VIX": 18 + rnd.normal(0, 2, n), "^TNX": 3 + rnd.normal(0, 0.1, n).cumsum() * 0.01,
                       "^IRX": 2.0, "IEF": 100 * np.cumprod(1 + rnd.normal(0, 0.003, n)),
                       "HYG": 100 * np.exp(-0.2 * f), "LQD": 100.0}, index=idx)
    for s in wm.CYCLICALS + wm.DEFENSIVES + ("XLK", "XLE"):
        px[s] = 100 * np.cumprod(1 + rnd.normal(0, 0.01, n))
    return px


def test_validation_keeps_informative_and_rejects_noise():
    px = _synthetic_market(n=3500)
    ind, st = wm.build_world(px, {}, None)
    st = st.dropna(how="all", subset=[c for c in st.columns if c.endswith("_score")])
    tg = wm.targets(px, list(st.index))
    mk = wm.market_frame(px).reindex(st.index)
    base = pd.DataFrame({"vix": mk["vix"], "spy_trend_200": mk["spy_trend_200"]}, index=st.index)
    v = wm.validate(st, base, tg, pd.Timestamp("2025-06-01"))
    assert "rv20" in v["better"], v["targets"]["rv20"]                              # Kredit-Dimension trägt Vola-Info
    assert v["verdict"] in ("KEEP", "MODIFY")
    # Rauschen: Welt-Zustand ohne Bezug
    st2 = st.copy()
    rnd = np.random.default_rng(3)
    for c in [c for c in st2.columns if c.endswith("_score")] + ["uncertainty"]:
        st2[c] = rnd.normal(size=len(st2))
    v2 = wm.validate(st2, base, tg, pd.Timestamp("2025-06-01"))
    assert v2["better"] == []


def test_targets_are_forward_and_end_dates_recorded():
    px = _synthetic_market(n=600)
    d = [px.index[300]]
    t = wm.targets(px, d)
    spy = px["SPY"]
    assert t.loc[d[0], "dd60"] == pytest.approx(spy.iloc[301:361].min() / spy.iloc[300] - 1)
    assert t.loc[d[0], "end_dd60"] == px.index[360]


def test_world_macro_connector_vintages(monkeypatch):
    import json
    from unittest.mock import patch
    from modules.external import http
    from modules.external.pit import utc_now
    from modules.external.registry import SourceRegistry
    from modules.external.sources import real_economy as re_
    from modules.external.sources.base import SourceStatus
    monkeypatch.setenv("FRED_API_KEY", "k")
    calls = []

    def fake(url, params=None, **kw):
        calls.append(params)
        content = json.dumps({"observations": [{"date": "2026-07-01", "realtime_start": "2026-08-07",
                                                "realtime_end": "9999-12-31", "value": "159000"}]}).encode()
        return http.FetchResult(url="https://fixture", status=200, content=content, content_type="application/json",
                                retrieved_at=utc_now(), content_hash="h", fingerprint="fp", bytes=len(content))
    with patch.object(re_.http, "fetch", side_effect=fake):
        res = re_.FredWorldMacroConnector(SourceRegistry().sources["fred_world_macro"]).fetch(datetime(2026, 9, 29, tzinfo=UTC))
    assert res.status == SourceStatus.PASS
    assert [c["series_id"] for c in calls] == ["PAYEMS", "ICSA", "RSAFS", "ISRATIO"]
    assert {o.metric for o in res.observations} == {"us_payems", "us_initial_claims", "us_retail_sales",
                                                   "us_inventory_sales_ratio"}
    assert all(o.available_at == datetime(2026, 8, 7, tzinfo=UTC) for o in res.observations)
