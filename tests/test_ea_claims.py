"""Expectation Alpha – maschinell prüfbare Claims: Normalisierung, deterministische PIT-Verifikation,
keine Selbstverifikation, keine erfundenen Werte, Widersprüche, kein Einfluss auf Research-Entscheidungen."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

import tests.ea_fixtures as fx
from modules import expectation_alpha as ea
from modules.expectation_alpha import claims as cl
from modules.expectation_alpha import config as eacfg
from modules.expectation_alpha import ledger as eal

UTC = timezone.utc
DT = datetime(2026, 10, 9, 15, 0, tzinfo=UTC)
CIK = "0000320193"


def _xrow(metric, value, start, end, filed, accn="acc-1", form="10-Q"):
    return {"entity_id": f"cik:{CIK}", "metric": metric, "value": str(value),
            "available_at": pd.Timestamp(filed, tz="UTC").isoformat(),
            "attrs": json.dumps({"start": start, "end": end, "accn": accn, "form": form})}


def _quarters(metric, values, filed_lag_days=35):
    """Quartale 2025-Q1 .. (len(values)) mit 3-Monats-Perioden; Filing = Periodenende + lag."""
    rows = []
    ends = pd.date_range("2025-03-31", periods=len(values), freq="QE")
    for v, e in zip(values, ends):
        s = (e - pd.offsets.QuarterBegin(startingMonth=1)).normalize()
        s = (e - pd.DateOffset(months=3) + pd.Timedelta(days=1)).date().isoformat()
        rows.append(_xrow(metric, v, s, e.date().isoformat(), (e + pd.Timedelta(days=filed_lag_days)).date().isoformat(),
                          accn=f"{metric}-{e.date()}"))
    return rows


def _stores(xrows=(), frows=()):
    return {"xbrl": pd.DataFrame(list(xrows)), "filings": pd.DataFrame(list(frows)), "ciks": {"AAPL": CIK}, "errors": []}


def _claim(ct, direction=None, value=None, ref="catalyst", conf=0.9):
    spec = cl.CLAIM_TYPES[ct]
    return cl.Claim(ct, "AAPL", spec.get("metric"), direction or spec["direction"], value, ref, conf)


def _ev(stores, t=DT):
    return cl.load_evidence("AAPL", t, stores=stores)


# ── Normalisierung ──────────────────────────────────────────────────────────
def test_normalize_drops_invented_values_and_bad_directions():
    texts = {"catalyst": "Umsatz stieg um 12,5 % im Quartal", "asymmetry_reasoning": "Marge sinkt deutlich"}
    raw = [{"claim_type": "revenue_up", "direction": "up", "value_if_known": 12.5, "source_reference": "catalyst",
            "confidence": 0.9},
           {"claim_type": "revenue_up", "direction": "up", "value_if_known": 31.0, "source_reference": "catalyst"},
           {"claim_type": "margin_down", "direction": "up", "source_reference": "asymmetry_reasoning"},
           {"claim_type": "neuer_typ", "direction": "up", "source_reference": "catalyst"},
           "kein objekt"]
    ok, dropped = cl.normalize(raw, texts, "AAPL")
    assert [c.claim_type for c in ok] == ["revenue_up", "revenue_up", "other"]
    assert ok[0].value_if_known == 12.5 and ok[1].value_if_known is None             # 31 steht nicht im Text
    reasons = " ".join(d["reason"] for d in dropped)
    assert "value_not_in_source" in reasons and "widerspricht Claim-Typ" in reasons and "kein Objekt" in reasons


# ── deterministische Verifikation ───────────────────────────────────────────
def test_revenue_verified_and_contradicted_with_deadband():
    up = _stores(_quarters("revenue", [100, 101, 102, 103, 110]))
    assert cl.verify([_claim("revenue_up")], _ev(up))[0]["status"] == cl.VERIFIED
    assert cl.verify([_claim("revenue_down")], _ev(up))[0]["status"] == cl.CONTRADICTED
    flat = _stores(_quarters("revenue", [100, 101, 102, 103, 100.5]))
    r = cl.verify([_claim("revenue_up")], _ev(flat))[0]
    assert r["status"] == cl.UNVERIFIED and "±" in r["reason"]
    v = cl.verify([_claim("revenue_up")], _ev(up))[0]
    assert v["evidence_id"].startswith(f"xbrl:{CIK}:revenue:2026-03-31") and v["evidence_family"] == "company_filing"


def test_pit_filing_after_decision_is_invisible():
    rows = _quarters("revenue", [100, 101, 102, 103, 110])
    # Q2-2026 mit Einbruch, aber erst NACH der Entscheidung eingereicht -> darf nicht zählen
    rows.append(_xrow("revenue", 50, "2026-04-01", "2026-06-30", "2026-10-20", accn="late"))
    res = cl.verify([_claim("revenue_up")], _ev(_stores(rows)))[0]
    assert res["status"] == cl.VERIFIED and res["evidence"]["period_end"] == "2026-03-31"
    # Revision derselben Periode nach der Entscheidung: damaliger Wert gilt
    rows.append(_xrow("revenue", 90, "2026-01-01", "2026-03-31", "2026-10-15", accn="restated", form="10-Q/A"))
    res = cl.verify([_claim("revenue_up")], _ev(_stores(rows)))[0]
    assert res["status"] == cl.VERIFIED and res["evidence"]["value"] == 110.0
    # dieselbe Revision VOR der Entscheidung bekannt -> widerspricht jetzt
    rows[-1] = _xrow("revenue", 90, "2026-01-01", "2026-03-31", "2026-09-01", accn="restated", form="10-Q/A")
    assert cl.verify([_claim("revenue_up")], _ev(_stores(rows)))[0]["status"] == cl.CONTRADICTED


def test_growth_acceleration_and_margin():
    accel = _stores(_quarters("revenue", [100, 100, 100, 100, 104, 112]))     # YoY 4 % -> 12 %
    assert cl.verify([_claim("revenue_growth_up")], _ev(accel))[0]["status"] == cl.VERIFIED
    assert cl.verify([_claim("revenue_growth_down")], _ev(accel))[0]["status"] == cl.CONTRADICTED
    rows = _quarters("revenue", [100, 100, 100, 100, 100]) + _quarters("net_income", [20, 20, 20, 20, 10])
    res = cl.verify([_claim("margin_down"), _claim("margin_up")], _ev(_stores(rows)))
    assert [r["status"] for r in res] == [cl.VERIFIED, cl.CONTRADICTED]


def test_filing_items_window_and_absence_never_contradicts():
    def f(items, accepted):
        return {"series_id": f"acc-{accepted}", "entity_id": f"cik:{CIK}", "available_at": accepted,
                "attrs": json.dumps({"form": "8-K", "items": items})}
    st = _stores(frows=[f(["2.02", "9.01"], "2026-10-02T21:00:00+00:00"), f(["5.02"], "2026-08-01T21:00:00+00:00"),
                        f(["1.01"], "2026-10-10T12:00:00+00:00")])           # nach der Entscheidung
    res = {r["claim_type"]: r for r in cl.verify([_claim("earnings_released"), _claim("management_change"),
                                                  _claim("material_agreement")], _ev(st))}
    assert res["earnings_released"]["status"] == cl.VERIFIED
    assert res["management_change"]["status"] == cl.UNVERIFIED                # außerhalb 14 T
    assert res["material_agreement"]["status"] == cl.UNVERIFIED                # erst nach der Entscheidung
    assert all(r["status"] != cl.CONTRADICTED for r in res.values())


def test_no_self_verification_unverifiable_types_stay_unverified():
    """Text und LLM-Konfidenz sind nie Evidenz: guidance_raised bleibt UNVERIFIED, egal wie sicher das LLM ist."""
    st = _stores(_quarters("revenue", [100, 101, 102, 103, 110]))
    for ct in ("guidance_raised", "orders_accelerating", "capex_increased", "other"):
        r = cl.verify([_claim(ct, conf=1.0)], _ev(st))[0]
        assert r["status"] == cl.UNVERIFIED and r["verification_method"] is None and r["evidence_id"] is None


def test_missing_entity_or_data_is_unverified_not_contradicted():
    st = _stores(_quarters("revenue", [100, 101, 102, 103, 110]))
    no_cik = cl.load_evidence("ZZZ", DT, stores=st)
    r = cl.verify([_claim("revenue_up")], no_cik)[0]
    assert r["status"] == cl.UNVERIFIED and no_cik["error"]
    r = cl.verify([_claim("eps_up")], _ev(st))[0]
    assert r["status"] == cl.UNVERIFIED and "eps_diluted" in r["reason"]


def test_summary_counts_and_none_without_extraction():
    st = _stores(_quarters("revenue", [100, 101, 102, 103, 110]))
    ver = cl.verify([_claim("revenue_up"), _claim("revenue_down"), _claim("guidance_raised")], _ev(st))
    s = cl.summarize(ver, extraction_status="OK")
    assert (s["n_claims"], s["n_verified"], s["n_contradicted"], s["n_unverified"]) == (3, 1, 1, 1)
    assert s["verified_claim_fraction"] == pytest.approx(1 / 3, abs=1e-4)
    none = cl.summarize(None, extraction_status="SKIPPED")
    assert none["n_claims"] is None and none["verified_claim_fraction"] is None          # nie 0 als Ersatz
    assert cl.summarize([], extraction_status="OK")["verified_claim_fraction"] is None


# ── Integration: nur Logging, keine Entscheidungswirkung ─────────────────────
@pytest.fixture(scope="module")
def market():
    px = fx.make_px(end="2027-06-30", years=7)
    px["AAPL"] = px["XLK"] * 1.3
    return px, fx.make_archive(), fx.make_commodity()


def _llm(system, user):
    return json.dumps({"claims": [
        {"claim_type": "revenue_up", "direction": "up", "value_if_known": None, "source_reference": "catalyst",
         "confidence": 0.8},
        {"claim_type": "guidance_raised", "direction": "up", "source_reference": "catalyst", "confidence": 1.0}]})


def test_enrich_logs_claims_without_changing_decisions(tmp_path, market):
    px, ar, cm = market
    cfg = eacfg.load()
    an = [fx.make_analysis("AAPL", catalyst="Umsatz steigt, Prognose angehoben")]
    stores = _stores(_quarters("revenue", [100, 101, 102, 103, 110]))
    base = dict(fx.loaders(px, ar, cm))
    s1 = ea.enrich_candidates(copy.deepcopy(an), decision_time=DT, cfg=cfg, root=tmp_path / "a", contracts=[],
                              registry=tmp_path / "r.jsonl", loaders={**base, "claims_llm": _llm, "claims_stores": stores})
    off = copy.deepcopy(cfg)
    off["claims"]["enabled"] = False
    s2 = ea.enrich_candidates(copy.deepcopy(an), decision_time=DT, cfg=off, root=tmp_path / "b", contracts=[],
                              registry=tmp_path / "r.jsonl", loaders=base)
    r1, r2 = eal.read_rows(tmp_path / "a")[0], eal.read_rows(tmp_path / "b")[0]
    assert r1["status"] == r2["status"] and r1["status_reasons"] == r2["status_reasons"]     # kein Einfluss
    c = r1["claims"]
    assert c["summary"]["n_claims"] == 2 and c["summary"]["n_verified"] == 1 and c["summary"]["n_unverified"] == 1
    assert {x["claim_type"]: x["status"] for x in c["claims"]} == {"revenue_up": cl.VERIFIED,
                                                                    "guidance_raised": cl.UNVERIFIED}
    assert r1["env"]["ea_verified_claim_fraction"] == 0.5
    assert r2["claims"]["summary"]["extraction_status"] == "SKIPPED" and r2["claims"]["summary"]["n_claims"] is None
    assert s1["claims"]["n_claims"] == 2 and s2["claims"]["extraction_status_counts"] == {"SKIPPED": 1}


def test_duplicate_ticker_events_are_not_cross_assigned(tmp_path, market):
    px, ar, cm = market
    an = [fx.make_analysis("AAPL", catalyst="Event A: Umsatz steigt"),
          fx.make_analysis("AAPL", catalyst="Event B: Marge sinkt")]
    calls = []

    def llm(system, user):
        calls.append(user)
        return _llm(system, user)
    ea.enrich_candidates(an, decision_time=DT, cfg=eacfg.load(), root=tmp_path, contracts=[],
                         registry=tmp_path / "r.jsonl",
                         loaders={**fx.loaders(px, ar, cm), "claims_llm": llm, "claims_stores": _stores()})
    rows = eal.read_rows(tmp_path)
    assert len(rows) == 2 and not calls
    assert all(r["claims"]["summary"]["extraction_status"] == "NOT_RUN" and "mehrfach" in r["claims"]["reason"]
               for r in rows)
