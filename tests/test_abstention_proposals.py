"""Self-Improvement-Eingang: Beobachtung -> Hypothese -> Walk-Forward -> Vertragsentwurf.
Nichts wird registriert oder angewendet."""
from __future__ import annotations

import random
from datetime import date, timedelta

from modules import abstention_proposals as ap
from modules import hypothesis_contract as hc


def _hist(n=120, effect=True, seed=3, reliable=True):
    rng = random.Random(seed)
    trades = []
    for i in range(n):
        s = rng.uniform(0.01, 0.05)
        bad = effect and s > 0.042
        trades.append({"entry_date": (date(2026, 1, 1) + timedelta(days=i)).isoformat(), "ticker": f"T{i}",
                       "features": {"sigma_30d": s, "impact": rng.randint(3, 8), "z_score": rng.gauss(0, 1)},
                       "outcome": (-0.6 if bad else 0.15) + rng.gauss(0, 0.1),
                       "outcome_reliable": reliable})
    return {"closed_trades": trades}


def test_real_effect_survives_walk_forward_and_yields_valid_draft():
    r = ap.propose(_hist(), existing=[])
    rules = [p["walk_forward"]["rule"] for p in r["proposals"]]
    assert any(x.startswith("sigma_30d >") for x in rules), r["rejected"]
    p = next(p for p in r["proposals"] if p["walk_forward"]["rule"].startswith("sigma_30d >"))
    assert p["status"] == "PROPOSED" and p["max_state_from_history"] == "HISTORICALLY_VALIDATED"
    assert p["contract_draft"]["registered_at"] is None                      # nie selbst registriert
    assert p["validation_errors"] == []                                       # vollständiger Vertrag
    assert r["data_kind"] == "historical_walk_forward"


def test_noise_yields_no_proposal():
    for seed in range(5):
        assert ap.propose(_hist(effect=False, seed=seed), existing=[])["proposals"] == []


def test_unreliable_outcomes_ignored_and_minimum():
    r = ap.propose(_hist(reliable=False), existing=[])
    assert r["n_trades"] == 0 and r["proposals"] == [] and "zu wenige" in r["status"]


def test_similar_to_registered_is_not_reproposed():
    first = ap.propose(_hist(), existing=[])
    draft = next(p["contract_draft"] for p in first["proposals"] if p["walk_forward"]["rule"].startswith("sigma_30d >"))
    again = ap.propose(_hist(), existing=[draft])
    assert not any(p["walk_forward"]["rule"].startswith("sigma_30d >") for p in again["proposals"])
    assert any("ähnlich" in x["reason"] for x in again["rejected"])
    assert hc.similarity(draft, draft) >= 0.9
