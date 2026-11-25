import pytest
from modules import risk_gates as rg
from modules import economics_first as ef

@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, 0, True, "bad"])
def test_invalid_vix_closes_gate(monkeypatch, value):
    gate = rg.RiskGates()
    monkeypatch.setattr(gate, "_fetch_vix", lambda: value)
    assert gate.global_ok() is False
    assert gate.last_vix is None

def test_vix_age_rejects_future_and_old():
    from datetime import date, timedelta
    assert rg._valid_age(date.today())
    assert not rg._valid_age(date.today() + timedelta(days=1))
    assert not rg._valid_age(date.today() - timedelta(days=7))

@pytest.mark.parametrize("content", [None, "[]", "max_quote_probes: 24", "max_quote_probes: -1", "mode: bogus", "max_quote_probes: true"])
def test_bad_economics_config_cannot_raise_budget(tmp_path, content):
    p = tmp_path / "policy.yaml"
    if content is not None:
        p.write_text(content)
    policy = ef.settings(p)
    assert policy["mode"] == "active"
    assert policy["max_quote_probes"] == policy["max_haiku_candidates"] == policy["max_sonnet_candidates"] == 0
