from datetime import date, timedelta
from types import SimpleNamespace
import pytest
from modules import risk_gates as rg

@pytest.mark.parametrize("calendar,expected",[
    (None,"UNKNOWN"), ({}, "UNKNOWN"), ({"Earnings Date":[]}, "UNKNOWN"),
    ({"Earnings Date":[date.today()-timedelta(days=1)]}, "UNKNOWN"),
    ({"Earnings Date":[date.today()+timedelta(days=15)]}, "CLEAR"),
    ({"Earnings Date":[date.today()+timedelta(days=14)]}, "UPCOMING"),
    ({"Earnings Date":[date.today()+timedelta(days=40),"bad"]}, "UNKNOWN"),
    ({"Earnings Date":[date.today()+timedelta(days=2),"bad"]}, "UPCOMING")])
def test_earnings_state(monkeypatch,calendar,expected):
    monkeypatch.setattr(rg.yf,"Ticker",lambda _:SimpleNamespace(calendar=calendar))
    assert rg.RiskGates().earnings_status("ABC")==expected

def test_outage_is_unknown(monkeypatch):
    def fail(_): raise RuntimeError("offline")
    monkeypatch.setattr(rg.yf,"Ticker",fail)
    assert rg.RiskGates().earnings_status("ABC")=="UNKNOWN"
