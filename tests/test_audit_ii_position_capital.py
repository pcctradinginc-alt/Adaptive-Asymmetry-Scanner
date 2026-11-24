from modules.position_sizing import apply_size_multiplier


def test_whole_contract_rounding_reconciles_committed_capital():
    base={"contracts":1,"position_usd":200,"position_pct":.02,"pct_of_depot":.02,"cost_per_contract":200,"max_risk_usd":200}
    a=apply_size_multiplier(base,.5)
    assert a["contracts"]==0
    assert a["position_usd"]==a["position_pct"]==a["pct_of_depot"]==a["max_risk_usd"]==0
    assert a["target_budget_usd"]==100
