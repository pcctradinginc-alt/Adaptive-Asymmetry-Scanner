from datetime import date
from modules.mirofish_simulation import trading_steps, paired_time_scale_shadow


def test_actual_calendar_includes_thanksgiving_and_weekend():
    assert trading_steps(date(2026,11,25),4)==1


def test_paired_shadow_is_reproducible_and_research_only():
    kw=dict(as_of=date(2026,11,24),seed=123,cap=.75,threshold=.5)
    a=paired_time_scale_shadow(100,.03,.001,.02,120,512,**kw)
    b=paired_time_scale_shadow(100,.03,.001,.02,120,512,**kw)
    assert a==b
    assert a["trading_steps"]<a["calendar_days"]
    assert a["trading_target"]<a["calendar_target"]
    assert a["production_influence"]=="NONE"
