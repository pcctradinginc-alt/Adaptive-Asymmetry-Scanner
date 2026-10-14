"""Schlanke CI-Jobs (Scanner-Guard, Watchdog) installieren nur pyyaml.

Regression 2026-10-13: workflow_health._market_open_day importiert market_snapshot, das `requests` auf
Modulebene importierte -> Guard und Watchdog brachen mit ModuleNotFoundError ab, der Scanner lief nicht.
Die Tests laufen in einem Subprozess, in dem `requests` nicht importierbar ist.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_BLOCK = "import sys; sys.modules['requests'] = None\n"


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", _BLOCK + code], cwd=ROOT, capture_output=True, text=True,
                          timeout=60)


def test_upstream_guard_ohne_requests_im_handelsfenster():
    r = _run(
        "from datetime import datetime, timezone\n"
        "from modules.workflow_health import upstream_ready\n"
        "res = upstream_ready('scanner', datetime(2026, 10, 13, 20, 18, tzinfo=timezone.utc))\n"
        "assert res['status'] != 'ERROR', res\n"
        "print(res['status'])\n"
    )
    assert r.returncode == 0, r.stderr


def test_kalender_ohne_requests():
    r = _run(
        "from datetime import date\n"
        "from modules.market_snapshot import is_trading_day\n"
        "assert is_trading_day(date(2026, 10, 13))\n"
        "assert not is_trading_day(date(2026, 12, 25))\n"
    )
    assert r.returncode == 0, r.stderr


def test_watchdog_module_ohne_requests():
    r = _run(
        "from datetime import datetime, timezone\n"
        "from modules.workflow_health import load_cfg, plan_recovery\n"
        "plan_recovery(datetime(2026, 10, 14, 7, 42, tzinfo=timezone.utc), {}, load_cfg(), artifacts={}, "
        "recoveries={})\n"
    )
    assert r.returncode == 0, r.stderr
