"""End-to-End-Orchestrierung von pipeline.main(): Fehlerpfade laufen fail-closed durch,
schreiben immer den Tages-Snapshot (Stats + Reject-Gründe + kanonischer SystemState) und
senden eine Status-Mail – nie ein Trade-Vorschlag, nie ein Absturz ohne Snapshot.
Alle externen Dienste sind gemockt; Arbeitsverzeichnis ist temporär (keine Repo-Outputs)."""
from __future__ import annotations

import json

import pytest

import pipeline

TODAY_GLOB = "outputs/daily_reports/*.json"


class _Gates:
    def __init__(self, ok=True, vix=18.0):
        self._ok, self.last_vix = ok, vix

    def global_ok(self):
        return self._ok


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sent = {"status": [], "trade": []}
    monkeypatch.setattr(pipeline, "send_status_email",
                        lambda stats, today, health=None: sent["status"].append(dict(stats)))
    from modules import email_reporter
    monkeypatch.setattr(email_reporter, "send_email", lambda p, t, s=None: sent["trade"].append(p))
    from modules import cost_telemetry, source_health, system_state
    monkeypatch.setattr(cost_telemetry, "install_http_counter", lambda: False)
    monkeypatch.setattr(source_health, "scanner_preflight",
                        lambda: {"proceed": True, "blocked_decisions": [], "safe_mode": False,
                                 "data_quality": 1.0, "reasons": [], "source": "test", "fallbacks": []})
    monkeypatch.setattr(system_state, "current", lambda **k: {"state_version": "t1"})
    monkeypatch.setattr(system_state, "safe_mode_view",
                        lambda s: {"state_version": "t1", "active": False, "reasons": [], "drift_level": "NORMAL",
                                   "known": True})
    monkeypatch.setattr(pipeline, "build_health_report", lambda **k: {"status": "test"})
    monkeypatch.setattr(pipeline, "RiskGates", lambda: _Gates())
    monkeypatch.setattr(pipeline, "get_macro_context", lambda: {"macro_regime": "test"})
    monkeypatch.setattr(pipeline, "OptionsDesigner", lambda **k: object())

    class _Ingest:
        def __init__(self, history=None):
            pass

        def run(self):
            raise AssertionError("darf in diesem Pfad nicht erreicht werden")
    monkeypatch.setattr(pipeline, "DataIngestion", _Ingest)
    return sent


def _snapshot():
    import glob
    files = glob.glob(TODAY_GLOB)
    assert len(files) == 1, "Tages-Snapshot muss auf jedem Exit-Pfad geschrieben werden"
    return json.loads(open(files[0]).read())


def test_broken_data_health_check_blocks_fail_closed(env, monkeypatch):
    from modules import source_health

    def boom():
        raise RuntimeError("API down")
    monkeypatch.setattr(source_health, "scanner_preflight", boom)
    pipeline.main()
    st = _snapshot()["stats"]
    assert st["stop_reason"].startswith("Data Health") and "API down" in st["stop_reason"]
    assert st["data_health"]["proceed"] is False and st["data_health"]["data_safe_mode"] is True
    assert "safe_mode" not in st["data_health"]                  # nur EIN Safe-Mode-Begriff (kanonisch)
    assert env["status"] and not env["trade"] and st["trades"] == 0


def test_broken_system_state_is_unknown_and_active(env, monkeypatch):
    from modules import system_state

    def boom(**k):
        raise ValueError("korrupt")
    monkeypatch.setattr(system_state, "current", boom)
    monkeypatch.setattr(pipeline, "RiskGates", lambda: _Gates(ok=False, vix=40.0))
    pipeline.main()
    st = _snapshot()["stats"]
    assert st["system_state"]["active"] is True and st["system_state"]["known"] is False
    assert st["stop_reason"].startswith("VIX-Gate") and st["vix"] == 40.0


def test_vix_unavailable_stops_without_trades(env, monkeypatch):
    monkeypatch.setattr(pipeline, "RiskGates", lambda: _Gates(ok=False, vix=None))
    pipeline.main()
    st = _snapshot()["stats"]
    assert st["stop_reason"] == "VIX-Gate (VIX nicht abrufbar)" and st["vix"] is None
    assert env["status"] and not env["trade"]


def test_empty_universe_stops_cleanly(env, monkeypatch):
    class _Empty:
        def __init__(self, history=None):
            pass

        def run(self):
            return []
    monkeypatch.setattr(pipeline, "DataIngestion", _Empty)
    pipeline.main()
    snap = _snapshot()
    assert snap["stats"]["stop_reason"] == "Keine Kandidaten nach Hard-Filter."
    assert snap["stats"]["system_state"]["state_version"] == "t1"
    assert env["status"] and not env["trade"]


def test_ingestion_crash_propagates_after_no_trade(env, monkeypatch):
    class _Crash:
        def __init__(self, history=None):
            pass

        def run(self):
            raise ConnectionError("Kursquelle weg")
    monkeypatch.setattr(pipeline, "DataIngestion", _Crash)
    with pytest.raises(ConnectionError):            # laut statt still: Workflow schlägt sichtbar fehl
        pipeline.main()
    assert not env["trade"]


def test_provider_outage_is_a_data_failure_not_a_market_finding(env, monkeypatch):
    """Audit 2026-11-26: 25.11.2026 lieferte yfinance für fast alle Ticker 401; 462 Werte galten als 'Market Cap
    < 2 Mrd.'. Ein Provider-Ausfall endet jetzt als data_failure – der Tag gilt dem Guard als NICHT versorgt."""
    outage = {"stage": "hard_filter", "kind": "provider_outage", "reason": "probe_failed_market_cap_missing",
              "total": 518, "no_data": 56, "market_cap_reject": 462, "share_suspect": 1.0,
              "probe": {"ok": False}, "passed": 0}

    class _Outage:
        def __init__(self, history=None):
            self.universe_size, self.provider_outage = 518, outage
            self.filter_stats = {"total": 518, "no_data": 56, "market_cap": 462, "passed": 0}

        def run(self):
            return []
    monkeypatch.setattr(pipeline, "DataIngestion", _Outage)
    pipeline.main()
    st = _snapshot()["stats"]
    assert st["data_failure"] == outage and st["stop_reason"].startswith("Datenausfall Stufe 1")
    assert st["hard_filter"]["market_cap"] == 462
    assert env["status"] and not env["trade"]
    from datetime import date
    from pathlib import Path

    from modules.workflow_health import artifact_ok, load_cfg
    repo = Path(pipeline.__file__).resolve().parent              # Test läuft im tmp-Verzeichnis
    spec = load_cfg(repo / "config" / "workflow_schedule.yaml")["workflows"]["scanner"]["artifact"]
    day = date.fromisoformat(Path(__import__("glob").glob(TODAY_GLOB)[0]).stem)
    assert artifact_ok(spec, day) is False           # Guard/Watchdog: Tag nicht versorgt -> Recovery möglich
