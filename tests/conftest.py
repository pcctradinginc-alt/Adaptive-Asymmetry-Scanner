"""Gemeinsame Test-Fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_system_state(tmp_path_factory, monkeypatch):
    """Tests dürfen den kanonischen SystemState des Repos (outputs/state/) nie schreiben:
    persistiert wird je Test in ein temporäres Verzeichnis. Die Eingaben bleiben unverändert –
    Tests, die einen eigenen Zustand brauchen, setzen DEFAULT_INPUTS selbst."""
    from modules import system_state as ss
    d = tmp_path_factory.mktemp("state")
    monkeypatch.setattr(ss, "STATE", d / "system_state.json")
    monkeypatch.setattr(ss, "HISTORY", d / "system_state_history.jsonl")


@pytest.fixture(autouse=True)
def _isolate_cost_telemetry(tmp_path_factory, monkeypatch):
    """Kosten-Ledger und Analyse-Cache schreiben in Tests nie nach outputs/costs/."""
    from modules import analysis_cache as ac
    from modules import cost_telemetry as ct
    d = tmp_path_factory.mktemp("costs")
    monkeypatch.setattr(ct, "LEDGER_DIR", d)
    monkeypatch.setattr(ac, "CACHE_PATH", d / "analysis_cache.json")
    ac._PENDING.clear()
