"""Unreparierbares LLM-JSON -> Kandidat verworfen, kein erfundener
BULLISH/PASSIERT-Datensatz (Review 2026-09-27)."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from modules.deep_analysis import DeepAnalysis


def _da(raw_text):
    da = DeepAnalysis.__new__(DeepAnalysis)
    da._macro = {"data_available": False, "macro_regime": "unknown"}
    da.client = MagicMock()
    da.client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(text=raw_text)])
    da._get_48h_move = lambda ticker: 0.0
    return da


def test_unrepairable_json_drops_candidate():
    # Schnitt am letzten '",' ergibt '{"list": [1, "xxx"}' -> auch repariert ungültig
    broken = '{"list": [1, "' + "x" * 300 + '", 2'
    da = _da(broken)
    cand = {"ticker": "AAA", "news": [], "info": {}, "features": {}}
    assert da._analyze(cand) is None
    assert da.run([cand]) == []
