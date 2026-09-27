"""Prescreener: leerer Batch darf den Lauf nicht abbrechen; API-Ausfall wird
als nicht bewertet gezählt, nicht als 'kein Signal' (Review 2026-09-27)."""
from unittest.mock import patch

from modules import prescreener as ps


def _p():
    with patch.object(ps.anthropic, "Anthropic"):
        return ps.Prescreener()


def test_empty_batch_result_does_not_crash():
    p = _p()
    with patch.object(p, "_call_with_retry", return_value=[]):
        assert p.run([{"ticker": "AAA"}]) == []
    assert p.failed_tickers == []


def test_failed_batch_is_recorded_as_not_evaluated():
    p = _p()
    with patch.object(p, "_call_with_retry", return_value=None):
        assert p.run([{"ticker": "AAA"}, {"ticker": "BBB"}]) == []
    assert sorted(p.failed_tickers) == ["AAA", "BBB"]
