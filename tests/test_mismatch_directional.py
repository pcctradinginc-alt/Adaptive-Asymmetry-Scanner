"""Gerichteter, news-verankerter Mismatch (SHADOW, Aktionsplan Punkt 4).

Geprüft: Produktionswert unverändert; Richtungsvariante zählt nur Bewegungen IN Thesenrichtung; die Basis der
news-verankerten Bewegung ist der letzte Schluss VOR der Nachricht (16:00 New York); Schattenwerte werden für
jede analysierte These notiert (auch Mismatch ≤ 0); Zeitstempel der Schlagzeilen werden PIT erfasst.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from modules import data_ingestion as di
from modules import mismatch_scorer as ms


def _hist(closes, start="2026-11-02"):
    idx = pd.bdate_range(start, periods=len(closes), tz="America/New_York")
    return pd.DataFrame({"Close": closes}, index=idx)


class _Tk:
    def __init__(self, df):
        self.df = df

    def history(self, period="15d"):
        return self.df


def _analysis(direction="BULLISH", impact=7, news_first=None):
    a = {"ticker": "ABC", "deep_analysis": {"impact": impact, "surprise": 5, "direction": direction}}
    if news_first:
        a["news_ts"] = {"first": news_first, "last": news_first, "n": 1, "source": "finnhub"}
    return a


def _score(a, move, sigma=0.02):
    sc = ms.MismatchScorer()
    with patch.object(sc, "_compute_sigma", return_value=sigma), \
         patch.object(sc, "_compute_48h_move", return_value=move), \
         patch.object(ms.MismatchScorer, "_note_shadow", lambda self: None):
        res = sc.run([a])
    return sc, res


def test_production_value_unchanged_and_direction_only_counts_moves_with_the_thesis():
    sc, res = _score(_analysis(), -0.04)                       # bullische These nach −4 %
    assert res == []                                           # Produktion: 7 − 5·2 = −3 → gefiltert (unverändert)
    sh = sc.shadow["ABC"]
    assert sh["mismatch_raw"] == pytest.approx(-3.0)
    assert sh["mismatch_dir_48h"] == pytest.approx(7.0)        # Gegenbewegung zählt nicht als "eingepreist"
    sc, res = _score(_analysis(), +0.01)
    assert res[0]["features"]["mismatch"] == pytest.approx(4.5)
    assert sc.shadow["ABC"]["mismatch_dir_48h"] == pytest.approx(4.5)
    sc, _ = _score(_analysis(direction="BEARISH"), -0.01)      # bärisch: Abwärtsbewegung ist Thesenrichtung
    assert sc.shadow["ABC"]["mismatch_dir_48h"] == pytest.approx(4.5)


def test_news_anchor_uses_last_close_before_the_news():
    df = _hist([100, 100, 100, 110, 112, 113])                 # 02.–09.11. (Mo–Mo), Ende = vorletzter Schluss 112
    sc = ms.MismatchScorer()
    with patch.object(ms.yf, "Ticker", lambda t: _Tk(df)):
        # Nachricht Do 05.11. 15:00 UTC = 10:00 NY (vor Schluss) -> Basis Mi 04.11. (100)
        move, base, sessions = sc._compute_move_since("ABC", "2026-11-05T15:00:00Z")
        assert base == "2026-11-04" and move == pytest.approx(0.12) and sessions == 2
        # Nachricht Do 05.11. 22:00 UTC = 17:00 NY (nach Schluss) -> Basis Do 05.11. (110)
        move, base, _ = sc._compute_move_since("ABC", "2026-11-05T22:00:00Z")
        assert base == "2026-11-05" and move == pytest.approx(112 / 110 - 1)
        # Nachricht nach dem letzten abgeschlossenen Schluss -> keine Bewegung messbar (None)
        assert sc._compute_move_since("ABC", "2026-11-09T22:00:00Z") is None
        assert sc._compute_move_since("ABC", "kein-datum") is None


def test_news_anchored_variant_is_recorded_and_filtered_theses_are_noted():
    df = _hist([100, 100, 100, 92, 92, 93])
    noted = []
    sc = ms.MismatchScorer()
    with patch.object(ms.yf, "Ticker", lambda t: _Tk(df)), \
         patch.object(sc, "_compute_sigma", return_value=0.02), \
         patch.object(sc, "_compute_48h_move", return_value=-0.08), \
         patch("modules.candidate_ledger.note", lambda t, **kw: noted.append((t, kw))):
        res = sc.run([_analysis(news_first="2026-11-05T15:00:00Z")])
    assert res == []                                           # Produktion verwirft (Überreaktion |−8 %|)
    (t, kw), = noted                                           # trotzdem notiert
    assert t == "ABC" and kw["mismatch_dir_news"] == pytest.approx(7.0) and kw["move_since_news"] == pytest.approx(-0.08)
    assert kw["news_anchor_close"] == "2026-11-04"


def test_news_window_and_yfinance_timestamps_are_point_in_time():
    t1, t2 = datetime.fromisoformat("2026-11-05T14:00:00"), datetime.fromisoformat("2026-11-05T09:30:00")
    assert di._news_window([t1, None, t2], "finnhub") == {"first": "2026-11-05T09:30:00Z",
                                                         "last": "2026-11-05T14:00:00Z", "n": 2, "source": "finnhub"}
    assert di._news_window([], "x") is None
    assert di._utc_from_iso("2026-11-05T10:00:00-05:00") == datetime.fromisoformat("2026-11-05T15:00:00")
    assert di._utc_from_epoch("bad") is None
    assert di._utc_from_epoch(1762354800) == datetime.fromisoformat("2025-11-05T15:00:00")   # Finnhub-Epoch -> UTC
    stamps = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    items = [{"title": "A", "providerPublishTime": int((now - timedelta(hours=3)).replace(tzinfo=timezone.utc).timestamp())},
             {"content": {"title": "B", "pubDate": (now - timedelta(hours=100)).strftime("%Y-%m-%dT%H:%M:%SZ")}}]
    assert di.parse_yfinance_news(items, now - timedelta(hours=48), published_out=stamps) == ["A"]
    assert len(stamps) == 1                                    # nur verwendete Schlagzeilen
