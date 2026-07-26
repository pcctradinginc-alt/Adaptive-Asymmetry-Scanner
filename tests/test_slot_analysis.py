"""
Tests für die Slot-Analyse (backtest_thresholds.slot_candidate_analysis,
monthly_report.build_slot_html) — Entscheidungsvorlage für den nächsten
Tuning-Slot. Synthetische history-Dicts, kein Netzwerk.
Ausführen: pytest tests/test_slot_analysis.py -v
"""

import pytest


# ── Helper ────────────────────────────────────────────────────────────────────

def _by_name(candidates: list[dict], name: str) -> dict:
    for c in candidates:
        if c["name"] == name:
            return c
    raise AssertionError(f"Kandidat '{name}' nicht gefunden in {candidates}")


TRAILING_NAME = "Trailing-Stop vs. harter Take-Profit"
STRATEGY_NAME = "Spreads vs. Long Calls"
SECTOR_NAME   = "Sektor-Konzentration"


# ── Analyse 1: Trailing vs. harter TP ────────────────────────────────────────

class TestTrailingAnalysis:
    def test_too_few_sims_not_ready(self):
        from backtest_thresholds import slot_candidate_analysis
        # Nur 3 abgeschlossene Simulationen → unter Guardrail n≥5
        history = {
            "trailing_sim": [
                {"ticker": "A", "tp_outcome": 0.5, "trailing_outcome": 0.8},
                {"ticker": "B", "tp_outcome": 0.5, "trailing_outcome": 0.7},
                {"ticker": "C", "tp_outcome": 0.5, "trailing_outcome": 0.6},
            ]
        }
        result = _by_name(slot_candidate_analysis(history), TRAILING_NAME)
        assert result["ready"] is False
        assert result["n"] == 3
        assert "daten sammeln" in result["recommendation"].lower()

    def test_clear_delta_gives_recommendation(self):
        from backtest_thresholds import slot_candidate_analysis
        # 5 Trades, Trailing im Schnitt deutlich besser (Δ ≥ 5pp)
        sims = [
            {"ticker": f"T{i}", "tp_outcome": 0.50, "trailing_outcome": 0.80}
            for i in range(5)
        ]
        history = {"trailing_sim": sims}
        result = _by_name(slot_candidate_analysis(history), TRAILING_NAME)
        assert result["ready"] is True
        assert result["n"] == 5
        assert result["recommendation"] is not None
        assert "trailing" in result["recommendation"].lower()

    def test_open_sims_are_ignored(self):
        from backtest_thresholds import slot_candidate_analysis
        # trailing_outcome=None → offene Simulation, zählt nicht mit
        history = {
            "trailing_sim": [
                {"ticker": "A", "tp_outcome": 0.5, "trailing_outcome": None},
                {"ticker": "B", "tp_outcome": 0.5, "trailing_outcome": None},
            ]
        }
        result = _by_name(slot_candidate_analysis(history), TRAILING_NAME)
        assert result["n"] == 0
        assert result["ready"] is False

    def test_no_delta_no_recommendation(self):
        from backtest_thresholds import slot_candidate_analysis
        # 5 Trades, Trailing kaum besser als TP → kein Vorschlag trotz ready
        sims = [
            {"ticker": f"T{i}", "tp_outcome": 0.50, "trailing_outcome": 0.51}
            for i in range(5)
        ]
        history = {"trailing_sim": sims}
        result = _by_name(slot_candidate_analysis(history), TRAILING_NAME)
        assert result["ready"] is True
        assert result["recommendation"] is None


# ── Analyse 2: Spreads vs. Long Calls ────────────────────────────────────────

class TestSpreadVsLongAnalysis:
    def _closed(self, spread_outcomes, long_outcomes):
        trades = (
            [{"ticker": f"S{i}", "strategy": "BULL_CALL_SPREAD", "outcome": o}
             for i, o in enumerate(spread_outcomes)]
            + [{"ticker": f"L{i}", "strategy": "LONG_CALL", "outcome": o}
               for i, o in enumerate(long_outcomes)]
        )
        return {"closed_trades": trades}

    def test_guardrail_blocks_small_groups(self):
        from backtest_thresholds import slot_candidate_analysis
        # Nur 6 Spreads, 6 Long Calls → unter Guardrail n≥10 je Gruppe
        history = self._closed([-0.3] * 6, [0.5] * 6)
        result = _by_name(slot_candidate_analysis(history), STRATEGY_NAME)
        assert result["ready"] is False
        assert result["recommendation"] is not None
        assert "daten sammeln" in result["recommendation"].lower()

    def test_spreads_clearly_worse_gives_recommendation(self):
        from backtest_thresholds import slot_candidate_analysis
        # 10 Spreads bei Ø -20%, 10 Long Calls bei Ø +20% → Gap 40pp ≥ 10pp Guardrail
        history = self._closed([-0.20] * 10, [0.20] * 10)
        result = _by_name(slot_candidate_analysis(history), STRATEGY_NAME)
        assert result["ready"] is True
        assert result["recommendation"] is not None
        assert "IV_SPREAD_GATE" in result["recommendation"]

    def test_similar_performance_no_recommendation(self):
        from backtest_thresholds import slot_candidate_analysis
        # 10 vs. 10, Gap deutlich unter 10pp → kein Vorschlag trotz ready
        history = self._closed([0.10] * 10, [0.12] * 10)
        result = _by_name(slot_candidate_analysis(history), STRATEGY_NAME)
        assert result["ready"] is True
        assert result["recommendation"] is None


# ── Analyse 3: Sektor-Konzentration ──────────────────────────────────────────

class TestSectorConcentrationAnalysis:
    def test_dominant_sector_detected(self):
        from backtest_thresholds import slot_candidate_analysis
        # XLK trägt fast alle positiven Outcomes, andere Sektoren negativ
        trades = (
            [{"ticker": f"K{i}", "sector_etf": "XLK", "outcome": 0.8} for i in range(15)]
            + [{"ticker": f"E{i}", "sector_etf": "XLE", "outcome": -0.3} for i in range(5)]
            + [{"ticker": f"V{i}", "sector_etf": "XLV", "outcome": -0.2} for i in range(3)]
        )
        history = {"closed_trades": trades, "shadow_trades": []}
        result = _by_name(slot_candidate_analysis(history), SECTOR_NAME)
        assert result["ready"] is True
        assert result["n"] == 23
        assert "XLK" in result["finding"]
        assert result["recommendation"] is not None
        assert "Sektor-Gate" in result["recommendation"]

    def test_below_total_guardrail_not_ready(self):
        from backtest_thresholds import slot_candidate_analysis
        # Nur 10 Trades mit sector_etf → unter Guardrail n≥20 gesamt
        trades = [{"ticker": f"X{i}", "sector_etf": "XLK", "outcome": 0.5} for i in range(10)]
        history = {"closed_trades": trades, "shadow_trades": []}
        result = _by_name(slot_candidate_analysis(history), SECTOR_NAME)
        assert result["ready"] is False
        assert result["recommendation"] is not None
        assert "daten sammeln" in result["recommendation"].lower()

    def test_shadow_final_mc_survivor_counted(self):
        from backtest_thresholds import slot_candidate_analysis
        # sector_etf-Trades kommen sowohl aus closed_trades als auch aus
        # final_mc_survivor-Schatten-Trades
        closed = [{"ticker": f"C{i}", "sector_etf": "XLK", "outcome": 0.3} for i in range(12)]
        shadow = (
            [{"ticker": f"S{i}", "sector_etf": "XLK", "reject_reason": "final_mc_survivor",
              "outcome": 0.4} for i in range(5)]
            + [{"ticker": f"O{i}", "sector_etf": "XLV", "reject_reason": "final_mc_survivor",
                "outcome": -0.1} for i in range(4)]
            # nicht final_mc_survivor → darf nicht mitzählen
            + [{"ticker": "N0", "sector_etf": "XLE", "reject_reason": "score_44", "outcome": 0.9}]
        )
        history = {"closed_trades": closed, "shadow_trades": shadow}
        result = _by_name(slot_candidate_analysis(history), SECTOR_NAME)
        assert result["n"] == 21  # 12 + 5 + 4, ohne den score_44-Eintrag
        assert result["ready"] is True


# ── Robustheit: leere/kaputte history ────────────────────────────────────────

class TestRobustness:
    def test_empty_history_no_exception_all_not_ready(self):
        from backtest_thresholds import slot_candidate_analysis
        result = slot_candidate_analysis({})
        assert len(result) == 3
        assert all(c["ready"] is False for c in result)
        assert all(c["recommendation"] is None or "daten sammeln" in c["recommendation"].lower()
                   for c in result)

    def test_missing_keys_no_exception(self):
        from backtest_thresholds import slot_candidate_analysis
        # Kaputte/unvollständige Einträge: fehlende Felder, falsche Typen, None
        history = {
            "closed_trades": [
                {"ticker": "A"},                                  # kein outcome
                {"ticker": "B", "outcome": None, "strategy": "LONG_CALL"},
                {"ticker": "C", "outcome": "nicht-numerisch", "strategy": "LONG_CALL"},
                None,
                {"ticker": "D", "outcome": 0.3, "strategy": "LONG_CALL", "sector_etf": None},
            ],
            "shadow_trades": [
                {"reject_reason": "final_mc_survivor"},            # kein outcome/sector_etf
                None,
                {"ticker": "E", "reject_reason": "final_mc_survivor",
                 "sector_etf": "XLK", "outcome": "kaputt"},
            ],
            "trailing_sim": [
                {"ticker": "F", "tp_outcome": None, "trailing_outcome": "kaputt"},
                None,
            ],
        }
        result = slot_candidate_analysis(history)
        assert len(result) == 3
        for c in result:
            assert set(c.keys()) == {"name", "n", "finding", "recommendation", "ready"}

    def test_non_dict_history_no_exception(self):
        from backtest_thresholds import slot_candidate_analysis
        # Völlig falscher Typ statt dict — darf nie crashen
        for bad in (None, [], "kaputt", 42):
            result = slot_candidate_analysis(bad)
            assert len(result) == 3
            assert all(c["ready"] is False for c in result)


# ── monthly_report.build_slot_html ───────────────────────────────────────────

class TestBuildSlotHtml:
    def test_renders_full_candidate_list(self):
        from backtest_thresholds import slot_candidate_analysis
        from monthly_report import build_slot_html
        trades = (
            [{"ticker": f"K{i}", "sector_etf": "XLK", "outcome": 0.8} for i in range(15)]
            + [{"ticker": f"E{i}", "sector_etf": "XLE", "outcome": -0.3} for i in range(5)]
        )
        candidates = slot_candidate_analysis({"closed_trades": trades, "shadow_trades": []})
        html = build_slot_html(candidates)
        assert "Slot-Analyse" in html
        assert "Tuning-Slot" in html
        for c in candidates:
            assert c["name"] in html

    def test_renders_empty_candidate_list(self):
        from monthly_report import build_slot_html
        html = build_slot_html([])
        assert "Slot-Analyse" in html
        assert "<h3>" in html

    def test_ready_false_entries_marked(self):
        from monthly_report import build_slot_html
        candidates = [
            {"name": "Test-Analyse", "n": 1, "finding": "kaum Daten",
             "recommendation": "weiter Daten sammeln", "ready": False},
        ]
        html = build_slot_html(candidates)
        assert "Test-Analyse" in html
        assert "⏳" in html
        # ausgegraut (nicht ✅)
        assert "✅" not in html
