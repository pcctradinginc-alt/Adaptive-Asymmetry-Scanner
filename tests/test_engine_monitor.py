"""
Tests für modules/engine_monitor.py (Überwachungs-Modul, reine Observability).

Alle Tests laufen auf synthetischen Fixtures via tmp_path — kein Netzwerk,
keine echten Daily-Reports/history.json. `today` wird als fester date()
übergeben, damit die Reife-Berechnung ("Zeitreise") deterministisch bleibt.
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path

import pytest

from modules.config import cfg
from modules.engine_monitor import (
    build_health_report,
    append_markdown_section,
    _killer_gate,
)

CLOSE_AFTER_DAYS = int(cfg.learning.close_after_days)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_daily(reports_dir: Path, day: date, stats: dict, rejects: dict | None = None) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{day.strftime('%Y-%m-%d')}.json"
    path.write_text(json.dumps({
        "date": day.strftime("%Y-%m-%d"),
        "stats": stats,
        "rejects": rejects or {},
    }))


def _base_stats(**overrides) -> dict:
    stats = {
        "vix": 18.5, "universe": 20, "candidates": 20, "prescreened": 6,
        "sector_ok": 18, "pre_mc": 5, "roi_precheck": 5, "analyzed": 3,
        "mismatch_ok": 0, "quick_mc": 0, "intraday_ok": 0, "final_mc": 0,
        "rl_scored": 0, "roi_ok": 0, "trades": 0, "stop_reason": "",
    }
    stats.update(overrides)
    return stats


def _empty_history(**overrides) -> dict:
    hist = {
        "feature_stats": {}, "active_trades": [], "closed_trades": [],
        "model_weights": {"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20},
        "sentiment_history": {}, "shadow_trades": [], "trailing_sim": [],
    }
    hist.update(overrides)
    return hist


# ── a) Dürre-Streak + Killer-Diagnose ────────────────────────────────────────

class TestKillerGate:
    def test_killer_is_stage_after_last_positive(self):
        stats = _base_stats(universe=20, prescreened=6, analyzed=3, mismatch_ok=0)
        assert _killer_gate(stats) == "mismatch_ok"

    def test_killer_when_prescreening_kills_everything(self):
        stats = _base_stats(universe=15, prescreened=0, analyzed=0)
        assert _killer_gate(stats) == "prescreened"

    def test_killer_when_universe_itself_is_empty(self):
        stats = _base_stats(universe=0, prescreened=0, analyzed=0)
        assert _killer_gate(stats) == "universe"


class TestDuerreStreak:
    def test_streak_and_dominant_killer(self, tmp_path):
        reports_dir = tmp_path / "daily_reports"
        today = date(2026, 7, 20)

        # 6 Tage in Folge ohne Trade (neuestes zuerst gezählt): 5x mismatch_ok-Killer,
        # 1x analyzed-Killer -> Dürre-Streak=6, dominanter Killer=mismatch_ok.
        _write_daily(reports_dir, today, _base_stats(analyzed=3, mismatch_ok=0, trades=0))
        _write_daily(reports_dir, today - timedelta(days=1), _base_stats(analyzed=3, mismatch_ok=0, trades=0))
        _write_daily(reports_dir, today - timedelta(days=2), _base_stats(analyzed=3, mismatch_ok=0, trades=0))
        _write_daily(reports_dir, today - timedelta(days=3), _base_stats(prescreened=4, analyzed=0, trades=0))
        _write_daily(reports_dir, today - timedelta(days=4), _base_stats(analyzed=3, mismatch_ok=0, trades=0))
        _write_daily(reports_dir, today - timedelta(days=5), _base_stats(analyzed=3, mismatch_ok=0, trades=0))
        # Streak-Ende: Tag mit Trade
        _write_daily(reports_dir, today - timedelta(days=6), _base_stats(
            analyzed=4, mismatch_ok=3, quick_mc=3, intraday_ok=3, final_mc=3, roi_ok=3, trades=3,
        ))

        result = build_health_report(history=_empty_history(), reports_dir=reports_dir, today=today)

        duerre = result["metrics"]["duerre"]
        assert duerre["streak"] == 6
        assert duerre["killer_counts"] == {"mismatch_ok": 5, "analyzed": 1}
        assert result["status"] == "WARN"
        assert len(result["warnings"]) == 1
        assert "Dürre-Streak" in result["warnings"][0]
        assert "mismatch_ok" in result["warnings"][0]

    def test_streak_below_threshold_gives_no_warning(self, tmp_path):
        reports_dir = tmp_path / "daily_reports"
        today = date(2026, 7, 20)
        for i in range(3):
            _write_daily(reports_dir, today - timedelta(days=i), _base_stats(analyzed=3, mismatch_ok=0, trades=0))
        # Vierter Tag hat einen Trade -> Streak bricht bei 3
        _write_daily(reports_dir, today - timedelta(days=3), _base_stats(
            analyzed=3, mismatch_ok=2, quick_mc=2, intraday_ok=2, final_mc=2, roi_ok=1, trades=1,
        ))

        result = build_health_report(history=_empty_history(), reports_dir=reports_dir, today=today)
        assert result["metrics"]["duerre"]["streak"] == 3
        assert not any("Dürre" in w for w in result["warnings"])

    def test_robust_against_corrupted_and_missing_files(self, tmp_path):
        reports_dir = tmp_path / "daily_reports"
        today = date(2026, 7, 20)
        reports_dir.mkdir(parents=True)
        # Kaputte Datei
        (reports_dir / f"{today.strftime('%Y-%m-%d')}.json").write_text("{not valid json")
        # Ein Tag fehlt komplett (today - 1)
        _write_daily(reports_dir, today - timedelta(days=2), _base_stats(trades=0, analyzed=0))

        # Darf nicht crashen, auch wenn die neueste Datei kaputt ist.
        result = build_health_report(history=_empty_history(), reports_dir=reports_dir, today=today)
        assert isinstance(result, dict)
        assert result["status"] in ("OK", "WARN")


# ── b) Paralleltest-Reife ─────────────────────────────────────────────────────

class TestParallelTestReife:
    def test_shadow_and_trailing_reife_counted_correctly(self, tmp_path):
        today = date(2026, 8, 15)

        def _entry_for_eval_offset(days_from_today: int) -> str:
            """entry_date so, dass Reife (entry+CLOSE_AFTER_DAYS) `days_from_today`
            Tage nach `today` liegt (negativ = überfällig)."""
            eval_date = today + timedelta(days=days_from_today)
            entry_date = eval_date - timedelta(days=CLOSE_AFTER_DAYS)
            return entry_date.strftime("%Y-%m-%d")

        shadow_trades = [
            # roi_gate: 1 bereits ausgewertet, 1 in 10 Tagen reif, 1 erst in 20 Tagen
            {"ticker": "A", "reject_reason": "roi_gate", "entry_date": "2026-01-01", "outcome": 0.1},
            {"ticker": "B", "reject_reason": "roi_gate", "entry_date": _entry_for_eval_offset(10), "outcome": None},
            {"ticker": "C", "reject_reason": "roi_gate", "entry_date": _entry_for_eval_offset(20), "outcome": None},
            # score_48: überfällig (Reife lag vor "today") -> zählt als "ready_in_14d"
            {"ticker": "D", "reject_reason": "score_48", "entry_date": _entry_for_eval_offset(-5), "outcome": None},
        ]
        trailing_sim = [
            {"ticker": "T1", "tp_outcome": 0.50, "trailing_outcome": None},               # offen
            {"ticker": "T2", "tp_outcome": 0.50, "trailing_outcome": 0.90},               # +0.40
            {"ticker": "T3", "tp_outcome": 0.50, "trailing_outcome": 0.30},               # -0.20
        ]
        history = _empty_history(shadow_trades=shadow_trades, trailing_sim=trailing_sim)

        result = build_health_report(history=history, reports_dir=tmp_path / "empty", today=today)
        pt = result["metrics"]["parallel_tests"]

        roi_gate = pt["shadow_by_reason"]["roi_gate"]
        assert roi_gate["total"] == 3
        assert roi_gate["evaluated"] == 1
        assert roi_gate["ready_in_14d"] == 1
        expected_next_eval = (today + timedelta(days=10)).strftime("%Y-%m-%d")
        assert roi_gate["next_eval_date"] == expected_next_eval

        score48 = pt["shadow_by_reason"]["score_48"]
        assert score48["total"] == 1
        assert score48["evaluated"] == 0
        assert score48["ready_in_14d"] == 1   # überfällig zählt als reif-bald

        trailing = pt["trailing_sim"]
        assert trailing["open"] == 1
        assert trailing["closed"] == 2
        assert abs(trailing["avg_delta"] - 0.10) < 1e-9

    def test_no_parallel_tests_gives_empty_metrics(self, tmp_path):
        today = date(2026, 7, 20)
        result = build_health_report(history=_empty_history(), reports_dir=tmp_path, today=today)
        pt = result["metrics"]["parallel_tests"]
        assert pt["shadow_by_reason"] == {}
        assert pt["trailing_sim"] == {"open": 0, "closed": 0, "avg_delta": None}


# ── c) Lern-Loop-Sanity ───────────────────────────────────────────────────────

class TestLearnLoopSanity:
    def test_nan_weight_warns(self, tmp_path):
        history = _empty_history(model_weights={"impact": float("nan"), "mismatch": 0.5, "eps_drift": 0.2})
        result = build_health_report(history=history, reports_dir=tmp_path, today=date(2026, 7, 20))
        assert result["status"] == "WARN"
        assert any("NaN" in w for w in result["warnings"])

    def test_all_zero_weights_warns(self, tmp_path):
        history = _empty_history(model_weights={"impact": 0.0, "mismatch": 0.0, "eps_drift": 0.0})
        result = build_health_report(history=history, reports_dir=tmp_path, today=date(2026, 7, 20))
        assert result["status"] == "WARN"
        assert any("model_weights" in w and "0" in w for w in result["warnings"])

    def test_valid_weights_no_sanity_warning(self, tmp_path):
        history = _empty_history(model_weights={"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20})
        result = build_health_report(history=history, reports_dir=tmp_path, today=date(2026, 7, 20))
        assert not any("NaN" in w or "model_weights" in w for w in result["warnings"])

    def test_stale_active_trade_warns(self, tmp_path):
        today = date(2026, 7, 26)
        stale_entry = today - timedelta(days=CLOSE_AFTER_DAYS + 31)   # > Cutoff (close_after+30)
        history = _empty_history(active_trades=[
            {"ticker": "ZOMBIE", "entry_date": stale_entry.strftime("%Y-%m-%d")},
        ])
        result = build_health_report(history=history, reports_dir=tmp_path, today=today)
        assert result["status"] == "WARN"
        assert any("ZOMBIE" in w for w in result["warnings"])

    def test_fresh_active_trade_no_warning(self, tmp_path):
        today = date(2026, 7, 26)
        fresh_entry = today - timedelta(days=5)
        history = _empty_history(active_trades=[
            {"ticker": "FRESH", "entry_date": fresh_entry.strftime("%Y-%m-%d")},
        ])
        result = build_health_report(history=history, reports_dir=tmp_path, today=today)
        assert not any("FRESH" in w for w in result["warnings"])


# ── d) Data-Health ────────────────────────────────────────────────────────────

class TestDataHealth:
    def test_stale_report_warns(self, tmp_path):
        reports_dir = tmp_path / "daily_reports"
        today = date(2026, 7, 20)
        _write_daily(reports_dir, today - timedelta(days=8), _base_stats(trades=1, roi_ok=1))
        result = build_health_report(history=_empty_history(), reports_dir=reports_dir, today=today)
        assert result["status"] == "WARN"
        assert any("Tage alt" in w or "läuft" in w for w in result["warnings"])

    def test_zero_universe_warns(self, tmp_path):
        reports_dir = tmp_path / "daily_reports"
        today = date(2026, 7, 20)
        _write_daily(reports_dir, today, _base_stats(universe=0))
        result = build_health_report(history=_empty_history(), reports_dir=reports_dir, today=today)
        assert any("universe=0" in w for w in result["warnings"])

    def test_missing_vix_warns(self, tmp_path):
        reports_dir = tmp_path / "daily_reports"
        today = date(2026, 7, 20)
        stats = _base_stats()
        stats["vix"] = None
        _write_daily(reports_dir, today, stats)
        result = build_health_report(history=_empty_history(), reports_dir=reports_dir, today=today)
        assert any("VIX" in w for w in result["warnings"])

    def test_no_reports_at_all_warns(self, tmp_path):
        result = build_health_report(history=_empty_history(), reports_dir=tmp_path / "nonexistent", today=date(2026, 7, 20))
        assert result["status"] == "WARN"
        assert any("Keine Daily-Reports" in w for w in result["warnings"])


# ── Gesunder Zustand ──────────────────────────────────────────────────────────

class TestHealthyState:
    def test_all_checks_pass_gives_ok(self, tmp_path):
        reports_dir = tmp_path / "daily_reports"
        today = date(2026, 7, 20)
        _write_daily(reports_dir, today, _base_stats(
            analyzed=3, mismatch_ok=2, quick_mc=2, intraday_ok=2, final_mc=1, roi_ok=1, trades=1,
        ))
        history = _empty_history(
            model_weights={"impact": 0.35, "mismatch": 0.45, "eps_drift": 0.20},
            active_trades=[{"ticker": "AAPL", "entry_date": (today - timedelta(days=3)).strftime("%Y-%m-%d")}],
        )
        result = build_health_report(history=history, reports_dir=reports_dir, today=today)
        assert result["status"] == "OK"
        assert result["warnings"] == []
        assert result["metrics"]["duerre"]["streak"] == 0


# ── append_markdown_section ───────────────────────────────────────────────────

class TestAppendMarkdownSection:
    def test_appends_section_to_existing_file(self, tmp_path):
        md_path = tmp_path / "2026-07-20.md"
        md_path.write_text("# Adaptive Asymmetry-Scanner – 2026-07-20\n\nUrsprünglicher Inhalt.\n")

        health = {
            "status": "WARN",
            "warnings": ["Dürre-Streak: 6 Tage ohne Trade in Folge (dominanter Killer: mismatch_ok, 5x)."],
            "metrics": {
                "duerre": {"streak": 6, "killer_counts": {"mismatch_ok": 5}},
                "parallel_tests": {
                    "shadow_by_reason": {"roi_gate": {"total": 5, "evaluated": 2, "ready_in_14d": 1, "next_eval_date": "2026-08-01"}},
                    "trailing_sim": {"open": 1, "closed": 2, "avg_delta": 0.1},
                },
            },
        }
        append_markdown_section(md_path, health)

        content = md_path.read_text()
        assert "Ursprünglicher Inhalt." in content   # additiv, nichts überschrieben
        assert "## Engine-Status" in content
        assert "WARN" in content
        assert "Dürre-Streak" in content
        assert "roi_gate" in content

    def test_missing_file_is_failsafe(self, tmp_path):
        md_path = tmp_path / "does_not_exist.md"
        health = {"status": "OK", "warnings": [], "metrics": {}}
        # Darf nicht raisen, darf die Datei auch nicht neu anlegen.
        append_markdown_section(md_path, health)
        assert not md_path.exists()

    def test_broken_health_dict_is_failsafe(self, tmp_path):
        md_path = tmp_path / "2026-07-20.md"
        md_path.write_text("# Report\n")
        # health ist absichtlich kaputt (kein dict) -> darf nicht crashen.
        append_markdown_section(md_path, health=None)  # type: ignore[arg-type]
        # Datei bleibt zumindest lesbar (keine Exception nach draussen).
        assert md_path.read_text().startswith("# Report")
