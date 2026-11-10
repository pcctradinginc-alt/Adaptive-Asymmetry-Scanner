"""
Tests für modules/engine_monitor.py (Überwachungs-Modul, reine Observability).

Alle Tests laufen auf synthetischen Fixtures via tmp_path — kein Netzwerk,
keine echten Daily-Reports/history.json. `today` wird als fester date()
übergeben, damit die Reife-Berechnung ("Zeitreise") deterministisch bleibt.
"""

import json
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

@pytest.fixture(autouse=True)
def _isolate_from_real_ledger(monkeypatch):
    """Seit dem ersten echten Lauf (2026-09-28) liegt ein Candidate Ledger im
    Repo; die Health-Checks dürfen in diesen Tests nicht davon abhängen."""
    import modules.engine_monitor as _em
    monkeypatch.setattr(_em, "_external_ledger_rows_last_30d", lambda today: [])


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

    def test_streak_longer_than_ten_reports_is_counted_in_full(self, tmp_path):
        """Live 2026-11-05: 13 Tage ohne Trade, gemeldet wurden 10 (nur 10 Reports geladen)."""
        reports_dir = tmp_path / "daily_reports"
        today = date(2026, 7, 31)
        for i in range(13):
            _write_daily(reports_dir, today - timedelta(days=i), _base_stats(analyzed=3, mismatch_ok=0, trades=0))
        _write_daily(reports_dir, today - timedelta(days=13), _base_stats(trades=1))

        result = build_health_report(history=_empty_history(), reports_dir=reports_dir, today=today)
        duerre = result["metrics"]["duerre"]
        assert duerre["streak"] == 13
        assert duerre["killer_counts"] == {"mismatch_ok": 13}
        assert len(duerre["last_10_days"]) == 10
        assert "13 Tage ohne Trade" in result["warnings"][0]

    def test_throughput_collapse_explains_drought_instead_of_gate(self, tmp_path):
        """Basis: 30 Tage mit 40 Analysen/Tag und 12 Trades (1 % je Analyse). Danach 8 Tage mit 2 Analysen/Tag:
        erwartet 0,16 Trades -> die Dürre folgt aus dem Durchsatz, nicht aus dem Killer-Gate."""
        reports_dir = tmp_path / "daily_reports"
        today = date(2026, 9, 30)
        for i in range(8):
            _write_daily(reports_dir, today - timedelta(days=i), _base_stats(analyzed=2, mismatch_ok=0, trades=0))
        for i in range(8, 38):
            _write_daily(reports_dir, today - timedelta(days=i),
                         _base_stats(analyzed=40, mismatch_ok=10, trades=1 if (i - 8) % 5 == 0 else 0))
        # Tag 8 hat einen Trade (i=8 -> (8-8)%5 == 0) und beendet den Streak.

        result = build_health_report(history=_empty_history(), reports_dir=reports_dir, today=today)
        tp = result["metrics"]["duerre"]["throughput"]
        assert result["metrics"]["duerre"]["streak"] == 8
        assert tp["baseline_days"] == 30 and tp["baseline_analyzed_per_day"] == 40.0
        assert tp["baseline_trades_per_analysis"] == 0.005          # 6 Trades / 1200 Analysen
        assert tp["expected_trades"] == 0.08 and tp["p_zero_trades"] > 0.9
        assert tp["throughput_collapsed_since"] == (today - timedelta(days=7)).strftime("%Y-%m-%d")
        warning = result["warnings"][0]
        assert "Durchsatz seit" in warning and "kein Beleg für ein zu scharfes Gate" in warning

    def test_no_throughput_claim_without_baseline(self, tmp_path):
        reports_dir = tmp_path / "daily_reports"
        today = date(2026, 7, 20)
        for i in range(6):
            _write_daily(reports_dir, today - timedelta(days=i), _base_stats(analyzed=3, mismatch_ok=0, trades=0))

        result = build_health_report(history=_empty_history(), reports_dir=reports_dir, today=today)
        assert result["metrics"]["duerre"]["throughput"] is None
        assert "Durchsatz" not in result["warnings"][0]

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

    def test_negative_feature_correlations_warns_frozen_loop(self, tmp_path):
        """
        25 synthetische Trades: höhere bins → niedrigere Outcomes.
        Alle Korrelationen sollten <= 0 sein → warning + frozen=True.
        """
        today = date(2026, 7, 20)

        closed_trades = []
        for i in range(25):
            # Höhere bins → negative Outcomes
            # bin_impact high (1.0) → outcome negative
            # bin_mismatch strong (1.0) → outcome negative
            # bin_eps_drift massive (1.0) → outcome negative
            bin_impact = "high" if i % 3 == 0 else ("mid" if i % 3 == 1 else "low")
            bin_mismatch = "strong" if i % 3 == 0 else ("good" if i % 3 == 1 else "weak")
            bin_eps_drift = "massive" if i % 3 == 0 else ("relevant" if i % 3 == 1 else "noise")

            # Inverses Mapping: high bin → negative outcome
            outcome = -0.30 if i % 3 == 0 else (-0.10 if i % 3 == 1 else 0.05)

            closed_trades.append({
                "ticker": f"STOCK{i}",
                "outcome": outcome,
                "outcome_method": "option_quote",   # RELIABLE (Freeze-Check nur auf RELIABLE)
                "features": {
                    "bin_impact": bin_impact,
                    "bin_mismatch": bin_mismatch,
                    "bin_eps_drift": bin_eps_drift,
                },
            })

        history = _empty_history(closed_trades=closed_trades)
        result = build_health_report(history=history, reports_dir=tmp_path, today=today)

        # Gewichts-Update ist entfernt: keine "eingefroren"-/"Gewichts-Update"-Warnung mehr
        assert not [w for w in result["warnings"]
                    if "eingefroren" in w.lower() or w.startswith("Gewichts-Update")], result["warnings"]

        # Diagnose bleibt: Korrelationen vorhanden, aber nie "frozen"
        learn_loop = result["metrics"].get("learn_loop", {})
        assert learn_loop.get("learn_loop_frozen") is False
        assert learn_loop.get("weight_update_status") == "REMOVED"
        assert learn_loop.get("weight_learning", "").startswith("REMOVED (2026-11-09")
        assert all(c <= 0 for c in learn_loop["feature_corr"].values())
        assert "feature_corr" in learn_loop
        assert len(learn_loop["feature_corr"]) == 3  # 3 Features

    def test_positive_feature_correlations_no_warning(self, tmp_path):
        """
        25 synthetische Trades: höhere bins → höhere Outcomes.
        Korrelationen sollten positiv sein → kein warning, frozen=False.
        """
        today = date(2026, 7, 20)

        closed_trades = []
        for i in range(25):
            # Höhere bins → positive Outcomes
            bin_impact = "high" if i % 3 == 0 else ("mid" if i % 3 == 1 else "low")
            bin_mismatch = "strong" if i % 3 == 0 else ("good" if i % 3 == 1 else "weak")
            bin_eps_drift = "massive" if i % 3 == 0 else ("relevant" if i % 3 == 1 else "noise")

            # Direktes Mapping: high bin → positive outcome
            outcome = 0.50 if i % 3 == 0 else (0.25 if i % 3 == 1 else 0.05)

            closed_trades.append({
                "ticker": f"STOCK{i}",
                "outcome": outcome,
                "outcome_method": "option_quote",
                "features": {
                    "bin_impact": bin_impact,
                    "bin_mismatch": bin_mismatch,
                    "bin_eps_drift": bin_eps_drift,
                },
            })

        history = _empty_history(closed_trades=closed_trades)
        result = build_health_report(history=history, reports_dir=tmp_path, today=today)

        # Keine Warnung über eingefrorene Loop
        frozen_warnings = [w for w in result["warnings"] if "eingefroren" in w.lower()]
        assert len(frozen_warnings) == 0

        # frozen sollte False sein
        learn_loop = result["metrics"].get("learn_loop", {})
        assert learn_loop.get("learn_loop_frozen") is False
        assert learn_loop.get("weight_update_status") == "REMOVED"
        assert "feature_corr" in learn_loop

    def test_insufficient_trades_no_correlation_warning(self, tmp_path):
        """
        Weniger als 20 closed_trades → keine Korrelations-Warnung,
        auch wenn die Korrelationen alle <= 0 sind.
        """
        today = date(2026, 7, 20)

        # Nur 15 Trades (unter Schwellwert von 20)
        closed_trades = []
        for i in range(15):
            outcome = -0.20
            closed_trades.append({
                "ticker": f"STOCK{i}",
                "outcome": outcome,
                "features": {
                    "bin_impact": "high",
                    "bin_mismatch": "strong",
                    "bin_eps_drift": "massive",
                },
            })

        history = _empty_history(closed_trades=closed_trades)
        result = build_health_report(history=history, reports_dir=tmp_path, today=today)

        # Keine Warnung über eingefrorene Loop (da < 20 trades)
        frozen_warnings = [w for w in result["warnings"] if "eingefroren" in w.lower()]
        assert len(frozen_warnings) == 0

        # 15 Trades ohne Preismethode = 0 RELIABLE; Gewichts-Lernpfad entfernt -> REMOVED, keine Warnung
        learn_loop = result["metrics"].get("learn_loop", {})
        assert learn_loop.get("weight_update_status") == "REMOVED"
        assert learn_loop.get("learn_loop_frozen") is False
        assert learn_loop.get("n_reliable") == 0 and learn_loop.get("n_non_reliable") == 15
        assert learn_loop.get("feature_corr_exploratory") == {}          # < 20 Trades: keine Korrelation
        assert not any("NEED_MORE_DATA" in w or w.startswith("Gewichts-Update") for w in result["warnings"])


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
