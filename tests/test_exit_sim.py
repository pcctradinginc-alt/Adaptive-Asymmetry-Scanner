"""
Tests für modules/exit_sim.py – Multi-Variant Exit-Counterfactual.

Synthetische Return-Pfade, keine Netzwerkzugriffe. Prüft, dass die sechs
Varianten (tp50_sl45, no_tp, trail35, trail50, partial50, time_only) auf
demselben Options-Return-Pfad unterschiedlich (und korrekt) schließen, sowie
register_exit_sim (Dedupe) und summarize_exit_sim (n<min_n-Guard).

Ausführen: pytest tests/test_exit_sim.py -v
"""

from datetime import datetime, timedelta

from modules.exit_sim import (
    VARIANT_KEYS,
    register_exit_sim,
    summarize_exit_sim,
    update_exit_sim_entry,
)

ENTRY_DATE = "2026-01-01"
CLOSE_AFTER_DAYS = 45


def _make_entry(entry_date: str = ENTRY_DATE) -> dict:
    """Minimaler exit_sim-Eintrag (Long Call, Expiry weit in der Zukunft, damit
    der DTE-basierte production Time-Exit im Test nicht ungewollt greift —
    das Schließen über die Zeit erfolgt hier ausschließlich über close_after_days)."""
    return {
        "ticker":      "TEST",
        "strategy":    "LONG_CALL",
        "option":      {"expiry": "2027-01-01", "strike": 100},
        "simulation":  {},
        "entry_debit": 1.0,
        "entry_date":  entry_date,
        "mfe": 0.0, "mae": 0.0, "closed": False,
        "variants": {},
    }


def _run_path(entry, outcomes, days_offsets, **kwargs):
    """Wendet eine Folge von (outcome, Tage-seit-Entry) Checkpoints an."""
    entry_dt = datetime.strptime(entry["entry_date"], "%Y-%m-%d")
    for outcome, offset in zip(outcomes, days_offsets):
        today = entry_dt + timedelta(days=offset)
        update_exit_sim_entry(entry, outcome, today, **kwargs)
    return entry


# ── Kern-Pfad: 0 → +50% → +200% → +130% → +40% (Max-Holding bei Tag 46) ──────
# Zeigt, dass A/B/C/D/E/F auf demselben Pfad unterschiedliche Outcomes/Close-
# Zeitpunkte liefern.

PATH = [0.0, 0.50, 2.00, 1.30, 0.40]
OFFSETS = [1, 5, 10, 20, 46]   # letzter Checkpoint >= close_after_days(45)


def test_variant_a_closes_at_hard_tp():
    entry = _make_entry()
    _run_path(entry, PATH, OFFSETS, close_after_days=CLOSE_AFTER_DAYS,
              trail_pcts=[0.35, 0.50], activation=0.50)
    a = entry["variants"]["tp50_sl45"]
    assert a["closed"] is True
    assert a["outcome"] == 0.50
    assert a["close_reason"] == "take_profit"


def test_variant_b_holds_to_max_holding():
    entry = _make_entry()
    _run_path(entry, PATH, OFFSETS, close_after_days=CLOSE_AFTER_DAYS,
              trail_pcts=[0.35, 0.50], activation=0.50)
    b = entry["variants"]["no_tp"]
    assert b["closed"] is True
    assert b["outcome"] == 0.40
    assert b["peak"] == 2.00


def test_variant_trail35_closes_on_pullback_below_65pct_of_peak():
    entry = _make_entry()
    _run_path(entry, PATH, OFFSETS, close_after_days=CLOSE_AFTER_DAYS,
              trail_pcts=[0.35, 0.50], activation=0.50)
    c = entry["variants"]["trail35"]
    # Peak=2.00, Trigger=2.00*0.65=1.30 -> Checkpoint 1.30 <= 1.30 löst aus.
    assert c["closed"] is True
    assert c["outcome"] == 1.30
    assert c["close_reason"] == "trail_stop"


def test_variant_trail50_gives_more_room_but_ends_worse_here():
    entry = _make_entry()
    _run_path(entry, PATH, OFFSETS, close_after_days=CLOSE_AFTER_DAYS,
              trail_pcts=[0.35, 0.50], activation=0.50)
    d = entry["variants"]["trail50"]
    # Peak=2.00, Trigger=2.00*0.50=1.00 -> 1.30 bleibt darüber, erst der letzte
    # Checkpoint (0.40, Max-Holding) schließt die Position.
    assert d["closed"] is True
    assert d["outcome"] == 0.40


def test_variant_partial50_averages_locked_half_with_remainder():
    entry = _make_entry()
    _run_path(entry, PATH, OFFSETS, close_after_days=CLOSE_AFTER_DAYS,
              trail_pcts=[0.35, 0.50], activation=0.50)
    e = entry["variants"]["partial50"]
    # Hälfte 1 verkauft bei +50% (erster Checkpoint >= Aktivierung), Hälfte 2
    # läuft bis Max-Holding bei +40% -> Mittel = (0.50+0.40)/2 = 0.45.
    assert e["closed"] is True
    assert e["outcome"] == 0.45


def test_variant_time_only_ignores_tp_and_sl():
    entry = _make_entry()
    _run_path(entry, PATH, OFFSETS, close_after_days=CLOSE_AFTER_DAYS,
              trail_pcts=[0.35, 0.50], activation=0.50)
    f = entry["variants"]["time_only"]
    assert f["closed"] is True
    assert f["outcome"] == 0.40
    assert f["peak"] == 2.00


def test_entry_closes_only_once_all_variants_closed():
    entry = _make_entry()
    # Nach dem ersten Checkpoint (+50%) ist nur A geschlossen, der Rest offen ->
    # entry insgesamt noch nicht "closed".
    update_exit_sim_entry(entry, PATH[0], datetime(2026, 1, 2),
                           close_after_days=CLOSE_AFTER_DAYS,
                           trail_pcts=[0.35, 0.50], activation=0.50)
    update_exit_sim_entry(entry, PATH[1], datetime(2026, 1, 6),
                           close_after_days=CLOSE_AFTER_DAYS,
                           trail_pcts=[0.35, 0.50], activation=0.50)
    assert entry["variants"]["tp50_sl45"]["closed"] is True
    assert entry["closed"] is False

    _run_path(entry, PATH[2:], OFFSETS[2:], close_after_days=CLOSE_AFTER_DAYS,
              trail_pcts=[0.35, 0.50], activation=0.50)
    assert entry["closed"] is True
    assert all(entry["variants"][k]["closed"] for k in VARIANT_KEYS)


def test_mfe_mae_track_extremes_of_path():
    entry = _make_entry()
    _run_path(entry, PATH, OFFSETS, close_after_days=CLOSE_AFTER_DAYS,
              trail_pcts=[0.35, 0.50], activation=0.50)
    assert entry["mfe"] == 2.00
    assert entry["mae"] == 0.0


# ── Stop-Loss-Pfad: Variante B/C/D/E schließen alle sofort am Stop ───────────

def test_stop_loss_closes_no_tp_and_trail_variants_before_activation():
    entry = _make_entry()
    today = datetime(2026, 1, 3)
    update_exit_sim_entry(entry, -0.50, today, close_after_days=CLOSE_AFTER_DAYS,
                           trail_pcts=[0.35, 0.50], activation=0.50)
    for key in ("tp50_sl45", "no_tp", "trail35", "trail50"):
        v = entry["variants"][key]
        assert v["closed"] is True, key
        assert v["outcome"] == -0.50, key
        assert v["close_reason"] == "stop_loss", key
    # partial50: nie aktiviert -> half1==half2==outcome -> combined == outcome
    e = entry["variants"]["partial50"]
    assert e["closed"] is True
    assert e["outcome"] == -0.50


# ── register_exit_sim: Dedupe über ticker+entry_date ─────────────────────────

def test_register_exit_sim_is_idempotent():
    history = {}
    trade = {"ticker": "ABC", "entry_date": "2026-02-01", "strategy": "LONG_CALL",
              "option": {"expiry": "2027-01-01"}, "entry_debit": 2.0}
    register_exit_sim(history, trade, datetime(2026, 2, 1))
    register_exit_sim(history, trade, datetime(2026, 2, 2))
    assert len(history["exit_sim"]) == 1
    assert history["exit_sim"][0]["ticker"] == "ABC"
    assert set(VARIANT_KEYS) == set(history["exit_sim"][0]["variants"].keys())


def test_register_exit_sim_separate_entries_for_different_dates():
    history = {}
    trade1 = {"ticker": "ABC", "entry_date": "2026-02-01", "strategy": "LONG_CALL",
               "option": {}, "entry_debit": 2.0}
    trade2 = {"ticker": "ABC", "entry_date": "2026-03-01", "strategy": "LONG_CALL",
               "option": {}, "entry_debit": 2.0}
    register_exit_sim(history, trade1, datetime(2026, 2, 1))
    register_exit_sim(history, trade2, datetime(2026, 3, 1))
    assert len(history["exit_sim"]) == 2


# ── summarize_exit_sim: n < min_n-Guard ───────────────────────────────────────

def test_summarize_reports_insufficient_data_below_min_n():
    history = {
        "exit_sim": [
            {"variants": {"tp50_sl45": {"closed": True, "outcome": 0.5}}},
            {"variants": {"tp50_sl45": {"closed": True, "outcome": -0.3}}},
        ]
    }
    summary = summarize_exit_sim(history, min_n=10)
    assert summary["tp50_sl45"]["n"] == 2
    assert "zu wenig Daten" in summary["tp50_sl45"]["note"]
    # Andere Varianten haben n=0
    assert summary["no_tp"]["n"] == 0


def test_summarize_computes_stats_at_or_above_min_n():
    outcomes = [0.5, -1.0, 0.2, 0.3, -0.2, 1.0, -0.95, 0.1, 0.0, 0.4]  # n=10
    history = {
        "exit_sim": [
            {"variants": {"no_tp": {"closed": True, "outcome": o}}} for o in outcomes
        ]
    }
    summary = summarize_exit_sim(history, min_n=10)
    row = summary["no_tp"]
    assert row["n"] == 10
    assert "note" not in row
    assert row["mean"] == round(sum(outcomes) / 10, 4)
    wins = sum(1 for o in outcomes if o > 0)
    assert row["win_rate"] == round(wins / 10, 4)
    total_losses = sum(1 for o in outcomes if o <= -0.95)
    assert row["total_loss_rate"] == round(total_losses / 10, 4)


def test_summarize_ignores_open_variants():
    history = {
        "exit_sim": [
            {"variants": {"tp50_sl45": {"closed": False, "outcome": None}}},
        ]
    }
    summary = summarize_exit_sim(history, min_n=1)
    assert summary["tp50_sl45"]["n"] == 0
