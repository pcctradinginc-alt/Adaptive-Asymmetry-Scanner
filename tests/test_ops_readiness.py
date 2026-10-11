"""Betriebsreife vor dem Merge (2026-10-11): DST-/kalenderfeste RTH-Marks, Options-Capture, NY-Zeit-Guard, Persistenz.

Synthetische Quotes prüfen nur die Mechanik (Zeitpunkte, Quellen, Idempotenz) – kein wirtschaftlicher Nachweis.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import rth_marks  # noqa: E402

from modules import candidate_ledger as cl  # noqa: E402
from modules import market_snapshot as ms  # noqa: E402
from modules import option_quotes as oq  # noqa: E402
from modules import workflow_health as wh  # noqa: E402

UTC = timezone.utc
L, S = "AAPL270319C00200000", "AAPL270319C00210000"


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def quote(root: Path, sym: str, at: str, bid: float, ask: float) -> None:
    """Beobachtete Quote zum Zeitpunkt `at` (UTC) archivieren; Börsenzeit = Abrufzeit."""
    ms_epoch = int(utc(at).timestamp() * 1000)
    oq.record([{"symbol": sym, "bid": bid, "ask": ask, "bid_date": ms_epoch, "ask_date": ms_epoch}],
              underlying="AAPL", source="tradier", context="test", retrieved_at=utc(at), root=root)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    root = tmp_path / "quotes"
    monkeypatch.setattr(oq, "ROOT", root)
    monkeypatch.setattr(ms, "fetch_option_quotes",                    # außerhalb der Sitzung nie live abrufen
                        lambda *a, **k: pytest.fail("Live-Abruf außerhalb der Sitzung"))
    return root


def opt_row(entry="2026-10-05", expiry="2027-03-19"):
    return {"signal_id": "s1", "ticker": "AAPL", "direction": "BULLISH", "outcomes": {},
            "real_option": {"symbol": L, "ask": 2.0, "mid": 1.9, "strike": 200, "entry_date": entry,
                            "expiry": expiry, "entry_pending": False}}


def spread_row(entry="2026-10-05"):
    return {"signal_id": "s2", "ticker": "AAPL", "direction": "BULLISH", "outcomes": {},
            "real_strategy": {"strategy": "BULL_CALL_SPREAD", "entry_date": entry, "expiry": "2027-03-19",
                              "entry_pending": False, "net_debit_entry": 1.0, "net_mid_entry": 1.0,
                              "legs": [{"side": "long", "symbol": L, "strike": 200},
                                       {"side": "short", "symbol": S, "strike": 210}]}}


# ── Kalender und Session ────────────────────────────────────────────────────
@pytest.mark.parametrize("ts, expected", [
    ("2026-03-06T14:29", "PRE"), ("2026-03-06T14:30", "RTH"),          # Freitag vor DST-Beginn: 09:30 EST
    ("2026-03-09T13:30", "RTH"), ("2026-03-09T13:29", "PRE"),          # Montag nach DST-Beginn: 09:30 EDT
    ("2026-10-30T19:59", "RTH"), ("2026-10-30T20:00", "POST"),         # letzter Freitag Sommerzeit
    ("2026-11-02T20:59", "RTH"), ("2026-11-02T21:00", "POST"),         # erster Montag Winterzeit
    ("2026-11-26T17:00", "CLOSED"), ("2025-01-09T17:00", "CLOSED"),    # Thanksgiving, Staatstrauer (Config)
    ("2026-11-27T17:59", "RTH"), ("2026-11-27T18:00", "POST"),         # verkürzt bis 13:00 ET
    ("2026-10-17T17:00", "CLOSED")])                                   # Samstag
def test_session_of_follows_nyse_calendar_across_dst(ts, expected):
    assert oq.session_of(utc(ts)) == expected


def test_calendar_override_file_missing_or_broken_keeps_rule_calendar(tmp_path):
    ms._calendar_overrides.cache_clear()
    try:
        bad = tmp_path / "bad.yaml"
        bad.write_text("closures: [kein-datum\n")
        assert ms._calendar_overrides(str(tmp_path / "fehlt.yaml")) == (frozenset(), {})
        assert ms._calendar_overrides(str(bad)) == (frozenset(), {})
    finally:
        ms._calendar_overrides.cache_clear()
    assert not ms.is_trading_day(datetime(2026, 12, 25).date())        # Regelfeiertag bleibt


# ── Marks aus archivierten RTH-Quotes (Erfassung und Mark entkoppelt) ─────────
def test_option_mark_after_close_uses_earliest_rth_quote_on_or_after_due_day(archive, monkeypatch):
    monkeypatch.setattr(cl, "_now_utc", lambda: utc("2026-10-13T22:00"))          # Dienstag nach Schluss
    quote(archive, L, "2026-10-09T15:00", 5.0, 5.1)       # RTH, aber vor Fälligkeit (10-10) -> ignoriert
    quote(archive, L, "2026-10-12T12:00", 9.0, 9.1)       # PRE -> ignoriert
    quote(archive, L, "2026-10-12T15:00", 2.4, 2.5)       # erster RTH-Handelstag ab Fälligkeit -> Mark
    quote(archive, L, "2026-10-13T15:00", 3.0, 3.1)       # später -> nicht gewählt
    row = opt_row()
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 13)) is True
    oc = row["outcomes"]
    assert oc["real_opt_ret_5d"] == pytest.approx(0.2) and oc["real_opt_mark_date_5d"] == "2026-10-12"
    assert oc["real_opt_mark_source_5d"] == "archive_rth" and "real_opt_late_5d" not in oc


def test_option_mark_stays_open_without_usable_rth_quote(archive, monkeypatch):
    monkeypatch.setattr(cl, "_now_utc", lambda: utc("2026-10-12T22:00"))
    quote(archive, L, "2026-10-12T21:30", 2.4, 2.5)       # POST
    quote(archive, L, "2026-10-12T15:00", 0.0, 2.5)       # Bid 0 -> nicht realisierbar
    row = opt_row()
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 12)) is False
    assert row["outcomes"] == {}                          # offen, nie ein Ersatzpreis


def test_late_flag_counts_from_mark_day_not_run_day(archive, monkeypatch):
    monkeypatch.setattr(cl, "_now_utc", lambda: utc("2026-10-13T22:00"))
    quote(archive, L, "2026-10-12T15:00", 2.4, 2.5)
    late, ontime = opt_row(entry="2026-09-20"), opt_row(entry="2026-10-05")     # fällig 09-25 bzw. 10-10
    cl._fill_real_option_marks([late, ontime], datetime(2026, 10, 13))
    assert late["outcomes"].get("real_opt_late_5d") is True                     # 17 Tage nach Fälligkeit
    assert "real_opt_late_5d" not in ontime["outcomes"]


def test_spread_mark_needs_both_legs_on_the_same_trading_day(archive, monkeypatch):
    monkeypatch.setattr(cl, "_now_utc", lambda: utc("2026-10-13T22:00"))
    quote(archive, L, "2026-10-12T15:00", 2.0, 2.1)       # nur Long-Leg am 10-12
    quote(archive, L, "2026-10-13T15:00", 2.5, 2.6)
    quote(archive, S, "2026-10-13T15:05", 0.9, 1.0)       # beide Legs erst am 10-13
    row = spread_row()
    assert cl._fill_real_strategy_marks([row], datetime(2026, 10, 13)) is True
    oc = row["outcomes"]
    assert oc["real_strat_mark_date_5d"] == "2026-10-13" and oc["real_strat_mark_source_5d"] == "archive_rth"
    assert oc["real_strat_ret_5d"] == pytest.approx(2.5 - 1.0 - 1.0)            # Long-Bid − Short-Ask − Debit


def test_due_mark_symbols_is_read_only(tmp_path):
    p = tmp_path / "2026-10.jsonl"
    p.write_text(json.dumps(opt_row()) + "\n" + json.dumps(spread_row()) + "\n"
                 + json.dumps(opt_row(expiry="2026-10-09")) + "\n")             # verfallen -> keine Quote nötig
    before = p.read_bytes()
    assert cl.due_mark_symbols("2026-10-12", root=tmp_path) == [L, S]
    assert p.read_bytes() == before


# ── Options-Capture: Fenster, Idempotenz, nie Ledger/History ─────────────────
@pytest.mark.parametrize("ts, expected", [
    ("2026-10-14T13:59", "BEFORE_WINDOW"), ("2026-10-14T14:00", "IN_WINDOW"),   # Sommer: 10:00 ET
    ("2026-11-04T14:59", "BEFORE_WINDOW"), ("2026-11-04T15:00", "IN_WINDOW"),   # Winter: 10:00 ET
    ("2026-11-04T20:50", "IN_WINDOW"), ("2026-11-04T20:51", "AFTER_WINDOW"),
    ("2026-11-27T17:50", "IN_WINDOW"), ("2026-11-27T17:51", "AFTER_WINDOW"),    # verkürzt 13:00 ET
    ("2026-11-26T17:00", "MARKET_CLOSED"), ("2026-10-18T17:00", "MARKET_CLOSED")])
def test_capture_window(ts, expected):
    assert rth_marks.capture_window(utc(ts))[1] == expected


def test_capture_only_is_idempotent_and_never_writes_ledger_or_history(archive, monkeypatch, tmp_path):
    hp = tmp_path / "history.json"
    hp.write_text(json.dumps({"active_trades": [{"ticker": "AAPL", "strategy": "LONG_CALL",
                                                 "option": {"option_symbol": L}}]}))
    before = hp.read_bytes()
    monkeypatch.setattr(cl, "due_mark_symbols", lambda today: [S])
    monkeypatch.setattr(cl, "update_outcomes", lambda *a, **k: pytest.fail("capture-only schreibt nie das Ledger"))
    calls = []

    def fetch(symbols, context="candidate_mark"):
        calls.append((list(symbols), context))
        for s in symbols:
            quote(archive, s, "2026-10-12T15:00", 1.0, 1.1)
        return {s: {"bid": 1.0, "ask": 1.1} for s in symbols}
    monkeypatch.setattr(ms, "fetch_option_quotes", fetch)
    now = utc("2026-10-12T15:00")
    first = rth_marks.run(now, hp, capture_only=True)
    assert calls == [([L], "trade_mark_rth"), ([S], "candidate_mark_rth")] and first["quotes"] == 2
    second = rth_marks.run(now + timedelta(hours=2), hp, capture_only=True)
    assert second["requested"] == 0 and second["already_captured"] == 2 and len(calls) == 2
    assert hp.read_bytes() == before


def _crons(wf: str) -> list[tuple[int, int]]:
    doc = yaml.safe_load((REPO / ".github/workflows" / wf).read_text(encoding="utf-8"))
    out = []
    for c in (doc.get(True) or doc.get("on"))["schedule"]:
        minute, hour, *_ = c["cron"].split()
        out.append((int(hour), int(minute)))
    return out


@pytest.mark.parametrize("day", ["2026-07-15", "2027-01-13", "2026-03-09", "2026-11-02",
                                 "2025-07-03", "2026-11-27", "2026-12-24"])
def test_capture_crons_hit_the_window_for_any_github_delay_up_to_7h(day):
    crons = _crons("options_capture.yml")
    for delay in range(0, 7 * 60 + 1, 5):
        assert any(rth_marks.capture_window(utc(f"{day}T{h:02d}:{m:02d}") + timedelta(minutes=delay))[0]
                   for h, m in crons), f"{day}: kein Cron im Fenster bei {delay} min Verzögerung"


# ── Scanner-Guard und Watchdog ──────────────────────────────────────────────
CFG = {"workflows": {"scanner": {"file": "scanner.yml", "cron_utc": "13:30", "days": [0, 1, 2, 3, 4],
                                 "trading_days_only": True, "max_delay_hours": 9, "artifact": None}},
       "gates": {"scanner": {"not_before_ny": "09:30", "requires": []}}}


@pytest.mark.parametrize("ts, status", [
    ("2026-10-14T13:31", "RUN"), ("2026-11-04T13:31", "SKIP_BEFORE_WINDOW"), ("2026-11-04T14:31", "RUN"),
    ("2026-03-06T14:29", "SKIP_BEFORE_WINDOW"), ("2026-03-09T13:31", "RUN"),
    ("2026-11-26T19:00", "SKIP_MARKET_CLOSED"), ("2025-01-09T19:00", "SKIP_MARKET_CLOSED")])
def test_scanner_guard_uses_new_york_time_and_calendar(tmp_path, ts, status):
    assert wh.upstream_ready("scanner", utc(ts), root=tmp_path, cfg=CFG)["status"] == status


def test_watchdog_expects_no_scanner_on_market_holiday():
    plan = wh.plan_recovery(utc("2026-11-26T23:50"), {"scanner": []}, cfg=CFG)["scanner"]
    assert plan["state"] == "NOT_SCHEDULED" and plan["market_closed"] is True and plan["action"] is None


def test_repo_schedule_config_uses_ny_gate():
    cfg = wh.load_cfg()
    assert cfg["gates"]["scanner"]["not_before_ny"] == "09:30" and "not_before_utc" not in cfg["gates"]["scanner"]
    assert cfg["workflows"]["scanner"]["trading_days_only"] is True
    assert sorted(_crons("scanner.yml")) == [(13, 30), (14, 30)]


# ── Persistenz ──────────────────────────────────────────────────────────────
def _steps(wf: str) -> list[dict]:
    doc = yaml.safe_load((REPO / ".github/workflows" / wf).read_text(encoding="utf-8"))
    return [s for job in doc["jobs"].values() for s in job["steps"]]


def test_scanner_persists_marks_before_pipeline_and_observations_after_failure():
    steps = _steps("scanner.yml")
    names = [s.get("name") for s in steps]
    assert names.index("Options-Marks (RTH)") < names.index("Commit Options-Marks") < names.index("Run Scanner")
    fail = [s for s in steps if s.get("if") == "failure()"]
    assert any("outputs/options_archive/" in s.get("run", "") and "outputs/monitoring/" in s.get("run", "")
               for s in fail)
    assert any(str(s.get("uses", "")).startswith("actions/upload-artifact@") for s in fail)


def test_feedback_persists_quote_archive_after_failure():
    fail = [s for s in _steps("feedback.yml") if s.get("if") == "failure()"]
    assert any("outputs/options_archive/" in s.get("run", "") for s in fail)
    assert any(str(s.get("uses", "")).startswith("actions/upload-artifact@") for s in fail)


def test_capture_workflow_is_isolated_and_append_only():
    doc = yaml.safe_load((REPO / ".github/workflows/options_capture.yml").read_text(encoding="utf-8"))
    assert doc["concurrency"]["group"] not in ("history-write", "") and doc["concurrency"]["cancel-in-progress"] is False
    runs = " ".join(s.get("run", "") for s in _steps("options_capture.yml"))
    assert "rth_marks.py --capture-only" in runs and "pipeline.py" not in runs and "feedback.py" not in runs
    pushes = [s["run"] for s in _steps("options_capture.yml") if "ci_push.sh" in s.get("run", "")]
    assert len(pushes) == 1 and pushes[0].rstrip().endswith("outputs/options_archive/")


@pytest.mark.parametrize("path", ["outputs/options_archive/quotes/2026-10-12.jsonl",
                                  "outputs/models/meta_label/registry.json",
                                  "outputs/monitoring/ops_history.jsonl",
                                  "outputs/candidate_ledger/2026-10.jsonl"])
def test_persistent_data_is_not_gitignored(path):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO)
    assert r.returncode == 1                                             # 1 = nicht ignoriert


def test_append_only_ledgers_use_union_merge():
    assert "outputs/**/*.jsonl merge=union" in (REPO / ".gitattributes").read_text(encoding="utf-8")


# ── Nachbesserungen nach dem unabhängigen Test (2026-10-11) ──────────────────
def test_calendar_override_entries_are_parsed_individually(tmp_path):
    p = tmp_path / "cal.yaml"
    p.write_text('closures: ["2025-01-09", "kein-datum"]\nearly_closes: {"2026-12-23": 13:00, "x": "13:00"}\n')
    ms._calendar_overrides.cache_clear()
    try:
        closures, early = ms._calendar_overrides(str(p))
    finally:
        ms._calendar_overrides.cache_clear()
    assert closures == frozenset({datetime(2025, 1, 9).date()})            # gültiger Eintrag bleibt erhalten
    assert early == {datetime(2026, 12, 23).date(): datetime(2026, 1, 1, 13, 0).time()}   # 13:00 unquotiert (=780)


def test_calendar_override_path_does_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ms._calendar_overrides.cache_clear()
    try:
        assert not ms.is_trading_day(datetime(2025, 1, 9).date())          # Staatstrauer auch außerhalb der Repo-Wurzel
    finally:
        ms._calendar_overrides.cache_clear()


def test_mark_day_is_the_quotes_exchange_day_not_the_retrieval_day(archive, monkeypatch):
    monkeypatch.setattr(cl, "_now_utc", lambda: utc("2026-10-13T22:00"))
    ms_epoch = int(utc("2026-10-12T19:00").timestamp() * 1000)              # Börsenzeit gestern (RTH, < 26 h alt)
    oq.record([{"symbol": L, "bid": 2.4, "ask": 2.5, "bid_date": ms_epoch, "ask_date": ms_epoch}],
              underlying="AAPL", source="tradier", context="test", retrieved_at=utc("2026-10-13T15:00"), root=archive)
    row = opt_row()
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 13)) is True
    assert row["outcomes"]["real_opt_mark_date_5d"] == "2026-10-12"


def test_zero_date_field_is_not_an_exchange_time():
    ra = utc("2026-10-14T15:00")
    rec = oq.normalize({"symbol": L, "bid": 1.0, "ask": 1.1, "bid_date": 0,
                        "ask_date": int(ra.timestamp() * 1000)}, underlying="AAPL", source="tradier",
                       context="test", retrieved_at=ra)
    assert rec["quote_time"].startswith("2026-10-14") and "STALE" not in rec["quality_flags"]


def test_capture_survives_broken_history_and_still_captures_candidates(archive, monkeypatch, tmp_path, capsys):
    hp = tmp_path / "history.json"
    hp.write_text("{kaputt")
    monkeypatch.setattr(cl, "due_mark_symbols", lambda today: [S])
    got = []
    monkeypatch.setattr(ms, "fetch_option_quotes", lambda symbols, context="": got.append((list(symbols), context)) or {})
    res = rth_marks.run(utc("2026-10-12T15:00"), hp, capture_only=True)
    assert got == [([S], "candidate_mark_rth")] and res["skipped"] is False
    assert "::warning::" in capsys.readouterr().out


def test_ci_push_pushes_pending_local_commit_when_nothing_new_is_staged(tmp_path):
    from tests.test_ops_readiness_independent import _ci_sandbox
    c1, _c2, ci_push, remote = _ci_sandbox(tmp_path)
    f = c1 / "outputs" / "options_archive" / "quotes" / "2026-10-14.jsonl"
    f.write_text(f.read_text() + '{"id": 1}\n')
    env_git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(env_git + ["add", "-A"], cwd=c1, check=True)
    subprocess.run(env_git + ["commit", "-q", "-m", "lokal, Push gescheitert"], cwd=c1, check=True)
    r = ci_push(c1, "Fehlerpfad", "outputs/options_archive/")             # nichts Neues gestagt
    assert r.returncode == 0 and "ausstehende lokale Commits" in r.stdout
    assert '{"id": 1}' in remote("outputs/options_archive/quotes/2026-10-14.jsonl")
    again = ci_push(c1, "nochmal", "outputs/options_archive/")
    assert again.returncode == 0 and "Keine Änderungen" in again.stdout
