"""Black-Box-Tests für modules/ops_monitor.py (read-only Betriebs- und Forward-Alpha-Monitoring).

ALLE Daten in dieser Datei sind SYNTHETISCHE TESTDATEN (tmp_path): erfundene Ticker, Preise, Trades,
Quotes und Ledger. Es werden weder echte Marktdaten noch echte Repo-Ausgaben (outputs/) als Eingabe
benötigt; nur der Read-only-Test gegen das Repo-Verzeichnis liest outputs/ – und prüft, dass es
danach unverändert ist.

Alle Pfade werden über `paths` von `collect()` überschrieben. Zusätzlich läuft jeder Test mit einem
leeren Arbeitsverzeichnis (cwd = tmp), damit die cwd-relativen Defaults (config.yaml,
outputs/external_data/normalized, Standard-Scorecard) nie echte Repo-Daten treffen.

Gruppen
  1  Quote-Abdeckung (Nenner, Status, Fenster, Exit-Marks, Provider/Strategie)
  2  Lebenszyklus (Trade-Zustände, Exactly-once, Kandidaten)
  3  Forward-Performance (Champion-Kennzahlen, Verdikte, Challenger auf derselben Population)
  4  Integrität (Hashes, Ketten, Duplikate, Quote-Archiv)
  5  Training (Manifest-Prüfungen)
  6  Promotion (unzulässiger Einfluss, Kill-Switch)
  7  Robustheit / Read-only
  8  Lauf- und Datenstatus

Die vom unabhängigen Test gefundenen Fehler (früher strikte xfail-Marker) sind behoben; die Tests sichern sie ab.
"""
from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
import os
import pickle
import shutil
import statistics
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from modules import learning_ledger as ll
from modules import ops_monitor as om
from modules import option_quotes as oq
from modules import promotion_controller as pc

NOW = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)      # Mittwoch, 11:00 New York
TODAY = NOW.date()
REPO = Path(__file__).resolve().parents[1]
SECTIONS = ("run", "quote_coverage", "lifecycle", "forward", "training", "promotion", "integrity", "data_health")
PATH_KEYS = ("history", "daily_reports", "decision_ledger", "decision_outcomes", "promotion_state",
             "promotion_transitions", "contract_registry", "workflow_status", "system_state", "source_health",
             "external_health", "trade_outcomes", "quotes", "candidate_ledger", "meta_label", "kill_switch")

OK_EVAL = {"status": "OK", "exit_session": "RTH", "flags": [], "tier": "REALIZED_RTH_REPRODUCIBLE", "reasons": []}
C1 = 2 * 1 * 0.65 / (1.0 * 100)       # Kommission je Trade als Rendite: 1 Leg, 2 Seiten, Debit 1.0
C2 = 2 * 2 * 0.65 / (1.0 * 100)       # Spread-Strategie: 2 Legs
QUALITY_RAW = {"OK": {"bid": 1.0, "ask": 1.1},            # CROSSED/NO_ASK -> UNUSABLE, ZERO_BID -> SUSPECT
               "SUSPECT": {"bid": 0.0, "ask": 0.5},
               "UNUSABLE": {"bid": 1.2, "ask": 1.0}}


def iso_day(offset: int) -> str:
    """Kalendertag `offset` Tage vor TODAY (negativ = Zukunft)."""
    return (TODAY - timedelta(days=offset)).isoformat()


# ── Synthetische Bausteine ──────────────────────────────────────────────────
class Env:
    """Synthetische Ablage: alle 16 `paths`-Schlüssel zeigen auf (zunächst nicht existierende) tmp-Orte."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "history": root / "history.json", "daily_reports": root / "daily_reports",
            "decision_ledger": root / "decision_ledger", "decision_outcomes": root / "decision_outcomes.jsonl",
            "promotion_state": root / "promotion_state.json",
            "promotion_transitions": root / "promotion_transitions.jsonl",
            "contract_registry": root / "contract_registry.jsonl", "workflow_status": root / "workflow_status.json",
            "system_state": root / "system_state.json", "source_health": root / "source_health_snapshot.json",
            "external_health": root / "external_source_health.json", "trade_outcomes": root / "trade_outcomes.jsonl",
            "quotes": root / "quotes", "candidate_ledger": root / "candidate_ledger",
            "meta_label": root / "meta_label", "kill_switch": root / "kill_switch.yaml"}

    # Aufbau (nur Testvorbereitung – collect() selbst schreibt nie)
    def write_json(self, key: str, obj) -> Path:
        p = self.paths[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj), encoding="utf-8")
        return p

    def write_jsonl(self, key: str, rows: list, name: str | None = None) -> Path:
        p = self.paths[key] / name if name else self.paths[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return p

    def history(self, active=(), closed=(), **extra) -> Path:
        return self.write_json("history", {"active_trades": list(active), "closed_trades": list(closed), **extra})

    def capture(self, sym: str, when: datetime, *, quality: str = "OK", source: str = "tradier",
                context: str = "entry") -> str:
        """Archiviert eine synthetische beobachtete Quote über die echte option_quotes.capture."""
        raw = {"symbol": sym, **QUALITY_RAW[quality], "iv": 0.4, "volume": 10, "open_interest": 100,
               "quote_time": (when - timedelta(minutes=1)).isoformat()}
        rid = oq.capture(raw, underlying=sym[:-15], source=source, context=context, retrieved_at=when,
                         root=self.paths["quotes"])
        assert rid, "synthetische Quote ließ sich nicht archivieren"
        return rid

    # Aufruf
    def collect(self, now: datetime = NOW) -> dict:
        return om.collect(now, self.paths)

    def cov(self) -> dict:
        return self.collect()["quote_coverage"]

    def champion(self) -> dict:
        return self.collect()["lifecycle"]["champion"]


def leg_symbols(trade: dict, right: str = "call") -> list[str]:
    opt = trade["option"]
    legs = [oq.occ_symbol(trade["ticker"], opt["expiry"], right, opt["strike"])]
    if opt.get("spread_leg"):
        legs.append(oq.occ_symbol(trade["ticker"], opt["expiry"], right, opt["spread_leg"]["strike"]))
    return legs


def option_trade(ticker: str, entry: str, *, strategy: str = "LONG_CALL", strike: float = 100.0,
                 expiry: str = "2027-01-15", debit: float = 1.1, spread: bool = False, contract: bool = True,
                 **extra) -> dict:
    """Synthetischer Champion-Trade (Long-Call oder Bull-Call-Spread); contract=False -> Kontrakt unbekannt."""
    opt: dict = {}
    if contract:
        opt = {"expiry": expiry, "strike": strike, "ask": debit, "bid": max(round(debit - 0.1, 2), 0.0)}
        if spread:
            opt["spread_leg"] = {"strike": strike + 10.0, "bid": 0.5, "ask": 0.6}
            opt["net_debit"] = debit
    t = {"ticker": ticker, "entry_date": entry, "strategy": strategy, "entry_debit": debit, "option": opt}
    t.update(extra)
    return t


def closed_trade(ticker: str, entry: str, close: str, outcome: float, *, method: str = "option_quote",
                 debit: float = 1.0, strategy: str = "LONG_CALL", strike: float = 100.0, spread: bool = False,
                 contract: bool = True, **extra) -> dict:
    """Synthetischer geschlossener Trade; Bewertung wird (wie im Altbestand) aus der Methode abgeleitet."""
    t = option_trade(ticker, entry, strategy=strategy, strike=strike, debit=debit, spread=spread, contract=contract)
    t.update(close_date=close, outcome=outcome, outcome_method=method, close_reason="target")
    t.update(extra)
    return t


def learnable(ticker: str = "LRN", close: str = "2026-10-12", **kw) -> dict:
    """Geschlossen, gespeicherte Bewertung OK, verlässliche Methode -> lernbar."""
    return closed_trade(ticker, "2026-09-01", close, 0.25, outcome_evaluation=dict(OK_EVAL), **kw)


def tid(trade: dict) -> str:
    return ll.trade_id(trade)


def trades_on_days(values: list[float], day_of) -> list[dict]:
    return [closed_trade(f"B{i:03d}", "2026-08-01", day_of(i), v) for i, v in enumerate(values)]


def snapshot(root: Path, *, content: bool = True) -> dict:
    """Rekursives Listing: Pfad -> (Verzeichnis?, Größe, mtime_ns, sha256)."""
    out = {}
    for p in sorted(root.rglob("*")):
        st = p.stat()
        isdir = p.is_dir()
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if content and not isdir else None
        out[str(p.relative_to(root))] = (isdir, None if isdir else st.st_size, st.st_mtime_ns, digest)
    return out


@pytest.fixture(autouse=True)
def cwd(tmp_path, monkeypatch) -> Path:
    """Leeres Arbeitsverzeichnis mit config.yaml (close_after_days=10); cwd-relative Defaults treffen kein Repo."""
    work = tmp_path / "cwd"
    work.mkdir()
    (work / "config.yaml").write_text("learning:\n  close_after_days: 10\n", encoding="utf-8")
    monkeypatch.chdir(work)
    return work


@pytest.fixture
def env(tmp_path) -> Env:
    return Env(tmp_path / "data")


def test_fixture_quality_levels_match_option_quotes():
    """Selbstprüfung der synthetischen Quotes: OK/SUSPECT/UNUSABLE entstehen wirklich über option_quotes.normalize."""
    sym = oq.occ_symbol("AAA", "2027-01-15", "call", 100.0)
    for quality in QUALITY_RAW:
        rec = oq.normalize({"symbol": sym, **QUALITY_RAW[quality], "iv": 0.4, "quote_time": NOW.isoformat()},
                           underlying="AAA", source="tradier", context="entry", retrieved_at=NOW)
        assert rec["quality_status"] == quality, (quality, rec["quality_flags"])


def test_path_keys_are_exactly_the_documented_ones(env):
    assert set(om.DEFAULT_PATHS) == set(PATH_KEYS) == set(env.paths)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Quote-Abdeckung
# ═══════════════════════════════════════════════════════════════════════════
def build_entry_grid(env: Env) -> list[dict]:
    """13 synthetische aktive Trades über alle Entry-Status, Fenstergrenzen, Provider und Strategien."""
    trades = []

    def add(ticker, entry, quality=None, *, source="tradier", strategy="LONG_CALL", contract=True, ids=None,
            spread=False, no_strategy=False):
        t = option_trade(ticker, entry, strategy=strategy, spread=spread, contract=contract)
        if quality:
            when = datetime.fromisoformat(f"{entry}T15:00:00+00:00")
            t["entry_quote_record_ids"] = [env.capture(leg_symbols(t)[0], when, quality=quality, source=source)]
        elif ids is not None:
            t["entry_quote_record_ids"] = ids
        if no_strategy:
            del t["strategy"]
        trades.append(t)

    spread = {"strategy": "BULL_CALL_SPREAD", "spread": True}
    add("AAPL", iso_day(0), "OK")                                            # A ENTRY_OK        tradier
    add("MSFT", iso_day(0), "SUSPECT")                                       # B ENTRY_SUSPECT   tradier
    add("NVDA", iso_day(0), "UNUSABLE", source="yfinance", **spread)         # C ENTRY_UNUSABLE  yfinance
    add("GOOG", iso_day(0), **spread)                                        # D ENTRY_NOT_CAPTURED
    add("TSLA", iso_day(0), contract=False)                                  # E NO_CONTRACT
    add("AMZN", iso_day(2), ids=["0123456789abcdef0123"])                    # F ENTRY_NOT_IN_ARCHIVE
    add("META", "2026-10-09", no_strategy=True)                              # G NOT_CAPTURED_BEFORE_ARCHIVE
    add("ORCL", iso_day(24), "OK", source="yfinance", **spread)              # H nur 30d (2026-09-20)
    add("IBM", iso_day(7), "OK")                                             # I 7d-Grenze (2026-10-07) drin
    add("CSCO", iso_day(8), "OK")                                            # J knapp außerhalb 7d (10-06)
    add("INTC", iso_day(30), "OK")                                           # K 30d-Grenze (2026-09-14) drin
    add("AMD", iso_day(31), "OK")                                            # L knapp außerhalb 30d (09-13)
    add("ADBE", iso_day(-1), "OK")                                           # M Zukunft: in keinem Fenster
    env.history(active=trades)
    return trades


@pytest.fixture
def entry_grid(env):
    build_entry_grid(env)
    return env.cov()


def test_entry_today_excludes_no_contract_from_denominator(entry_grid):
    e = entry_grid["windows"]["today"]["entry"]
    assert e["denominator"] == 4 and e["excluded_no_contract"] == 1
    assert e["by_status"] == {"ENTRY_OK": 1, "ENTRY_SUSPECT": 1, "ENTRY_UNUSABLE": 1, "ENTRY_NOT_CAPTURED": 1}
    assert (e["observed"], e["ok"], e["suspect"], e["unusable"], e["missing"]) == (2, 1, 1, 1, 1)
    assert e["coverage"] == 0.5                                       # observed / denominator, UNUSABLE zählt nicht


def test_entry_7d_counts_every_status(entry_grid):
    e = entry_grid["windows"]["7d"]["entry"]
    assert e["denominator"] == 7 and e["excluded_no_contract"] == 1
    assert e["by_status"] == {"ENTRY_OK": 2, "ENTRY_SUSPECT": 1, "ENTRY_UNUSABLE": 1, "ENTRY_NOT_CAPTURED": 1,
                              "ENTRY_NOT_IN_ARCHIVE": 1, "ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE": 1}
    assert (e["observed"], e["ok"], e["suspect"], e["unusable"], e["missing"]) == (3, 2, 1, 1, 3)
    assert e["coverage"] == round(3 / 7, 4)


def test_entry_30d_window_includes_boundary_and_excludes_future(entry_grid):
    e = entry_grid["windows"]["30d"]["entry"]
    # in: A–K (K = exakt 30 Tage her); out: L (31 Tage), M (Zukunft)
    assert e["denominator"] == 10 and e["excluded_no_contract"] == 1
    assert (e["observed"], e["ok"], e["suspect"], e["unusable"], e["missing"]) == (6, 5, 1, 1, 3)
    assert e["coverage"] == 0.6
    assert sum(e["by_status"].values()) == e["denominator"]


def test_entry_by_provider_and_strategy_use_30d_denominators(entry_grid):
    by_provider = entry_grid["entry_by_provider_30d"]
    assert by_provider == {"tradier": {"denominator": 5, "observed": 5, "coverage": 1.0},
                           "yfinance": {"denominator": 2, "observed": 1, "coverage": 0.5},   # UNUSABLE nicht beobachtet
                           "unknown": {"denominator": 3, "observed": 0, "coverage": 0.0}}    # nie archiviert -> unbekannt
    by_strategy = entry_grid["entry_by_strategy_30d"]
    assert by_strategy == {"LONG_CALL": {"denominator": 6, "observed": 5, "coverage": round(5 / 6, 4)},
                           "BULL_CALL_SPREAD": {"denominator": 3, "observed": 1, "coverage": round(1 / 3, 4)},
                           "unknown": {"denominator": 1, "observed": 0, "coverage": 0.0}}


@pytest.mark.parametrize("offset, in_today, in_7d, in_30d", [
    (-1, 0, 0, 0),      # Zukunft
    (0, 1, 1, 1),
    (1, 0, 1, 1),
    (7, 0, 1, 1),       # 7d: today-7 inklusive
    (8, 0, 0, 1),
    (30, 0, 0, 1),      # 30d: today-30 inklusive
    (31, 0, 0, 0)])
def test_entry_window_boundaries(env, offset, in_today, in_7d, in_30d):
    env.history(active=[option_trade("WIN", iso_day(offset))])
    w = env.cov()["windows"]
    assert [w[k]["entry"]["denominator"] for k in ("today", "7d", "30d")] == [in_today, in_7d, in_30d]


@pytest.mark.parametrize("entry, status", [("2026-10-09", "ENTRY_NOT_CAPTURED_BEFORE_ARCHIVE"),
                                           ("2026-10-10", "ENTRY_NOT_CAPTURED")])
def test_not_captured_before_archive_starts_on_2026_10_10(env, entry, status):
    env.history(active=[option_trade("ARC", entry)])
    assert env.cov()["windows"]["30d"]["entry"]["by_status"] == {status: 1}


def test_coverage_is_none_not_zero_when_denominator_is_zero(env):
    env.history(active=[option_trade("NOC", iso_day(0), contract=False)])          # nur Trade ohne Kontrakt
    cov = env.cov()
    for name in ("today", "7d", "30d"):
        w = cov["windows"][name]
        assert w["entry"]["denominator"] == 0 and w["entry"]["excluded_no_contract"] == 1
        assert w["entry"]["coverage"] is None and w["entry"]["missing"] == 0
        assert w["exit_closed"]["denominator"] == 0 and w["exit_closed"]["coverage"] is None
        assert w["complete"]["denominator"] == 0 and w["complete"]["coverage"] is None
    assert cov["exit_active"]["denominator"] == 0
    assert cov["exit_active"]["coverage"] is None and cov["exit_active"]["oldest_mark_age_days"] is None
    assert cov["entry_by_provider_30d"] == {} and cov["entry_by_strategy_30d"] == {}


def test_coverage_is_zero_not_none_when_denominator_positive_but_nothing_observed(env):
    env.history(active=[option_trade("ZERO", iso_day(0))])
    e = env.cov()["windows"]["today"]["entry"]
    assert e["denominator"] == 1 and e["coverage"] == 0.0 and e["coverage"] is not None


def test_empty_history_gives_none_everywhere(env):
    cov = env.cov()
    assert all(cov["windows"][w][k]["coverage"] is None for w in ("today", "7d", "30d")
               for k in ("entry", "exit_closed", "complete"))
    assert cov["exit_active"]["coverage"] is None and cov["candidate_marks"] == {}


@pytest.mark.parametrize("legs, expected", [
    (["OK", "OK"], "ENTRY_OK"),
    (["OK", "SUSPECT"], "ENTRY_SUSPECT"),                 # schwächstes Leg bestimmt den Status
    (["SUSPECT", "UNUSABLE"], "ENTRY_UNUSABLE"),
    (["OK", None], "ENTRY_NOT_IN_ARCHIVE")])              # ein Leg-Verweis ohne Archiv-Record
def test_spread_entry_status_is_the_worst_leg(env, legs, expected):
    t = option_trade("SPR", iso_day(0), strategy="BULL_CALL_SPREAD", spread=True)
    when = NOW - timedelta(hours=2)
    t["entry_quote_record_ids"] = [
        env.capture(sym, when, quality=q, source=src) if q else "0123456789abcdef0123"
        for sym, q, src in zip(leg_symbols(t), legs, ("tradier", "yfinance"))]
    env.history(active=[t])
    cov = env.cov()
    assert list(cov["windows"]["today"]["entry"]["by_status"]) == [expected]
    if expected != "ENTRY_NOT_IN_ARCHIVE":
        assert list(cov["entry_by_provider_30d"]) == ["tradier"]            # Provider = Quelle des ersten Legs


def test_entry_coverage_counts_active_and_closed_trades_by_entry_date(env):
    closed = closed_trade("CLO", iso_day(0), iso_day(0), 0.1)
    closed["entry_quote_record_ids"] = [env.capture(leg_symbols(closed)[0], NOW - timedelta(hours=3))]
    env.history(active=[option_trade("ACT", iso_day(0))], closed=[closed])
    e = env.cov()["windows"]["today"]["entry"]
    assert (e["denominator"], e["observed"], e["coverage"]) == (2, 1, 0.5)


def build_exit_active(env: Env) -> None:
    """Aktive synthetische Trades mit unterschiedlich alten archivierten Marks (jetzt = NOW)."""
    trades = []

    def add(ticker, marks_ago_hours, *, spread=False, contract=True, future=False):
        t = option_trade(ticker, iso_day(2), spread=spread, contract=contract)
        if contract:
            for sym, hours in zip(leg_symbols(t), marks_ago_hours):
                if hours is not None:
                    env.capture(sym, NOW + timedelta(hours=1) if future else NOW - timedelta(hours=hours),
                                context="mark")
        trades.append(t)

    add("AAA", [5])                                      # Mark vor 5 h        -> EXIT_MARK_TODAY
    add("BBB", [29])                                     # vor 29 h            -> EXIT_MARK_STALE (1.21 d)
    add("CCC", [])                                       # nie ein Mark        -> NO_EXIT_MARK
    add("DDD", [], contract=False)                       # Kontrakt unbekannt  -> ausgeschlossen
    add("EEE", [1, None], spread=True)                   # nur ein Leg         -> NO_EXIT_MARK
    add("FFF", [5, 48], spread=True)                     # ältestes Leg zählt  -> STALE (2.0 d)
    add("GGG", [0], future=True)                         # Mark liegt NACH jetzt (PIT) -> NO_EXIT_MARK
    add("HHH", [11 * 24])                                # 11 Tage (Lookback 10) -> NO_EXIT_MARK
    add("III", [10 * 24])                                # exakt 10 Tage        -> STALE (10.0 d)
    env.history(active=trades)


def test_exit_active_today_stale_and_missing_marks(env):
    build_exit_active(env)
    ex = env.cov()["exit_active"]
    assert ex["denominator"] == 8                               # 9 aktive minus 1 ohne Kontrakt
    assert ex["by_status"] == {"EXIT_MARK_TODAY": 1, "EXIT_MARK_STALE": 3, "NO_EXIT_MARK": 4,
                               "NO_CONTRACT": 1}                # ausgewiesen, aber nicht im Nenner
    assert ex["mark_today"] == 1 and ex["coverage"] == 0.125
    assert ex["oldest_mark_age_days"] == 10.0
    assert ex["trading_day"] is True                            # Mittwoch


def test_exit_active_trading_day_flag_on_weekend(env):
    env.history(active=[option_trade("WKD", "2026-10-14")])
    sat = datetime(2026, 10, 17, 15, tzinfo=timezone.utc)
    assert env.collect(sat)["quote_coverage"]["exit_active"]["trading_day"] is False


@pytest.mark.parametrize("age, expected", [               # NOW = 15:00 UTC -> Tagesgrenze bei 15 h
    (timedelta(minutes=5), "EXIT_MARK_TODAY"),
    (timedelta(hours=14, minutes=59, seconds=59), "EXIT_MARK_TODAY"),
    (timedelta(hours=15), "EXIT_MARK_TODAY"),                  # 00:00 UTC desselben Tages
    (timedelta(hours=15, seconds=1), "EXIT_MARK_STALE"),       # 23:59:59 UTC des Vortags
    (timedelta(hours=23, minutes=55), "EXIT_MARK_STALE"),
    (timedelta(hours=24), "EXIT_MARK_STALE"),
    (timedelta(hours=25), "EXIT_MARK_STALE")])
def test_exit_active_today_stale_boundary(env, age, expected):
    t = option_trade("BND", iso_day(3))
    env.capture(leg_symbols(t)[0], NOW - age, context="mark")
    env.history(active=[t])
    assert list(env.cov()["exit_active"]["by_status"]) == [expected]


@pytest.mark.parametrize("strategy, spread", [("LONG_PUT", False), ("BEAR_PUT_SPREAD", True)])
def test_put_strategies_look_up_put_symbols(env, strategy, spread):
    t = option_trade("PUT", iso_day(2), strategy=strategy, spread=spread)
    for sym in leg_symbols(t, right="put"):
        env.capture(sym, NOW - timedelta(hours=2), context="mark")
    env.history(active=[t])
    assert env.cov()["exit_active"]["by_status"] == {"EXIT_MARK_TODAY": 1}


def test_explicit_option_symbol_is_used_instead_of_reconstruction(env):
    explicit = oq.occ_symbol("ZZZ", "2027-03-19", "call", 250.0)           # passt NICHT zu ticker/strike des Trades
    t = option_trade("AAA", iso_day(2))
    t["option"]["option_symbol"] = explicit
    env.capture(explicit, NOW - timedelta(hours=2), context="mark")
    env.history(active=[t])
    assert env.cov()["exit_active"]["by_status"] == {"EXIT_MARK_TODAY": 1}


def build_closed_grid(env: Env) -> None:
    """Geschlossene synthetische Trades: Bewertungs-Mix heute, nur 7d, nur 30d (Entry weit außerhalb der Fenster)."""
    e = "2026-08-01"
    env.history(closed=[
        closed_trade("S01", e, iso_day(0), 0.4, method="expiry_intrinsic"),                    # Settlement
        closed_trade("S02", e, iso_day(0), 0.3, outcome_evaluation=dict(OK_EVAL)),              # OK, archiviert
        closed_trade("S03", e, iso_day(0), 0.2),                                               # OK, Exit nicht archiviert
        closed_trade("S04", e, iso_day(0), 0.5, method="delta_approx"),                        # modelliert -> UNUSABLE
        closed_trade("S05", e, iso_day(0), 0.1, debit=0.0),                                    # Entry-Debit 0 -> DATA_BAD
        closed_trade("S06", e, iso_day(4), 0.1),                                               # nur 7d/30d
        closed_trade("S07", e, iso_day(20), 0.6, method="delta_approx")])                      # nur 30d


def test_closed_exit_status_buckets_today(env):
    build_closed_grid(env)
    w = env.cov()["windows"]["today"]
    assert w["exit_closed"]["denominator"] == 5
    assert w["exit_closed"]["by_status"] == {"EXIT_SETTLEMENT": 1, "EXIT_OBSERVED_ARCHIVED": 1,
                                             "EXIT_OBSERVED_NOT_ARCHIVED": 1, "EXIT_NOT_REALIZABLE": 1,
                                             "EXIT_UNKNOWN_DATA_BAD": 1}      # S05: DATA_BAD vor der Exit-Prüfung
    assert w["exit_closed"]["observed"] == 3 and w["exit_closed"]["coverage"] == 0.6


def test_complete_counts_only_evaluation_status_ok(env):
    build_closed_grid(env)
    w = env.cov()["windows"]
    c = w["today"]["complete"]
    assert c["denominator"] == 5 and c["evaluable"] == 3 and c["coverage"] == 0.6
    assert c["by_status"] == {"OK": 3, "UNUSABLE": 1, "DATA_BAD": 1}
    assert c["by_tier"] == {"OBSERVED": 2, "REALIZED_RTH_REPRODUCIBLE": 1, "NOT_EVALUABLE": 2}
    assert c["reasons"] == {"EXIT_PRICE_MODELED": 1, "ENTRY_DEBIT_INVALID": 1}
    assert (w["7d"]["complete"]["denominator"], w["7d"]["complete"]["evaluable"]) == (6, 4)
    assert w["7d"]["complete"]["coverage"] == round(4 / 6, 4)
    assert (w["30d"]["complete"]["denominator"], w["30d"]["complete"]["evaluable"]) == (7, 4)
    assert w["30d"]["complete"]["coverage"] == round(4 / 7, 4)
    assert w["30d"]["exit_closed"]["coverage"] == round(4 / 7, 4)           # 3 ohne realisierbare Exit-Beobachtung von 7


def test_complete_ignores_stored_ok_lookalikes_with_bad_status(env):
    """Gespeicherte Bewertung mit Settlement-Exit, aber DATA_BAD, ist nie 'vollständig bewertbar'."""
    bad = {"status": "DATA_BAD", "exit_session": "SETTLEMENT", "flags": [], "tier": "NOT_EVALUABLE",
           "reasons": ["ENTRY_DEBIT_INVALID"]}
    env.history(closed=[closed_trade("DBD", "2026-08-01", iso_day(0), 0.1, outcome_evaluation=bad)])
    c = env.cov()["windows"]["today"]["complete"]
    assert c["evaluable"] == 0 and c["coverage"] == 0.0 and c["by_status"] == {"DATA_BAD": 1}


@pytest.mark.parametrize("offset, in_today, in_7d, in_30d", [
    (-1, 0, 0, 0), (0, 1, 1, 1), (1, 0, 1, 1), (7, 0, 1, 1), (8, 0, 0, 1), (30, 0, 0, 1), (31, 0, 0, 0)])
def test_closed_window_boundaries_use_close_date(env, offset, in_today, in_7d, in_30d):
    env.history(closed=[closed_trade("CLS", "2026-08-01", iso_day(offset), 0.1)])
    w = env.cov()["windows"]
    assert [w[k]["exit_closed"]["denominator"] for k in ("today", "7d", "30d")] == [in_today, in_7d, in_30d]
    assert [w[k]["complete"]["denominator"] for k in ("today", "7d", "30d")] == [in_today, in_7d, in_30d]


def test_closed_trade_without_outcome_is_not_counted(env):
    t = closed_trade("PEN", "2026-08-01", iso_day(0), 0.1)
    t["outcome"] = None
    env.history(closed=[t, closed_trade("FIN", "2026-08-01", iso_day(0), 0.0)])     # Outcome 0.0 zählt (nicht None)
    assert env.cov()["windows"]["today"]["complete"]["denominator"] == 1


def test_closed_trade_without_contract_is_excluded_from_exit_denominator(env):
    env.history(closed=[closed_trade("NCT", "2026-08-01", iso_day(0), 0.1, contract=False)])
    assert env.cov()["windows"]["today"]["exit_closed"]["denominator"] == 0


def test_candidate_option_marks_due_pending_and_late(env):
    env.write_jsonl("candidate_ledger", CANDIDATES, name="2026-10.jsonl")
    assert env.cov()["candidate_marks"] == {
        "real_opt_ret_5d": {"due": 4, "marked": 2, "pending": 2, "late_marks": 1, "oldest_pending_days": 2}}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Lebenszyklus
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("age, overdue", [(10, False), (12, False), (13, True), (40, True)])
def test_active_trade_is_exit_pending_and_overdue_after_close_after_days_plus_two(env, age, overdue):
    t = option_trade("ACT", iso_day(age))
    env.history(active=[t])                                   # close_after_days = 10 (cwd/config.yaml) + 2 Karenz
    ch = env.champion()
    assert ch["by_state"]["EXIT_PENDING"] == 1
    assert (ch["issues"].get("EXIT_OVERDUE", 0) == 1) is overdue
    assert [b["ticker"] for b in ch["blocked"]] == (["ACT"] if overdue else [])
    if overdue:
        assert ch["blocked"][0]["state"] == "EXIT_PENDING" and "EXIT_OVERDUE" in ch["blocked"][0]["issues"]


@pytest.mark.parametrize("config, age, overdue", [
    ("learning:\n  close_after_days: 3\n", 5, False),
    ("learning:\n  close_after_days: 3\n", 6, True),
    (None, 47, False),                                        # config.yaml fehlt -> Default 45 (+2)
    (None, 48, True),
    ("learning: {}\n", 47, False),                            # kein close_after_days -> Default 45 (+2)
    ("learning: {}\n", 48, True),
    ("learning: [unclosed\n", 48, True)])                     # defekte YAML -> Default 45
def test_close_after_days_is_read_from_config_with_default_45(env, cwd, config, age, overdue):
    if config is None:
        (cwd / "config.yaml").unlink()
    else:
        (cwd / "config.yaml").write_text(config, encoding="utf-8")
    env.history(active=[option_trade("CFG", iso_day(age))])
    assert (env.champion()["issues"].get("EXIT_OVERDUE", 0) == 1) is overdue


def test_trade_without_contract_stays_signal(env):
    env.history(active=[option_trade("SIG", iso_day(1), contract=False)],
                closed=[closed_trade("SIG2", "2026-09-01", iso_day(1), 0.2, contract=False)])
    ch = env.champion()
    assert ch["by_state"]["SIGNAL"] == 2 and ch["by_state"]["EXIT_PENDING"] == 0
    assert ch["issues"] == {"NO_CONTRACT": 2}


@pytest.mark.parametrize("close, issue", [("2026-10-09", "LEGACY_LEARNED_UNTRACKED"),
                                          ("2026-10-10", "ELIGIBLE_NOT_LEARNED"),
                                          ("2026-10-13", "ELIGIBLE_NOT_LEARNED")])
def test_eligible_not_learned_depends_on_consume_once_start(env, close, issue):
    env.history(closed=[learnable(close=close)])
    ch = env.champion()
    other = "ELIGIBLE_NOT_LEARNED" if issue == "LEGACY_LEARNED_UNTRACKED" else "LEGACY_LEARNED_UNTRACKED"
    assert ch["by_state"]["LEARNING_ELIGIBLE"] == 1
    assert ch["issues"].get(issue) == 1 and other not in ch["issues"]
    assert ch["legacy"]["learned_untracked"] == (1 if issue == "LEGACY_LEARNED_UNTRACKED" else 0)
    assert ch["exactly_once"]["eligible_not_learned"] == (0 if issue == "LEGACY_LEARNED_UNTRACKED" else 1)
    assert bool(ch["blocked"]) == (issue == "ELIGIBLE_NOT_LEARNED")          # Altbestand blockiert nicht


def test_learned_trade_with_consume_once_key(env):
    t = learnable()
    env.history(closed=[t], processed_outcomes={f"{tid(t)}:close:feature_stats": "2026-10-13",
                                                f"{tid(t)}:close:feature_stats_external": "2026-10-13"},
                processed_outcomes_meta={"duplicate_skips": 3, "since": "2026-10-13"})
    ch = env.champion()
    assert ch["by_state"]["LEARNED"] == 1 and ch["by_state"]["LEARNING_ELIGIBLE"] == 0
    assert not {"ELIGIBLE_NOT_LEARNED", "LEARNED_NOT_ELIGIBLE", "LEGACY_LEARNED_UNTRACKED"} & set(ch["issues"])
    assert ch["exactly_once"] == {"learned_keys": 2, "duplicate_closed_trades": 0, "duplicate_learnable_trades": 0,
                                  "learned_not_eligible": 0, "eligible_not_learned": 0, "keys_without_trade": 0,
                                  "duplicate_skips_total": 3, "ok": True}


def test_key_of_other_trade_or_other_target_does_not_count_as_learned(env):
    t, other = learnable("AAA"), learnable("BBB")
    env.history(closed=[t], processed_outcomes={f"{tid(other)}:close:feature_stats": "2026-10-13"})
    ch = env.champion()
    assert ch["by_state"]["LEARNED"] == 0 and ch["issues"].get("ELIGIBLE_NOT_LEARNED") == 1


def test_learned_but_not_eligible_breaks_exactly_once(env):
    t = closed_trade("MOD", "2026-09-01", "2026-10-12", 0.4, method="delta_approx")        # modelliert -> nie lernbar
    env.history(closed=[t], processed_outcomes={f"{tid(t)}:close:feature_stats": "2026-10-13"})
    ch = env.champion()
    assert ch["issues"]["LEARNED_NOT_ELIGIBLE"] == 1 and ch["issues"]["OUTCOME_UNUSABLE"] == 1
    assert ch["exactly_once"]["learned_not_eligible"] == 1 and ch["exactly_once"]["ok"] is False
    assert [b["ticker"] for b in ch["blocked"]] == ["MOD"]


@pytest.mark.parametrize("extra, method, debit, state, outcome_issue", [
    ({}, "option_quote", 1.0, "LEARNING_ELIGIBLE", None),
    ({}, "delta_approx", 1.0, "OUTCOME_FINAL", "OUTCOME_UNUSABLE"),                   # Bewertung nicht OK
    ({}, "option_quote", 0.0, "OUTCOME_FINAL", "OUTCOME_DATA_BAD"),                   # Methode ok, Debit 0
    ({"outcome_method_reconstructed": True}, "option_quote", 1.0, "OUTCOME_FINAL", None),   # Bewertung OK, nicht verlässlich
    ({"outcome_reliable": False}, "option_quote", 1.0, "OUTCOME_FINAL", None)])
def test_learning_eligibility_needs_ok_evaluation_and_reliable_method(env, extra, method, debit, state,
                                                                       outcome_issue):
    env.history(closed=[closed_trade("ELG", "2026-09-01", "2026-10-12", 0.2, method=method, debit=debit, **extra)])
    ch = env.champion()
    assert ch["by_state"][state] == 1
    assert [i for i in ch["issues"] if i.startswith("OUTCOME_")] == ([outcome_issue] if outcome_issue else [])
    assert ch["issues"].get("LEGACY_NO_STORED_EVALUATION") == 1                        # abgeleitet, nie gespeichert


def test_stored_evaluation_removes_legacy_issue(env):
    env.history(closed=[learnable()])
    assert "LEGACY_NO_STORED_EVALUATION" not in env.champion()["issues"]


def test_entry_problem_is_reported_as_lifecycle_issue(env):
    ok = option_trade("OKE", iso_day(1))
    ok["entry_quote_record_ids"] = [env.capture(leg_symbols(ok)[0], NOW - timedelta(days=1))]
    env.history(active=[ok, option_trade("MIS", iso_day(1))])
    issues = env.champion()["issues"]
    assert issues == {"ENTRY_NOT_CAPTURED": 1}


def test_duplicate_learnable_trades_break_exactly_once(env):
    t = learnable()
    env.history(closed=[t, dict(t)], processed_outcomes={f"{tid(t)}:close:feature_stats": "2026-10-13"})
    ex = env.champion()["exactly_once"]
    assert ex["duplicate_closed_trades"] == 1 and ex["duplicate_learnable_trades"] == 1 and ex["ok"] is False


def test_duplicate_unlearnable_legacy_trades_only_count_as_legacy(env):
    t = closed_trade("OLD", "2026-04-11", "2026-04-20", -0.3, method="delta_approx")
    env.history(closed=[t, dict(t), dict(t)])
    ch = env.champion()
    assert ch["legacy"]["duplicate_unlearnable_trades"] == 1
    assert ch["exactly_once"]["duplicate_learnable_trades"] == 0 and ch["exactly_once"]["ok"] is True
    assert ch["by_state"]["OUTCOME_FINAL"] == 3 and ch["blocked"] == []


def test_distinct_trades_are_no_duplicates(env):
    env.history(closed=[learnable("AAA"), learnable("BBB")])
    ch = env.champion()
    assert ch["exactly_once"]["duplicate_closed_trades"] == 0 and ch["legacy"]["duplicate_unlearnable_trades"] == 0


def test_keys_without_trade_are_counted_but_shadow_keys_are_not(env):
    env.history(closed=[learnable()], processed_outcomes={
        "Tdeadbeefdeadbeef:close:feature_stats": "2026-10-13",                    # Champion-Schlüssel ohne Trade
        "shadow:ab12cd34ef56ab12:45d:feature_stats_external": "2026-10-13",       # Shadow-Beobachtung: kein Trade
        "obs-77:close:other_target": "2026-10-13"})                               # fremde Beobachtung (kein 'T…')
    ex = env.champion()["exactly_once"]
    assert ex["keys_without_trade"] == 1 and ex["learned_keys"] == 3


def test_exactly_once_ok_does_not_react_to_eligible_not_learned_or_orphan_keys(env):
    """AMBIGUE SEMANTIK (Charakterisierung): Die lifecycle-Doku nennt 'jeder lernbare geschlossene Trade wurde gelernt'
    und 'jeder gelernte Schlüssel gehört zu einem lernbaren Trade' als Exactly-once-Prüfung; `ok` wird aber nur von
    LEARNED_NOT_ELIGIBLE, Duplikaten lernbarer Trades und doppelten Zielen gesetzt. ELIGIBLE_NOT_LEARNED und
    Schlüssel ohne Trade stehen als Zähler da, kippen `ok` jedoch nicht (evtl. gewollt: Lernen läuft zeitversetzt)."""
    env.history(closed=[learnable()], processed_outcomes={"Tdeadbeefdeadbeef:close:feature_stats": "2026-10-13"})
    ex = env.champion()["exactly_once"]
    assert ex["eligible_not_learned"] == 1 and ex["keys_without_trade"] == 1
    assert ex["ok"] is True


def test_same_target_learned_under_two_horizon_spellings_breaks_exactly_once(env):
    t = learnable()
    env.history(closed=[t], processed_outcomes={f"{tid(t)}:close:feature_stats": "2026-10-13",
                                                f"{tid(t)}:45d:feature_stats": "2026-10-13"})
    assert env.champion()["exactly_once"]["ok"] is False


def test_distinct_shadow_observations_do_not_break_exactly_once(env):
    env.history(processed_outcomes={"shadow:aaaaaaaaaaaaaaaa:45d:feature_stats_external": "2026-10-13",
                                    "shadow:bbbbbbbbbbbbbbbb:45d:feature_stats_external": "2026-10-13"})
    assert env.champion()["exactly_once"]["ok"] is True


def test_champion_by_state_has_all_states_and_sums_to_rows(env):
    env.history(active=[option_trade("ACT", iso_day(1))],
                closed=[learnable("AAA"), closed_trade("MOD", "2026-09-01", "2026-10-12", 0.4, method="delta_approx"),
                        closed_trade("NOC", "2026-09-01", "2026-10-12", 0.4, contract=False)])
    ch = env.champion()
    assert list(ch["by_state"]) == ["SIGNAL", "CONTRACT_SELECTED", "ENTRY_OBSERVED", "EXIT_PENDING", "OUTCOME_FINAL",
                                    "LEARNING_ELIGIBLE", "LEARNED"]
    assert ch["by_state"] == {"SIGNAL": 1, "CONTRACT_SELECTED": 0, "ENTRY_OBSERVED": 0, "EXIT_PENDING": 1,
                              "OUTCOME_FINAL": 1, "LEARNING_ELIGIBLE": 1, "LEARNED": 0}


CANDIDATES = [
    {"ticker": "REJ", "direction": None},                                                      # kein Signal
    {"ticker": "C01", "direction": "BULLISH", "real_option_skip_reason": "no_chain"},          # EXCLUDED_NO_CHAIN
    {"ticker": "C02", "direction": "BEARISH"},                                                 # EXCLUDED_NO_CONTRACT
    {"ticker": "C03", "direction": "BULLISH", "real_option": {"symbol": "X", "entry_pending": True}},
    {"ticker": "C04", "direction": "BULLISH", "real_option": {"ask": 0, "entry_date": iso_day(2)}},   # ohne Ask
    {"ticker": "C05", "direction": "BULLISH", "real_option": {"ask": 1.0, "entry_date": iso_day(2)}},
    {"ticker": "C06", "direction": "BULLISH", "real_option": {"ask": 1.0, "entry_date": iso_day(6)}},  # 6 d < 5+2
    {"ticker": "C07", "direction": "BEARISH", "real_option": {"ask": 1.0, "entry_date": iso_day(7)}},  # 7 d -> überfällig
    {"ticker": "C08", "direction": "BULLISH", "real_option": {"ask": 1.0, "entry_date": iso_day(13)},
     "outcomes": {"real_opt_ret_5d": 0.1}},
    {"ticker": "C09", "direction": "BULLISH", "real_option": {"ask": 1.0, "entry_date": iso_day(13)},
     "outcomes": {"real_opt_ret_5d": -0.4, "real_opt_late_5d": True}, "late_mark": True},
    {"ticker": "C10", "direction": "BULLISH", "real_option": {"ask": 1.0}},                      # ohne Entry-Datum
]


def test_candidate_lifecycle_counts(env):
    env.write_jsonl("candidate_ledger", CANDIDATES[:6], name="2026-09.jsonl")
    env.write_jsonl("candidate_ledger", CANDIDATES[6:], name="2026-10.jsonl")              # mehrere Dateien
    cand = env.collect()["lifecycle"]["candidates"]
    assert cand == {"SIGNAL": 10, "EXCLUDED_NO_CHAIN": 1, "EXCLUDED_NO_CONTRACT": 1, "CONTRACT_SELECTED": 8,
                    "ENTRY_PENDING": 2, "ENTRY_OBSERVED": 6, "EXIT_PENDING": 3, "EXIT_OVERDUE": 1,
                    "OUTCOME_FINAL": 2, "LATE_MARK": 1}


def test_candidate_lifecycle_empty_without_ledger(env):
    assert env.collect()["lifecycle"]["candidates"] == {}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Forward-Performance
# ═══════════════════════════════════════════════════════════════════════════
def test_champion_observed_metrics_use_only_ok_evaluations(env):
    env.history(closed=[                                        # absichtlich NICHT in Tagesreihenfolge
        closed_trade("F03", "2026-09-01", "2026-10-03", -0.40),
        closed_trade("F01", "2026-09-01", "2026-10-04", 0.30),
        closed_trade("F04", "2026-09-01", "2026-10-02", -0.20, outcome_net=-0.25),          # Netto zählt
        closed_trade("F02", "2026-09-01", "2026-10-01", 0.50),
        closed_trade("F05", "2026-09-01", "2026-10-05", 1.00, method="delta_approx"),       # modelliert -> raus
        closed_trade("F06", "2026-09-01", "2026-10-06", 0.20, debit=0.0)])                  # DATA_BAD -> raus
    champ = env.collect()["forward"]["champion"]
    m = champ["observed"]
    assert m["n"] == 4 and m["n_independent_days"] == 4
    # Netto = Brutto - Kommission (C1 je 1-Leg-Trade bei Debit 1.0); gespeichertes outcome_net (F04) hat Vorrang
    net = [0.5 - C1, -0.25, -0.4 - C1, 0.3 - C1]                       # Tagesreihenfolge F02, F04, F03, F01
    assert m["sum_net"] == pytest.approx(sum(net)) and m["mean_net"] == pytest.approx(sum(net) / 4)
    assert m["median_net"] == pytest.approx((-0.25 + 0.3 - C1) / 2) and m["hit_rate"] == 0.5
    assert m["worst"] == pytest.approx(-0.4 - C1)
    assert m["stdev"] == pytest.approx(statistics.stdev(net), abs=1e-6)
    # Drawdown der kumulierten Summe in TAGESREIHENFOLGE: Peak 0.487, Tief -0.176 -> -0.663
    # (in Eingabereihenfolge wäre es -0.413)
    assert m["max_drawdown_cum"] == pytest.approx(-0.663)
    assert m["loss_distribution"] == {"n_losses": 2, "p50_loss": -0.25, "share_total_loss": 0.0}
    assert m["verdict"] == "NEED_MORE_DATA" and m["ci"] is None
    assert champ["not_evaluable"] == {"UNUSABLE": 1, "DATA_BAD": 1}


def test_champion_drawdown_starts_from_zero_peak(env):
    env.history(closed=[closed_trade("N01", "2026-09-01", "2026-10-01", -0.3),
                        closed_trade("N02", "2026-09-01", "2026-10-02", -0.2)])
    m = env.collect()["forward"]["champion"]["observed"]
    assert m["max_drawdown_cum"] == pytest.approx(-0.5 - 2 * C1) and m["hit_rate"] == 0.0
    assert m["loss_distribution"]["share_total_loss"] == 0.0


def test_champion_total_loss_share_counts_minus_99_percent_and_worse(env):
    env.history(closed=[closed_trade("TL1", "2026-09-01", "2026-10-01", -1.0, method="option_quote_bid_zero"),
                        closed_trade("TL2", "2026-09-01", "2026-10-02", -0.99 + C1),            # netto -0.99 zählt
                        closed_trade("TL3", "2026-09-01", "2026-10-03", -0.989 + C1),
                        closed_trade("TL4", "2026-09-01", "2026-10-04", 0.5)])
    assert env.collect()["forward"]["champion"]["observed"]["loss_distribution"]["share_total_loss"] == 0.5


def test_champion_zero_return_is_no_hit_and_no_loss_and_single_trade_has_no_stdev(env):
    env.history(closed=[closed_trade("Z01", "2026-09-01", "2026-10-01", C1)])       # netto genau 0
    m = env.collect()["forward"]["champion"]["observed"]
    assert m["n"] == 1 and m["hit_rate"] == 0.0 and m["stdev"] is None
    assert m["loss_distribution"] == {"n_losses": 0, "p50_loss": None, "share_total_loss": 0.0}
    assert m["max_drawdown_cum"] == 0.0 and m["worst"] == 0.0


def test_champion_without_ok_trades_is_no_data(env):
    env.history(closed=[closed_trade("MOD", "2026-09-01", "2026-10-01", 0.4, method="delta_approx")])
    champ = env.collect()["forward"]["champion"]
    assert champ["observed"] == {"n": 0, "n_independent_days": 0, "verdict": "NO_DATA"}
    assert champ["not_evaluable"] == {"UNUSABLE": 1}


def bulk(n: int, n_days: int, value_of) -> list[dict]:
    """n OK-Trades auf genau min(n, n_days) verschiedene Schlusstage."""
    start = date(2026, 9, 1)
    return trades_on_days([value_of(i) for i in range(n)], lambda i: (start + timedelta(days=i % n_days)).isoformat())


@pytest.mark.parametrize("n, n_days, value_of, verdict", [
    (30, 20, lambda i: 0.05 + 0.01 * (i % 5), "POSITIVE"),                    # Mindestdaten exakt erreicht
    (40, 25, lambda i: 0.05 + 0.01 * (i % 5), "POSITIVE"),
    (30, 20, lambda i: -(0.05 + 0.01 * (i % 5)), "NEGATIVE"),
    (30, 20, lambda i: 0.1 if i % 2 == 0 else -0.1, "INCONCLUSIVE"),
    (29, 25, lambda i: 0.1, "NEED_MORE_DATA"),                                # zu wenige Trades
    (30, 19, lambda i: 0.1, "NEED_MORE_DATA")])                               # zu wenige unabhängige Tage
def test_champion_verdict_thresholds(env, n, n_days, value_of, verdict):
    env.history(closed=bulk(n, n_days, value_of))
    m = env.collect()["forward"]["champion"]["observed"]
    assert m["n"] == n and m["n_independent_days"] == n_days
    assert m["verdict"] == verdict
    if verdict == "NEED_MORE_DATA":
        assert m["ci"] is None
    else:
        lo, hi = m["ci"]
        assert lo <= m["mean_net"] <= hi
        assert (lo > 0) == (verdict == "POSITIVE") and (hi < 0) == (verdict == "NEGATIVE")


def test_champion_split_by_strategy_and_regime(env):
    def tr(name, day, v, strategy, vix=None, **kw):
        return closed_trade(name, "2026-09-01", f"2026-10-0{day}", v, strategy=strategy,
                            features={"vix_structure": vix} if vix else {}, **kw)
    env.history(closed=[tr("G01", 1, 0.2, "LONG_CALL", "CONTANGO"), tr("G02", 2, -0.1, "LONG_CALL", "CONTANGO"),
                        tr("G03", 3, 0.3, "LONG_CALL"), tr("G04", 4, 0.4, "BULL_CALL_SPREAD", "BACKWARDATION"),
                        tr("G05", 5, 0.1, "BULL_CALL_SPREAD", "BACKWARDATION"),
                        tr("G06", 6, 9.9, "BULL_CALL_SPREAD", "BACKWARDATION", method="delta_approx")])
    champ = env.collect()["forward"]["champion"]
    assert {k: (v["n"], round(v["sum_net"], 6)) for k, v in champ["by_strategy"].items()} == {
        "LONG_CALL": (3, round(0.4 - 3 * C1, 6)), "BULL_CALL_SPREAD": (2, round(0.5 - 2 * C2, 6))}  # ohne Modell
    assert {k: (v["n"], round(v["sum_net"], 6)) for k, v in champ["by_regime"].items()} == {
        "CONTANGO": (2, round(0.1 - 2 * C1, 6)), "BACKWARDATION": (2, round(0.5 - 2 * C2, 6)),
        "unknown": (1, round(0.3 - C1, 6))}


def write_decisions(env: Env, rows: list[tuple], hyps: dict | None = None) -> None:
    """rows: (decision_id, {hypothese: (evaluable, fired)}, outcome-oder-'none'-oder-None, outcome_method)."""
    ledger, outs = [], []
    for i, (did, intel, outcome, method) in enumerate(rows):
        ledger.append({"decision_id": did, "date": "2026-10-01", "ticker": f"D{i:03d}",
                       "intelligence": {k: {"evaluable": e, "fired": f} for k, (e, f) in intel.items()}})
        if outcome is not None:
            outs.append({"decision_id": did, "outcome": None if outcome == "none" else outcome,
                         "outcome_method": method})
    env.write_jsonl("decision_ledger", ledger, name="2026-10.jsonl")
    env.write_jsonl("decision_outcomes", outs)
    if hyps is not None:
        env.write_json("promotion_state", {"hypotheses": hyps})


def test_challenger_value_of_abstention_on_frozen_population(env):
    T, F = True, False
    write_decisions(env, [
        ("d1", {"A": (T, T), "B": (T, F)}, -0.5, "option_quote"),
        ("d2", {"A": (T, T), "B": (T, F)}, 0.2, "option_quote"),
        ("d3", {"A": (T, T), "B": (T, F)}, -0.1, "option_quote"),
        ("d4", {"A": (T, F), "B": (T, T)}, 0.3, "option_quote"),
        ("d5", {"A": (T, F), "B": (T, T)}, -0.1, "option_quote"),
        ("d6", {"A": (F, T)}, -0.9, "delta_approx"),                    # nicht auswertbar: feuert, zählt aber nicht
        ("d7", {"A": (T, T)}, None, None),                              # Outcome noch offen
        ("d8", {"A": (T, T)}, "none", "option_quote")],                 # Outcome-Zeile ohne Wert
        hyps={"A": {"state": "GUARDED_PRODUCTION", "influence_level": "ABSTENTION_ONLY"},
              "C": {"state": "PROSPECTIVE_CHALLENGER", "influence_level": "NONE", "production_class": "abstention",
                    "eligible_stage": "CHAMPION_TRADE", "evidence": {"n_observations": 3}}})
    fwd = env.collect()["forward"]
    assert fwd["population"]["decisions"] == 8 and fwd["population"]["resolved"] == 6
    assert fwd["population"]["outcome_methods"] == {"option_quote": 5, "delta_approx": 1}
    a, b, c = fwd["challengers"]["A"], fwd["challengers"]["B"], fwd["challengers"]["C"]
    # A feuert auf d1 (-0.5), d2 (+0.2), d3 (-0.1); behält d4 (+0.3), d5 (-0.1)
    assert a["evaluable_resolved"] == 5 and a["would_act_on"] == 3
    assert a["avoided_losses"] == pytest.approx(0.6) and a["foregone_gains"] == pytest.approx(0.2)
    assert a["incremental_vs_champion_sum"] == pytest.approx(0.4)
    assert a["incremental_vs_champion_mean"] == pytest.approx(0.08)                # 0.4 / 5 auswertbare
    assert a["fired_mean"] == pytest.approx(-0.4 / 3, abs=1e-6) and a["kept_mean"] == pytest.approx(0.1)
    assert a["state"] == "GUARDED_PRODUCTION" and a["influence_level"] == "ABSTENTION_ONLY"
    assert a["verdict"] == "NEED_MORE_DATA" and a["observed_outcome_share"] == round(5 / 6, 4)
    # B feuert auf dieselbe Population nur auf d4 (+0.3) und d5 (-0.1): Abstinenz hätte Gewinn gekostet
    assert b["evaluable_resolved"] == 5 and b["would_act_on"] == 2
    assert b["avoided_losses"] == pytest.approx(0.1) and b["foregone_gains"] == pytest.approx(0.3)
    assert b["incremental_vs_champion_sum"] == pytest.approx(-0.2)
    assert b["incremental_vs_champion_mean"] == pytest.approx(-0.04)
    assert b["state"] is None and b["population"] == "champion_decisions"
    # C steht nur im Promotion-State: keine Entscheidung -> Kennzahlen None (nie 0)
    assert c["evaluable_resolved"] == 0 and c["would_act_on"] == 0
    assert c["avoided_losses"] is None and c["foregone_gains"] is None and c["fired_mean"] is None
    assert c["incremental_vs_champion_sum"] is None and c["incremental_vs_champion_mean"] is None
    assert c["population"] == "CHAMPION_TRADE" and c["controller_evidence"]["n_observations"] == 3


def test_challenger_that_never_fires_has_no_fired_metrics(env):
    write_decisions(env, [("d1", {"Q": (True, False)}, 0.2, "option_quote"),
                          ("d2", {"Q": (True, False)}, -0.2, "option_quote")])
    q = env.collect()["forward"]["challengers"]["Q"]
    assert q["evaluable_resolved"] == 2 and q["would_act_on"] == 0 and q["kept_mean"] == pytest.approx(0.0)
    assert q["avoided_losses"] is None and q["foregone_gains"] is None and q["incremental_vs_champion_sum"] is None


def test_challenger_with_only_non_evaluable_decisions_is_empty(env):
    write_decisions(env, [("d1", {"Z": (False, True)}, -0.5, "option_quote")])
    z = env.collect()["forward"]["challengers"]["Z"]
    assert z["evaluable_resolved"] == 0 and z["would_act_on"] == 0 and z["avoided_losses"] is None


@pytest.mark.parametrize("n_eval, n_fired, verdict", [(30, 10, "SEE_CONTROLLER"), (29, 10, "NEED_MORE_DATA"),
                                                      (30, 9, "NEED_MORE_DATA"), (60, 40, "SEE_CONTROLLER")])
def test_challenger_verdict_needs_30_evaluable_and_10_fired(env, n_eval, n_fired, verdict):
    write_decisions(env, [(f"d{i}", {"V": (True, i < n_fired)}, 0.1 if i % 2 else -0.1, "option_quote")
                          for i in range(n_eval)])
    v = env.collect()["forward"]["challengers"]["V"]
    assert v["evaluable_resolved"] == n_eval and v["would_act_on"] == n_fired and v["verdict"] == verdict


def test_forward_meta_label_status_defaults_to_never_run(env):
    assert env.collect()["forward"]["meta_label"] == {"status": "NEVER_RUN"}


def test_scorecard_honours_path_overrides(env, cwd):
    env.history(closed=[closed_trade("SCR", "2026-09-01", "2026-09-10", 0.2)])           # liegt unter paths['history']
    cells = env.collect()["forward"]["scorecard"]
    assert "CHAMPION_TRADE|ALL" in cells


def test_scorecard_ignores_cwd_defaults_when_paths_are_given(env, cwd):
    """Mit Overrides liest die Scorecard nie cwd-relative Defaults (outputs/history.json)."""
    (cwd / "outputs").mkdir()
    (cwd / "outputs" / "history.json").write_text(json.dumps(
        {"closed_trades": [closed_trade("SCR", "2026-09-01", "2026-09-10", 0.2)]}), encoding="utf-8")
    assert "CHAMPION_TRADE|ALL" not in env.collect()["forward"]["scorecard"]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integrität
# ═══════════════════════════════════════════════════════════════════════════
def signed_state(**body) -> dict:
    st = {"generated": "2026-10-13T00:00:00+00:00", "hypotheses": {}, **body}
    st["state_hash"] = pc.state_digest(st)
    return st


def test_promotion_state_hash_ok_has_no_issue(env):
    env.write_json("promotion_state", signed_state(max_automatic_influence="ABSTENTION_ONLY"))
    i = env.collect()["integrity"]
    assert i["promotion_state_hash_ok"] is True and "PROMOTION_STATE_HASH" not in i["issues"]


def test_promotion_state_hash_mismatch_is_an_issue(env):
    st = signed_state(max_automatic_influence="ABSTENTION_ONLY")
    st["max_automatic_influence"] = "WEIGHT_25"                       # nach dem Signieren verändert
    env.write_json("promotion_state", st)
    i = env.collect()["integrity"]
    assert i["promotion_state_hash_ok"] is False and i["issues"] == ["PROMOTION_STATE_HASH"]


def test_promotion_state_missing_is_not_checked_and_not_an_issue(env):
    i = env.collect()["integrity"]
    assert i["promotion_state_hash_ok"] is None and i["issues"] == []


def test_clean_integrity_has_no_issues_and_zero_counters(env):
    i = env.collect()["integrity"]
    assert i["issues"] == [] and i["transition_chain_problems"] == [] and i["registry_problems"] == []
    assert (i["decision_ledger_duplicates"], i["trade_outcome_duplicates"]) == (0, 0)
    assert (i["quote_records_30d"], i["quote_duplicate_ids"], i["quote_non_observed_prices"]) == (0, 0, 0)
    assert i["closed_trade_quarantine"] == 0 and i["external_archive"] == {}


def test_transition_chain_manipulation_is_an_issue(env):
    p = env.paths["promotion_transitions"]
    kw = {"path": p, "code_ver": "c", "data_ver": "d"}
    pc.transition("K@v1", "IDEA", reason="synthetic one", now="2026-10-10T00:00:00+00:00", **kw)
    pc.transition("K@v1", "PROSPECTIVE_CHALLENGER", reason="synthetic two", now="2026-10-10T01:00:00+00:00", **kw)
    assert env.collect()["integrity"]["issues"] == []                   # intakte Kette
    p.write_text(p.read_text(encoding="utf-8").replace("synthetic two", "tampered"), encoding="utf-8")
    i = env.collect()["integrity"]
    assert i["issues"] == ["TRANSITION_CHAIN"] and i["transition_chain_problems"]


def test_contract_registry_chain_break_is_an_issue(env):
    env.write_jsonl("contract_registry", [{"key": "K@v1", "contract": {}, "spec_hash": "bad", "prev_hash": "nope",
                                           "entry_hash": "zzz"}])
    i = env.collect()["integrity"]
    assert i["issues"] == ["CONTRACT_REGISTRY"] and i["registry_problems"]


def test_decision_ledger_duplicate_date_ticker_across_files(env):
    rows_a = [{"date": "2026-10-13", "ticker": "AAA", "decision_id": "d1"},
              {"date": "2026-10-13", "ticker": "BBB", "decision_id": "d2"},
              {"date": "2026-10-14", "ticker": "AAA", "decision_id": "d3"}]          # gleicher Ticker, anderer Tag: ok
    env.write_jsonl("decision_ledger", rows_a, name="2026-10.jsonl")
    assert env.collect()["integrity"]["decision_ledger_duplicates"] == 0
    env.write_jsonl("decision_ledger", [{"date": "2026-10-13", "ticker": "AAA", "decision_id": "d4"}],
                    name="2026-11.jsonl")                                              # (Datum, Ticker) doppelt
    i = env.collect()["integrity"]
    assert i["decision_ledger_duplicates"] == 1 and i["issues"] == ["DECISION_LEDGER_DUPLICATES"]


def test_duplicate_trade_id_in_trade_outcomes(env):
    env.write_jsonl("trade_outcomes", [{"trade_id": "T1"}, {"trade_id": "T1"}, {"trade_id": "T1"},
                                       {"trade_id": "T2"}, {"trade_id": "T3"}, {"trade_id": "T3"}])
    i = env.collect()["integrity"]
    assert i["trade_outcome_duplicates"] == 2 and i["issues"] == ["TRADE_OUTCOME_DUPLICATES"]     # T1 und T3


def quote_record(sym: str, retrieved: str, context: str = "mark", price_type: str | None = None) -> dict:
    rec = oq.normalize({"symbol": sym, "bid": 1.0, "ask": 1.1, "iv": 0.4, "quote_time": retrieved},
                       underlying=sym[:-15], source="tradier", context=context, retrieved_at=retrieved)
    if price_type:
        rec["price_type"] = price_type
    return rec


def test_quote_archive_duplicate_ids_and_non_observed_prices(env):
    a = quote_record(oq.occ_symbol("AAA", "2027-01-15", "call", 100.0), "2026-10-13T15:00:00+00:00")
    b = quote_record(oq.occ_symbol("BBB", "2027-01-15", "call", 100.0), "2026-10-13T15:00:00+00:00",
                     price_type="MODELED")
    old = quote_record(oq.occ_symbol("CCC", "2027-01-15", "call", 100.0), "2026-08-01T15:00:00+00:00")
    env.write_jsonl("quotes", [a, a, a, b], name="2026-10-13.jsonl")        # direkt geschrieben (archive wäre idempotent)
    env.write_jsonl("quotes", [old, old], name="2026-08-01.jsonl")           # außerhalb der 30 Tage -> ignoriert
    i = env.collect()["integrity"]
    assert i["quote_records_30d"] == 4 and i["quote_duplicate_ids"] == 1 and i["quote_non_observed_prices"] == 1
    assert i["issues"] == ["QUOTE_ARCHIVE"]


@pytest.mark.parametrize("day_offset, counted", [(30, True), (31, False)])
def test_quote_archive_integrity_window_is_30_days(env, day_offset, counted):
    day = iso_day(day_offset)
    rec = quote_record(oq.occ_symbol("AAA", "2027-01-15", "call", 100.0), f"{day}T15:00:00+00:00")
    env.write_jsonl("quotes", [rec, rec], name=f"{day}.jsonl")
    i = env.collect()["integrity"]
    assert (i["quote_duplicate_ids"] == 1) is counted and (i["issues"] == ["QUOTE_ARCHIVE"]) is counted


def test_archive_written_by_option_quotes_has_no_integrity_issue(env):
    sym = oq.occ_symbol("AAA", "2027-01-15", "call", 100.0)
    for hours in (1, 2, 3):
        env.capture(sym, NOW - timedelta(hours=hours), context=f"mark{hours}")
    i = env.collect()["integrity"]
    assert i["quote_records_30d"] == 3 and i["issues"] == []


def test_closed_trade_quarantine_is_counted(env):
    env.history(closed_trades_quarantine=[{"ticker": "QQQ"}, {"ticker": "RRR"}])
    assert env.collect()["integrity"]["closed_trade_quarantine"] == 2


def test_external_archive_hash_failure_is_an_issue(cwd, env):
    """Zeilen unter cwd/outputs/external_data/normalized: nur jüngst geschriebene Dateien werden geprüft."""
    ts = NOW.timestamp()
    recent = cwd / "outputs/external_data/normalized/src/recent.jsonl"
    stale = cwd / "outputs/external_data/normalized/src/stale.jsonl"
    recent.parent.mkdir(parents=True)
    bad_row = {"payload_hash": "abc", "quality_status": "BOGUS", "value": 1}
    recent.write_text(json.dumps(bad_row) + "\n" + json.dumps({"value": 2}) + "\nkein json\n", encoding="utf-8")
    stale.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")
    os.utime(recent, (ts, ts))
    os.utime(stale, (ts - 4 * 86400, ts - 4 * 86400))                      # älter als 3 Tage -> übersprungen
    i = env.collect()["integrity"]
    assert i["external_archive"] == {"failed": 1, "legacy": 1, "unparsable": 1}
    assert i["issues"] == ["EXTERNAL_ARCHIVE_HASH"]


def test_integrity_issue_order_is_stable(env):
    st = signed_state()
    st["generated"] = "tampered"
    env.write_json("promotion_state", st)
    env.write_jsonl("trade_outcomes", [{"trade_id": "T1"}, {"trade_id": "T1"}])
    env.write_jsonl("decision_ledger", [{"date": "d", "ticker": "A"}, {"date": "d", "ticker": "A"}], name="x.jsonl")
    assert env.collect()["integrity"]["issues"] == ["PROMOTION_STATE_HASH", "DECISION_LEDGER_DUPLICATES",
                                                    "TRADE_OUTCOME_DUPLICATES"]


def test_corrupt_contract_registry_is_reported_as_issue_not_lost(env):
    env.write_json("promotion_state", signed_state())
    env.paths["contract_registry"].write_text('{"key": "K@v1"\n{"key": "L@v1"}\n', encoding="utf-8")
    i = env.collect()["integrity"]
    assert "CONTRACT_REGISTRY" in i["issues"] and i["promotion_state_hash_ok"] is True


def test_string_paths_for_transition_and_registry_files_are_accepted(env):
    paths = {**env.paths, "promotion_transitions": str(env.paths["promotion_transitions"]),
             "contract_registry": str(env.paths["contract_registry"])}
    assert "error" not in om.collect(NOW, paths)["integrity"]


def test_string_paths_are_accepted_for_the_other_keys(env):
    env.history(active=[option_trade("STR", iso_day(1))])
    paths = {k: str(v) for k, v in env.paths.items() if k not in ("promotion_transitions", "contract_registry")}
    res = om.collect(NOW, {**env.paths, **paths})
    assert all("error" not in res[s] for s in SECTIONS)
    assert res["lifecycle"]["champion"]["by_state"]["EXIT_PENDING"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. Training (Meta-Label-Manifeste)
# ═══════════════════════════════════════════════════════════════════════════
def write_meta_version(root: Path, version: str, *, trained_at: str = "2026-10-10T08:00:00+00:00",
                       label_cutoff: str = "2026-10-10", n_train: int = 250, n_train_dates: int = 30,
                       readiness: dict | None = None, manifest: bool = True) -> dict:
    meta = {"version": version, "spec_hash": "x", "trained_at": trained_at, "label_cutoff": label_cutoff,
            "n_train": n_train, "train_ids_hash": "h", "new_labels_since_previous": 0, "active_model": "logit_l2"}
    if manifest:
        (root / "versions" / version).mkdir(parents=True, exist_ok=True)
        full = {**meta, "n_train_dates": n_train_dates, "spec": {"readiness": readiness or
                                                                {"min_labels": 200, "min_dates": 20}}}
        (root / "versions" / version / "manifest.json").write_text(json.dumps(full), encoding="utf-8")
    return meta


def write_meta_registry(root: Path, versions: list[dict], active: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "registry.json").write_text(json.dumps({"spec_hash": "x", "versions": versions, "active": active,
                                                    "history": []}), encoding="utf-8")


def test_training_never_run_without_meta_label_root(env):
    t = env.collect()["training"]
    assert t["status"] == "NEVER_RUN" and t["versions"] == 0 and t["manifest_violations"] == []
    assert t["active"] is None and t["active_artifact_ok"] is None


def test_training_manifest_violations(env):
    root = env.paths["meta_label"]
    write_meta_registry(root, [
        write_meta_version(root, "v-ok", label_cutoff="2026-10-10", n_train=200, n_train_dates=20),   # exakt an den Grenzen
        write_meta_version(root, "v-leak", label_cutoff="2026-10-11"),                                # Label NACH Training
        write_meta_version(root, "v-few-labels", n_train=199),
        write_meta_version(root, "v-few-dates", n_train_dates=19),
        write_meta_version(root, "v-both", label_cutoff="2026-10-12", n_train=10, n_train_dates=2),
        write_meta_version(root, "v-no-manifest", manifest=False)])                                   # wird übersprungen
    (root / "status.json").write_text(json.dumps({"status": "TRAINED", "n_labels": 250, "n_label_dates": 30,
                                                  "required": {"min_labels": 200, "min_dates": 20},
                                                  "as_of": "2026-10-10"}), encoding="utf-8")
    t = env.collect()["training"]
    assert t["versions"] == 6 and t["status"] == "TRAINED" and t["n_labels"] == 250 and t["n_label_dates"] == 30
    assert t["required"] == {"min_labels": 200, "min_dates": 20} and t["as_of"] == "2026-10-10"
    assert t["manifest_violations"] == [
        {"version": "v-leak", "violations": ["LABEL_AFTER_TRAINING"]},
        {"version": "v-few-labels", "violations": ["TRAINED_BELOW_READINESS"]},
        {"version": "v-few-dates", "violations": ["TRAINED_BELOW_READINESS"]},
        {"version": "v-both", "violations": ["LABEL_AFTER_TRAINING", "TRAINED_BELOW_READINESS"]}]


def test_training_label_cutoff_on_training_day_is_not_leakage(env):
    root = env.paths["meta_label"]
    write_meta_registry(root, [write_meta_version(root, "v1", trained_at="2026-10-10T23:59:59+00:00",
                                                  label_cutoff="2026-10-10")])
    assert env.collect()["training"]["manifest_violations"] == []


def test_training_active_artifact_integrity(env, tmp_path):
    root = env.paths["meta_label"]
    blob = pickle.dumps({"synthetic": "model"})
    art = root / "versions" / "v1" / "logit_l2.pkl"
    meta = write_meta_version(root, "v1")
    art.write_bytes(blob)
    meta["artifacts"] = {"logit_l2": {"path": "versions/v1/logit_l2.pkl", "sha256": hashlib.sha256(blob).hexdigest()}}
    write_meta_registry(root, [meta], active="v1")
    t = env.collect()["training"]
    assert t["active"] == "v1" and t["active_artifact_ok"] is True
    # Hash weicht ab -> fail-safe: Artefakt nicht in Ordnung (neue Wurzel, damit der Modell-Cache nicht greift)
    root2 = tmp_path / "meta2"
    (root2 / "versions" / "v1").mkdir(parents=True)
    (root2 / "versions" / "v1" / "logit_l2.pkl").write_bytes(blob)
    meta2 = {**meta, "artifacts": {"logit_l2": {"path": "versions/v1/logit_l2.pkl", "sha256": "0" * 64}}}
    write_meta_registry(root2, [meta2], active="v1")
    assert om.training_status({**env.paths, "meta_label": root2})["active_artifact_ok"] is False


def test_training_active_without_artifacts_is_not_ok(env):
    root = env.paths["meta_label"]
    write_meta_registry(root, [write_meta_version(root, "v1")], active="v1")
    t = env.collect()["training"]
    assert t["active_artifact_ok"] is False and t["hold"] is None


# ═══════════════════════════════════════════════════════════════════════════
# 6. Promotion
# ═══════════════════════════════════════════════════════════════════════════
HYPS = {
    "G-ok": {"state": "GUARDED_PRODUCTION", "influence_level": "ABSTENTION_ONLY", "integrity_ok": True},
    "L-ok": {"state": "LIMITED_PRODUCTION", "influence_level": "RERANK_ONLY", "integrity_ok": True},
    "G-bad-integrity": {"state": "GUARDED_PRODUCTION", "influence_level": "ABSTENTION_ONLY", "integrity_ok": False},
    "G-no-integrity-flag": {"state": "GUARDED_PRODUCTION", "influence_level": "ABSTENTION_ONLY"},
    "P-shadow-with-influence": {"state": "PROSPECTIVE_CHALLENGER", "influence_level": "ABSTENTION_ONLY",
                                "integrity_ok": True},
    "F-forward-validated": {"state": "FORWARD_VALIDATED", "influence_level": "RERANK_ONLY", "integrity_ok": True},
    "D-demoted": {"state": "DEMOTED", "influence_level": "WEIGHT_10", "integrity_ok": True},
    "S-shadow": {"state": "PROSPECTIVE_CHALLENGER", "influence_level": "NONE", "integrity_ok": False},
    "S-no-level": {"state": "IDEA", "integrity_ok": False},
}


def test_unjustified_influence_requires_integrity_and_active_production_state(env):
    env.write_json("promotion_state", {"hypotheses": HYPS, "max_automatic_influence": "ABSTENTION_ONLY",
                                       "safe_mode": {"active": True}, "generated": "2026-10-13T00:00:00+00:00"})
    env.write_json("system_state", {"champion_version": {"version": "champ-7"}})
    p = env.collect()["promotion"]
    assert p["unjustified_influence"] == ["D-demoted", "F-forward-validated", "G-bad-integrity",
                                          "G-no-integrity-flag", "P-shadow-with-influence"]
    assert p["active_influence"] == sorted(k for k, h in HYPS.items()
                                           if h.get("influence_level", "NONE") != "NONE")
    assert "S-shadow" not in p["unjustified_influence"] and "S-no-level" not in p["unjustified_influence"]
    assert p["max_automatic_influence"] == "ABSTENTION_ONLY" and p["safe_mode"] is True
    assert p["champion_version"] == "champ-7" and p["generated"] == "2026-10-13T00:00:00+00:00"


def test_promotion_counters_and_blocking_reason(env):
    env.write_json("promotion_state", {"hypotheses": {
        "A": {"state": "PROSPECTIVE_CHALLENGER", "influence_level": "NONE", "next_requirement": "mehr Daten",
              "reasons": ["x"], "decision": "KEEP_SHADOW", "recommendation": None, "integrity_ok": True},
        "B": {"state": "PROSPECTIVE_CHALLENGER", "influence_level": "NONE", "reasons": ["n zu klein", "CI offen"]},
        "C": {"state": "REJECTED", "influence_level": "NONE"}}})
    p = env.collect()["promotion"]
    assert p["by_state"] == {"PROSPECTIVE_CHALLENGER": 2, "REJECTED": 1} and p["by_influence"] == {"NONE": 3}
    assert p["hypotheses"]["A"]["blocking"] == "mehr Daten" and p["hypotheses"]["A"]["decision"] == "KEEP_SHADOW"
    assert p["hypotheses"]["B"]["blocking"] == "n zu klein; CI offen" and p["hypotheses"]["C"]["blocking"] is None
    assert p["unjustified_influence"] == [] and p["active_influence"] == []


def test_full_production_with_intact_integrity_is_justified(env):
    assert "FULL_PRODUCTION" in pc.ACTIVE_STATES
    env.write_json("promotion_state", {"hypotheses": {
        "FULL": {"state": "FULL_PRODUCTION", "influence_level": "WEIGHT_25", "integrity_ok": True}}})
    assert env.collect()["promotion"]["unjustified_influence"] == []


def test_every_active_state_of_the_controller_except_full_is_justified(env):
    env.write_json("promotion_state", {"hypotheses": {
        s: {"state": s, "influence_level": "ABSTENTION_ONLY", "integrity_ok": True}
        for s in pc.ACTIVE_STATES if s != "FULL_PRODUCTION"}})
    assert env.collect()["promotion"]["unjustified_influence"] == []


@pytest.mark.parametrize("content, active", [
    (None, False),                                    # Datei fehlt -> nicht aktiv
    ("active: true\n", True),
    ("active: false\n", False),
    ("{}\n", False),
    ("active: [unclosed\n", True),                    # defekt -> fail-safe AKTIV
    ("- 1\n- 2\n", True)])                            # falscher Typ -> fail-safe AKTIV
def test_kill_switch_file(env, content, active):
    if content is not None:
        env.paths["kill_switch"].write_text(content, encoding="utf-8")
    assert env.collect()["promotion"]["kill_switch_active"] is active


def test_promotion_without_state_file_is_empty_but_well_formed(env):
    p = env.collect()["promotion"]
    assert p["hypotheses"] == {} and p["by_state"] == {} and p["unjustified_influence"] == []
    assert p["max_automatic_influence"] is None and p["safe_mode"] is None and p["champion_version"] is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. Robustheit / Read-only
# ═══════════════════════════════════════════════════════════════════════════
def one_challenger_state() -> dict:
    return {"A": {"state": "PROSPECTIVE_CHALLENGER", "influence_level": "NONE"}}


def populate_everything(env: Env, cwd: Path) -> None:
    """Befüllt jeden einzelnen `paths`-Ort mit kleinen synthetischen Daten (inkl. cwd-relativer Defaults)."""
    t_act = option_trade("ACT", iso_day(1))
    t_act["entry_quote_record_ids"] = [env.capture(leg_symbols(t_act)[0], NOW - timedelta(days=1))]
    env.capture(leg_symbols(t_act)[0], NOW - timedelta(hours=2), context="mark")
    lrn = learnable()
    env.history(active=[t_act, option_trade("NOC", iso_day(1), contract=False)],
                closed=[lrn, closed_trade("MOD", "2026-09-01", "2026-10-12", 0.3, method="delta_approx")],
                processed_outcomes={f"{tid(lrn)}:close:feature_stats": "2026-10-13"},
                closed_trades_quarantine=[{"ticker": "QQQ"}])
    env.write_jsonl("candidate_ledger", CANDIDATES, name="2026-10.jsonl")
    write_decisions(env, [("d1", {"A": (True, True)}, -0.5, "option_quote"),
                          ("d2", {"A": (True, False)}, 0.2, "option_quote")], hyps=one_challenger_state())
    env.write_jsonl("trade_outcomes", [{"trade_id": "T1"}, {"trade_id": "T1"}])
    pc.transition("K@v1", "IDEA", reason="synthetic", path=env.paths["promotion_transitions"], code_ver="c",
                  data_ver="d", now="2026-10-10T00:00:00+00:00")
    env.write_json("workflow_status", {"days": {TODAY.isoformat(): {"scanner": {"state": "OK", "conclusion": "success"}}}})
    env.write_json("system_state", {"champion_version": {"version": "v1"}})
    env.write_json("source_health", {"generated": "x", "sources": {"a": {"status": "HEALTHY"}}})
    env.write_json("external_health", {"sources": {"a": {"status": "OK"}}})
    env.paths["daily_reports"].mkdir()
    (env.paths["daily_reports"] / f"{TODAY.isoformat()}.json").write_text(
        json.dumps({"stats": {"universe": 5, "final_mc": 2, "roi_ok": 1, "roi_rejects": {"n": 1}}}), encoding="utf-8")
    root = env.paths["meta_label"]
    write_meta_registry(root, [write_meta_version(root, "v1", label_cutoff="2026-10-12")])
    env.paths["kill_switch"].write_text("active: false\n", encoding="utf-8")
    ext = cwd / "outputs/external_data/normalized/src/recent.jsonl"
    ext.parent.mkdir(parents=True)
    ext.write_text(json.dumps({"payload_hash": "abc", "quality_status": "BOGUS"}) + "\n", encoding="utf-8")
    os.utime(ext, (NOW.timestamp(), NOW.timestamp()))


@contextlib.contextmanager
def forbid_writes(monkeypatch):
    """Jeder Schreib-/Anlege-/Lösch-Versuch über open/Path/os/shutil/tempfile schlägt mit AssertionError fehl."""
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and set(mode) & set("wax+"):
            raise AssertionError(f"verbotener Schreibzugriff: open({file!r}, {mode!r})")
        return real_open(file, mode, *args, **kwargs)

    def blocked(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"verbotener Schreibzugriff: {name}")
        return _raise

    with monkeypatch.context() as m:
        m.setattr(builtins, "open", guarded_open)
        m.setattr(io, "open", guarded_open)
        for owner, names in ((Path, ("write_text", "write_bytes", "touch", "mkdir", "unlink", "rename", "replace",
                                     "rmdir")),
                             (os, ("replace", "rename", "remove", "unlink", "mkdir", "makedirs", "rmdir")),
                             (shutil, ("rmtree", "copy", "copy2", "copyfile", "move")),
                             (tempfile, ("mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryDirectory"))):
            for name in names:
                m.setattr(owner, name, blocked(f"{owner.__name__}.{name}"))
        yield


def test_write_guard_blocks_every_write_route_but_not_reads(tmp_path, monkeypatch):
    """Selbsttest des Schreibwächters: ohne ihn wäre der Read-only-Test wertlos."""
    f = tmp_path / "vorhanden.txt"
    f.write_text("x", encoding="utf-8")
    with forbid_writes(monkeypatch):
        assert f.read_text(encoding="utf-8") == "x" and f.read_bytes() == b"x" and f.exists()
        for attempt in (lambda: f.write_text("y"), lambda: f.write_bytes(b"y"), lambda: open(f, "a"),
                        lambda: f.open("w"), lambda: io.open(f, "r+"), lambda: (tmp_path / "neu").mkdir(),
                        lambda: os.replace(f, tmp_path / "b"), lambda: f.unlink(), lambda: tempfile.mkstemp(dir=tmp_path),
                        lambda: shutil.rmtree(tmp_path), lambda: (tmp_path / "t").touch()):
            with pytest.raises(AssertionError, match="verboten"):
                attempt()
    assert f.read_text(encoding="utf-8") == "x"


def test_a_section_that_writes_would_be_caught(env, cwd, tmp_path, monkeypatch):
    """Canary: schreibt eine Sektion, melden Wächter (als Sektionsfehler) UND Verzeichnis-Snapshot den Verstoß."""
    populate_everything(env, cwd)

    def writing_section(*args, **kwargs):
        (tmp_path / "data" / "neu.json").write_text("{}", encoding="utf-8")
        return {}

    monkeypatch.setattr(om, "data_health", writing_section)
    before = snapshot(tmp_path)
    with forbid_writes(monkeypatch):
        res = env.collect()
    assert res["data_health"]["error"].startswith("AssertionError: verbotener Schreibzugriff")
    assert snapshot(tmp_path) == before                       # Wächter hat das Schreiben verhindert
    (tmp_path / "data" / "neu.json").write_text("{}", encoding="utf-8")
    with_new = snapshot(tmp_path)
    assert with_new != before                                 # der Snapshot erkennt eine neue Datei
    (tmp_path / "data" / "neu.json").write_text("[]", encoding="utf-8")
    assert snapshot(tmp_path) != with_new                     # … und gleich große, geänderte Inhalte


def test_collect_writes_nothing_and_changes_no_input(env, cwd, tmp_path, monkeypatch):
    populate_everything(env, cwd)
    baseline = env.collect()                                  # wärmt Imports auf (kein __pycache__-Rauschen)
    before = snapshot(tmp_path)
    with forbid_writes(monkeypatch):
        result = env.collect()
    assert snapshot(tmp_path) == before                      # kein neues/geändertes/gelöschtes File oder Verzeichnis
    assert result == baseline and result["read_only"] is True
    assert all("error" not in result[s] for s in SECTIONS)


def test_collect_against_repo_checkout_leaves_repo_outputs_untouched(env, monkeypatch):
    """cwd = Repo-Wurzel: cwd-relative Defaults (Scorecard, externes Archiv) lesen echte Daten – nur lesend."""
    outputs = REPO / "outputs"
    if not outputs.is_dir():
        pytest.skip("kein outputs/-Verzeichnis im Checkout")
    monkeypatch.chdir(REPO)
    baseline = env.collect()                                  # Aufwärmen
    before = snapshot(outputs, content=False)
    config_before = (REPO / "config.yaml").stat().st_mtime_ns
    with forbid_writes(monkeypatch):
        result = env.collect()
    assert snapshot(outputs, content=False) == before
    assert (REPO / "config.yaml").stat().st_mtime_ns == config_before
    assert all(isinstance(result[sec], dict) and isinstance(baseline[sec], dict) for sec in SECTIONS)


def test_collect_is_deterministic_for_a_fixed_now(env, cwd):
    populate_everything(env, cwd)
    assert env.collect() == env.collect()


def test_collect_result_shape_and_defaults(env):
    res = env.collect()
    assert res["schema"] == "ops-status-v1" and res["read_only"] is True
    assert res["generated"] == "2026-10-14T15:00:00+00:00" and res["as_of"] == "2026-10-14"
    assert list(res)[:4] == ["schema", "generated", "as_of", "read_only"] and set(SECTIONS) <= set(res)
    live = om.collect(paths=env.paths)                         # now=None -> aktuelle UTC-Zeit
    assert datetime.fromisoformat(live["generated"]).tzinfo is not None
    assert live["as_of"] == live["generated"][:10]


def test_nothing_exists_every_section_is_a_dict_without_error(env):
    res = env.collect()
    for s in SECTIONS:
        assert isinstance(res[s], dict) and "error" not in res[s], s
    assert res["run"]["workflows"] == {} and res["run"]["report_date"] is None and res["run"]["report_is_today"] is False


def test_override_wins_over_default_location(env, cwd):
    (cwd / "outputs").mkdir()
    (cwd / "outputs" / "history.json").write_text(json.dumps({"active_trades": [option_trade("DEF", iso_day(1))]}),
                                                  encoding="utf-8")
    assert env.collect()["lifecycle"]["champion"]["by_state"]["EXIT_PENDING"] == 0       # Override (leer) gewinnt
    assert om.collect(NOW, None)["lifecycle"]["champion"]["by_state"]["EXIT_PENDING"] == 1   # ohne Override: Default
    assert om.collect(NOW, {})["lifecycle"]["champion"]["by_state"]["EXIT_PENDING"] == 1


def test_defaults_resolve_via_option_quotes_root_and_cwd(env, cwd, tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    monkeypatch.setattr(oq, "ROOT", archive / "quotes")
    sym = oq.occ_symbol("AAA", "2027-01-15", "call", 100.0)
    rec = quote_record(sym, (NOW - timedelta(hours=1)).isoformat())
    (archive / "quotes").mkdir(parents=True)
    (archive / "quotes" / f"{TODAY.isoformat()}.jsonl").write_text((json.dumps(rec) + "\n") * 2, encoding="utf-8")
    (archive / "trade_outcomes.jsonl").write_text('{"trade_id": "T1"}\n{"trade_id": "T1"}\n', encoding="utf-8")
    (cwd / "config").mkdir()
    (cwd / "config" / "production_kill_switch.yaml").write_text("active: true\n", encoding="utf-8")
    res = om.collect(NOW, None)
    assert res["integrity"]["quote_duplicate_ids"] == 1                      # Quote-Archiv = option_quotes.ROOT
    assert res["integrity"]["trade_outcome_duplicates"] == 1                 # trade_outcomes.jsonl neben dem Archiv
    assert res["promotion"]["kill_switch_active"] is True                    # config/production_kill_switch.yaml
    assert res["data_health"]["quote_archive_days"][TODAY.isoformat()]["n"] == 2


@pytest.mark.parametrize("key", PATH_KEYS)
def test_each_path_key_can_point_to_a_nonexistent_location(env, key):
    paths = {**env.paths, key: env.root / "does" / "not" / "exist" / key}
    res = om.collect(NOW, paths)
    assert all(isinstance(res[s], dict) and "error" not in res[s] for s in SECTIONS)


GARBAGE = "{{{ kein json"
MID_FILE_GARBAGE = 'kaputt\n{"a": 1}\n'                      # defekte Zeile MITTEN in einem JSONL -> CorruptLedgerError


@pytest.mark.parametrize("key, kind", [
    ("history", "garbage"), ("history", "binary"), ("history", "directory"), ("history", "list"),
    ("history", "string"), ("promotion_state", "garbage"), ("promotion_state", "list"),
    ("workflow_status", "garbage"), ("system_state", "garbage"), ("source_health", "garbage"),
    ("source_health", "list"), ("external_health", "garbage"), ("external_health", "list"),
    ("decision_outcomes", "midfile"), ("trade_outcomes", "midfile"), ("promotion_transitions", "midfile"),
    ("contract_registry", "midfile"), ("kill_switch", "garbage"), ("daily_reports", "garbage_file"),
    ("decision_ledger", "midfile_file"), ("candidate_ledger", "midfile_file"), ("quotes", "midfile_file"),
    ("meta_label", "garbage_registry"), ("meta_label", "garbage_status")])
def test_collect_never_raises_on_corrupt_inputs(env, key, kind):
    p = env.paths[key]
    p.parent.mkdir(parents=True, exist_ok=True)
    if kind == "garbage":
        p.write_text(GARBAGE, encoding="utf-8")
    elif kind == "binary":
        p.write_bytes(b"\xff\xfe\x00\x80")
    elif kind == "directory":
        p.mkdir()
    elif kind == "list":
        p.write_text("[1, 2]", encoding="utf-8")
    elif kind == "string":
        p.write_text('"nur ein String"', encoding="utf-8")
    elif kind == "midfile":
        p.write_text(MID_FILE_GARBAGE, encoding="utf-8")
    elif kind == "garbage_file":
        p.mkdir()
        (p / f"{TODAY.isoformat()}.json").write_text(GARBAGE, encoding="utf-8")
    elif kind == "midfile_file":
        p.mkdir()
        (p / f"{TODAY.isoformat()}.jsonl").write_text(MID_FILE_GARBAGE, encoding="utf-8")
    elif kind == "garbage_registry":
        p.mkdir()
        (p / "registry.json").write_text(GARBAGE, encoding="utf-8")
    elif kind == "garbage_status":
        p.mkdir()
        (p / "status.json").write_text(GARBAGE, encoding="utf-8")
    res = env.collect()                                       # darf nie werfen
    assert res["schema"] == "ops-status-v1"
    for s in SECTIONS:
        assert isinstance(res[s], dict), s
        if "error" in res[s]:
            assert isinstance(res[s]["error"], str) and ":" in res[s]["error"]       # '<Ausnahmeklasse>: <Text>'


@pytest.mark.parametrize("section, func", [
    ("run", "run_status"), ("quote_coverage", "quote_coverage"), ("lifecycle", "lifecycle"),
    ("forward", "forward_performance"), ("training", "training_status"), ("promotion", "promotion_status"),
    ("integrity", "integrity"), ("data_health", "data_health")])
def test_forced_exception_in_one_section_does_not_break_the_others(env, cwd, monkeypatch, section, func):
    populate_everything(env, cwd)
    baseline = env.collect()
    assert all("error" not in baseline[s] for s in SECTIONS)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(om, func, boom)
    res = env.collect()
    assert res[section] == {"error": "RuntimeError: boom"}
    for other in SECTIONS:
        if other != section:
            assert res[other] == baseline[other], other


def test_forced_exception_inside_forward_scorecard_keeps_champion_and_challengers(env, cwd, monkeypatch):
    from modules import evidence_ledger as el
    populate_everything(env, cwd)

    def boom(*args, **kwargs):
        raise ValueError("score kaputt")

    monkeypatch.setattr(el, "scorecard", boom)
    fwd = env.collect()["forward"]
    assert fwd["scorecard"] == {"error": "ValueError: score kaputt"}
    assert "observed" in fwd["champion"] and "A" in fwd["challengers"] and "error" not in fwd


def test_forced_exception_inside_external_archive_check_keeps_the_integrity_section(env, cwd, monkeypatch):
    populate_everything(env, cwd)

    def boom(*args, **kwargs):
        raise OSError("platte weg")

    monkeypatch.setattr(om, "_external_rows_check", boom)
    i = env.collect()["integrity"]
    assert i["external_archive"] == {"error": "OSError: platte weg"}
    assert "error" not in i and "EXTERNAL_ARCHIVE_HASH" not in i["issues"] and "TRADE_OUTCOME_DUPLICATES" in i["issues"]


def test_unreadable_champion_history_degrades_only_history_based_sections(env):
    env.paths["history"].write_text(GARBAGE, encoding="utf-8")
    res = env.collect()
    assert res["lifecycle"]["champion"]["by_state"]["EXIT_PENDING"] == 0       # behandelt wie 'keine Historie'
    assert all("error" not in res[s] for s in SECTIONS)


def test_naive_now_is_treated_as_utc(env):
    t = option_trade("NAI", iso_day(0))
    env.capture(leg_symbols(t)[0], NOW - timedelta(hours=1), context="mark")
    env.history(active=[t])
    res = om.collect(NOW.replace(tzinfo=None), env.paths)
    assert "error" not in res["quote_coverage"]


# ═══════════════════════════════════════════════════════════════════════════
# 8. Lauf- und Datenstatus
# ═══════════════════════════════════════════════════════════════════════════
def test_run_status_picks_latest_report_not_after_today(env):
    reports = env.paths["daily_reports"]
    reports.mkdir()
    stats = {"universe": 372, "analyzed": 96, "final_mc": 21, "roi_ok": 3, "roi_rejects": {"n": 18}, "trades": 1,
             "stop_reason": "x", "session": "regular"}
    for day, s in ((iso_day(1), {"universe": 1}), (TODAY.isoformat(), stats), (iso_day(-1), {"universe": 999})):
        (reports / f"{day}.json").write_text(json.dumps({"stats": s}), encoding="utf-8")
    env.write_json("workflow_status", {"days": {TODAY.isoformat(): {
        "scanner": {"state": "OK", "conclusion": "success", "actual_last_run": "2026-10-14T13:40:00Z",
                    "delay_hours": 0.2}, "external_data": {"state": "PENDING"}}, iso_day(1): {"scanner": {}}}})
    run = env.collect()["run"]
    assert run["report_date"] == TODAY.isoformat() and run["report_is_today"] is True
    assert run["scanner"] == {"universe": 372, "analyzed": 96, "champion_candidates": 21, "contracts_designed": 21,
                              "trades": 1, "stop_reason": "x", "session": "regular"}
    assert run["workflows"] == {"scanner": {"state": "OK", "conclusion": "success",
                                            "last_run": "2026-10-14T13:40:00Z", "delay_hours": 0.2},
                                "external_data": {"state": "PENDING", "conclusion": None, "last_run": None,
                                                  "delay_hours": None}}


def test_run_status_falls_back_to_older_report_and_missing_counts_are_none(env):
    reports = env.paths["daily_reports"]
    reports.mkdir()
    (reports / f"{iso_day(2)}.json").write_text(json.dumps({"stats": {"roi_ok": 3}}), encoding="utf-8")
    run = env.collect()["run"]
    assert run["report_date"] == iso_day(2) and run["report_is_today"] is False
    assert run["scanner"]["contracts_designed"] is None and run["scanner"]["universe"] is None


def test_data_health_counts_archive_days_and_source_status(env):
    sym = oq.occ_symbol("AAA", "2027-01-15", "call", 100.0)
    sym2 = oq.occ_symbol("BBB", "2027-01-15", "call", 100.0)
    env.capture(sym, NOW - timedelta(hours=1), context="mark")
    env.capture(sym2, NOW - timedelta(hours=1), context="entry", quality="UNUSABLE")
    for h in (1, 2, 3, 4):
        env.capture(sym, NOW - timedelta(days=1, hours=h), context="mark")
    env.capture(sym, NOW - timedelta(days=13), context="mark")                     # 13 Tage: noch im Fenster
    env.capture(sym, NOW - timedelta(days=14), context="mark")                     # 14 Tage: außerhalb
    env.write_json("source_health", {"generated": "2026-10-14T10:00:00+00:00", "sources": {
        "a": {"status": "HEALTHY"}, "b": {"status": "DEGRADED"}, "c": {"status": "HEALTHY"}}})
    env.write_json("external_health", {"sources": {"x": {"status": "OK"}, "y": {"last_status": "FAILED"},
                                                   "z": {"status": "OK"}}})
    dh = env.collect()["data_health"]
    assert dh["source_health"] == {"generated": "2026-10-14T10:00:00+00:00",
                                   "by_status": {"HEALTHY": 2, "DEGRADED": 1}}
    assert dh["external_data"] == {"by_status": {"OK": 2, "FAILED": 1}}
    assert dh["quote_archive_days"] == {
        iso_day(0): {"n": 2, "by_context": {"mark": 1, "entry": 1}, "by_quality": {"OK": 1, "UNUSABLE": 1}},
        iso_day(1): {"n": 4, "by_context": {"mark": 4}, "by_quality": {"OK": 4}},
        iso_day(13): {"n": 1, "by_context": {"mark": 1}, "by_quality": {"OK": 1}}}
    assert dh["quote_archive_median_14d"] == 2.0 and dh["quote_archive_root_exists"] is True


def test_data_health_empty_archive_has_no_median(env):
    dh = env.collect()["data_health"]
    assert dh["quote_archive_days"] == {} and dh["quote_archive_median_14d"] is None


def test_quote_archive_root_exists_refers_to_the_configured_archive(env):
    assert oq.ROOT.exists()                                   # conftest: Standard-Archiv ist vorhanden
    assert not env.paths["quotes"].exists()
    assert env.collect()["data_health"]["quote_archive_root_exists"] is False
