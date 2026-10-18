"""Unabhängiger Test der Entry-/Exit-Zeitlogik (timing-v2) – Tester != Autor.

Geprüft wird gegen die SPEZIFIKATION (Nutzerentscheidungen A–I) und gegen OFFIZIELLE FAKTEN (NYSE-Kalender,
US-/EU-Zeitumstellung), nicht gegen das Zufallsverhalten des Codes:
  A  versionierte Policy (Altbestand = timing-v1, current() bei defekter Config = timing-v2, build_trade_record)
  B  zulässige Optionsquote (option_quotes.usable_rth_quote): Mark-Fenster je Instrument, DST, Feiertage, Qualität
  C  Entry (timing_policy.evaluate_entry): 15-min-Toleranz nur im Fenster, früheste Sitzung, Entry-Zeit, Ask-Preis
  D  Exit (evaluate_exit): 5-min-Toleranz, Bid-Preis, Verfall -> offizielles Settlement, nie Basiswert-/Modellpreis
  E  evaluate(): Statusmodell, EXIT_BEFORE_ENTRY, Point-in-Time (retrieved_at <= now)
  F  feedback.py: geschlossene Börsentage, close_v2/resolve_pending_exits, Lernen genau einmal, Altbestand v1
  G  candidate_ledger: neue Zeilen timing-v2 (nur Archiv-Fills), Altzeilen unverändert über die alten Fill-Funktionen
  H  scripts/rth_marks.trade_targets: Mindestabrufzeit je Symbol
  I  evidence_ledger.scorecard: Zellen je Timing-Policy getrennt
  +  Property-Test über zufällige Zeiten (kein rückwirkender Fill, keine Quote vor der jeweiligen Grenze)

Regeln: kein Netzwerk (fetch_option_quotes, unadjusted_close_on, yfinance, requests werden in JEDEM Test ersetzt bzw.
gesperrt), Dateien nur in tmp_path, option_quotes.ROOT zeigt auf tmp_path. Synthetische Quotes prüfen nur die Mechanik.
Die gefundenen Fehler (Settlement vor dem offiziellen Schluss, Quellen im Outcome-Record, Label-PIT) sind behoben; die Tests sichern sie ab.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import requests
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import feedback  # noqa: E402
import pipeline  # noqa: E402
import rth_marks  # noqa: E402

from modules import candidate_ledger as cl  # noqa: E402
from modules import evidence_ledger as el  # noqa: E402
from modules import market_snapshot as ms  # noqa: E402
from modules import option_quotes as oq  # noqa: E402
from modules import timing_policy as tp  # noqa: E402

UTC = timezone.utc
NY = ZoneInfo("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════════════════════════════
def utc(s: str) -> datetime:
    """'2026-10-14T13:30[:SS]' -> aware UTC."""
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def ny(s: str) -> datetime:
    """New-Yorker Wanduhrzeit '2026-10-14T09:45' -> aware UTC (DST über zoneinfo, unabhängig vom Code unter Test)."""
    return datetime.fromisoformat(s).replace(tzinfo=NY).astimezone(UTC)


def ts(t: datetime) -> str:
    return t.astimezone(UTC).isoformat(timespec="seconds")


def occ(root: str, expiry: str, cp: str, strike: float) -> str:
    """OCC-Symbol von Hand gebaut (unabhängig von option_quotes.occ_symbol)."""
    return f"{root}{expiry[2:4]}{expiry[5:7]}{expiry[8:10]}{cp}{int(round(strike * 1000)):08d}"


KINDS = {  # kind -> (Strategie, Call/Put, Long-Strike, Short-Strike)
    "call": ("LONG_CALL", "C", 200.0, None),
    "put": ("LONG_PUT", "P", 200.0, None),
    "bull": ("BULL_CALL_SPREAD", "C", 200.0, 210.0),
    "bear": ("BEAR_PUT_SPREAD", "P", 200.0, 190.0),
}
FAR = "2027-01-15"       # weit entfernter Verfall (Freitag) für Tests ohne Verfallsbezug


def syms(kind="call", expiry=FAR, root="AAPL", strikes=None) -> list[str]:
    _, cp, lk, sk = KINDS[kind]
    if strikes:
        lk, sk = strikes[0], (strikes[1] if sk else None)
    return [occ(root, expiry, cp, lk)] + ([occ(root, expiry, cp, sk)] if sk else [])


def make_trade(kind="call", *, decision, close=None, expiry=FAR, ticker="AAPL", policy="timing-v2",
               with_symbols=True, strikes=None, **extra) -> dict:
    """Champion-Trade-Datensatz wie pipeline.build_trade_record (+ close_time nach dem Exit-Entscheid)."""
    strat, cp, lk, sk = KINDS[kind]
    if strikes:
        lk, sk = strikes[0], (strikes[1] if sk else None)
    opt = {"strike": lk, "expiry": expiry, "bid": 1.9, "ask": 2.0, "last": 1.95}
    if with_symbols:
        opt["option_symbol"] = occ(ticker, expiry, cp, lk)
    if sk:
        opt["spread_leg"] = {"strike": sk, "expiry": expiry, "bid": 0.9, "ask": 1.0}
        if with_symbols:
            opt["spread_leg"]["symbol"] = occ(ticker, expiry, cp, sk)
        opt["net_debit"] = 1.1
    t = {"ticker": ticker, "entry_date": decision.astimezone(UTC).strftime("%Y-%m-%d"), "strategy": strat,
         "entry_debit": 1.1 if sk else 2.0, "option": opt, "outcome": None, "features": {},
         "decision_time": ts(decision)}
    if policy:
        t["timing_policy"] = policy
    if close is not None:
        t["close_time"] = ts(close)
        t["close_date"] = close.astimezone(UTC).strftime("%Y-%m-%d")
    t.update(extra)
    return t


def put(sym: str, at: datetime, bid=1.0, ask=1.2, *, quote_at="=", **extra) -> None:
    """Eine BEOBACHTETE Quote ins Archiv (oq.ROOT = tmp_path). at = Abrufzeit, quote_at = Börsenzeit ('=' gleich, None fehlt)."""
    raw = {"symbol": sym, "bid": bid, "ask": ask, **extra}
    qt = at if quote_at == "=" else quote_at
    if qt is not None:
        raw["quote_time"] = qt.isoformat()
    n = oq.record([raw], underlying=sym[:sym.index(next(c for c in sym if c.isdigit()))], source="test",
                  context=f"t{at.timestamp():.0f}", retrieved_at=at)
    assert n == 1, "Quote wurde nicht (neu) geschrieben – Testdaten kollidieren (gleiche record_id)"


def put_all(kind: str, at: datetime, long=(1.0, 1.2), short=(0.5, 0.7), expiry=FAR, **kw) -> None:
    """Quote(s) aller Legs derselben Erfassungsrunde."""
    s = syms(kind, expiry)
    put(s[0], at, *long, **kw)
    if len(s) > 1:
        put(s[1], at, *short, **kw)


def rec(sym="AAPL270115C00200000", at=None, bid=1.0, ask=1.2, *, quote_at="=", **extra) -> dict:
    """Normalisierter Archiv-Record (nicht archiviert) für usable_rth_quote."""
    raw = {"symbol": sym, "bid": bid, "ask": ask, **extra}
    qt = at if quote_at == "=" else quote_at
    if qt is not None:
        raw["quote_time"] = qt.isoformat()
    r = oq.normalize(raw, underlying="X", source="test", context="t", retrieved_at=at)
    assert r is not None
    return r


def ev(trade, now, close_fn=None):
    return tp.evaluate(trade, now, close_fn=close_fn)


def entry_of(trade, now):
    return tp.evaluate_entry(trade, now)


class Netz:
    """Ersatz für alle Netzwerkzugriffe; protokolliert Aufrufe (Test schlägt NICHT still auf echtes Netz durch)."""

    def __init__(self):
        self.fetch_calls: list[list[str]] = []
        self.close_calls: list[tuple] = []
        self.closes: dict[tuple, float] = {}
        self.on_fetch = None            # optional: callable(symbols) -> dict

    def fetch(self, symbols, context="candidate_mark"):
        self.fetch_calls.append(list(symbols))
        return self.on_fetch(symbols) if self.on_fetch else {}

    def close(self, ticker, day):
        d = day if isinstance(day, date) else date.fromisoformat(str(day)[:10])
        self.close_calls.append((ticker, d))
        return self.closes.get((ticker, d))


@pytest.fixture(autouse=True)
def net(monkeypatch, tmp_path):
    """Isolation: Quote-Archiv und Arbeitsverzeichnis in tmp_path, Netz gesperrt, Policy-Cache sauber."""
    monkeypatch.setattr(oq, "ROOT", tmp_path / "quotes")
    monkeypatch.chdir(tmp_path)
    n = Netz()
    monkeypatch.setattr(ms, "fetch_option_quotes", n.fetch)
    monkeypatch.setattr(ms, "unadjusted_close_on", n.close)

    def boom(*a, **k):
        raise AssertionError("Netzwerkzugriff im Test")
    monkeypatch.setattr(requests.sessions.Session, "request", boom)
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", boom)
    monkeypatch.setattr(yfinance, "download", boom)
    tp._load.cache_clear()
    yield n
    tp._load.cache_clear()


@pytest.fixture
def policy_cfg(tmp_path):
    """Setzt timing_policy.CONFIG auf eine Testdatei (None = Datei fehlt) und stellt den Cache danach wieder her."""
    orig = tp.CONFIG

    def setter(text: str | None):
        p = tmp_path / "timing_policy_test.yaml"
        if text is not None:
            p.write_text(text, encoding="utf-8")
        tp.CONFIG = p
        tp._load.cache_clear()
    yield setter
    tp.CONFIG = orig
    tp._load.cache_clear()


# ══════════════════════════════════════════════════════════════════════════════════════════════
# A. Versionierte Policy
# ══════════════════════════════════════════════════════════════════════════════════════════════
def test_A_datensatz_ohne_feld_ist_timing_v1():
    assert tp.policy_of({}) == "timing-v1"
    assert tp.policy_of(None) == "timing-v1"
    assert tp.policy_of({"timing_policy": None}) == "timing-v1"
    assert tp.policy_of({"ticker": "AAPL", "outcome": 0.5}) == "timing-v1"
    assert tp.policy_of({"timing_policy": "timing-v2"}) == "timing-v2"
    assert tp.policy_of({"timing_policy": "timing-v1"}) == "timing-v1"


def test_A_echte_config_current_ist_v2_und_werte_laut_spezifikation():
    cfg = yaml.safe_load((REPO / "config" / "timing_policy.yaml").read_text(encoding="utf-8"))
    assert cfg["current"] == "timing-v2"
    assert tp.current() == "timing-v2"
    assert tp.params() == {"decision_quote_max_age_minutes": 15, "decision_quote_slack_minutes": 5,
                           "fill_session_grace_hours": 12, "spread_leg_pairing_minutes": 15,
                           "settlement_retry_days": 5}
    v2 = cfg["policies"]["timing-v2"]
    assert v2["entry"]["long_price"] == "ask" and v2["exit"]["long_price"] == "bid"
    assert cfg["policies"]["timing-v1"]["status"] == "LEGACY"


@pytest.mark.parametrize("inhalt", [
    None,                                   # Datei fehlt
    "",                                     # leer
    "current: [unclosed\n  - x: : :\n",     # kein gültiges YAML
    "- nur\n- eine\n- liste\n",              # kein Mapping
    "current: timing-v9\npolicies: {}\n",   # unbekannte Policy
    "current: null\n",                      # Feld leer
    "policies:\n  timing-v2: {}\n",         # current fehlt
])
def test_A_current_bei_fehlender_oder_defekter_config_ist_v2(policy_cfg, inhalt):
    policy_cfg(inhalt)
    assert tp.current() == "timing-v2"
    # Parameter fallen auf die Standardwerte der Spezifikation zurück
    assert tp.params()["decision_quote_max_age_minutes"] == 15
    assert tp.params()["fill_session_grace_hours"] == 12


def test_A_config_werte_ueberschreiben_nur_gueltige_zahlen(policy_cfg):
    policy_cfg("current: timing-v2\npolicies:\n  timing-v2:\n    entry: {decision_quote_max_age_minutes: 7}\n"
               "    settlement_retry_days: true\n    fill_session_grace_hours: abc\n")
    p = tp.params()
    assert p["decision_quote_max_age_minutes"] == 7
    assert p["settlement_retry_days"] == 5          # bool ist keine Zahl
    assert p["fill_session_grace_hours"] == 12      # Text ist keine Zahl


def _pipeline_p(strategy="LONG_CALL", **opt_kw):
    opt = {"strike": 100.0, "expiry": "2026-11-20", "bid": 4.9, "ask": 5.0, "option_symbol": "AAPL261120C00100000"}
    opt.update(opt_kw)
    return {"ticker": "AAPL", "strategy": strategy, "option": opt, "features": {}}


def test_A_build_trade_record_traegt_v2_decision_time_utc_und_laesst_entry_debit_unveraendert():
    vor = datetime.now(UTC).replace(microsecond=0)
    r = pipeline.build_trade_record(_pipeline_p(), "2026-10-14")
    nach = datetime.now(UTC)
    assert r["timing_policy"] == "timing-v2"
    d = datetime.fromisoformat(r["decision_time"])
    assert d.utcoffset() == timedelta(0), "decision_time muss ein UTC-Zeitpunkt sein"
    assert vor <= d <= nach + timedelta(seconds=1)
    assert r["entry_debit"] == 5.0 and r["entry_date"] == "2026-10-14"        # Champion-Entscheidung unverändert
    assert r["entry_quote"]["assumed_fill"] == "ask"
    assert tp.policy_of(r) == "timing-v2"


def test_A_build_trade_record_spread_entry_debit_unveraendert():
    sl = {"strike": 110.0, "bid": 2.0, "ask": 2.2}
    r = pipeline.build_trade_record(_pipeline_p("BULL_CALL_SPREAD", net_debit=3.1, spread_leg=sl), "2026-10-14")
    assert r["entry_debit"] == 3.1 and r["timing_policy"] == "timing-v2"
    r2 = pipeline.build_trade_record(_pipeline_p("BULL_CALL_SPREAD", spread_leg=sl), "2026-10-14")
    assert r2["entry_debit"] == round(5.0 - 2.0, 2)


def test_A_build_trade_record_bei_defekter_config_v2(policy_cfg):
    policy_cfg("current: [kaputt")
    assert pipeline.build_trade_record(_pipeline_p(), "2026-10-14")["timing_policy"] == "timing-v2"


def test_A_altbestand_wird_nie_umgeschrieben_oder_bewertet():
    """Ein Datensatz ohne Feld bleibt ohne Feld; die Policy-Funktionen mutieren ihre Eingabe nicht."""
    t = make_trade(decision=ny("2026-10-13T11:00"), close=ny("2026-10-14T11:00"), policy=None)
    vorher = json.dumps(t, sort_keys=True)
    assert "timing_policy" not in t and tp.policy_of(t) == "timing-v1"
    ev(t, ny("2026-10-15T12:00"))
    assert json.dumps(t, sort_keys=True) == vorher


# ══════════════════════════════════════════════════════════════════════════════════════════════
# B. Zulässige Optionsquote und Mark-Fenster (offizielle Fakten, UTC-Zeiten von Hand gerechnet)
# ══════════════════════════════════════════════════════════════════════════════════════════════
# (Handelstag, Wurzel, Fensterbeginn UTC, Fensterende UTC, Begründung)
FENSTER = [
    ("2026-03-06", "AAPL", "15:00:00", "20:50:00", "EST (UTC-5) vor der US-Umstellung 2026-03-08"),
    ("2026-03-09", "AAPL", "14:00:00", "19:50:00", "EDT: erster Handelstag nach der US-Umstellung"),
    ("2026-03-27", "AAPL", "14:00:00", "19:50:00", "US EDT, Europa noch CET (Woche zwischen den Umstellungen)"),
    ("2026-03-30", "AAPL", "14:00:00", "19:50:00", "nach der EU-Umstellung 2026-03-29: für die USA unverändert"),
    ("2026-10-23", "AAPL", "14:00:00", "19:50:00", "EDT, Europa noch Sommerzeit"),
    ("2026-10-26", "AAPL", "14:00:00", "19:50:00", "nach EU-Rückstellung 2026-10-25, US noch EDT bis 11-01"),
    ("2026-10-30", "AAPL", "14:00:00", "19:50:00", "letzter Freitag vor der US-Rückstellung"),
    ("2026-11-02", "AAPL", "15:00:00", "20:50:00", "EST: erster Handelstag nach der US-Rückstellung 2026-11-01"),
    ("2026-12-02", "AAPL", "15:00:00", "20:50:00", "Winter"),
    ("2025-11-28", "AAPL", "15:00:00", "17:50:00", "Tag nach Thanksgiving: Schluss 13:00 ET"),
    ("2025-12-24", "AAPL", "15:00:00", "17:50:00", "Heiligabend 2025: Schluss 13:00 ET"),
    ("2026-12-24", "AAPL", "15:00:00", "17:50:00", "Heiligabend 2026 (Donnerstag): Schluss 13:00 ET"),
    ("2025-07-03", "AAPL", "14:00:00", "16:50:00", "3. Juli 2025 (EDT): Schluss 13:00 ET"),
    # instrumentenspezifisch: SPY/SPX/SPXW schließen 15 min später (16:15 bzw. 13:15 ET)
    ("2026-10-14", "SPY", "14:00:00", "20:05:00", "SPY EDT: 16:15 - 10 min = 16:05 ET"),
    ("2026-12-02", "SPXW", "15:00:00", "21:05:00", "SPXW EST: 16:05 ET = 21:05 UTC"),
    ("2026-10-14", "SPX", "14:00:00", "20:05:00", "SPX EDT"),
    ("2025-11-28", "SPY", "15:00:00", "18:05:00", "SPY am verkürzten Tag: 13:15 - 10 min = 13:05 ET"),
    ("2025-07-03", "SPXW", "14:00:00", "17:05:00", "SPXW am 3. Juli 2025"),
    ("2026-03-30", "SPY", "14:00:00", "20:05:00", "SPY nach EU-Umstellung"),
]


@pytest.mark.parametrize("tag,root,beginn,ende,warum", FENSTER)
def test_B_fenstergrenzen_je_instrument_und_dst(tag, root, beginn, ende, warum):
    sym = occ(root, "2027-06-18", "C", 100.0)
    b, e = utc(f"{tag}T{beginn}"), utc(f"{tag}T{ende}")
    s = timedelta(seconds=1)
    assert not oq.usable_rth_quote(rec(sym, b - s)), f"{warum}: 1 s vor Fensterbeginn"
    assert oq.usable_rth_quote(rec(sym, b)), f"{warum}: Fensterbeginn inklusive"
    assert oq.usable_rth_quote(rec(sym, e)), f"{warum}: Fensterende inklusive"
    assert not oq.usable_rth_quote(rec(sym, e + s)), f"{warum}: 1 s nach Fensterende"
    assert oq.usable_rth_quote(rec(sym, b + (e - b) / 2))
    assert oq.mark_window(b, sym) == "IN_WINDOW" and oq.mark_window(e + s, sym) == "AFTER_WINDOW"
    assert oq.mark_window(b - s, sym) == "BEFORE_WINDOW"
    # timing_policy rechnet mit denselben Grenzen
    w = tp._window(date.fromisoformat(tag), sym)
    assert w == (b, e)


@pytest.mark.parametrize("tag", [
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25", "2024-03-29", "2026-10-17", "2026-10-18",
    "2027-01-18", "2027-03-26", "2027-06-18", "2027-07-05", "2027-11-25", "2027-12-24", "2028-04-14",
])
def test_B_feiertage_sonderschliessung_und_wochenende_keine_zulaessige_quote(tag):
    """Offizielle NYSE-Schließtage: Neujahr, MLK, Presidents, Good Friday, Memorial, Juneteenth, 4. Juli (2026 am
    Freitag 3. Juli beobachtet), Labor, Thanksgiving, Weihnachten; 2025-01-09 Staatstrauer Carter; Wochenende."""
    d = date.fromisoformat(tag)
    for stunde in ("15:00:00", "17:00:00", "19:00:00"):
        r = rec(at=utc(f"{tag}T{stunde}"))
        assert not oq.usable_rth_quote(r), f"{tag} {stunde}"
        assert oq.mark_window(utc(f"{tag}T{stunde}")) == "MARKET_CLOSED"
    assert tp._window(d) is None
    assert not ms.is_trading_day(d)


@pytest.mark.parametrize("tag", ["2025-01-08", "2025-01-10", "2025-11-28", "2025-12-26", "2026-04-02", "2026-04-06",
                                 "2026-06-18", "2026-07-02", "2026-11-27", "2026-12-24", "2027-12-31", "2027-06-17",
                                 "2027-07-02", "2028-07-03"])
def test_B_nachbartage_der_feiertage_sind_handelstage(tag):
    assert ms.is_trading_day(date.fromisoformat(tag)) and tp._window(date.fromisoformat(tag)) is not None


def test_B_verkuerzte_sitzungen_nur_an_den_offiziellen_tagen():
    """Early Closes 13:00 ET: 3. Juli 2025, Tag nach Thanksgiving, Heiligabend – 2026-07-03 ist Feiertag (kein
    Early Close), 2026-07-02 und 2027-12-24 (beobachteter Feiertag) sind keine Early Closes."""
    def schluss(tag):
        return ms.market_close_time(date.fromisoformat(tag)).hour
    assert [schluss(x) for x in ("2025-07-03", "2025-11-28", "2025-12-24", "2026-11-27", "2026-12-24")] == [13] * 5
    assert schluss("2026-07-02") == 16 and schluss("2025-12-23") == 16 and schluss("2025-11-26") == 16
    assert not ms.is_trading_day(date(2027, 12, 24))


def test_B_late_close_wurzeln_aus_der_config_und_aktienoptionen_nicht():
    cfg = yaml.safe_load((REPO / "config" / "market_calendar.yaml").read_text(encoding="utf-8"))
    roots = set(cfg["instrument_sessions"]["late_close_roots"])
    assert {"SPY", "SPX", "SPXW"} <= roots and cfg["instrument_sessions"]["late_close_minutes"] == 15
    for aktie in ("AAPL", "MSFT", "TSLA", "NVDA"):
        assert aktie not in roots
        # 16:05 ET: für Aktienoptionen schon nach dem Fensterende, für SPY/SPX/SPXW noch drin
        t = ny("2026-10-14T16:05:00")
        assert not oq.usable_rth_quote(rec(occ(aktie, "2027-06-18", "C", 100.0), t))
    for wurzel in ("SPY", "SPX", "SPXW"):
        assert oq.usable_rth_quote(rec(occ(wurzel, "2027-06-18", "C", 100.0), ny("2026-10-14T16:05:00")))
        assert not oq.usable_rth_quote(rec(occ(wurzel, "2027-06-18", "C", 100.0), ny("2026-10-14T16:05:01")))


def test_B_qualitaetskriterien():
    t = ny("2026-10-14T11:00")
    ok = rec(at=t, bid=1.0, ask=1.2)
    assert oq.usable_rth_quote(ok)
    # UNUSABLE: gekreuzt, Ask fehlt/0, negativer Preis, nicht endlich
    assert not oq.usable_rth_quote(rec(at=t, bid=2.0, ask=1.0))
    assert not oq.usable_rth_quote(rec(at=t, bid=1.0, ask=None))
    assert not oq.usable_rth_quote(rec(at=t, bid=1.0, ask=0))
    assert not oq.usable_rth_quote(rec(at=t, bid=-1.0, ask=1.0))
    assert not oq.usable_rth_quote(rec(at=t, bid=float("nan"), ask=1.0))
    # Bid fehlt oder 0 -> nicht zulässig (Exit zum Bid müsste realisierbar sein)
    assert not oq.usable_rth_quote(rec(at=t, bid=0, ask=1.0))
    assert not oq.usable_rth_quote(rec(at=t, bid=None, ask=1.0))
    # SUSPECT (weiter Spread, locked) ist NICHT ausgeschlossen: nur UNUSABLE/STALE/Bid-Ask<=0 sind es
    assert oq.usable_rth_quote(rec(at=t, bid=0.1, ask=2.0))
    assert oq.usable_rth_quote(rec(at=t, bid=1.0, ask=1.0))
    # STALE: Börsenzeit mehr als 26 h vor dem Abruf
    stale = rec(at=t, quote_at=t - timedelta(hours=27))
    assert "STALE" in stale["quality_flags"] and not oq.usable_rth_quote(stale)
    # leer / ohne Zeit
    assert not oq.usable_rth_quote(None) and not oq.usable_rth_quote({})
    kaputt = dict(ok)
    kaputt["retrieved_at"] = None
    assert not oq.usable_rth_quote(kaputt)


def test_B_boersenzeit_der_quote_muss_in_der_regulaeren_sitzung_liegen():
    t = ny("2026-10-14T10:30")
    assert not oq.usable_rth_quote(rec(at=t, quote_at=ny("2026-10-14T08:00"))), "Pre-Market-Quote"
    assert not oq.usable_rth_quote(rec(at=t, quote_at=ny("2026-10-13T16:30"))), "nachbörsliche Quote vom Vorabend"
    assert not oq.usable_rth_quote(rec(at=t, quote_at=ny("2026-10-14T16:30"))), "Börsenzeit nach dem Abruf, POST"
    assert oq.usable_rth_quote(rec(at=t, quote_at=ny("2026-10-14T10:20")))
    # Abruf außerhalb des Fensters, Börsenzeit im Fenster -> trotzdem nicht zulässig (Abruf zählt)
    assert not oq.usable_rth_quote(rec(at=ny("2026-10-14T16:30"), quote_at=ny("2026-10-14T15:45")))
    assert not oq.usable_rth_quote(rec(at=ny("2026-10-15T08:00"), quote_at=ny("2026-10-14T15:45")))


def test_B_rth_quote_history_ist_point_in_time_aufsteigend_und_nur_zulaessige():
    s = syms("call")[0]
    put(s, ny("2026-10-14T11:30"), 1.0, 1.2)
    put(s, ny("2026-10-14T10:30"), 0.9, 1.1)
    put(s, ny("2026-10-14T11:00"), 2.0, 1.0)                         # gekreuzt -> nie zulässig
    put(s, ny("2026-10-14T16:30"), 1.0, 1.2)                         # nachbörslich abgerufen
    h = oq.rth_quote_history([s], ny("2026-10-14T11:29:59"))
    assert [r["retrieved_at"][:16] for _, r in h[s]] == ["2026-10-14T14:30"]       # 11:30 noch nicht abgerufen
    h = oq.rth_quote_history([s], ny("2026-10-14T11:30:00"))
    assert [r["retrieved_at"][:16] for _, r in h[s]] == ["2026-10-14T14:30", "2026-10-14T15:30"]
    assert all(d == "2026-10-14" for d, _ in h[s])
    assert oq.rth_quote_history([s], ny("2026-10-14T10:29:59")) == {}


def test_B_rth_quote_history_quote_day_ist_der_tag_der_boersenzeit():
    s = syms("call")[0]
    put(s, ny("2026-10-15T10:05"), 1.0, 1.2, quote_at=ny("2026-10-14T15:30"))      # seit gestern unveränderte Quote
    h = oq.rth_quote_history([s], ny("2026-10-15T12:00"))
    assert [d for d, _ in h[s]] == ["2026-10-14"]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# C. Entry
# ══════════════════════════════════════════════════════════════════════════════════════════════
CALL = syms("call")[0]
D0 = ny("2026-10-13T11:00")           # Dienstag, EDT, im Fenster (10:00–15:50 ET)


def entry(kind="call", *, decision, now, expiry=FAR, **kw):
    return tp.evaluate_entry(make_trade(kind, decision=decision, expiry=expiry, **kw), now)


@pytest.mark.parametrize("sekunden_vorher,erlaubt", [(0, True), (60, True), (14 * 60 + 59, True), (15 * 60, True),
                                                      (15 * 60 + 1, False), (16 * 60, False), (3600, False)])
def test_C_toleranz_15_minuten_nur_wenn_entscheidung_im_fenster(sekunden_vorher, erlaubt):
    put(CALL, D0 - timedelta(seconds=sekunden_vorher), 1.0, 1.4)
    blk, recs = entry(decision=D0, now=D0 + timedelta(minutes=1))
    if erlaubt:
        assert blk["status"] == "FILLED" and len(recs) == 1
        assert blk["price"] == 1.4
        assert blk["time"] == ts(D0), "Entry-Zeit = max(Entscheidung, Abruf) – nie vor der Entscheidung"
    else:
        assert blk["status"] == "PENDING" and recs == [] and "price" not in blk


def test_C_entry_zeit_ist_max_aus_entscheidung_und_abruf():
    put(CALL, D0 + timedelta(minutes=20), 1.0, 1.4)
    blk, _ = entry(decision=D0, now=D0 + timedelta(hours=1))
    assert blk["status"] == "FILLED" and blk["time"] == ts(D0 + timedelta(minutes=20))
    # früheste zulässige Quote gewinnt: eine frische Quote 5 min VOR der Entscheidung (Toleranz) kommt zuerst
    put(CALL, D0 - timedelta(minutes=5), 1.1, 1.3)
    blk, _ = entry(decision=D0, now=D0 + timedelta(hours=1))
    assert blk["time"] == ts(D0) and blk["price"] == 1.3


def test_C_entscheidung_vor_dem_fenster_am_selben_tag():
    dec = ny("2026-10-13T09:00")
    put(CALL, ny("2026-10-13T08:50"), 1.0, 1.1)           # Pre-Market: nie zulässig
    put(CALL, ny("2026-10-13T09:59:59"), 1.0, 1.2)        # 1 s vor Fensterbeginn
    blk, _ = entry(decision=dec, now=ny("2026-10-13T10:00"))
    assert blk["status"] == "PENDING" and blk["day"] == "2026-10-13"
    put(CALL, ny("2026-10-13T10:00:00"), 1.0, 1.3)
    blk, _ = entry(decision=dec, now=ny("2026-10-13T10:00"))
    assert blk["status"] == "FILLED" and blk["price"] == 1.3 and blk["time"] == ts(ny("2026-10-13T10:00"))


def test_C_entscheidung_nach_fensterende_nur_naechste_sitzung_ohne_toleranz():
    dec = ny("2026-10-13T15:50:01")                         # 1 s nach Fensterende
    put(CALL, ny("2026-10-13T15:49:00"), 1.0, 1.1)          # nur 1 min vor der Entscheidung, aber Entscheidung außerhalb
    put(CALL, ny("2026-10-13T15:50:00"), 1.0, 1.15)
    blk, recs = entry(decision=dec, now=ny("2026-10-13T20:00"))
    assert blk["status"] == "PENDING" and recs == [] and blk["day"] == "2026-10-14"
    put(CALL, ny("2026-10-14T09:59:59"), 1.0, 1.2)
    put(CALL, ny("2026-10-14T10:00:00"), 1.0, 1.25)
    blk, _ = entry(decision=dec, now=ny("2026-10-14T10:30"))
    assert blk["status"] == "FILLED" and blk["price"] == 1.25 and blk["day"] == "2026-10-14"
    assert blk["time"] == ts(ny("2026-10-14T10:00"))


def test_C_entscheidung_exakt_am_fensterende_zaehlt_noch_als_im_fenster():
    dec = ny("2026-10-13T15:50:00")
    put(CALL, ny("2026-10-13T15:35:00"), 1.0, 1.1)
    blk, _ = entry(decision=dec, now=ny("2026-10-13T15:51"))
    assert blk["status"] == "FILLED" and blk["day"] == "2026-10-13" and blk["price"] == 1.1
    assert blk["time"] == ts(dec)


# (Entscheidung New Yorker Zeit, frühester Fill-Tag) – Freitagabend, Wochenende, Feiertage, Early Closes, UTC-Mitternacht
FILL_TAGE = [
    ("2026-10-16T20:00", "2026-10-19"),     # Freitagabend -> Montag
    ("2026-10-16T15:51", "2026-10-19"),     # Freitag direkt nach Fensterende -> Montag
    ("2026-10-17T12:00", "2026-10-19"),     # Samstag
    ("2026-10-18T22:00", "2026-10-19"),     # Sonntagabend (UTC schon Montag)
    ("2026-10-13T19:59", "2026-10-14"),     # kurz vor UTC-Mitternacht (23:59 UTC)
    ("2026-10-13T20:00", "2026-10-14"),     # genau UTC-Mitternacht des Folgetags
    ("2026-10-13T20:01", "2026-10-14"),     # nach UTC-Mitternacht
    ("2026-10-13T23:59", "2026-10-14"),     # NY-Mitternacht
    ("2026-10-14T00:00", "2026-10-14"),     # Beginn des Folgetags in NY, vor der Eröffnung
    ("2025-11-26T20:00", "2025-11-28"),     # vor Thanksgiving: Donnerstag (geschlossen) übersprungen
    ("2025-11-27T12:00", "2025-11-28"),     # Thanksgiving selbst
    ("2025-11-28T12:55", "2025-12-01"),     # Early Close: Fensterende 12:50 ET, danach Montag
    ("2025-11-28T12:45", "2025-11-28"),     # Early Close: noch im Fenster
    ("2025-12-24T12:55", "2025-12-26"),     # Heiligabend (Early Close) -> 25.12. geschlossen -> 26.12.
    ("2025-12-24T12:45", "2025-12-24"),
    ("2026-04-02T17:00", "2026-04-06"),     # Donnerstag vor Good Friday -> Montag
    ("2026-04-03T10:00", "2026-04-06"),     # Good Friday selbst
    ("2025-06-18T17:00", "2025-06-20"),     # Juneteenth 2025 (Donnerstag) übersprungen
    ("2026-06-18T17:00", "2026-06-22"),     # Juneteenth 2026 (Freitag) -> Montag
    ("2025-01-08T17:00", "2025-01-10"),     # 2025-01-09 Staatstrauer Carter übersprungen
    ("2025-07-03T12:55", "2025-07-07"),     # 3. Juli 2025 Early Close, 4. Juli Feiertag (Freitag) -> Montag
    ("2026-07-02T17:00", "2026-07-06"),     # 2026-07-03 beobachteter Feiertag -> Montag
    ("2026-03-08T16:00", "2026-03-09"),     # US-Umstellungstag (Sonntag) -> Montag
    ("2026-03-06T16:00", "2026-03-09"),     # Freitag vor der US-Umstellung
    ("2026-10-30T17:00", "2026-11-02"),     # Freitag vor der US-Rückstellung
    ("2026-11-01T12:00", "2026-11-02"),     # US-Rückstellungstag (25 h)
    ("2026-03-28T12:00", "2026-03-30"),     # Samstag vor der EU-Umstellung
    ("2026-10-24T12:00", "2026-10-26"),     # Samstag vor der EU-Rückstellung
]


@pytest.mark.parametrize("entscheidung,fill_tag", FILL_TAGE)
def test_C_fill_ausschliesslich_in_der_fruehesten_moeglichen_sitzung(entscheidung, fill_tag):
    dec = ny(entscheidung)
    d = date.fromisoformat(fill_tag)
    assert tp.fill_day(dec, CALL) == d
    # Quote am Folgehandelstag des Fill-Tags wäre zulässig, darf aber nie genutzt werden, wenn der Fill-Tag leer bleibt
    nxt = ms.next_trading_day(d)
    w_nxt = tp._window(nxt)
    put(CALL, w_nxt[0] + timedelta(hours=1), 1.0, 1.9)
    blk, recs = entry(decision=dec, now=w_nxt[1])
    assert blk["status"] in ("PENDING", "UNUSABLE") and recs == [], "Fill nie in einer späteren Sitzung als der frühesten"
    assert blk["day"] == fill_tag and "price" not in blk
    # Quote im Fenster des Fill-Tages -> Fill dort, Zeit >= Entscheidung
    w = tp._window(d)
    q_at = max(w[0], dec) + timedelta(minutes=2)
    put(CALL, q_at, 1.0, 1.5)
    blk, recs = entry(decision=dec, now=w_nxt[1])
    assert blk["status"] == "FILLED" and blk["day"] == fill_tag and blk["price"] == 1.5
    assert _t(blk["time"]) >= dec and blk["time"] == ts(q_at)


def _t(s) -> datetime:
    return datetime.fromisoformat(s)


def test_C_dst_sprung_us_umstellung_fenster_gilt_in_edt_nicht_in_est():
    """Montag 2026-03-09 (EDT): 14:05 UTC = 10:05 ET ist zulässig. Wer noch EST (UTC-5) annähme, sähe 09:05 ET."""
    dec = utc("2026-03-08T20:00")                                       # Sonntagabend
    put(CALL, utc("2026-03-09T13:55"), 1.0, 1.1)                        # 09:55 EDT -> vor dem Fenster
    blk, _ = entry(decision=dec, now=utc("2026-03-09T14:04"))
    assert blk["status"] == "PENDING"
    put(CALL, utc("2026-03-09T14:05"), 1.0, 1.2)
    blk, _ = entry(decision=dec, now=utc("2026-03-09T14:06"))
    assert blk["status"] == "FILLED" and blk["time"] == "2026-03-09T14:05:00+00:00"


def test_C_dst_sprung_us_rueckstellung_fenster_gilt_in_est_nicht_in_edt():
    """Montag 2026-11-02 (EST): Fenster beginnt 15:00 UTC; 14:30 UTC (= 09:30 EST) ist zu früh."""
    dec = utc("2026-10-30T21:00")                                       # Freitag 17:00 EDT
    put(CALL, utc("2026-11-02T14:30"), 1.0, 1.1)
    put(CALL, utc("2026-11-02T14:59"), 1.0, 1.15)
    blk, _ = entry(decision=dec, now=utc("2026-11-02T15:00"))
    assert blk["status"] == "PENDING"
    put(CALL, utc("2026-11-02T15:05"), 1.0, 1.2)
    blk, _ = entry(decision=dec, now=utc("2026-11-02T15:06"))
    assert blk["status"] == "FILLED" and blk["time"] == "2026-11-02T15:05:00+00:00" and blk["price"] == 1.2


@pytest.mark.parametrize("quelle", ["2026-03-30T16:00:00+02:00", "2026-03-30T14:00:00Z", "2026-03-30T14:00:00",
                                    "2026-03-30T10:00:00-04:00"])
def test_C_decision_time_formate_mit_offset_sind_derselbe_zeitpunkt(quelle):
    """decision_time in MESZ (+02:00), UTC, naiv (= UTC) oder EDT bezeichnet denselben Zeitpunkt 14:00 UTC."""
    t = make_trade(decision=utc("2026-03-30T14:00"))
    t["decision_time"] = quelle
    put(CALL, utc("2026-03-30T14:10"), 1.0, 1.3)
    blk, _ = tp.evaluate_entry(t, utc("2026-03-30T14:30"))
    assert blk["status"] == "FILLED" and blk["decision_time"] == "2026-03-30T14:00:00+00:00"
    assert blk["time"] == "2026-03-30T14:10:00+00:00"


def test_C_point_in_time_spaeter_abgerufene_quote_ist_erst_ab_retrieved_at_sichtbar():
    put(CALL, ny("2026-10-13T11:30"), 1.0, 1.4)
    t = make_trade(decision=D0)
    for now, status in ((ny("2026-10-13T11:29:59"), "PENDING"), (ny("2026-10-13T11:30:00"), "FILLED"),
                        (ny("2026-10-13T12:00"), "FILLED")):
        blk, _ = tp.evaluate_entry(t, now)
        assert blk["status"] == status, now
    # und nach dem Verstreichen gilt weiter die FRÜHESTE Quote (eine noch spätere ändert nichts)
    put(CALL, ny("2026-10-13T12:30"), 0.5, 3.0)
    blk, _ = tp.evaluate_entry(t, ny("2026-10-13T13:00"))
    assert blk["price"] == 1.4 and blk["time"] == ts(ny("2026-10-13T11:30"))


def test_C_pending_dann_unusable_nach_fensterende_plus_12_stunden():
    t = make_trade(decision=D0)
    ende = ny("2026-10-13T15:50")
    assert tp.evaluate_entry(t, ende + timedelta(hours=11, minutes=59))[0]["status"] == "PENDING"
    blk, recs = tp.evaluate_entry(t, ende + timedelta(hours=12, minutes=1))
    assert blk["status"] == "UNUSABLE" and blk["reason"] == "NO_ADMISSIBLE_QUOTE_IN_SESSION" and recs == []
    # Montag-Fenster nach Freitagabend-Entscheidung: gleiche Regel relativ zum Montag-Fensterende
    t2 = make_trade(decision=ny("2026-10-16T20:00"))
    ende2 = ny("2026-10-19T15:50")
    assert tp.evaluate_entry(t2, ende2 + timedelta(hours=11, minutes=59))[0]["status"] == "PENDING"
    assert tp.evaluate_entry(t2, ende2 + timedelta(hours=12, minutes=1))[0]["status"] == "UNUSABLE"


def test_C_pending_gruende_vor_und_im_fenster():
    t = make_trade(decision=ny("2026-10-16T20:00"))
    assert tp.evaluate_entry(t, ny("2026-10-17T12:00"))[0]["reason"] == "AWAITING_SESSION"
    assert tp.evaluate_entry(t, ny("2026-10-19T10:30"))[0]["reason"] == "AWAITING_ADMISSIBLE_QUOTE"


def test_C_unzulaessige_quotes_werden_uebersprungen_dann_greift_die_naechste_zulaessige():
    bad = [dict(bid=2.0, ask=1.0),                                  # gekreuzt
           dict(bid=1.0, ask=0),                                    # Ask 0
           dict(bid=0, ask=1.0),                                    # Bid 0
           dict(bid=None, ask=1.0),                                 # Bid fehlt
           dict(bid=1.0, ask=1.2, quote_at=ny("2026-10-11T15:00")),                  # STALE (> 26 h)
           dict(bid=1.0, ask=1.2, quote_at=ny("2026-10-12T16:30"))]                  # Börsenzeit nachbörslich
    for i, kw in enumerate(bad):
        put(CALL, D0 + timedelta(minutes=i), **kw)
    put(CALL, D0 + timedelta(minutes=30), 1.0, 1.9)                 # erst diese ist zulässig
    blk, _ = entry(decision=D0, now=D0 + timedelta(hours=1))
    assert blk["status"] == "FILLED" and blk["price"] == 1.9 and blk["time"] == ts(D0 + timedelta(minutes=30))


def test_C_nur_unzulaessige_quotes_nie_ein_ersatzpreis():
    put(CALL, D0 + timedelta(minutes=1), 2.0, 1.0)                  # gekreuzt
    put(CALL, ny("2026-10-13T16:30"), 1.0, 1.2)                     # nachbörslich abgerufen
    t = make_trade(decision=D0, entry_debit=2.0)
    blk, recs = tp.evaluate_entry(t, ny("2026-10-14T05:00"))
    assert blk["status"] == "UNUSABLE" and recs == [] and "price" not in blk, "kein Rückgriff auf option.ask/entry_debit"


def test_C_preis_ist_der_ask_nicht_bid_mid_oder_last():
    put(CALL, D0 + timedelta(minutes=1), 1.0, 1.4, last=1.25)
    blk, _ = entry(decision=D0, now=D0 + timedelta(minutes=5), entry_debit=9.99)
    assert blk["price"] == 1.4


@pytest.mark.parametrize("kind", ["bull", "bear"])
def test_C_spread_preis_long_ask_minus_short_bid(kind):
    put_all(kind, D0 + timedelta(minutes=3), long=(3.0, 3.4), short=(1.1, 1.3))
    blk, recs = entry(kind, decision=D0, now=D0 + timedelta(minutes=10))
    assert blk["status"] == "FILLED" and len(recs) == 2
    assert blk["price"] == pytest.approx(3.4 - 1.1)                  # nicht Mid−Mid (2.1) und nicht Ask−Ask (2.1)
    assert [r["occ_symbol"] for r in recs] == syms(kind), "Records in Leg-Reihenfolge Long, Short"


def test_C_spread_legs_muessen_aus_derselben_erfassungsrunde_stammen():
    long_s, short_s = syms("bull")
    put(long_s, D0 + timedelta(minutes=1), 3.0, 3.4)
    put(short_s, D0 + timedelta(minutes=17), 1.1, 1.3)               # 16 min Abstand
    now = D0 + timedelta(hours=1)
    assert entry("bull", decision=D0, now=now)[0]["status"] == "PENDING"
    put(short_s, D0 + timedelta(minutes=16), 1.0, 1.2)               # exakt 15 min Abstand -> gepaart
    blk, _ = entry("bull", decision=D0, now=now)
    assert blk["status"] == "FILLED" and blk["price"] == pytest.approx(3.4 - 1.0)
    assert blk["time"] == ts(D0 + timedelta(minutes=16)), "Entry-Zeit = späteste Leg-Abrufzeit"


def test_C_spread_ein_leg_fehlt_oder_nur_in_spaeterer_sitzung():
    long_s, short_s = syms("bull")
    put(long_s, D0 + timedelta(minutes=1), 3.0, 3.4)
    assert entry("bull", decision=D0, now=ny("2026-10-13T15:55"))[0]["status"] == "PENDING"
    put(short_s, ny("2026-10-14T10:30"), 1.0, 1.2)                   # Short-Leg erst am Folgetag
    blk, _ = entry("bull", decision=D0, now=ny("2026-10-14T11:00"))
    assert blk["status"] in ("PENDING", "UNUSABLE") and blk["status"] != "FILLED"
    assert entry("bull", decision=D0, now=ny("2026-10-14T05:00"))[0]["status"] == "UNUSABLE"


def test_C_spread_mit_nicht_positivem_preis_ist_unusable_ohne_ausnahme():
    put_all("bull", D0 + timedelta(minutes=1), long=(0.9, 1.0), short=(1.1, 1.2))      # Ask 1.0 − Bid 1.1 < 0
    blk, recs = entry("bull", decision=D0, now=D0 + timedelta(minutes=5))
    assert blk["status"] == "UNUSABLE" and recs == [] and "price" not in blk


def test_C_entry_nach_verfall_ist_unusable():
    exp = "2026-10-16"                                                # Freitag
    s = syms("call", exp)[0]
    dec = ny("2026-10-16T16:30")                                     # Verfallstag nach Schluss
    t = make_trade("call", decision=dec, expiry=exp)
    assert tp.fill_day(dec, s) == date(2026, 10, 19)
    # Phantom-Quote nach Verfall im Archiv darf keinen Entry ergeben
    put(s, ny("2026-10-19T10:30"), 1.0, 1.2)
    blk, recs = tp.evaluate_entry(t, ny("2026-10-19T11:00"))
    assert blk["status"] == "UNUSABLE" and recs == [] and "price" not in blk
    # ohne Quote: nie FILLED, nach Karenz UNUSABLE
    t2 = make_trade("call", decision=ny("2026-10-16T16:31"), expiry="2026-10-16", ticker="MSFT")
    assert tp.evaluate_entry(t2, ny("2026-10-20T05:00"))[0]["status"] == "UNUSABLE"


def test_C_entry_am_verfallstag_im_fenster_ist_erlaubt():
    exp = "2026-10-16"
    s = syms("call", exp)[0]
    put(s, ny("2026-10-16T11:05"), 0.4, 0.5)
    blk, _ = tp.evaluate_entry(make_trade("call", decision=ny("2026-10-16T11:00"), expiry=exp), ny("2026-10-16T11:10"))
    assert blk["status"] == "FILLED" and blk["price"] == 0.5


def test_C_kontrakt_aus_strike_und_verfall_rekonstruiert_und_unbekannter_kontrakt():
    s = syms("call")[0]
    put(s, D0 + timedelta(minutes=2), 1.0, 1.3)
    t = make_trade("call", decision=D0, with_symbols=False)
    assert [g["symbol"] for g in tp.trade_legs(t)] == [s]
    assert tp.evaluate_entry(t, D0 + timedelta(minutes=5))[0]["status"] == "FILLED"
    put_t = make_trade("put", decision=D0, with_symbols=False)
    assert [g["symbol"] for g in tp.trade_legs(put_t)] == [syms("put")[0]]
    bear = make_trade("bear", decision=D0, with_symbols=False)
    assert [g["symbol"] for g in tp.trade_legs(bear)] == syms("bear")
    t["option"].pop("strike")
    blk, recs = tp.evaluate_entry(t, D0 + timedelta(minutes=5))
    assert blk["status"] == "UNUSABLE" and blk["reason"] == "CONTRACT_UNKNOWN" and recs == []


@pytest.mark.parametrize("dt", [None, "", "kein Datum", float("nan")])
def test_C_decision_time_fehlt_oder_unlesbar(dt):
    t = make_trade(decision=D0)
    t["decision_time"] = dt
    blk, recs = tp.evaluate_entry(t, D0 + timedelta(hours=1))
    assert blk["status"] == "UNUSABLE" and blk["reason"] == "DECISION_TIME_MISSING" and recs == []


# ══════════════════════════════════════════════════════════════════════════════════════════════
# D. Exit
# ══════════════════════════════════════════════════════════════════════════════════════════════
def trade_with_entry(kind="call", *, close, expiry=FAR, entry_at=None, ticker="AAPL", **kw):
    """Trade, dessen Entry am Dienstag 2026-10-13 ~11:05 ET zu Long-Ask 2.0 (Spread: Short-Bid 0.8 -> 1.2) füllt."""
    entry_at = entry_at or ny("2026-10-13T11:05")
    s = syms(kind, expiry, ticker)
    put(s[0], entry_at, 1.8, 2.0)
    if len(s) > 1:
        put(s[1], entry_at, 0.8, 1.0)
    return make_trade(kind, decision=ny("2026-10-13T11:00"), close=close, expiry=expiry, ticker=ticker, **kw)


ENTRY_PREIS = {"call": 2.0, "put": 2.0, "bull": 1.2, "bear": 1.2}


@pytest.mark.parametrize("kind", ["call", "put", "bull", "bear"])
def test_D_exit_preis_ist_long_bid_bzw_long_bid_minus_short_ask(kind):
    close = ny("2026-10-14T11:00")
    t = trade_with_entry(kind, close=close)
    s = syms(kind)
    put(s[0], ny("2026-10-14T11:02"), 3.0, 3.6, last=3.3)
    if len(s) > 1:
        put(s[1], ny("2026-10-14T11:02"), 1.0, 1.4)
    blk, e, x = ev(t, ny("2026-10-14T12:00"))
    assert blk["status"] == "OK" and blk["policy"] == "timing-v2"
    exp_exit = 3.0 if len(s) == 1 else 3.0 - 1.4
    assert blk["exit"]["price"] == pytest.approx(exp_exit)
    assert blk["entry"]["price"] == pytest.approx(ENTRY_PREIS[kind])
    assert blk["outcome"] == pytest.approx(round(exp_exit / ENTRY_PREIS[kind] - 1, 6))
    assert blk["method"] == ("option_quote" if len(s) == 1 else "spread_quote") and blk["price_type"] == "OBSERVED_QUOTE"
    assert len(x) == len(s) and [r["occ_symbol"] for r in x] == s


def test_D_spread_exit_preis_wird_bei_null_gekappt_nie_negativ():
    close = ny("2026-10-14T11:00")
    t = trade_with_entry("bull", close=close)
    s = syms("bull")
    put(s[0], ny("2026-10-14T11:02"), 0.5, 0.7)
    put(s[1], ny("2026-10-14T11:02"), 0.6, 0.9)                      # Long-Bid 0.5 − Short-Ask 0.9 < 0
    blk, _, _ = ev(t, ny("2026-10-14T12:00"))
    assert blk["exit"]["price"] == 0.0 and blk["status"] == "OK" and blk["outcome"] == pytest.approx(-1.0)


@pytest.mark.parametrize("sekunden,erlaubt", [(0, True), (180, True), (300, True), (301, False), (600, False)])
def test_D_toleranz_5_minuten_nur_wenn_exit_entscheid_im_fenster(sekunden, erlaubt):
    close = ny("2026-10-14T14:00")
    t = trade_with_entry(close=close)
    put(CALL, close - timedelta(seconds=sekunden), 4.0, 4.4)
    blk, _, x = ev(t, ny("2026-10-14T14:10"))
    if erlaubt:
        assert blk["status"] == "OK" and blk["exit"]["price"] == 4.0
        assert blk["exit"]["time"] == ts(close), "Exit-Zeit = max(Entscheid, Abruf) – nie vor dem Exit-Entscheid"
    else:
        assert blk["status"] == "PENDING" and x == [] and blk["outcome"] is None


def test_D_exit_entscheid_nach_fensterende_naechste_sitzung_ohne_toleranz():
    close = ny("2026-10-14T16:30")
    t = trade_with_entry(close=close)
    put(CALL, ny("2026-10-14T15:49"), 4.0, 4.4)                      # vor dem Entscheid, Entscheid außerhalb des Fensters
    assert ev(t, ny("2026-10-14T20:00"))[0]["status"] == "PENDING"
    put(CALL, ny("2026-10-15T09:59:59"), 5.0, 5.4)
    put(CALL, ny("2026-10-15T10:00:00"), 6.0, 6.4)
    blk, _, _ = ev(t, ny("2026-10-15T10:30"))
    assert blk["status"] == "OK" and blk["exit"]["price"] == 6.0 and blk["exit"]["day"] == "2026-10-15"
    assert blk["exit"]["time"] == ts(ny("2026-10-15T10:00"))


def test_D_exit_freitagabend_wird_montag():
    close = ny("2026-10-16T21:30")                                   # Freitag 21:30 ET = Samstag 01:30 UTC
    t = trade_with_entry(close=close)
    put(CALL, ny("2026-10-16T15:45"), 4.0, 4.4)
    put(CALL, ny("2026-10-19T10:05"), 7.0, 7.4)
    assert ev(t, ny("2026-10-17T12:00"))[0]["status"] == "PENDING"
    blk, _, _ = ev(t, ny("2026-10-19T10:06"))
    assert blk["status"] == "OK" and blk["exit"]["day"] == "2026-10-19" and blk["exit"]["price"] == 7.0


def test_D_exit_ohne_zulaessige_quote_pending_dann_unusable_mit_grund():
    close = ny("2026-10-14T16:30")
    t = trade_with_entry(close=close)
    ende = ny("2026-10-15T15:50")
    assert ev(t, ende + timedelta(hours=11, minutes=59))[0]["status"] == "PENDING"
    blk, _, x = ev(t, ende + timedelta(hours=12, minutes=1))
    assert blk["status"] == "UNUSABLE" and x == []
    assert "EXIT_NO_ADMISSIBLE_QUOTE_IN_SESSION" in blk["reasons"] and blk["outcome"] is None


def test_D_exit_pit_spaete_quote_erst_ab_retrieved_at():
    close = ny("2026-10-14T11:00")
    t = trade_with_entry(close=close)
    put(CALL, ny("2026-10-14T13:00"), 4.0, 4.4)
    assert ev(t, ny("2026-10-14T12:59:59"))[0]["status"] == "PENDING"
    blk, _, _ = ev(t, ny("2026-10-14T13:00"))
    assert blk["status"] == "OK" and blk["exit"]["time"] == ts(ny("2026-10-14T13:00"))


def test_D_exit_nie_basiswert_oder_modellpreis(net):
    """Ohne Optionsquote nie OK; der Basiswert-Schluss (close_fn) wird vor dem Verfall nie herangezogen."""
    close = ny("2026-10-14T11:00")
    t = trade_with_entry(close=close, close_price=250.0, outcome_at_decision=0.9)
    def close_fn(*a):
        raise AssertionError("Basiswertpreis darf vor dem Verfall nie verwendet werden")
    for now in (ny("2026-10-14T11:30"), ny("2026-10-14T20:00"), ny("2026-10-15T05:00"), ny("2026-12-01T12:00")):
        blk, _, _ = ev(t, now, close_fn=close_fn)
        assert blk["status"] in ("PENDING", "UNUSABLE") and blk["outcome"] is None
    assert net.close_calls == [] and net.fetch_calls == []


# ── Verfall / Settlement ─────────────────────────────────────────────────────
EXP = "2026-10-16"                                    # Freitag


def settle_trade(kind, *, close, entry_at=None):
    return trade_with_entry(kind, close=close, expiry=EXP, entry_at=entry_at)


@pytest.mark.parametrize("kind,schluss,innerer_wert", [
    ("call", 230.0, 30.0), ("call", 200.0, 0.0), ("call", 150.0, 0.0), ("call", 200.01, 0.01),
    ("put", 170.0, 30.0), ("put", 200.0, 0.0), ("put", 260.0, 0.0),
    ("bull", 195.0, 0.0), ("bull", 205.0, 5.0), ("bull", 210.0, 10.0), ("bull", 260.0, 10.0),     # Breite gekappt
    ("bear", 205.0, 0.0), ("bear", 195.0, 5.0), ("bear", 190.0, 10.0), ("bear", 120.0, 10.0),
])
def test_D_settlement_innerer_wert_aus_unbereinigtem_schluss(kind, schluss, innerer_wert, net):
    t = settle_trade(kind, close=ny("2026-10-16T16:30"))             # Exit-Entscheid am Verfallstag nach Schluss
    net.closes[("AAPL", date(2026, 10, 16))] = schluss
    blk, e, x = ev(t, ny("2026-10-16T22:00"))
    assert blk["status"] == "OK" and blk["exit"]["status"] == "SETTLED" and x == []
    assert blk["exit"]["price"] == pytest.approx(innerer_wert)
    assert blk["method"] == "expiry_intrinsic" and blk["price_type"] == "SETTLEMENT_INTRINSIC"
    assert blk["outcome"] == pytest.approx(round(innerer_wert / ENTRY_PREIS[kind] - 1, 6))
    assert net.close_calls == [("AAPL", date(2026, 10, 16))], "Schluss genau vom Verfallstag, nicht Vor-/Folgetag"
    assert blk["exit"]["underlying_close"] == schluss


@pytest.mark.parametrize("close_entscheid", [
    "2026-10-16T16:00",      # Verfallstag unmittelbar nach Schluss
    "2026-10-16T23:00",      # Verfallstag spätabends (UTC schon Samstag)
    "2026-10-17T12:00",      # Samstag
    "2026-10-19T10:30",      # Montag nach Verfall
    "2026-12-01T10:30",      # Wochen später (verspäteter Lauf)
])
def test_D_exit_entscheid_nach_dem_fenster_des_verfallstags_wird_settlement(close_entscheid, net):
    t = settle_trade("call", close=ny(close_entscheid))
    net.closes[("AAPL", date(2026, 10, 16))] = 215.0
    # Auch wenn nach dem Verfall (nie existierende) Quoten im Archiv liegen: Settlement, nie eine Quote
    put(syms("call", EXP)[0], ny("2026-10-19T10:30"), 14.0, 14.5)
    blk, _, x = ev(t, max(ny(close_entscheid), ny("2026-10-16T16:30")) + timedelta(hours=1))
    assert blk["exit"]["status"] == "SETTLED" and blk["exit"]["price"] == 15.0 and x == []
    assert blk["status"] == "OK" and blk["outcome"] == pytest.approx(15.0 / 2.0 - 1)
    assert net.close_calls and all(c == ("AAPL", date(2026, 10, 16)) for c in net.close_calls)


def test_D_verfallstag_im_fenster_mit_quote_wird_zur_quote_ausgefuehrt_nicht_settled(net):
    t = settle_trade("call", close=ny("2026-10-16T11:00"))
    put(syms("call", EXP)[0], ny("2026-10-16T10:58"), 9.0, 9.4)
    blk, _, x = ev(t, ny("2026-10-16T11:05"))
    assert blk["status"] == "OK" and blk["exit"]["status"] == "FILLED" and blk["exit"]["price"] == 9.0 and len(x) == 1
    assert net.close_calls == []


def test_D_verfallstag_im_fenster_ohne_quote_pending_bis_fensterende_dann_settlement(net):
    t = settle_trade("call", close=ny("2026-10-16T11:00"))
    net.closes[("AAPL", date(2026, 10, 16))] = 230.0
    blk, _, _ = ev(t, ny("2026-10-16T15:49"))
    assert blk["status"] == "PENDING" and net.close_calls == []
    # offizieller Schluss 16:00 ET; abgefragt erst nach der Karenz (SETTLEMENT_CLOSE_DELAY = 1 h)
    blk, _, _ = ev(t, ny("2026-10-16T16:30"))
    assert blk["status"] == "PENDING" and net.close_calls == []
    blk, _, x = ev(t, ny("2026-10-16T17:05"))
    assert blk["status"] == "OK" and blk["exit"]["status"] == "SETTLED" and blk["exit"]["price"] == 30.0 and x == []


def test_D_settlement_nie_vor_dem_offiziellen_schluss(net):
    t = settle_trade("call", close=ny("2026-10-16T11:00"))
    net.closes[("AAPL", date(2026, 10, 16))] = 230.0                  # Intraday-Kurs 15:55 ET (Schluss noch offen)
    blk, _, _ = ev(t, ny("2026-10-16T15:55"))
    assert blk["status"] == "PENDING" and net.close_calls == [], "vor 16:00 ET gibt es keinen offiziellen Schluss"


def test_D_exit_entscheid_zwischen_fensterende_und_handelsschluss_wartet_auf_offiziellen_schluss(net):
    t = settle_trade("call", close=ny("2026-10-16T15:55"))
    net.closes[("AAPL", date(2026, 10, 16))] = 230.0
    blk, _, _ = ev(t, ny("2026-10-16T15:57"))
    assert blk["status"] == "PENDING" and net.close_calls == []


def test_D_settlement_schluss_fehlt_pending_dann_nach_5_tagen_unusable(net):
    t = settle_trade("call", close=ny("2026-10-16T16:30"))
    for now, status in ((ny("2026-10-16T22:00"), "PENDING"), (utc("2026-10-21T12:00"), "PENDING"),
                        (utc("2026-10-22T12:00"), "UNUSABLE")):
        blk, _, _ = ev(t, now)
        assert blk["status"] == status and blk["outcome"] is None, now
    assert blk["reasons"] == ["EXIT_SETTLEMENT_CLOSE_MISSING"]
    # sobald der Schluss vorliegt, wird abgerechnet (auch spät)
    net.closes[("AAPL", date(2026, 10, 16))] = 205.0
    blk, _, _ = ev(t, utc("2026-10-22T12:00"))
    assert blk["status"] == "OK" and blk["exit"]["price"] == 5.0


def test_D_settlement_ist_pending_bis_zum_verfall_nicht_frueher():
    """Exit-Entscheid VOR dem Verfall: nie ein Settlement, bevor der Verfallstag gelaufen ist."""
    t = settle_trade("call", close=ny("2026-10-14T16:30"))           # Fill-Tag Donnerstag 10-15 (vor Verfall)
    blk, _, _ = ev(t, ny("2026-10-15T11:00"))
    assert blk["status"] == "PENDING" and not blk["exit"].get("settlement")


def test_D_early_close_verfallstag_settlement_nach_fensterende(net):
    exp = "2025-11-28"                                               # Tag nach Thanksgiving, Schluss 13:00 ET
    s = syms("call", exp)[0]
    put(s, ny("2025-11-26T11:05"), 1.8, 2.0)
    t = make_trade("call", decision=ny("2025-11-26T11:00"), close=ny("2025-11-28T12:55"), expiry=exp)
    net.closes[("AAPL", date(2025, 11, 28))] = 210.0
    blk, _, _ = ev(t, ny("2025-11-28T14:00"))
    assert blk["exit"]["status"] == "SETTLED" and blk["exit"]["price"] == 10.0
    # Entscheid 12:45 ET (im Fenster bis 12:50) mit Quote 12:46 -> Ausführung zur Quote
    put(s, ny("2025-11-28T12:46"), 11.0, 11.4)
    t2 = make_trade("call", decision=ny("2025-11-26T11:00"), close=ny("2025-11-28T12:45"), expiry=exp)
    blk, _, _ = ev(t2, ny("2025-11-28T14:00"))
    assert blk["exit"]["status"] == "FILLED" and blk["exit"]["price"] == 11.0


def test_D_exit_vor_dem_verfallstag_ohne_quote_am_fill_tag_wird_nicht_settled(net):
    """Entscheid Donnerstag 15:00 ET (Verfall Freitag): Fill-Tag = Donnerstag; ohne Quote dort nach Karenz UNUSABLE –
    das Settlement gilt nur, wenn die früheste Ausführungs-Sitzung nach dem Verfall liegt bzw. am Verfallstag selbst."""
    t = settle_trade("call", close=ny("2026-10-15T15:00"))
    net.closes[("AAPL", date(2026, 10, 16))] = 230.0
    blk, _, _ = ev(t, ny("2026-10-16T05:00"))
    assert blk["status"] == "UNUSABLE" and blk["exit"].get("settlement") is None and net.close_calls == []


# ── Exit-Entscheid vs. Entry (E) ──────────────────────────────────────────────
def test_E_exit_entscheid_vor_gefuelltem_entry_ist_unusable_exit_before_entry():
    dec = ny("2026-10-13T17:00")                                     # nach Schluss -> Entry erst Mittwoch 10:30
    put(CALL, ny("2026-10-14T10:30"), 1.8, 2.0)
    put(CALL, ny("2026-10-14T11:00"), 3.0, 3.4)
    t = make_trade(decision=dec, close=ny("2026-10-13T18:00"))        # Exit-Entscheid vor dem Entry-Fill
    blk, e, x = ev(t, ny("2026-10-14T12:00"))
    assert blk["status"] == "UNUSABLE" and "EXIT_BEFORE_ENTRY" in blk["reasons"] and blk["outcome"] is None
    assert blk["entry"]["status"] == "FILLED"


def test_E_exit_entscheid_exakt_zur_entry_zeit_ist_nicht_vor_dem_entry():
    put(CALL, D0, 1.8, 2.0)
    put(CALL, D0 + timedelta(minutes=1), 3.0, 3.4)
    t = make_trade(decision=D0, close=D0)                            # Entry-Zeit == Exit-Entscheid
    blk, _, _ = ev(t, D0 + timedelta(minutes=30))
    assert "EXIT_BEFORE_ENTRY" not in blk.get("reasons", []) and blk["status"] == "OK"
    assert blk["entry"]["time"] == blk["exit"]["decision_time"] == ts(D0)


def test_E_entry_noch_pending_bei_vorhandenem_exit_entscheid_ist_pending():
    t = make_trade(decision=D0, close=D0 + timedelta(minutes=5))
    blk, e, x = ev(t, D0 + timedelta(minutes=6))
    assert blk["status"] == "PENDING" and blk["outcome"] is None and e == [] and x == []
    assert blk["entry"]["status"] == "PENDING"
    # Entry füllt erst NACH dem Exit-Entscheid (Quote 11:30 > Entscheid 11:05): nie ausgeführt -> EXIT_BEFORE_ENTRY
    put(CALL, D0 + timedelta(minutes=30), 1.8, 2.0)
    put(CALL, D0 + timedelta(minutes=40), 3.0, 3.4)
    blk, _, _ = ev(t, D0 + timedelta(minutes=35))
    assert blk["status"] == "UNUSABLE" and "EXIT_BEFORE_ENTRY" in blk["reasons"] and blk["outcome"] is None


def test_E_entry_gefuellt_exit_noch_ausstehend_ist_pending_dann_ok():
    put(CALL, D0 + timedelta(minutes=30), 1.8, 2.0)
    put(CALL, D0 + timedelta(minutes=50), 3.0, 3.4)
    t = make_trade(decision=D0, close=D0 + timedelta(minutes=45))
    blk, _, _ = ev(t, D0 + timedelta(minutes=35))                   # Entry gefüllt, Exit-Quote 11:50 noch unbekannt (PIT)
    assert blk["status"] == "PENDING" and blk["entry"]["status"] == "FILLED" and blk["exit"]["status"] == "PENDING"
    blk, _, _ = ev(t, D0 + timedelta(minutes=51))
    assert blk["status"] == "OK" and blk["outcome"] == pytest.approx(3.0 / 2.0 - 1)


def test_E_ohne_exit_entscheid_wird_nur_der_entry_bewertet_status_pending():
    put(CALL, D0 + timedelta(minutes=2), 1.8, 2.0)
    blk, e, x = ev(make_trade(decision=D0), D0 + timedelta(hours=1))
    assert blk["status"] == "PENDING" and blk["outcome"] is None and x == [] and len(e) == 1
    assert blk["entry"]["status"] == "FILLED" and blk["exit"] is None


def test_E_block_struktur_und_outcome_nur_bei_ok():
    put(CALL, D0 + timedelta(minutes=2), 1.8, 2.0)
    put(CALL, D0 + timedelta(minutes=30), 2.4, 2.6)
    t = make_trade(decision=D0, close=D0 + timedelta(minutes=29))
    blk, _, _ = ev(t, D0 + timedelta(hours=1))
    assert blk["policy"] == "timing-v2" and blk["version"] == "timing-eval-v1"
    assert blk["status"] == "OK" and blk["reasons"] == []
    assert blk["outcome"] == pytest.approx(round(2.4 / 2.0 - 1, 6)) and blk["evaluated_at"] == ts(D0 + timedelta(hours=1))
    for st_now in (D0 + timedelta(minutes=10), ny("2026-10-20T12:00")):
        b2, _, _ = ev(make_trade(decision=D0, close=D0 + timedelta(minutes=5), ticker="MSFT"), st_now)
        assert b2["status"] in ("PENDING", "UNUSABLE") and b2["outcome"] is None and b2["method"] is None


def test_E_evaluate_wirft_nie_und_ist_idempotent():
    put(CALL, D0 + timedelta(minutes=2), 1.8, 2.0)
    put(CALL, D0 + timedelta(minutes=30), 2.4, 2.6)
    t = make_trade(decision=D0, close=D0 + timedelta(minutes=29))
    a = ev(t, D0 + timedelta(hours=1))
    b = ev(t, D0 + timedelta(hours=1))
    assert a == b
    for müll in ({}, {"option": "x"}, {"decision_time": "x", "option": None}, {"ticker": "A", "option": {"strike": "abc"}}):
        blk, e, x = ev(müll, D0)
        assert blk["status"] in ("PENDING", "UNUSABLE") and blk["outcome"] is None


def test_E_point_in_time_nur_records_mit_retrieved_at_bis_now_werden_verwendet():
    put(CALL, D0 + timedelta(minutes=2), 1.8, 2.0)
    put(CALL, D0 + timedelta(minutes=30), 2.4, 2.6)
    t = make_trade(decision=D0, close=D0 + timedelta(minutes=29))
    for now in (D0 + timedelta(minutes=29), D0 + timedelta(minutes=30), D0 + timedelta(hours=3)):
        blk, e, x = ev(t, now)
        for r in e + x:
            assert tp._ts(r["retrieved_at"]) <= now
    assert ev(t, D0 + timedelta(minutes=29))[0]["status"] == "PENDING"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# F. feedback.py
# ══════════════════════════════════════════════════════════════════════════════════════════════
class FakeDT(datetime):
    """Fester 'Jetzt'-Zeitpunkt für feedback.main (now/utcnow)."""
    now_value: datetime = utc("2026-10-13T12:00")

    @classmethod
    def now(cls, tz=None):
        return cls.now_value.astimezone(tz) if tz else cls.now_value.replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return cls.now_value.replace(tzinfo=None)


FEATURES = {"bin_impact": "high", "bin_mismatch": "strong", "bin_eps_drift": "massive"}


def champion(kind="call", *, decision, close=None, ticker="AAPL", policy="timing-v2", entry_date=None, **extra) -> dict:
    t = make_trade(kind, decision=decision, close=close, ticker=ticker, policy=policy, **extra)
    t["features"] = dict(FEATURES)
    t["entry_date"] = entry_date or decision.astimezone(UTC).strftime("%Y-%m-%d")
    t["deep_analysis"] = {"direction": "BULLISH"}
    return t


def run_main(monkeypatch, history, now, *, outcome=0.6, exit_reason="take_profit", price=100.0):
    """feedback.main() mit festem 'Jetzt' und ohne Netz/Dateisystem; zählt kursbasierte Aufrufe und Lern-Updates."""
    from modules import email_reporter, final_mc_ledger, rl_robust_shadow, universe_v2_ledger
    from modules.expectation_alpha import config as ea_cfg
    FakeDT.now_value = now
    calls = {"price": [], "compute": [], "learn": [], "record": []}
    real_learn, real_record = feedback.learn_on_close, feedback.record_trade_outcome

    def get_price(tk):
        calls["price"].append(tk)
        return price

    def comp(t, cur, meta=None):
        calls["compute"].append(t["ticker"])
        if meta is not None:
            meta.update(method="option_quote", price_type="OBSERVED_QUOTE")
        return outcome

    def learn(h, t, o, r):
        calls["learn"].append((t["ticker"], round(o, 6), r))
        return real_learn(h, t, o, r)

    def record(t, o, meta, *a, **k):
        calls["record"].append((t["ticker"], round(o, 6)))
        return real_record(t, o, meta, *a, **k)
    monkeypatch.setattr(feedback, "datetime", FakeDT)
    monkeypatch.setattr(feedback, "load_history", lambda: history)
    monkeypatch.setattr(feedback, "save_history", lambda h: None)
    for name, fn in (("get_current_price", get_price), ("compute_outcome", comp), ("learn_on_close", learn),
                     ("record_trade_outcome", record), ("check_exit_rules", lambda t, o, d: exit_reason),
                     ("decide_exit", lambda t, o, m, d, n: exit_reason),   # Exit-Regel (V5) getrennt getestet
                     ("register_exit_sim", lambda *a, **k: None), ("_spread_execution_monitor", lambda *a, **k: None),
                     ("_spread_execution_exit", lambda *a, **k: None), ("compute_pearson_weights", lambda h: {}),
                     ("evaluate_shadow_trades", lambda *a, **k: None), ("maybe_notify_rl_arming", lambda h: None),
                     ("retrain_rl_agent", lambda h: None)):
        monkeypatch.setattr(feedback, name, fn)
    monkeypatch.setattr(feedback.candidate_ledger, "update_outcomes", lambda *a, **k: None)
    monkeypatch.setattr(universe_v2_ledger, "resolve_outcomes", lambda **k: 0)
    monkeypatch.setattr(final_mc_ledger, "resolve_outcomes", lambda **k: 0)
    monkeypatch.setattr(ea_cfg, "enabled", lambda: False)
    monkeypatch.setattr(rl_robust_shadow, "train_robust_shadow", lambda h: None)
    monkeypatch.setattr(email_reporter, "send_exit_alert_email", lambda *a, **k: None)
    feedback.main()
    return calls


def bins(history) -> int:
    return sum(b["count"] for f in (history.get("feature_stats") or {}).values() for b in f.values())


def outcomes_file(tmp_path) -> list[dict]:
    p = tmp_path / "trade_outcomes.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


@pytest.mark.parametrize("jetzt,offen", [
    ("2026-11-26T19:00", False),      # Thanksgiving
    ("2026-11-27T02:00", False),      # 21:00 ET Thanksgiving (UTC schon Freitag!)
    ("2026-11-27T12:00", True),       # Freitag nach Thanksgiving
    ("2026-11-28T00:30", True),       # 19:30 ET Freitag (UTC Samstag)
    ("2026-04-03T15:00", False),      # Good Friday
    ("2026-04-04T02:00", False),      # Karfreitag 22:00 ET (UTC Samstag)
    ("2026-06-19T15:00", False),      # Juneteenth 2026
    ("2025-06-19T15:00", False),      # Juneteenth 2025
    ("2025-01-09T15:00", False),      # Sonderschließung (Staatstrauer Carter)
    ("2025-01-10T03:00", False),      # 22:00 ET am 09.01.
    ("2026-07-03T15:00", False),      # beobachteter Unabhängigkeitstag
    ("2026-10-17T15:00", False),      # Samstag
    ("2026-10-19T00:30", False),      # Sonntag 20:30 ET (UTC Montag!)
    ("2026-10-17T00:30", True),       # Freitag 20:30 ET (UTC Samstag)
    ("2026-10-13T00:30", True),       # Montag 20:30 ET (Columbus Day ist NYSE-Handelstag)
    ("2025-11-28T18:00", True),       # Early Close ist ein Handelstag
    ("2026-03-09T13:00", True),       # Montag nach US-Umstellung
])
def test_F_market_open_day_nutzt_new_york_datum_und_nyse_kalender(jetzt, offen):
    assert feedback._market_open_day(utc(jetzt)) is offen
    assert feedback._market_open_day(utc(jetzt).replace(tzinfo=None)) is offen        # naiv = UTC


@pytest.mark.parametrize("jetzt", ["2026-11-26T19:00", "2025-01-09T15:00", "2026-04-03T15:00", "2026-10-17T15:00",
                                   "2026-10-18T12:00"])
def test_F_geschlossener_boersentag_keine_kursbasierte_exit_pruefung(monkeypatch, tmp_path, jetzt):
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)
    aktiv = champion(decision=ny("2026-10-13T11:00"))
    cf = champion(decision=ny("2026-10-13T11:00"), ticker="MSFT")
    hist = {"active_trades": [aktiv, champion(decision=ny("2026-10-13T11:00"), ticker="NVDA", policy=None)],
            "counterfactual_trades": [cf], "closed_trades": []}
    c = run_main(monkeypatch, hist, utc(jetzt))
    assert c["price"] == [] and c["compute"] == [], "an NYSE-geschlossenen Tagen keine kursbasierte Prüfung"
    assert len(hist["active_trades"]) == 2 and hist["closed_trades"] == []
    assert hist.get("exit_pending_trades", []) == [] and hist.get("closed_trades_unusable", []) == []
    assert len(hist["counterfactual_trades"]) == 1 and c["learn"] == [] and bins(hist) == 0


def test_F_geschlossener_tag_laesst_ausstehende_exits_ohne_zulaessige_quote_ausstehend(monkeypatch, tmp_path):
    """Thanksgiving: der Exit-Entscheid vom Mittwochabend braucht eine Quote aus der nächsten Sitzung (Freitag);
    am geschlossenen Feiertag gibt es keine -> weiter ausstehend, kein Outcome aus Vortagesquotes."""
    put(CALL, ny("2025-11-26T11:05"), 1.8, 2.0)
    put(CALL, ny("2025-11-26T15:45"), 5.0, 5.4)                      # Vortagesquote (vor dem Entscheid)
    t = champion(decision=ny("2025-11-26T11:00"), close=ny("2025-11-26T20:00"))
    t["outcome"] = None
    hist = {"active_trades": [], "exit_pending_trades": [t], "closed_trades": []}
    c = run_main(monkeypatch, hist, ny("2025-11-27T14:00"))
    assert hist["exit_pending_trades"] == [t] and hist["closed_trades"] == [] and c["learn"] == []
    assert t["outcome"] is None
    # Freitag (Early Close) mit Quote im Fenster 10:00–12:50 -> Ausführung
    put(CALL, ny("2025-11-28T10:30"), 6.0, 6.4)
    run_main(monkeypatch, hist, ny("2025-11-28T14:00"), exit_reason=None)
    assert hist["exit_pending_trades"] == [] and len(hist["closed_trades"]) == 1
    assert hist["closed_trades"][0]["outcome"] == pytest.approx(6.0 / 2.0 - 1, abs=1e-4)


def test_F_v1_altbestand_schliesst_unveraendert_sofort_zum_entscheidungs_outcome(monkeypatch, tmp_path):
    v1 = champion(decision=ny("2026-10-13T11:00"), policy=None, ticker="MSFT")
    hist = {"active_trades": [v1], "closed_trades": []}
    c = run_main(monkeypatch, hist, ny("2026-10-14T17:30"), outcome=0.55)          # nach Handelsschluss
    assert [t["ticker"] for t in hist["closed_trades"]] == ["MSFT"]
    got = hist["closed_trades"][0]
    assert got["outcome"] == 0.55 and "timing" not in got and "timing_policy" not in got and "close_time" not in got
    assert hist.get("exit_pending_trades", []) == [] and hist.get("closed_trades_unusable", []) == []
    assert c["learn"] == [("MSFT", 0.55, True)]


def test_F_v2_nach_handelsschluss_wartet_dann_ausfuehrung_lernen_genau_einmal(monkeypatch, tmp_path):
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)                       # Entry-Fill (Ask 2.0)
    t = champion(decision=ny("2026-10-13T11:00"))
    hist = {"active_trades": [t], "closed_trades": []}
    # Exit-Entscheid Dienstag 2026-10-14 17:30 ET (nach dem Fenster): Outcome zum Entscheid 0.6 darf NICHT gelten
    c1 = run_main(monkeypatch, hist, ny("2026-10-14T17:30"), outcome=0.6)
    assert hist["closed_trades"] == [] and len(hist["exit_pending_trades"]) == 1
    pend = hist["exit_pending_trades"][0]
    assert pend["outcome"] is None and pend["outcome_at_decision"] == 0.6 and pend["close_reason"] == "take_profit"
    assert pend["close_time"] == ts(ny("2026-10-14T17:30")) and c1["learn"] == [] and bins(hist) == 0
    # kein zulässiger Preis -> noch nicht geschlossen; nächste Sitzung: Quote 10:20 ET
    put(CALL, ny("2026-10-14T15:45"), 9.0, 9.4)                       # vor dem Entscheid: nie
    put(CALL, ny("2026-10-15T10:20"), 3.0, 3.4)
    c2 = run_main(monkeypatch, hist, ny("2026-10-15T10:30"), exit_reason=None)
    assert hist["exit_pending_trades"] == [] and len(hist["closed_trades"]) == 1
    got = hist["closed_trades"][0]
    assert got["outcome"] == pytest.approx(3.0 / 2.0 - 1, abs=1e-4), "v2-Outcome (Entry-Ask, Exit-Bid), nicht der Entscheids-Return"
    assert got["outcome_at_decision"] == 0.6
    assert c2["learn"] == [("AAPL", pytest.approx(0.5), True)] and bins(hist) == 3     # impact/mismatch/eps_drift je 1
    assert len(c2["record"]) == 1 and len(outcomes_file(tmp_path)) == 1
    # wiederholte Scheduler-Läufe (auch an späteren Tagen): kein zweites Outcome, kein zweites Lernen
    for jetzt in ("2026-10-15T23:30", "2026-10-16T14:00", "2026-10-19T14:00"):
        c = run_main(monkeypatch, hist, ny(jetzt), exit_reason=None)
        assert c["learn"] == [] and c["record"] == []
    assert len(hist["closed_trades"]) == 1 and bins(hist) == 3 and len(outcomes_file(tmp_path)) == 1


def test_F_v2_exit_im_fenster_mit_quote_schliesst_sofort(monkeypatch, tmp_path):
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)
    put(CALL, ny("2026-10-14T13:58"), 4.0, 4.4)                       # 2 min vor dem Exit-Entscheid
    t = champion(decision=ny("2026-10-13T11:00"))
    hist = {"active_trades": [t], "closed_trades": []}
    c = run_main(monkeypatch, hist, ny("2026-10-14T14:00"))
    assert len(hist["closed_trades"]) == 1 and hist.get("exit_pending_trades", []) == []
    assert hist["closed_trades"][0]["outcome"] == pytest.approx(4.0 / 2.0 - 1, abs=1e-4)
    assert hist["closed_trades"][0]["timing"]["status"] == "OK"
    assert c["learn"] == [("AAPL", pytest.approx(1.0), True)] and bins(hist) == 3


def test_F_v2_ohne_zulaessigen_preis_geht_nach_karenz_in_unusable_ohne_outcome_und_ohne_lernen(monkeypatch, tmp_path):
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)
    t = champion(decision=ny("2026-10-13T11:00"))
    hist = {"active_trades": [t], "closed_trades": []}
    run_main(monkeypatch, hist, ny("2026-10-14T17:30"))
    assert len(hist["exit_pending_trades"]) == 1
    # Mittwoch: Fenster verstrichen + 12 h Karenz, keine zulässige Quote
    c = run_main(monkeypatch, hist, ny("2026-10-16T05:00"), exit_reason=None)
    assert hist["exit_pending_trades"] == [] and hist["closed_trades"] == []
    assert len(hist["closed_trades_unusable"]) == 1
    u = hist["closed_trades_unusable"][0]
    assert u["outcome"] is None and u["timing"]["status"] == "UNUSABLE"
    assert "EXIT_NO_ADMISSIBLE_QUOTE_IN_SESSION" in u["timing"]["reasons"]
    assert c["learn"] == [] and c["record"] == [] and bins(hist) == 0 and outcomes_file(tmp_path) == []
    run_main(monkeypatch, hist, ny("2026-10-16T14:00"), exit_reason=None)           # wiederholter Lauf
    assert len(hist["closed_trades_unusable"]) == 1 and hist["closed_trades"] == []


def test_F_close_v2_und_resolve_pending_exits_direkt_idempotent(monkeypatch, tmp_path):
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)
    t = champion(decision=ny("2026-10-13T11:00"), close=ny("2026-10-16T21:00"))     # Freitagabend
    hist = {"active_trades": [], "closed_trades": []}
    assert feedback.close_v2(hist, t, ny("2026-10-16T21:01"), kind="champion") == "PENDING"
    assert hist["exit_pending_trades"] == [t]
    for jetzt in ("2026-10-17T12:00", "2026-10-18T12:00", "2026-10-19T09:00"):       # Wochenende + vor dem Fenster
        res = feedback.resolve_pending_exits(hist, ny(jetzt))
        assert res["champion"] == {"CLOSED": 0, "UNUSABLE": 0, "PENDING": 1}
        assert hist["exit_pending_trades"] == [t] and hist.get("closed_trades") == []
    put(CALL, ny("2026-10-19T10:10"), 2.4, 2.6)
    res = feedback.resolve_pending_exits(hist, ny("2026-10-19T10:30"))
    assert res["champion"]["CLOSED"] == 1 and hist["exit_pending_trades"] == []
    assert len(hist["closed_trades"]) == 1 and bins(hist) == 3
    for _ in range(3):
        res = feedback.resolve_pending_exits(hist, ny("2026-10-19T11:30"))
        assert res["champion"] == {"CLOSED": 0, "UNUSABLE": 0, "PENDING": 0}
    assert len(hist["closed_trades"]) == 1 and bins(hist) == 3
    assert sum(1 for v in hist["processed_outcomes"] if v.endswith(":close:feature_stats")) == 1


def test_F_v2_settlement_exit_schliesst_mit_innerem_wert_und_lernt_einmal(monkeypatch, tmp_path, net):
    exp = "2026-10-16"
    s = syms("call", exp)[0]
    put(s, ny("2026-10-13T11:05"), 1.8, 2.0)
    t = champion(decision=ny("2026-10-13T11:00"), expiry=exp)
    hist = {"active_trades": [t], "closed_trades": []}
    net.closes[("AAPL", date(2026, 10, 16))] = 230.0
    c = run_main(monkeypatch, hist, ny("2026-10-16T18:00"))                          # Verfallstag nach Schluss
    assert len(hist["closed_trades"]) == 1
    got = hist["closed_trades"][0]
    assert got["outcome"] == pytest.approx(30.0 / 2.0 - 1, abs=1e-4)
    assert got["outcome_method"] == "expiry_intrinsic" and got["outcome_price_type"] == "SETTLEMENT_INTRINSIC"
    assert net.close_calls and all(x == ("AAPL", date(2026, 10, 16)) for x in net.close_calls)
    assert len(c["learn"]) == 1 and bins(hist) == 3
    run_main(monkeypatch, hist, ny("2026-10-17T12:00"), exit_reason=None)
    run_main(monkeypatch, hist, ny("2026-10-19T12:00"), exit_reason=None)
    assert len(hist["closed_trades"]) == 1 and bins(hist) == 3


def test_F_counterfactual_v2_gleicher_lebenszyklus_aber_ohne_lern_updates(monkeypatch, tmp_path):
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)
    cf = champion(decision=ny("2026-10-13T11:00"))
    hist = {"active_trades": [], "counterfactual_trades": [cf], "closed_trades": []}
    c = run_main(monkeypatch, hist, ny("2026-10-14T17:30"))                           # Exit-Entscheid nach Schluss
    assert hist["counterfactual_trades"] == [] and len(hist["counterfactual_exit_pending"]) == 1
    assert hist["closed_trades"] == [] and c["learn"] == []
    put(CALL, ny("2026-10-15T10:20"), 3.0, 3.4)
    run_main(monkeypatch, hist, ny("2026-10-15T10:30"), exit_reason=None)
    assert hist["counterfactual_exit_pending"] == [] and len(hist["counterfactual_closed"]) == 1
    assert hist["counterfactual_closed"][0]["outcome"] == pytest.approx(0.5, abs=1e-4)
    assert hist["closed_trades"] == [] and bins(hist) == 0 and outcomes_file(tmp_path) == []


def test_F_fehler_in_einem_ausstehenden_trade_blockiert_die_anderen_nicht(monkeypatch):
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)
    put(CALL, ny("2026-10-19T10:10"), 2.4, 2.6)
    gut = champion(decision=ny("2026-10-13T11:00"), close=ny("2026-10-16T21:00"))
    schlecht = {"ticker": "XXX", "option": None, "decision_time": 5, "close_time": [], "timing_policy": "timing-v2"}
    hist = {"exit_pending_trades": [schlecht, gut], "closed_trades": []}
    feedback.resolve_pending_exits(hist, ny("2026-10-19T10:30"))
    assert len(hist["closed_trades"]) == 1 and hist["closed_trades"][0]["ticker"] == "AAPL"


def test_F_outcome_record_verweist_auf_die_fill_quotes_nicht_auf_die_buchungs_quote(monkeypatch, tmp_path):
    """Reproduzierbarkeit: trade_outcomes.jsonl muss die Quotes nennen, aus denen Entry- und Exit-Preis stammen."""
    chain = oq.capture({"symbol": CALL, "bid": 1.9, "ask": 2.1, "quote_time": ny("2026-10-13T17:00").isoformat()},
                       underlying="AAPL", source="tradier", context="trade_entry", retrieved_at=ny("2026-10-13T17:00"))
    put(CALL, ny("2026-10-14T10:05"), 1.8, 2.0)                       # Entry-Fill am nächsten Morgen
    put(CALL, ny("2026-10-14T14:00"), 3.0, 3.4)                       # Exit-Fill
    t = champion(decision=ny("2026-10-13T17:00"), close=ny("2026-10-14T13:59"))
    t["entry_quote_record_ids"] = [chain]                             # wie pipeline.book_proposals: Chain-Quote der Buchung
    hist = {"exit_pending_trades": [t], "closed_trades": []}
    feedback.resolve_pending_exits(hist, ny("2026-10-14T14:30"))
    assert len(hist["closed_trades"]) == 1
    rec_ = outcomes_file(tmp_path)[0]
    fill_ids = hist["closed_trades"][0]["timing"]["entry"]["record_ids"]
    assert chain not in fill_ids and len(fill_ids) == 1
    assert rec_["exit_quote_record_ids"] == hist["closed_trades"][0]["timing"]["exit"]["record_ids"]
    assert rec_["entry_quote_record_ids"] == fill_ids, "Entry-Preis stammt aus dem Fill-Record, nicht aus der Buchungs-Quote"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# G. candidate_ledger
# ══════════════════════════════════════════════════════════════════════════════════════════════
LEG_L, LEG_S = syms("bull")
LEGACY_OPT = occ("MSFT", FAR, "C", 400.0)
LEGACY_L, LEGACY_S = occ("MSFT", FAR, "C", 400.0), occ("MSFT", FAR, "C", 420.0)


def ledger_row(sid: str, signal_at: datetime, *, v2: bool, option=CALL, legs=(LEG_L, LEG_S), ticker="AAPL",
               opt=True, spread=True) -> dict:
    row = {"date": signal_at.astimezone(UTC).strftime("%Y-%m-%d"), "ticker": ticker, "signal_id": sid,
           "status": "proposed", "direction": "BULLISH", "features": {}, "entry_price": 100.0, "entry_basis": "close",
           "session": "post", "signal_timestamp": ts(signal_at), "outcomes": {}}
    if v2:
        row["timing_policy"] = "timing-v2"
    if opt:
        row["real_option"] = {"symbol": option, "strike": 200.0, "expiry": FAR, "entry_pending": True,
                              "bid": None, "ask": None, "mid": None,
                              "snapshot_quote": {"bid": 9.0, "ask": 9.9, "mid": 9.5, "quote_ts": "x"}}
    if spread:
        row["real_strategy"] = {"strategy": "BULL_CALL_SPREAD", "expiry": FAR, "entry_pending": True,
                                "net_debit_entry": None, "net_mid_entry": None,
                                "legs": [{"side": "long", "symbol": legs[0], "strike": 200.0, "bid": None, "ask": None,
                                          "mid": None},
                                         {"side": "short", "symbol": legs[1], "strike": 210.0, "bid": None, "ask": None,
                                          "mid": None}]}
    return row


def write_ledger(root: Path, rows: list[dict], month="2026-10") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{month}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def read_ledger(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.fixture
def ledger(monkeypatch, tmp_path, net):
    """Ledger-Wurzel in tmp_path; History-/Preis-Abrufe gesperrt (leer), 'Jetzt' über cl._now_utc steuerbar."""
    root = tmp_path / "candidate_ledger"
    clock = {"now": ny("2026-10-13T11:00")}
    monkeypatch.setattr(cl, "_now_utc", lambda: clock["now"])
    monkeypatch.setattr(cl, "_fetch_history_batch", lambda *a, **k: {})
    monkeypatch.setattr(cl, "_fetch_history_open_batch", lambda *a, **k: {})
    monkeypatch.setattr(cl, "_fetch_prices_batch", lambda tickers: {t: None for t in tickers})
    clock["root"] = root
    return clock


def update(ledger_state, today="2026-10-13"):
    cl.update_outcomes(today, root=ledger_state["root"])


def test_G_neue_zeilen_tragen_timing_v2_und_entry_nur_ueber_das_archiv(monkeypatch, ledger, net, tmp_path):
    FakeDT.now_value = ny("2026-10-13T11:00")
    monkeypatch.setattr(cl, "datetime", FakeDT)
    monkeypatch.setattr(cl.market_snapshot, "us_market_session", lambda t: "regular")
    monkeypatch.setattr(cl.market_snapshot, "fetch_underlying_quotes",
                        lambda tk: {t: {"bid": 100.0, "ask": 100.4, "mid": 100.2, "last": 100.1, "prev_close": 99.0,
                                        "open": 100.0, "quote_ts": "t", "source": "tradier"} for t in tk})
    monkeypatch.setattr(cl.market_snapshot, "select_contract", lambda *a: {
        "symbol": CALL, "strike": 200.0, "expiry": FAR, "dte": 95, "bid": 5.0, "ask": 5.4, "mid": 5.2, "iv": 0.3,
        "delta": 0.5, "open_interest": 500, "quote_ts": "t"})
    monkeypatch.setattr(cl.market_snapshot, "select_spread_short_leg", lambda *a, **k: {
        "symbol": LEG_S, "strike": 210.0, "bid": 2.0, "ask": 2.2, "mid": 2.1})
    monkeypatch.setattr(cl.market_snapshot, "fetch_term_iv_point", lambda *a, **k: None)
    root = ledger["root"]
    cl.start_run("2026-10-13")
    cl.note("AAPL", stage="deep_analysis", direction="BULLISH", ttm="4-8 Wochen")
    cl.flush(reports_dir_root=root)
    row = read_ledger(root / "2026-10.jsonl")[0]
    assert row["timing_policy"] == "timing-v2"
    # selbst bei 'regular' Session und Chain-Quote KEIN sofortiger Entry: ohne archivierte zulässige Quote bleibt er offen
    for k in ("real_option", "real_strategy"):
        if k in row:
            assert row[k]["entry_pending"] is True and row[k].get("entry_date") is None
            assert row[k].get("entry_status") == "PENDING"
    assert row["real_option"]["ask"] is None and row["real_option"]["bid"] is None


def test_G_v2_zeile_signal_nach_fensterende_fill_erst_naechster_tag_nur_aus_dem_archiv(ledger, net, tmp_path):
    sig = ny("2026-10-13T16:00")                                     # nach Fensterende 15:50
    path = write_ledger(ledger["root"], [ledger_row("s1", sig, v2=True)])
    put(CALL, ny("2026-10-13T15:40"), 3.0, 3.2)                      # frisch (20 min vor Signal) aber Signal außerhalb
    put_all("bull", ny("2026-10-13T15:45"), long=(4.0, 4.2), short=(1.0, 1.2))
    ledger["now"] = ny("2026-10-13T16:30")
    update(ledger)
    r = read_ledger(path)[0]
    assert r["real_option"]["entry_pending"] is True and r["real_option"]["ask"] is None
    assert net.fetch_calls == [], "außerhalb des Mark-Fensters wird nichts abgerufen"
    # nächster Handelstag: Quote 10:30 ET im Archiv
    put(CALL, ny("2026-10-14T10:30"), 5.0, 5.4)
    put_all("bull", ny("2026-10-14T10:31"), long=(5.0, 5.4), short=(2.0, 2.3))
    ledger["now"] = ny("2026-10-14T10:45")
    update(ledger, "2026-10-14")
    r = read_ledger(path)[0]
    ro, rs = r["real_option"], r["real_strategy"]
    assert ro["entry_pending"] is False and ro["entry_status"] == "FILLED" and ro["entry_date"] == "2026-10-14"
    assert ro["ask"] == 5.4 and ro["bid"] == 5.0 and ro["entry_filled_at"] == ts(ny("2026-10-14T10:30"))
    assert len(ro["entry_quote_record_ids"]) == 1
    assert rs["entry_pending"] is False and rs["entry_date"] == "2026-10-14"
    assert rs["net_debit_entry"] == pytest.approx(5.4 - 2.0), "Long-Ask − Short-Bid"
    assert rs["entry_filled_at"] == ts(ny("2026-10-14T10:31")) and len(rs["entry_quote_record_ids"]) == 2
    # Idempotenz: weiterer Lauf ändert die Datei nicht
    vorher = path.read_bytes()
    ledger["now"] = ny("2026-10-14T11:30")
    update(ledger, "2026-10-14")
    assert path.read_bytes() == vorher


def test_G_v2_signal_im_fenster_toleranz_15_minuten_und_fetch_nur_im_fenster(ledger, net):
    sig = ny("2026-10-13T11:00")
    path = write_ledger(ledger["root"], [ledger_row("s1", sig, v2=True, spread=False)])
    put(CALL, ny("2026-10-13T10:44"), 3.0, 3.2)                      # 16 min vor dem Signal -> nicht erlaubt
    ledger["now"] = ny("2026-10-13T11:05")
    held = []

    def live(symbols):
        held.append(list(symbols))
        put(CALL, ledger["now"], 3.5, 3.7)                            # der Abruf archiviert wie der echte Hook
        return {CALL: {"bid": 3.5, "ask": 3.7, "mid": 3.6, "ts": None}}
    net.on_fetch = live
    update(ledger)
    r = read_ledger(path)[0]
    assert held == [[CALL]], "im Mark-Fenster werden frische Quotes abgerufen (landen im Archiv)"
    assert r["real_option"]["ask"] == 3.7 and r["real_option"]["entry_filled_at"] == ts(ny("2026-10-13T11:05"))


@pytest.mark.parametrize("signal,jetzt", [
    ("2026-10-13T09:40", "2026-10-13T09:45"),     # reguläre Session, aber vor dem Mark-Fenster
    ("2026-10-13T09:40", "2026-10-13T09:59"),
    ("2026-10-16T17:00", "2026-10-17T12:00"),     # Wochenende
    ("2025-11-26T17:00", "2025-11-27T12:00"),     # Thanksgiving
    ("2026-04-02T17:00", "2026-04-03T12:00"),     # Good Friday
    ("2026-10-13T15:00", "2026-10-13T15:55"),     # nach Fensterende, vor Handelsschluss
])
def test_G_v2_ausserhalb_des_mark_fensters_kein_abruf_und_kein_entry_aus_der_live_quote(ledger, net, signal, jetzt):
    sig = ny(signal)
    path = write_ledger(ledger["root"], [ledger_row("s1", sig, v2=True)], month=sig.strftime("%Y-%m"))
    ledger["now"] = ny(jetzt)
    net.on_fetch = lambda s: {CALL: {"bid": 3.5, "ask": 3.7, "mid": 3.6, "ts": None}}
    cl.update_outcomes(ledger["now"].strftime("%Y-%m-%d"), root=ledger["root"])
    r = read_ledger(path)[0]
    assert r["real_option"]["entry_pending"] is True and r["real_option"]["ask"] is None
    assert r["real_strategy"]["entry_pending"] is True and r["real_strategy"]["net_debit_entry"] is None
    assert net.fetch_calls == []


def test_G_v2_spread_mit_vertauschter_leg_reihenfolge_nutzt_long_ask_minus_short_bid(ledger, net):
    sig = ny("2026-10-13T11:00")
    row = ledger_row("s1", sig, v2=True, opt=False)
    row["real_strategy"]["legs"].reverse()                            # Short zuerst
    path = write_ledger(ledger["root"], [row])
    put(LEG_L, ny("2026-10-13T11:02"), 4.0, 4.2)
    put(LEG_S, ny("2026-10-13T11:03"), 1.0, 1.3)
    ledger["now"] = ny("2026-10-13T11:10")
    net.on_fetch = lambda s: {}
    update(ledger)
    rs = read_ledger(path)[0]["real_strategy"]
    assert rs["entry_pending"] is False and rs["net_debit_entry"] == pytest.approx(4.2 - 1.0)
    assert rs["entry_filled_at"] == ts(ny("2026-10-13T11:03"))


def test_G_v2_ohne_zulaessige_quote_nach_fensterende_plus_karenz_unusable(ledger, net):
    sig = ny("2026-10-13T11:00")
    path = write_ledger(ledger["root"], [ledger_row("s1", sig, v2=True)])
    ledger["now"] = ny("2026-10-13T15:55")
    update(ledger)
    assert read_ledger(path)[0]["real_option"]["entry_pending"] is True
    ledger["now"] = ny("2026-10-14T04:00")                           # Fensterende 15:50 + 12 h = 03:50 ET
    update(ledger, "2026-10-14")
    ro = read_ledger(path)[0]["real_option"]
    assert ro["entry_pending"] is False and ro["entry_status"] == "UNUSABLE"
    assert ro["entry_unusable_reason"] == "NO_ADMISSIBLE_QUOTE_IN_SESSION" and ro["ask"] is None
    assert "entry_date" not in ro or ro.get("entry_date") is None
    vorher = (ledger["root"] / "2026-10.jsonl").read_bytes()
    ledger["now"] = ny("2026-10-14T11:00")
    update(ledger, "2026-10-14")
    assert (ledger["root"] / "2026-10.jsonl").read_bytes() == vorher


def test_G_altzeilen_ohne_feld_laufen_unveraendert_ueber_die_alten_fill_funktionen(ledger, net):
    """Altbestand: sofortiger Entry zur Live-Quote in jeder 'regulären' Session, Feld timing_policy bleibt fehlend –
    während die v2-Zeile daneben ausschließlich Archiv-Fills bekommt."""
    sig = ny("2026-10-13T09:40")
    alt = ledger_row("alt", sig, v2=False, option=LEGACY_OPT, legs=(LEGACY_L, LEGACY_S), ticker="MSFT")
    neu = ledger_row("neu", sig, v2=True)
    path = write_ledger(ledger["root"], [alt, neu])
    ledger["now"] = ny("2026-10-13T09:45")                           # reguläre Session, vor dem Mark-Fenster
    quotes = {LEGACY_L: {"bid": 6.0, "ask": 6.4, "mid": 6.2, "ts": None},
              LEGACY_S: {"bid": 2.0, "ask": 2.4, "mid": 2.2, "ts": None}}
    net.on_fetch = lambda s: {k: v for k, v in quotes.items() if k in s}
    update(ledger)
    zeilen = {r["signal_id"]: r for r in read_ledger(path)}
    a, n = zeilen["alt"], zeilen["neu"]
    assert "timing_policy" not in a and "entry_status" not in a["real_option"]
    assert a["real_option"]["entry_pending"] is False and a["real_option"]["ask"] == 6.4
    assert a["real_option"]["entry_filled_at"] == ts(ledger["now"]) and a["real_option"]["entry_date"] == "2026-10-13"
    assert a["real_strategy"]["net_debit_entry"] == pytest.approx(6.4 - 2.0)
    assert n["real_option"]["entry_pending"] is True and n["real_option"]["ask"] is None
    # die v2-Symbole wurden von den alten Fill-Funktionen nie abgefragt
    assert all(CALL not in c and LEG_L not in c and LEG_S not in c for c in net.fetch_calls)


def test_G_altzeile_ausserhalb_der_session_bleibt_wie_frueher_offen(ledger, net):
    alt = ledger_row("alt", ny("2026-10-13T17:00"), v2=False, option=LEGACY_OPT, legs=(LEGACY_L, LEGACY_S), ticker="MSFT")
    path = write_ledger(ledger["root"], [alt])
    ledger["now"] = ny("2026-10-13T18:00")
    net.on_fetch = lambda s: {LEGACY_OPT: {"bid": 6.0, "ask": 6.4, "mid": 6.2, "ts": None}}
    update(ledger)
    r = read_ledger(path)[0]
    assert r["real_option"]["entry_pending"] is True and "timing_policy" not in r and net.fetch_calls == []


def test_G_pending_entry_targets_und_due_mark_symbols_sind_nur_lesend(ledger, net):
    sig = ny("2026-10-13T11:00")
    filled = ledger_row("f", ny("2026-10-01T11:00"), v2=True, option=LEGACY_OPT, legs=(LEGACY_L, LEGACY_S), ticker="MSFT")
    filled["real_option"].update(entry_pending=False, entry_date="2026-10-01", bid=5.0, ask=5.4, mid=5.2)
    filled["real_strategy"].update(entry_pending=False, entry_date="2026-10-01")
    path = write_ledger(ledger["root"], [ledger_row("p", sig, v2=True), filled,
                                         ledger_row("alt", sig, v2=False, option=occ("TSLA", FAR, "C", 300.0),
                                                    legs=(occ("TSLA", FAR, "C", 300.0), occ("TSLA", FAR, "C", 310.0)),
                                                    ticker="TSLA")])
    vorher = path.read_bytes()
    targets = cl.pending_entry_targets("2026-10-13", root=ledger["root"])
    assert {k: ts(v) for k, v in targets.items()} == {CALL: ts(sig), LEG_L: ts(sig), LEG_S: ts(sig)}
    assert LEGACY_OPT not in targets and occ("TSLA", FAR, "C", 300.0) not in targets, "nur offene v2-Entries"
    due = cl.due_mark_symbols("2026-10-13", root=ledger["root"])
    assert LEGACY_OPT in due and LEGACY_L in due, "gefüllter v2-Entry mit fälligem 5d-Horizont"
    assert CALL not in due and LEG_L not in due, "offene v2-Entries werden nicht als Mark-Ziel geführt"
    assert path.read_bytes() == vorher and net.fetch_calls == []


def test_G_zwei_zeilen_gleiches_symbol_pending_target_ist_das_spaeteste_signal(ledger):
    s1, s2 = ny("2026-10-13T11:00"), ny("2026-10-13T14:00")
    write_ledger(ledger["root"], [ledger_row("a", s1, v2=True, spread=False), ledger_row("b", s2, v2=True, spread=False)])
    t = cl.pending_entry_targets("2026-10-13", root=ledger["root"])
    assert {k: ts(v) for k, v in t.items()} == {CALL: ts(s2)}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# H. scripts/rth_marks.trade_targets
# ══════════════════════════════════════════════════════════════════════════════════════════════
def v2_active(kind="call", *, decision, filled: bool, ticker="AAPL", **kw):
    t = make_trade(kind, decision=decision, ticker=ticker, **kw)
    t["timing"] = {"entry": {"status": "FILLED" if filled else "PENDING"}}
    return t


def test_H_mindestabrufzeit_ist_ny_tagesbeginn_bei_offenem_v2_entry_nicht_vor_der_entscheidung():
    now = ny("2026-10-13T12:00")
    day0 = ny("2026-10-13T00:00")
    dec = ny("2026-10-13T11:00")
    h = {"active_trades": [v2_active("call", decision=dec, filled=False, ticker="AAPL"),
                           v2_active("call", decision=dec, filled=True, ticker="MSFT"),
                           v2_active("call", decision=dec, filled=False, ticker="NVDA", policy=None),
                           v2_active("bull", decision=dec, filled=False, ticker="TSLA"),
                           make_trade("call", decision=dec, ticker="AMD")]}           # frisch gebucht, noch ohne 'timing'
    t = rth_marks.trade_targets(h, now)
    assert t[occ("AMD", FAR, "C", 200.0)] == dec, "v2 ohne Entry-Block gilt als offener Entry"
    assert t[occ("AAPL", FAR, "C", 200.0)] == dec, "offener v2-Entry: nie vor der Entscheidung"
    assert t[occ("MSFT", FAR, "C", 200.0)] == day0, "gefüllter Entry: Tagesbeginn"
    assert t[occ("NVDA", FAR, "C", 200.0)] == day0, "Altbestand v1: Tagesbeginn"
    assert t[occ("TSLA", FAR, "C", 200.0)] == dec and t[occ("TSLA", FAR, "C", 210.0)] == dec, "beide Spread-Legs"


def test_H_entscheidung_von_gestern_ergibt_heutigen_tagesbeginn():
    now = ny("2026-10-14T12:00")
    h = {"active_trades": [v2_active("call", decision=ny("2026-10-13T11:00"), filled=False)]}
    assert rth_marks.trade_targets(h, now)[CALL] == ny("2026-10-14T00:00")


def test_H_exit_pending_nicht_vor_close_time_und_counterfactual_gleich():
    now = ny("2026-10-13T12:00")
    cl_ = ny("2026-10-13T11:30")
    h = {"exit_pending_trades": [make_trade("call", decision=ny("2026-10-12T11:00"), close=cl_)],
         "counterfactual_exit_pending": [make_trade("put", decision=ny("2026-10-12T11:00"), close=cl_)],
         "counterfactual_trades": [v2_active("bear", decision=ny("2026-10-13T11:45"), filled=False, ticker="MSFT")]}
    t = rth_marks.trade_targets(h, now)
    assert t[syms("call")[0]] == cl_ and t[syms("put")[0]] == cl_
    assert t[syms("bear", root="MSFT")[0]] == ny("2026-10-13T11:45") and t[syms("bear", root="MSFT")[1]] == ny("2026-10-13T11:45")
    # Exit-Entscheid von gestern -> heutiger Tagesbeginn
    t2 = rth_marks.trade_targets(h, ny("2026-10-14T12:00"))
    assert t2[syms("call")[0]] == ny("2026-10-14T00:00")


def test_H_symbol_in_mehreren_listen_gewinnt_die_spaeteste_grenze():
    now = ny("2026-10-13T12:00")
    a = v2_active("call", decision=ny("2026-10-13T10:30"), filled=False)
    b = make_trade("call", decision=ny("2026-10-12T11:00"), close=ny("2026-10-13T11:30"))
    assert rth_marks.trade_targets({"active_trades": [a], "exit_pending_trades": [b]}, now)[CALL] == ny("2026-10-13T11:30")


@pytest.mark.parametrize("jetzt,tagesbeginn_utc", [
    ("2026-03-08T12:00", "2026-03-08T05:00"),     # US-Umstellungstag: Mitternacht noch EST (UTC-5)
    ("2026-03-09T12:00", "2026-03-09T04:00"),     # danach EDT
    ("2026-11-01T12:00", "2026-11-01T04:00"),     # US-Rückstellungstag: Mitternacht noch EDT
    ("2026-11-02T15:30", "2026-11-02T05:00"),     # danach EST
    ("2026-03-29T12:00", "2026-03-29T04:00"),     # EU-Umstellungstag: US unverändert EDT
    ("2026-10-25T12:00", "2026-10-25T04:00"),     # EU-Rückstellungstag: US unverändert EDT
])
def test_H_tagesbeginn_ist_new_york_mitternacht_auch_an_umstellungstagen(jetzt, tagesbeginn_utc):
    h = {"active_trades": [make_trade("call", decision=utc("2026-01-05T15:00"))]}
    h["active_trades"][0]["timing"] = {"entry": {"status": "FILLED"}}
    assert rth_marks.trade_targets(h, utc(jetzt))[CALL] == utc(tagesbeginn_utc)


def test_H_capture_fordert_quote_an_wenn_vorhandene_vor_der_entscheidung_liegt(net, tmp_path):
    """run(capture_only): eine zulässige Quote VOR dem Entscheid zählt nicht als 'bereits erfasst' (kein Fill möglich)."""
    dec = ny("2026-10-13T11:00")
    hp = tmp_path / "history.json"
    hp.write_text(json.dumps({"active_trades": [v2_active("call", decision=dec, filled=False)]}), encoding="utf-8")
    put(CALL, ny("2026-10-13T10:20"), 1.0, 1.2)                      # zulässig, aber VOR dem Entscheid
    now = ny("2026-10-13T11:10")
    res = rth_marks.run(now, hp, capture_only=True)
    assert res["window"] == "IN_WINDOW" and res["already_captured"] == 0 and res["requested"] == 1
    assert net.fetch_calls == [[CALL]]
    put(CALL, ny("2026-10-13T11:05"), 1.0, 1.2)                      # nach dem Entscheid: erfasst
    net.fetch_calls.clear()
    res = rth_marks.run(now, hp, capture_only=True)
    assert res["already_captured"] == 1 and res["requested"] == 0 and net.fetch_calls == []


def test_H_capture_ausserhalb_des_fensters_tut_nichts(net, tmp_path):
    hp = tmp_path / "history.json"
    hp.write_text(json.dumps({"active_trades": [v2_active("call", decision=D0, filled=False)]}), encoding="utf-8")
    for jetzt in ("2026-10-13T16:30", "2026-10-13T09:45", "2026-10-17T15:00", "2026-11-26T15:00"):
        res = rth_marks.run(ny(jetzt), hp, capture_only=True)
        assert res["skipped"] is True
    assert net.fetch_calls == []


# ══════════════════════════════════════════════════════════════════════════════════════════════
# I. evidence_ledger.scorecard
# ══════════════════════════════════════════════════════════════════════════════════════════════
def ev_rec(pop, pid, day, net_, policy, *, h=None, **meta):
    m = dict(meta)
    if policy is not None:
        m["timing_policy"] = policy
    return el._rec(prediction_id=pid, source_store="test", population=pop, entity=pid, decision_time=day, direction=1,
                   features={}, horizon=h, horizon_unit="calendar", outcome=net_ + 0.013, outcome_net=net_, cost=0.013,
                   outcome_method="option_quote", price_type="OBSERVED_QUOTE", quality="RELIABLE",
                   label_available_at=day, meta=m)


def test_I_scorecard_zellen_je_timing_policy_getrennt_altbestand_zaehlt_als_v1():
    recs = [ev_rec("CHAMPION_TRADE", "a", "2026-09-01", 0.10, None),               # Altbestand ohne Feld
            ev_rec("CHAMPION_TRADE", "b", "2026-09-02", 0.20, "timing-v1"),
            ev_rec("CHAMPION_TRADE", "c", "2026-09-03", -0.40, "timing-v2"),
            ev_rec("CHAMPION_TRADE", "d", "2026-09-04", -0.20, "timing-v2"),
            ev_rec("CANDIDATE_OPTION", "e", "2026-09-05", 0.05, None, h=5),
            ev_rec("CANDIDATE_OPTION", "f", "2026-09-06", 0.30, "timing-v2", h=5)]
    sc = el.scorecard(recs, as_of="2026-12-31")
    c = sc["cells"]
    assert c["CHAMPION_TRADE|ALL"]["n"] == 4
    assert c["CHAMPION_TRADE|timing=timing-v1"]["n"] == 2 and c["CHAMPION_TRADE|timing=timing-v2"]["n"] == 2
    assert c["CHAMPION_TRADE|timing=timing-v1"]["mean_net"] == pytest.approx(0.15)
    assert c["CHAMPION_TRADE|timing=timing-v2"]["mean_net"] == pytest.approx(-0.30)
    assert c["CANDIDATE_OPTION_5d|timing=timing-v1"]["n"] == 1 and c["CANDIDATE_OPTION_5d|timing=timing-v2"]["n"] == 1
    assert c["CANDIDATE_OPTION_5d|timing=timing-v1"]["mean_net"] == pytest.approx(0.05)
    assert not any(k.endswith("timing=None") for k in c)


def test_I_nur_v1_daten_erzeugen_keine_v2_zelle_und_umgekehrt():
    sc = el.scorecard([ev_rec("CHAMPION_TRADE", "a", "2026-09-01", 0.1, None)], as_of="2026-12-31")
    assert "CHAMPION_TRADE|timing=timing-v1" in sc["cells"] and "CHAMPION_TRADE|timing=timing-v2" not in sc["cells"]
    sc = el.scorecard([ev_rec("CHAMPION_TRADE", "a", "2026-09-01", 0.1, "timing-v2")], as_of="2026-12-31")
    assert "CHAMPION_TRADE|timing=timing-v2" in sc["cells"] and "CHAMPION_TRADE|timing=timing-v1" not in sc["cells"]


def test_I_kandidaten_meta_und_champion_adapter_uebernehmen_die_policy():
    assert el._candidate_meta({"signal_id": "x", "status": "proposed"})["timing_policy"] == "timing-v1"
    assert el._candidate_meta({"signal_id": "x", "timing_policy": "timing-v2"})["timing_policy"] == "timing-v2"
    base = {"ticker": "AAPL", "entry_date": "2026-09-01", "close_date": "2026-09-08", "strategy": "LONG_CALL",
            "entry_debit": 2.0, "option": {"strike": 200, "expiry": FAR, "ask": 2.0}, "outcome": 0.5,
            "outcome_method": "option_quote", "outcome_price_type": "OBSERVED_QUOTE", "outcome_reliable": True,
            "features": {}, "deep_analysis": {"direction": "BULLISH"}}
    alt, neu = dict(base), dict(base, ticker="MSFT", timing_policy="timing-v2", timing={"entry": {"price": 2.0}})
    got = {r["entity"]: r["meta"]["timing_policy"] for r in el.from_champion_trades(history={"closed_trades": [alt, neu]})}
    assert got == {"AAPL": "timing-v1", "MSFT": "timing-v2"}


def test_I_label_available_at_nicht_vor_dem_exit_fill_bei_timing_v2():
    t = {"ticker": "AAPL", "entry_date": "2026-10-13", "close_date": "2026-10-16", "strategy": "LONG_CALL",
         "entry_debit": 2.0, "option": {"strike": 200, "expiry": FAR, "ask": 2.0}, "outcome": 0.5,
         "outcome_method": "option_quote", "outcome_price_type": "OBSERVED_QUOTE", "outcome_reliable": True,
         "timing_policy": "timing-v2", "close_time": "2026-10-17T01:00:00+00:00",
         "exit_fill_time": "2026-10-19T14:20:00+00:00", "features": {}, "deep_analysis": {"direction": "BULLISH"},
         "timing": {"entry": {"price": 2.0}, "status": "OK"}}
    (r,) = el.from_champion_trades(history={"closed_trades": [t]})
    # das Outcome ist erst am Montag 2026-10-19 (Exit-Fill) bekannt -> frühestens ab dem Folgetag als Label verfügbar
    assert r["label_available_at"] >= "2026-10-20"
    assert not el.labeled([r], as_of="2026-10-18", min_quality=("RELIABLE",))


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Property-Tests über zufällige Zeiten (Orakel unabhängig vom Code unter Test, nur Kalender aus market_snapshot)
# ══════════════════════════════════════════════════════════════════════════════════════════════
LATE = {"SPY", "SPX", "SPXW"}
PROP_EXPIRY = "2029-01-19"          # Freitag, weit hinter allen zufälligen Zeitpunkten (Ziel: kein Verfall im Spiel)


def o_window(d: date, root: str):
    """Mark-Fenster [Eröffnung+30 min, Schluss−10 min] (UTC) oder None – unabhängig von timing_policy/option_quotes."""
    if not ms.is_trading_day(d):
        return None
    schluss = ms.market_close_time(d).strftime("%H:%M")
    end = ny(f"{d.isoformat()}T{schluss}") + timedelta(minutes=(15 if root in LATE else 0) - 10)
    return ny(f"{d.isoformat()}T10:00"), end


def o_in_window(t: datetime, root: str) -> bool:
    w = o_window(t.astimezone(NY).date(), root)
    return bool(w and w[0] <= t <= w[1])


def o_fill_day(t: datetime, root: str) -> date:
    d = t.astimezone(NY).date()
    for _ in range(20):
        w = o_window(d, root)
        if w and t <= w[1]:
            return d
        d += timedelta(days=1)
    raise AssertionError("kein Handelstag gefunden")


class Q:
    def __init__(self, sym, at, bid, ask, kind):
        self.sym, self.at, self.bid, self.ask, self.kind = sym, at, bid, ask, kind

    def eligible(self, root) -> bool:
        return self.kind == "ok" and o_in_window(self.at, root)


def random_decision(rng: random.Random, root: str) -> datetime:
    d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 910))
    if rng.random() < 0.5:
        return (datetime.combine(d, datetime.min.time(), tzinfo=UTC) + timedelta(seconds=rng.randint(0, 86399)))
    w = o_window(d, root) or (ny(f"{d.isoformat()}T10:00"), ny(f"{d.isoformat()}T15:50"))
    anchor = rng.choice(w)
    return anchor + timedelta(seconds=rng.randint(-1800, 1800))


def random_quotes(rng: random.Random, symbols: list[str], dec: datetime, root: str) -> list[Q]:
    out, seen = [], set()
    for _ in range(rng.randint(0, 9)):
        m = rng.random()
        if m < 0.45:                                   # gezielt im Fenster des frühesten Fill-Tages (bzw. des Folgetags)
            d = o_fill_day(dec, root) + timedelta(days=rng.choice([0, 0, 0, 1]))
            w = o_window(d, root)
            if not w:
                continue
            at = w[0] + timedelta(seconds=rng.randint(-120, int((w[1] - w[0]).total_seconds()) + 120))
        elif m < 0.75:
            at = dec + timedelta(seconds=rng.randint(-2400, 2400))
        else:
            at = dec + timedelta(seconds=rng.randint(-7200, 3 * 86400))
        kind = rng.choices(["ok", "crossed", "noask", "zerobid", "stale", "postqt"], [70, 6, 6, 6, 6, 6])[0]
        for i, sym in enumerate(symbols):
            t_leg = at if i == 0 or rng.random() < 0.5 else at + timedelta(seconds=rng.randint(-1200, 1200))
            if (sym, int(t_leg.timestamp())) in seen:
                continue
            seen.add((sym, int(t_leg.timestamp())))
            base = rng.uniform(3.0, 6.0) if i == 0 else rng.uniform(0.5, 2.5)
            bid, ask = round(base, 2), round(base + rng.uniform(0.02, 0.5), 2)
            out.append(Q(sym, t_leg, bid, ask, kind if i == 0 or rng.random() < 0.8 else "ok"))
    return out


def archive_q(q: Q) -> None:
    kw = {}
    bid, ask = q.bid, q.ask
    if q.kind == "crossed":
        bid, ask = ask, bid
    elif q.kind == "noask":
        ask = None
    elif q.kind == "zerobid":
        bid = 0
    elif q.kind == "stale":
        kw["quote_at"] = q.at - timedelta(hours=30)
    elif q.kind == "postqt":
        kw["quote_at"] = ny(f"{(q.at.astimezone(NY).date() - timedelta(days=1)).isoformat()}T17:00")
    put(q.sym, q.at, bid, ask, **kw)


def o_first_fill(quotes: list[Q], symbols: list[str], after: datetime, now: datetime, root: str):
    """Frühester zulässiger Fill (Liste Q je Symbol) in der frühesten möglichen Sitzung oder None."""
    day = o_fill_day(after, root)
    by = {s: sorted([q for q in quotes if q.sym == s and q.eligible(root) and after <= q.at <= now
                     and q.at.astimezone(NY).date() == day], key=lambda q: q.at) for s in symbols}
    if not all(by.values()):
        return day, None
    if len(symbols) == 1:
        return day, [by[symbols[0]][0]]
    for q0 in by[symbols[0]]:
        m = next((q for q in by[symbols[1]] if abs((q.at - q0.at).total_seconds()) <= 900), None)
        if m:
            return day, [q0, m]
    return day, None


def _scenario(rng, i, kind, root, *, with_close):
    strikes = (300.0 + i, (310.0 + i) if KINDS[kind][1] == "C" else (290.0 + i))
    symbols = syms(kind, PROP_EXPIRY, root, strikes)
    dec = random_decision(rng, root).replace(microsecond=0)
    quotes = random_quotes(rng, symbols, dec, root)
    for q in quotes:
        archive_q(q)
    return strikes, symbols, dec, quotes


@pytest.mark.parametrize("seed", [11, 22, 33])
def test_property_entry_nie_rueckwirkend_und_nur_aus_der_fruehesten_zulaessigen_quote(seed):
    rng = random.Random(seed)
    n_filled = n_pending = n_unusable = 0
    for i in range(90):
        kind = rng.choice(["call", "put", "bull", "bear"])
        root = rng.choice(["AAPL", "AAPL", "SPY"])
        strikes, symbols, dec, quotes = _scenario(rng, i, kind, root, with_close=False)
        t = make_trade(kind, decision=dec, ticker=root, strikes=strikes, expiry=PROP_EXPIRY)
        now = dec + timedelta(seconds=rng.choice([0, 3600, rng.randint(0, 6 * 86400), rng.randint(0, 6 * 86400),
                                                  rng.randint(86400, 6 * 86400)]))
        blk, recs = tp.evaluate_entry(t, now)
        after = dec - timedelta(minutes=15) if o_in_window(dec, root) else dec
        day, fill = o_first_fill(quotes, symbols, after, now, root)
        ctx = f"seed={seed} i={i} {kind} {root} dec={dec.isoformat()} now={now.isoformat()}"
        if fill is not None:
            price = fill[0].ask if len(fill) == 1 else round(fill[0].ask - fill[1].bid, 4)
            if price <= 0:
                assert blk["status"] == "UNUSABLE" and "price" not in blk, ctx
                continue
            assert blk["status"] == "FILLED", ctx
            n_filled += 1
            assert blk["price"] == pytest.approx(price), ctx
            assert [tp._ts(r["retrieved_at"]) for r in recs] == [q.at for q in fill], ctx
            # keine rückwirkende Ausführung und keine Quote vor der jeweiligen Grenze
            fill_time = tp._ts(blk["time"])
            assert fill_time >= dec and fill_time <= now, ctx
            assert fill_time == max(dec, max(q.at for q in fill)), ctx
            for r in recs:
                ra = tp._ts(r["retrieved_at"])
                assert ra >= after and ra <= now, ctx
                assert ra >= dec - timedelta(minutes=15) and (o_in_window(dec, root) or ra >= dec), ctx
                assert ra.astimezone(NY).date() == day, "Fill ausschließlich in der frühesten möglichen Sitzung"
                assert oq.usable_rth_quote(r), ctx
            if len(recs) == 2:
                assert abs((tp._ts(recs[0]["retrieved_at"]) - tp._ts(recs[1]["retrieved_at"])).total_seconds()) <= 900
            # Stabilität (Point-in-Time): ein späteres 'now' ändert einen bestehenden Fill nicht
            blk2, recs2 = tp.evaluate_entry(t, now + timedelta(seconds=rng.randint(1, 5 * 86400)))
            assert (blk2["status"], blk2["time"], blk2["price"]) == ("FILLED", blk["time"], blk["price"]), ctx
        else:
            assert blk["status"] in ("PENDING", "UNUSABLE") and recs == [] and "price" not in blk, ctx
            w = o_window(day, root)
            erwartet = "UNUSABLE" if now > w[1] + timedelta(hours=12) else "PENDING"
            if abs((now - (w[1] + timedelta(hours=12))).total_seconds()) > 1:
                assert blk["status"] == erwartet, ctx
            n_pending += blk["status"] == "PENDING"
            n_unusable += blk["status"] == "UNUSABLE"
    assert n_filled >= 10 and n_pending >= 5 and n_unusable >= 5, (n_filled, n_pending, n_unusable)


@pytest.mark.parametrize("seed", [44, 55, 66])
def test_property_exit_nie_rueckwirkend_und_bid_preis(seed):
    rng = random.Random(seed)
    n_filled = 0
    for i in range(90):
        kind = rng.choice(["call", "put", "bull", "bear"])
        root = rng.choice(["AAPL", "AAPL", "SPY"])
        strikes, symbols, close, quotes = _scenario(rng, i, kind, root, with_close=True)
        dec = close - timedelta(days=2)
        t = make_trade(kind, decision=dec, close=close, ticker=root, strikes=strikes, expiry=PROP_EXPIRY)
        now = close + timedelta(seconds=rng.choice([0, 3600, rng.randint(0, 6 * 86400), rng.randint(0, 6 * 86400),
                                                    rng.randint(86400, 6 * 86400)]))
        # Entry-Seite neutralisieren: nur der Exit-Teil wird verglichen
        blk, recs = tp.evaluate_exit(t, now)
        after = close - timedelta(minutes=5) if o_in_window(close, root) else close
        day, fill = o_first_fill(quotes, symbols, after, now, root)
        ctx = f"seed={seed} i={i} {kind} {root} close={close.isoformat()} now={now.isoformat()}"
        if fill is not None:
            price = fill[0].bid if len(fill) == 1 else round(max(fill[0].bid - fill[1].ask, 0.0), 4)
            assert blk["status"] == "FILLED" and blk["price"] == pytest.approx(price), ctx
            n_filled += 1
            ft = tp._ts(blk["time"])
            assert close <= ft <= now and ft == max(close, max(q.at for q in fill)), ctx
            for r in recs:
                ra = tp._ts(r["retrieved_at"])
                assert after <= ra <= now and ra.astimezone(NY).date() == day, ctx
                assert ra >= close - timedelta(minutes=5) and (o_in_window(close, root) or ra >= close), ctx
        else:
            assert blk["status"] in ("PENDING", "UNUSABLE") and recs == [] and "price" not in blk, ctx
    assert n_filled >= 10


@pytest.mark.parametrize("seed", [77, 88])
def test_property_ohne_optionsquote_nie_ok_und_nie_ein_basiswertpreis(seed, net):
    """Kein Archiv-Eintrag: weder Entry noch Exit noch Outcome; close_fn wird nur beim Settlement abgerufen."""
    rng = random.Random(seed)
    for i in range(120):
        kind = rng.choice(["call", "put", "bull", "bear"])
        dec = random_decision(rng, "AAPL").replace(microsecond=0)
        close = dec + timedelta(seconds=rng.randint(0, 5 * 86400))
        now = close + timedelta(seconds=rng.randint(0, 20 * 86400))
        verfall = (date(2025, 1, 1) + timedelta(days=rng.randint(0, 960))).isoformat()
        t = make_trade(kind, decision=dec, close=close, expiry=verfall)
        blk, e, x = ev(t, now, close_fn=lambda tk, d: 123.45)
        assert e == [] and x == []
        assert not (blk["status"] == "OK" and blk["exit"]["status"] != "SETTLED")
        if blk["status"] == "OK":                       # nur über das Settlement – und das setzt einen gefüllten Entry voraus
            raise AssertionError(f"Entry ohne Optionsquote darf nie gefüllt sein: {blk}")
        assert blk["outcome"] is None


@pytest.mark.parametrize("seed", [101, 202])
def test_property_verfall_settlement_nur_aus_dem_offiziellen_schluss_und_nie_vor_der_ausfuehrung(seed):
    """Zufällige Verfallstage/Exit-Entscheide ohne Quotes: Orakel nach Spezifikation D (früheste Ausführungs-Sitzung
    nach dem Verfall oder am Verfallstag nach Fensterende -> Settlement aus dem unbereinigten Schluss)."""
    rng = random.Random(seed)
    n_settled = n_pending = 0
    for i in range(160):
        kind = rng.choice(["call", "put", "bull", "bear"])
        root = rng.choice(["AAPL", "AAPL", "SPY"])
        d = date(2025, 1, 1) + timedelta(days=rng.randint(0, 900))
        w = o_window(d, root)
        if w is None:
            continue
        exp = d.isoformat()
        schluss_close = rng.uniform(150.0, 260.0)
        strikes = (300.0 + i, (310.0 + i) if KINDS[kind][1] == "C" else (290.0 + i))
        # Exit-Entscheid: vor, am und nach dem Verfallstag
        close = w[0] + timedelta(seconds=rng.randint(-3 * 86400, 3 * 86400))
        # Settlement erst nach dem offiziellen Schluss des Basiswerts + Karenz (tp.SETTLEMENT_CLOSE_DELAY)
        official_close = ny(f"{exp}T{ms.market_close_time(d).strftime('%H:%M')}") + tp.SETTLEMENT_CLOSE_DELAY
        now = close + timedelta(seconds=rng.randint(0, 9 * 86400))
        if w[1] < now < official_close:
            continue                                   # Fensterende bis Schluss + Karenz: eigene Tests (PENDING)
        t = make_trade(kind, decision=close - timedelta(days=3), close=close, ticker=root, strikes=strikes, expiry=exp)
        calls = []

        def cf(tk, day, _v=schluss_close):
            calls.append((tk, day))
            return _v
        blk, recs = tp.evaluate_exit(t, now, close_fn=cf)
        after = close - timedelta(minutes=5) if o_in_window(close, root) else close
        ds = o_fill_day(after, root)
        settle = ds > d or (ds == d and now > w[1])
        ctx = f"seed={seed} i={i} {kind} {root} exp={exp} close={close.isoformat()} now={now.isoformat()}"
        assert recs == [], ctx
        if settle:
            assert calls and all(c == (root, d) for c in calls), ctx
            assert blk["status"] == "SETTLED" and blk.get("settlement") is True, ctx
            _, cp, lk, sk = KINDS[kind]
            iv = (lambda k: max(k - schluss_close, 0.0)) if cp == "P" else (lambda k: max(schluss_close - k, 0.0))
            erwartet = iv(strikes[0]) if sk is None else max(iv(strikes[0]) - iv(strikes[1]), 0.0)
            assert blk["price"] == pytest.approx(erwartet, abs=1e-4), ctx
            n_settled += 1
        else:
            assert calls == [], f"vor der Ausführungs-Sitzung am Verfallstag kein Settlement: {ctx}"
            assert blk["status"] in ("PENDING", "UNUSABLE") and "price" not in blk, ctx
            n_pending += 1
    assert n_settled >= 15 and n_pending >= 10, (n_settled, n_pending)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Ergänzungen: verzögerte Läufe um UTC-Mitternacht, Informationsblock aktiver Trades, Kosten, Monitoring
# ══════════════════════════════════════════════════════════════════════════════════════════════
def test_F_verspaeteter_lauf_nach_utc_mitternacht_freitagabend_wird_montag(monkeypatch, tmp_path):
    """Feedback-Lauf 00:30 UTC am Samstag = Freitag 20:30 ET (Handelstag!): Exit-Entscheid, Ausführung erst Montag."""
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)
    t = champion(decision=ny("2026-10-13T11:00"))
    hist = {"active_trades": [t], "closed_trades": []}
    put(CALL, ny("2026-10-16T15:45"), 8.0, 8.4)                        # Freitagsquote VOR dem Entscheid: nie
    c = run_main(monkeypatch, hist, utc("2026-10-17T00:30"))
    assert "AAPL" in c["compute"] and c["price"], "NY-Datum Freitag = Handelstag -> kursbasierte Prüfung läuft"
    assert hist["closed_trades"] == [] and len(hist["exit_pending_trades"]) == 1
    assert hist["exit_pending_trades"][0]["close_time"] == "2026-10-17T00:30:00+00:00"
    run_main(monkeypatch, hist, ny("2026-10-17T12:00"), exit_reason=None)          # Samstag: nichts passiert
    run_main(monkeypatch, hist, ny("2026-10-18T22:00"), exit_reason=None)          # Sonntagabend (UTC Montag 02:00)
    assert hist["closed_trades"] == [] and len(hist["exit_pending_trades"]) == 1
    put(CALL, ny("2026-10-19T10:05"), 3.0, 3.4)
    run_main(monkeypatch, hist, ny("2026-10-19T22:30"), exit_reason=None)
    assert len(hist["closed_trades"]) == 1
    assert hist["closed_trades"][0]["outcome"] == pytest.approx(3.0 / 2.0 - 1, abs=1e-4)
    assert hist["closed_trades"][0]["timing"]["exit"]["day"] == "2026-10-19"


def test_F_aktiver_v2_trade_ohne_exit_bleibt_aktiv_mit_informativem_entry_block(monkeypatch, tmp_path):
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)
    t = champion(decision=ny("2026-10-13T11:00"))
    hist = {"active_trades": [t], "closed_trades": []}
    c = run_main(monkeypatch, hist, ny("2026-10-14T14:00"), outcome=0.1, exit_reason=None)
    assert hist["active_trades"] == [t] and hist["closed_trades"] == [] and c["learn"] == []
    assert t["timing"]["entry"]["status"] == "FILLED" and t["timing"]["entry"]["price"] == 2.0
    assert t["timing"]["status"] == "PENDING" and t["timing"]["outcome"] is None
    assert "close_time" not in t and t.get("exit_fill_time") is None


def test_F_ok_schliessung_bewertung_und_kosten_basieren_auf_dem_fill_preis(monkeypatch, tmp_path):
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)
    put(CALL, ny("2026-10-14T13:58"), 4.0, 4.4)
    t = champion(decision=ny("2026-10-13T11:00"), entry_debit=7.77)              # abweichender Champion-Debit (Chain-Quote)
    hist = {"active_trades": [t], "closed_trades": []}
    run_main(monkeypatch, hist, ny("2026-10-14T14:00"))
    got = hist["closed_trades"][0]
    assert got["outcome"] == pytest.approx(4.0 / 2.0 - 1), "Entry-Preis = Fill-Ask (2.0), nicht entry_debit der Entscheidung"
    ev_ = got["outcome_evaluation"]
    assert ev_["status"] == "OK" and ev_["rth_executable"] is True and ev_["reproducible"] is True
    commission = float(feedback.cfg.costs.commission_per_contract)
    kosten = 2 * commission / (2.0 * 100)
    assert got["outcome_net"] == pytest.approx(round(got["outcome"] - kosten, 6), abs=1e-5)
    assert outcomes_file(tmp_path)[0]["entry_debit"] == 2.0


def test_F_exit_pending_nur_in_pending_liste_nie_gleichzeitig_geschlossen(monkeypatch, tmp_path):
    put(CALL, ny("2026-10-13T11:05"), 1.8, 2.0)
    t = champion(decision=ny("2026-10-13T11:00"))
    hist = {"active_trades": [t], "closed_trades": []}
    run_main(monkeypatch, hist, ny("2026-10-14T17:30"))
    ids = lambda key: [x["ticker"] for x in hist.get(key, [])]                    # noqa: E731
    assert ids("active_trades") == [] and ids("exit_pending_trades") == ["AAPL"] and ids("closed_trades") == []
    put(CALL, ny("2026-10-15T10:20"), 3.0, 3.4)
    run_main(monkeypatch, hist, ny("2026-10-15T11:00"), exit_reason=None)
    assert ids("active_trades") == [] and ids("exit_pending_trades") == [] and ids("closed_trades") == ["AAPL"]
    assert ids("closed_trades_unusable") == []


def test_monitoring_timing_status_trennt_die_policies():
    from modules import ops_monitor
    hist = {"closed_trades": [{"ticker": "A", "outcome": 0.2, "outcome_at_decision": 0.5, "timing_policy": "timing-v2"},
                              {"ticker": "B", "outcome": 0.1}],
            "exit_pending_trades": [{"ticker": "C", "timing_policy": "timing-v2", "close_time": "2026-10-13T20:00:00+00:00",
                                     "timing": {"reasons": ["EXIT_AWAITING_SESSION"]}}],
            "closed_trades_unusable": [{"ticker": "D", "timing_policy": "timing-v2",
                                        "timing": {"reasons": ["EXIT_NO_ADMISSIBLE_QUOTE_IN_SESSION"]}}],
            "active_trades": [{"ticker": "E", "timing_policy": "timing-v2", "timing": {"entry": {"status": "PENDING"}}},
                              {"ticker": "F"}]}
    r = ops_monitor.timing_status(hist, utc("2026-10-15T20:00"))
    assert r["current_policy"] == "timing-v2"
    assert r["by_policy"]["closed"] == {"timing-v2": 1, "timing-v1": 1}
    assert r["by_policy"]["active"] == {"timing-v2": 1, "timing-v1": 1}
    assert r["v2_closed"] == 1 and r["v2_unusable"] == 1 and r["v2_exit_pending"] == 1
    assert r["v2_exit_pending_oldest_days"] == pytest.approx(2.0, abs=0.01)
    assert r["v2_deviation_vs_decision"]["n"] == 1 and r["v2_deviation_vs_decision"]["mean"] == pytest.approx(-0.3)
    assert r["v2_unusable_reasons"] == {"EXIT_NO_ADMISSIBLE_QUOTE_IN_SESSION": 1}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Weitere Grenzfälle
# ══════════════════════════════════════════════════════════════════════════════════════════════
def test_D_exit_donnerstagabend_vor_verfall_fill_tag_ist_der_verfallstag(net):
    """Entscheid Donnerstag 16:30 ET (nach Fensterende), Verfall Freitag: früheste Sitzung = Verfallstag selbst."""
    t = settle_trade("call", close=ny("2026-10-15T16:30"))
    s = syms("call", EXP)[0]
    net.closes[("AAPL", date(2026, 10, 16))] = 230.0
    assert ev(t, ny("2026-10-16T09:00"))[0]["status"] == "PENDING"
    put(s, ny("2026-10-16T10:30"), 9.0, 9.4)
    blk, _, x = ev(t, ny("2026-10-16T10:45"))
    assert blk["exit"]["status"] == "FILLED" and blk["exit"]["price"] == 9.0 and net.close_calls == [] and len(x) == 1
    # ohne Quote am Verfallstag: nach Fensterende + offiziellem Schluss Settlement (anderer Ticker, eigene Entry-Quote)
    t2 = trade_with_entry("call", close=ny("2026-10-15T16:31"), expiry=EXP, ticker="MSFT")
    net.closes[("MSFT", date(2026, 10, 16))] = 230.0
    blk, _, _ = ev(t2, ny("2026-10-16T17:05"))                      # nach offiziellem Schluss + 1 h Karenz
    assert blk["exit"]["status"] == "SETTLED" and blk["exit"]["price"] == 30.0


def test_C_zehn_minuten_alte_quote_bei_signal_im_fenster_wird_genutzt_entry_zeit_ist_das_signal(ledger, net):
    sig = ny("2026-10-13T11:00")
    path = write_ledger(ledger["root"], [ledger_row("s1", sig, v2=True, spread=False)])
    put(CALL, ny("2026-10-13T10:50"), 3.0, 3.2)
    ledger["now"] = ny("2026-10-13T11:02")
    update(ledger)
    ro = read_ledger(path)[0]["real_option"]
    assert ro["entry_pending"] is False and ro["ask"] == 3.2
    assert ro["entry_filled_at"] == ts(sig), "Entry-Zeit nie vor dem Signal, auch wenn die Quote 10 min älter ist"
    assert ro["entry_date"] == "2026-10-13"


def test_E_evaluate_mit_naiver_zeit_gilt_als_utc():
    put(CALL, D0 + timedelta(minutes=2), 1.8, 2.0)
    t = make_trade(decision=D0)
    naiv = (D0 + timedelta(hours=1)).replace(tzinfo=None)
    assert ev(t, naiv)[0]["entry"] == ev(t, D0 + timedelta(hours=1))[0]["entry"]


def test_E_verfallene_option_mit_entry_nach_dem_verfall_wird_nie_zu_ok(net):
    """Entscheid Freitag 16:30 (Verfallstag, nach Schluss): Entry-Fill-Tag Montag liegt nach Verfall -> UNUSABLE, auch wenn
    Exit-Settlement möglich wäre – ein nie ausgeführter Entry hat kein Outcome."""
    exp = "2026-10-16"
    s = syms("call", exp)[0]
    put(s, ny("2026-10-19T10:30"), 0.0 + 0.5, 0.6)
    net.closes[("AAPL", date(2026, 10, 16))] = 230.0
    t = make_trade("call", decision=ny("2026-10-16T16:30"), close=ny("2026-10-16T17:00"), expiry=exp)
    blk, _, _ = ev(t, ny("2026-10-20T12:00"))
    assert blk["status"] == "UNUSABLE" and blk["outcome"] is None
    assert any(r.startswith("ENTRY_") for r in blk["reasons"])


def test_C_uebernommene_vortagesquote_ist_kein_fill_boersenzeit_muss_am_fill_tag_liegen():
    """Entscheid gestern nach Schluss; heute 10:05 abgerufen, aber seit gestern 15:30 unverändert (Börsenzeit gestern, nicht
    stale): das ist der Preis VOR der Entscheidung -> kein Fill. Erst die echte Quote von heute zählt."""
    dec = ny("2026-10-13T17:00")
    put(CALL, ny("2026-10-14T10:05"), 5.0, 5.4, quote_at=ny("2026-10-13T15:30"))
    assert oq.usable_rth_quote(oq.latest_quote(CALL, ny("2026-10-14T10:30"))), "als Mark-Quote formal zulässig"
    blk, recs = entry(decision=dec, now=ny("2026-10-14T10:30"))
    assert blk["status"] == "PENDING" and recs == []
    put(CALL, ny("2026-10-14T10:20"), 4.0, 4.2)
    blk, _ = entry(decision=dec, now=ny("2026-10-14T10:30"))
    assert blk["status"] == "FILLED" and blk["price"] == 4.2 and blk["time"] == ts(ny("2026-10-14T10:20"))


def test_D_uebernommene_vortagesquote_ist_kein_exit_fill():
    close = ny("2026-10-14T17:00")
    t = trade_with_entry(close=close)
    put(CALL, ny("2026-10-15T10:05"), 5.0, 5.4, quote_at=ny("2026-10-14T15:30"))
    assert ev(t, ny("2026-10-15T10:30"))[0]["status"] == "PENDING"
    put(CALL, ny("2026-10-15T10:20"), 4.0, 4.2)
    blk, _, _ = ev(t, ny("2026-10-15T10:30"))
    assert blk["status"] == "OK" and blk["exit"]["price"] == 4.0


def test_C_policy_parameter_aus_der_config_gelten_auch_ohne_code_defaults(policy_cfg):
    """Spezifikationswerte stehen in der Config (15/5/12/15/5); die Auswertung folgt der Config, nicht festen Konstanten."""
    policy_cfg("current: timing-v2\npolicies:\n  timing-v2:\n    entry: {decision_quote_max_age_minutes: 3}\n"
               "    fill_session_grace_hours: 1\n")
    put(CALL, D0 - timedelta(minutes=5), 1.0, 1.4)
    blk, _ = entry(decision=D0, now=D0 + timedelta(minutes=1))
    assert blk["status"] == "PENDING", "5 min alte Quote ist bei 3-min-Toleranz nicht zulässig"
    assert tp.evaluate_entry(make_trade(decision=D0), ny("2026-10-13T16:55"))[0]["status"] == "UNUSABLE"


def test_F_settlement_outcome_record_nennt_keine_exit_quote(monkeypatch, tmp_path, net):
    exp = "2026-10-16"
    s = syms("call", exp)[0]
    put(s, ny("2026-10-13T11:05"), 1.8, 2.0)
    put(s, ny("2026-10-16T15:00"), 9.0, 9.4)                          # irgendeine Quote vom Verfallstag (nicht die Exit-Basis)
    FakeDT.now_value = ny("2026-10-16T17:00")
    monkeypatch.setattr(feedback, "datetime", FakeDT)                 # 'Jetzt' des Laufs = Bewertungszeit (deterministisch)
    t = champion(decision=ny("2026-10-13T11:00"), close=ny("2026-10-16T16:30"), expiry=exp)
    net.closes[("AAPL", date(2026, 10, 16))] = 230.0
    hist = {"exit_pending_trades": [t], "closed_trades": []}
    feedback.resolve_pending_exits(hist, FakeDT.now_value)
    rec_ = outcomes_file(tmp_path)[0]
    assert rec_["price_type"] == "SETTLEMENT_INTRINSIC"
    assert [x for x in rec_["exit_quote_record_ids"] if x] == [], "ein Settlement beruht auf keiner Exit-Quote"



def test_G_v2_zeile_ohne_kontrakt_oder_signalzeit_wird_unusable_nie_mit_ersatzpreis(ledger, net):
    sig = ny("2026-10-13T11:00")
    ohne_symbol = ledger_row("a", sig, v2=True)
    ohne_symbol["real_option"]["symbol"] = None
    ohne_symbol["real_strategy"]["legs"][1]["symbol"] = None
    ohne_zeit = ledger_row("b", sig, v2=True, option=occ("TSLA", FAR, "C", 300.0), ticker="TSLA", spread=False)
    ohne_zeit["signal_timestamp"] = None
    path = write_ledger(ledger["root"], [ohne_symbol, ohne_zeit])
    put(CALL, ny("2026-10-13T11:02"), 1.0, 1.2)                      # Quote existiert, darf aber nicht 'geraten' werden
    ledger["now"] = ny("2026-10-13T11:10")
    update(ledger)
    a, b = read_ledger(path)
    assert a["real_option"]["entry_status"] == "UNUSABLE" and a["real_option"]["entry_unusable_reason"] == "CONTRACT_UNKNOWN"
    assert a["real_strategy"]["entry_status"] == "UNUSABLE" and a["real_strategy"]["entry_pending"] is False
    assert a["real_option"]["ask"] is None and a["real_strategy"]["net_debit_entry"] is None
    assert b["real_option"]["entry_unusable_reason"] == "SIGNAL_TIME_MISSING" and b["real_option"]["ask"] is None
