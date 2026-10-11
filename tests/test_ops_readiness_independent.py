"""Unabhängiger Betriebsreife-Test des Options-Daten-Betriebs (Tester != Autor).

Geprüft wird gegen die SPEZIFIKATION und gegen offizielle NYSE-Fakten, nicht gegen das Zufallsverhalten des Codes:
  1. NYSE-Kalender (market_snapshot): Feiertage, verkürzte Sitzungen, Sonderschließungen, DST-feste Sitzungszeiten
  2. option_quotes: session_of / ny_date / usable_rth_quote / rth_quote_history (Point-in-Time)
  3. candidate_ledger: Marks aus archivierten RTH-Quotes, Settlement, due_mark_symbols (nur lesend)
  4. scripts/rth_marks.py: Capture-Fenster, Idempotenz, nie Ledger/history.json
  5. workflow_health: Scanner-Guard in New-York-Zeit, Börsenfeiertage, Watchdog
  6. Workflows/Persistenz: options_capture.yml, scanner.yml, feedback.yml, .gitattributes, Sicherheits-Schalter

Regeln: kein Netzwerk (Tradier/yfinance per monkeypatch ersetzt), Dateien nur in tmp_path, nie outputs/ des Repos.
Die gefundenen Fehler (Settlement-Herkunft, Datumsfeld 0) sind behoben; die Tests sichern sie ab.
Synthetische Quotes prüfen nur die Mechanik, keine wirtschaftliche Aussage.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path

import pytest
import requests
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import rth_marks  # noqa: E402
import upstream_guard  # noqa: E402

from modules import candidate_ledger as cl  # noqa: E402
from modules import market_snapshot as ms  # noqa: E402
from modules import option_quotes as oq  # noqa: E402
from modules import workflow_health as wh  # noqa: E402

UTC = timezone.utc
LONG = "AAPL270115C00200000"
SHORT = "AAPL270115C00210000"
CAND = "MSFT270115C00400000"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen und Fixtures
# ══════════════════════════════════════════════════════════════════════════════════════════════
def utc(s: str) -> datetime:
    """'2026-10-14T13:30[:SS]' -> aware UTC."""
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def _dt(x) -> datetime:
    return x if isinstance(x, datetime) else utc(x)


def epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def d(s: str) -> date:
    return date.fromisoformat(s)


def mdset(year: int, mds: str) -> set[date]:
    """'01-01 07-04' -> {date(year,1,1), date(year,7,4)}."""
    return {date(year, int(x[:2]), int(x[3:])) for x in mds.split()}


def werktage(s) -> set[date]:
    """Nur Mo–Fr (nyse_holidays enthält ein auf Samstag fallendes Neujahr als Datum – bedeutungslos, da ohnehin zu)."""
    return {x for x in s if x.weekday() < 5}


def archiviere(root: Path, sym: str, at, bid, ask, *, quote_at="=", context="test", **extra) -> None:
    """Eine beobachtete Quote ins Archiv schreiben. at = Abrufzeit; quote_at = Börsenzeit ('=' gleich Abrufzeit, None = fehlt)."""
    ra = _dt(at)
    raw = {"symbol": sym, "bid": bid, "ask": ask, **extra}
    qt = ra if quote_at == "=" else (None if quote_at is None else _dt(quote_at))
    if qt is not None:
        raw["bid_date"] = raw["ask_date"] = epoch_ms(qt)
    n = oq.record([raw], underlying=re.match(r"[A-Z]+", sym).group(), source="tradier", context=context,
                  retrieved_at=ra, root=root)
    assert n == 1, "Quote wurde nicht (neu) geschrieben – Testdaten kollidieren (gleiche record_id)"


def norm(raw: dict, at, **kw) -> dict:
    return oq.normalize(raw, underlying="AAPL", source="tradier", context="test", retrieved_at=_dt(at), **kw)


def opt_row(entry: str, *, expiry="2027-01-15", ask=2.0, mid=1.9, symbol=LONG, direction="BULLISH", strike=200,
            pending=False, outcomes=None, ticker="AAPL") -> dict:
    return {"signal_id": f"s-{symbol}-{entry}", "ticker": ticker, "direction": direction, "date": entry,
            "outcomes": dict(outcomes or {}),
            "real_option": {"symbol": symbol, "ask": ask, "mid": mid, "strike": strike, "entry_date": entry,
                            "expiry": expiry, "entry_pending": pending}}


def spread_row(entry: str, *, expiry="2027-01-15", debit=1.2, mid=1.1, direction="BULLISH", long_strike=200,
               short_strike=210, long_sym=LONG, short_sym=SHORT, outcomes=None) -> dict:
    return {"signal_id": f"p-{long_sym}-{entry}", "ticker": "AAPL", "direction": direction, "date": entry,
            "outcomes": dict(outcomes or {}),
            "real_strategy": {"strategy": "BULL_CALL_SPREAD" if direction == "BULLISH" else "BEAR_PUT_SPREAD",
                              "entry_date": entry, "expiry": expiry, "entry_pending": False,
                              "net_debit_entry": debit, "net_mid_entry": mid,
                              "legs": [{"side": "long", "symbol": long_sym, "strike": long_strike},
                                       {"side": "short", "symbol": short_sym, "strike": short_strike}]}}


def single_leg_row(entry: str, *, expiry="2027-01-15", symbol=LONG, ask=2.0, mid=1.9, outcomes=None) -> dict:
    return {"signal_id": f"l-{symbol}-{entry}", "ticker": "AAPL", "direction": "BULLISH", "date": entry,
            "outcomes": dict(outcomes or {}),
            "real_strategy": {"strategy": "LONG_CALL", "entry_date": entry, "expiry": expiry, "entry_pending": False,
                              "legs": [{"side": "long", "symbol": symbol, "strike": 200, "ask": ask, "mid": mid}]}}


class LiveQuotes:
    """Ersatz für market_snapshot.fetch_option_quotes (Tradier): liefert vorgegebene Quotes, protokolliert Aufrufe."""

    def __init__(self, quotes: dict | None = None):
        self.quotes = quotes or {}
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, symbols, context="candidate_mark"):
        self.calls.append((list(symbols), context))
        return {s: self.quotes[s] for s in symbols if s in self.quotes}


@pytest.fixture(autouse=True)
def _kalender_aus_repo_config(monkeypatch):
    """Overrides immer aus der Repo-Config (absoluter Pfad, unabhängig vom Arbeitsverzeichnis); lru_cache vor und
    nach jedem Test leeren, damit keine Testdatei in den nächsten Test durchschlägt."""
    ms._calendar_overrides.cache_clear()
    monkeypatch.setattr(ms, "CALENDAR_OVERRIDES", REPO / "config" / "market_calendar.yaml")
    yield
    ms._calendar_overrides.cache_clear()


@pytest.fixture(autouse=True)
def _isolation(tmp_path, monkeypatch):
    """Archiv in tmp, kein Netzwerk, kein Tradier-Key, nie ein Modellpreis (Black-Scholes) in den Marks."""
    monkeypatch.setattr(oq, "ROOT", tmp_path / "options_archive" / "quotes")
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)

    def _kein_netz(*a, **k):
        raise AssertionError("Netzwerkzugriff im Test (Tradier/yfinance muss per monkeypatch ersetzt werden)")

    def _kein_modellpreis(*a, **k):
        raise AssertionError("Modellpreis (Black-Scholes) in einem Mark – verboten")

    monkeypatch.setattr(requests.sessions.Session, "request", _kein_netz)
    monkeypatch.setattr(ms, "_fetch_quotes_raw", _kein_netz)
    monkeypatch.setattr(ms, "unadjusted_close_on", _kein_netz)
    monkeypatch.setattr(cl, "bs_price", _kein_modellpreis)


@pytest.fixture
def archiv(tmp_path) -> Path:
    return oq.ROOT


@pytest.fixture
def kein_live(monkeypatch):
    """Außerhalb der Sitzung darf nie live bei Tradier abgerufen werden."""
    def _boom(*a, **k):
        pytest.fail("Live-Abruf (Tradier) außerhalb der regulären Sitzung")
    monkeypatch.setattr(ms, "fetch_option_quotes", _boom)


def jetzt(monkeypatch, ts: str) -> datetime:
    """candidate_ledger._now_utc auf ts (UTC) setzen."""
    t = utc(ts)
    monkeypatch.setattr(cl, "_now_utc", lambda: t)
    return t


def kalender_datei(monkeypatch, tmp_path: Path, text: str | bytes) -> Path:
    """Eigene Kalender-Overrides statt der Repo-Config."""
    p = tmp_path / "market_calendar_test.yaml"
    if isinstance(text, bytes):
        p.write_bytes(text)
    else:
        p.write_text(text, encoding="utf-8")
    monkeypatch.setattr(ms, "CALENDAR_OVERRIDES", p)
    ms._calendar_overrides.cache_clear()
    return p


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 1. NYSE-Kalender
# ══════════════════════════════════════════════════════════════════════════════════════════════
OFFIZIELL = {  # Jahr: (Feiertage, verkürzte Sitzungen 13:00 ET) laut NYSE
    2022: ("01-17 02-21 04-15 05-30 06-20 07-04 09-05 11-24 12-26", "11-25"),
    2023: ("01-02 01-16 02-20 04-07 05-29 06-19 07-04 09-04 11-23 12-25", "07-03 11-24"),
    2024: ("01-01 01-15 02-19 03-29 05-27 06-19 07-04 09-02 11-28 12-25", "07-03 11-29 12-24"),
    2025: ("01-01 01-09 01-20 02-17 04-18 05-26 06-19 07-04 09-01 11-27 12-25", "07-03 11-28 12-24"),
    2026: ("01-01 01-19 02-16 04-03 05-25 06-19 07-03 09-07 11-26 12-25", "11-27 12-24"),
    2027: ("01-01 01-18 02-15 03-26 05-31 06-18 07-05 09-06 11-25 12-24", "11-26"),
}


@pytest.mark.parametrize("jahr", sorted(OFFIZIELL))
def test_nyse_feiertage_entsprechen_der_offiziellen_liste(jahr):
    erwartet = mdset(jahr, OFFIZIELL[jahr][0])
    assert werktage(ms.nyse_holidays(jahr)) == erwartet, (
        f"{jahr}: zu viel {sorted(werktage(ms.nyse_holidays(jahr)) - erwartet)}, "
        f"fehlt {sorted(erwartet - werktage(ms.nyse_holidays(jahr)))}")


@pytest.mark.parametrize("jahr", sorted(OFFIZIELL))
def test_nyse_verkuerzte_sitzungen_entsprechen_der_offiziellen_liste(jahr):
    assert ms.nyse_early_closes(jahr) == mdset(jahr, OFFIZIELL[jahr][1])


def test_2026_kein_verkuerzter_2_juli_und_3_juli_ist_ganzer_feiertag():
    assert d("2026-07-03") in ms.nyse_holidays(2026)                 # Independence Day (observed, 4.7. = Samstag)
    assert d("2026-07-03") not in ms.nyse_early_closes(2026)
    assert d("2026-07-02") not in ms.nyse_early_closes(2026)
    assert ms.market_close_time(d("2026-07-02")) == dt_time(16, 0)
    assert ms.is_trading_day(d("2026-07-02")) and not ms.is_trading_day(d("2026-07-03"))


def test_2027_heiligabend_ist_feiertag_ohne_kurztag_und_juneteenth_observed():
    assert d("2027-12-24") in ms.nyse_holidays(2027) and d("2027-12-24") not in ms.nyse_early_closes(2027)
    assert not ms.is_trading_day(d("2027-12-24"))
    assert not ms.is_trading_day(d("2027-06-18")) and ms.is_trading_day(d("2027-06-17"))  # 19.6. = Samstag -> Freitag
    assert ms.market_close_time(d("2027-11-26")) == dt_time(13, 0)                        # Tag nach Thanksgiving


def test_neujahr_auf_samstag_wird_nicht_am_31_12_beobachtet():
    assert ms.is_trading_day(d("2021-12-31"))        # Neujahr 2022 = Samstag
    assert ms.is_trading_day(d("2027-12-31"))        # Neujahr 2028 = Samstag (gleiche NYSE-Regel)
    assert d("2021-12-31") not in ms.nyse_holidays(2021) and d("2027-12-31") not in ms.nyse_holidays(2027)
    assert not ms.is_trading_day(d("2023-01-02"))    # Neujahr 2023 = Sonntag -> Montag


@pytest.mark.parametrize("tag", ["2020-07-03", "2021-12-24", "2022-12-26", "2023-01-02", "2021-07-05", "2022-06-20"])
def test_beobachtete_feiertage_wochenende_wird_verschoben(tag):
    """Feiertag auf Samstag -> Freitag, auf Sonntag -> Montag (Juneteenth erst ab 2022)."""
    assert not ms.is_trading_day(d(tag))


@pytest.mark.parametrize("tag", ["2021-06-18", "2026-10-12", "2026-11-11", "2026-07-02", "2025-12-26", "2026-01-02"])
def test_normale_handelstage_sind_keine_feiertage(tag):
    """Juneteenth gab es 2021 noch nicht; Columbus Day/Veterans Day und der 26.12. sind keine NYSE-Feiertage."""
    assert ms.is_trading_day(d(tag))


def test_wochenenden_sind_keine_handelstage_und_next_trading_day_ueberspringt_feiertage():
    assert not ms.is_trading_day(d("2026-10-17")) and not ms.is_trading_day(d("2026-10-18"))
    assert ms.next_trading_day(d("2026-11-25")) == d("2026-11-27")      # Thanksgiving 26.11.
    assert ms.next_trading_day(d("2026-12-24")) == d("2026-12-28")      # 25.12. Feiertag, dann Wochenende
    assert ms.next_trading_day(d("2027-12-23")) == d("2027-12-27")      # 24.12.2027 Feiertag


@pytest.mark.parametrize("tag, schluss", [
    ("2025-07-03", dt_time(13, 0)), ("2025-11-28", dt_time(13, 0)), ("2025-12-24", dt_time(13, 0)),
    ("2026-11-27", dt_time(13, 0)), ("2026-12-24", dt_time(13, 0)), ("2027-11-26", dt_time(13, 0)),
    ("2026-10-14", dt_time(16, 0)), ("2026-11-04", dt_time(16, 0)), ("2026-12-23", dt_time(16, 0))])
def test_market_close_time(tag, schluss):
    assert ms.market_close_time(d(tag)) == schluss


# (Tag, Eröffnung UTC, Schluss UTC) – DST über zoneinfo, beidseits beider Zeitumstellungen 2026 und 2027
SITZUNGEN = [
    ("2026-03-06", "14:30", "21:00"),   # Fr vor DST-Beginn (08.03.): EST
    ("2026-03-09", "13:30", "20:00"),   # Mo nach DST-Beginn: EDT
    ("2026-10-30", "13:30", "20:00"),   # Fr vor DST-Ende (01.11.): EDT
    ("2026-11-02", "14:30", "21:00"),   # Mo nach DST-Ende: EST
    ("2027-03-12", "14:30", "21:00"),   # Fr vor DST-Beginn (14.03.2027)
    ("2027-03-15", "13:30", "20:00"),
    ("2027-11-05", "13:30", "20:00"),   # Fr vor DST-Ende (07.11.2027)
    ("2027-11-08", "14:30", "21:00"),
    ("2025-07-03", "13:30", "17:00"),   # verkürzt im Sommer
    ("2026-11-27", "14:30", "18:00"),   # verkürzt im Winter
    ("2026-12-24", "14:30", "18:00"),
]


@pytest.mark.parametrize("tag, auf, zu", SITZUNGEN)
def test_regular_session_utc_ist_dst_fest(tag, auf, zu):
    o, c = ms.regular_session_utc(d(tag))
    assert (o, c) == (utc(f"{tag}T{auf}"), utc(f"{tag}T{zu}"))
    assert o.utcoffset() == timedelta(0) and c.utcoffset() == timedelta(0)


@pytest.mark.parametrize("tag", ["2026-11-26", "2026-07-03", "2025-01-09", "2026-12-25", "2027-12-24",
                                 "2026-03-08", "2026-11-01", "2026-10-17", "2026-10-18"])
def test_regular_session_utc_ist_none_an_geschlossenen_tagen(tag):
    assert ms.regular_session_utc(d(tag)) is None


@pytest.mark.parametrize("ts, erwartet", [
    # Sommerzeit (Mi 14.10.2026, EDT = UTC−4)
    ("2026-10-14T00:00:00", "post"), ("2026-10-14T04:00:00", "pre"), ("2026-10-14T13:29:59", "pre"), ("2026-10-14T13:30:00", "regular"),
    ("2026-10-14T19:59:59", "regular"), ("2026-10-14T20:00:00", "post"), ("2026-10-14T23:59:59", "post"),
    # Winterzeit (Mi 04.11.2026, EST = UTC−5)
    ("2026-11-04T13:30:00", "pre"), ("2026-11-04T14:29:59", "pre"), ("2026-11-04T14:30:00", "regular"),
    ("2026-11-04T20:59:59", "regular"), ("2026-11-04T21:00:00", "post"),
    # verkürzt: Tag nach Thanksgiving 2026 (EST), 3. Juli 2025 (EDT)
    ("2026-11-27T17:59:59", "regular"), ("2026-11-27T18:00:00", "post"),
    ("2025-07-03T16:59:59", "regular"), ("2025-07-03T17:00:00", "post"),
    # geschlossen: Feiertag (ganztägig, auch abends), Sonderschließung, Wochenende
    ("2026-11-26T15:00:00", "closed"), ("2026-11-26T23:00:00", "closed"), ("2026-07-03T15:00:00", "closed"),
    ("2025-01-09T15:00:00", "closed"), ("2026-10-17T15:00:00", "closed"), ("2026-10-18T15:00:00", "closed"),
    # Randfall: UTC-Datum ist der verkürzte Tag, NY-Datum noch Thanksgiving
    ("2026-11-27T03:00:00", "closed"),
])
def test_us_market_session(ts, erwartet):
    assert ms.us_market_session(utc(ts)) == erwartet


def test_us_market_session_naive_zeit_gilt_als_utc():
    assert ms.us_market_session(datetime(2026, 10, 14, 13, 30)) == "regular"
    assert ms.us_market_session(datetime(2026, 10, 14, 13, 29)) == "pre"


def test_session_funktionen_sind_untereinander_konsistent():
    """Kreuzprüfung dreier Implementierungen (session_of, us_market_session, regular_session_utc) in Minutenraster
    rund um alle Zeitumstellungen, Feiertage und verkürzte Sitzungen."""
    bereiche = [("2026-03-05", "2026-03-12"), ("2026-10-29", "2026-11-05"), ("2026-11-22", "2026-11-30"),
                ("2026-12-21", "2026-12-29"), ("2027-03-11", "2027-03-18"), ("2027-11-04", "2027-11-11"),
                ("2025-07-01", "2025-07-08"), ("2025-01-06", "2025-01-13")]
    fehler = []
    for von, bis in bereiche:
        t, ende = utc(von + "T00:00"), utc(bis + "T00:00")
        punkte = []
        while t < ende:
            punkte.append(t)
            t += timedelta(minutes=3)
        for tag in range((ende - utc(von + "T00:00")).days + 1):         # ±1 min um alle Grenzen
            tag_ny = d(von) + timedelta(days=tag)
            b = ms.regular_session_utc(tag_ny)
            if b:
                for g in b:
                    punkte += [g - timedelta(minutes=1), g, g + timedelta(minutes=1)]
        for t in punkte:
            ny_tag = t.astimezone(ms.NY_TZ).date()
            b = ms.regular_session_utc(ny_tag)
            s_opt, s_mkt = oq.session_of(t), ms.us_market_session(t)
            in_sitzung = bool(b and b[0] <= t < b[1])
            if (s_opt == "RTH") != in_sitzung or (s_mkt == "regular") != in_sitzung:
                fehler.append((t.isoformat(), s_opt, s_mkt, in_sitzung))
            elif b is None and not (s_opt == "CLOSED" and s_mkt == "closed"):
                fehler.append((t.isoformat(), s_opt, s_mkt, "geschlossener Tag"))
            elif b and s_opt == "PRE" and s_mkt != "pre":
                fehler.append((t.isoformat(), s_opt, s_mkt, "PRE"))
            elif b and s_opt == "POST" and s_mkt != "post":
                fehler.append((t.isoformat(), s_opt, s_mkt, "POST"))
    assert not fehler, fehler[:5]


# ── Overrides aus config/market_calendar.yaml ─────────────────────────────────
def test_repo_config_traegt_staatstrauer_carter_und_zeigt_wirkung():
    schliessungen, kurztage = ms._calendar_overrides(str(REPO / "config" / "market_calendar.yaml"))
    assert d("2025-01-09") in schliessungen and kurztage == {}
    assert not ms.is_trading_day(d("2025-01-09")) and d("2025-01-09") in ms.nyse_holidays(2025)
    assert ms.is_trading_day(d("2025-01-08")) and ms.is_trading_day(d("2025-01-10"))


def test_overrides_schliesstag_wirkt_auf_alle_kalenderfunktionen(tmp_path, monkeypatch):
    kalender_datei(monkeypatch, tmp_path, 'closures:\n  - "2026-10-14"\nearly_closes: {}\n')
    tag = d("2026-10-14")
    assert tag in ms.nyse_holidays(2026) and tag not in ms.nyse_holidays(2027)
    assert not ms.is_trading_day(tag) and ms.regular_session_utc(tag) is None
    assert ms.us_market_session(utc("2026-10-14T15:00")) == "closed"
    assert oq.session_of(utc("2026-10-14T15:00")) == "CLOSED"
    assert rth_marks.capture_window(utc("2026-10-14T16:00")) == (False, "MARKET_CLOSED")
    assert wh.upstream_ready("scanner", utc("2026-10-14T15:00"), root=tmp_path,
                             cfg=scanner_cfg())["status"] == "SKIP_MARKET_CLOSED"


def test_overrides_verkuerzte_sitzung_mit_eigener_schlusszeit(tmp_path, monkeypatch):
    kalender_datei(monkeypatch, tmp_path, 'early_closes:\n  "2026-10-15": "12:00"\n  2026-10-16: "12:30"\n')
    assert ms.market_close_time(d("2026-10-15")) == dt_time(12, 0)
    assert ms.market_close_time(d("2026-10-16")) == dt_time(12, 30)      # YAML-Datum ohne Anführungszeichen
    assert ms.nyse_early_closes(2026) >= {d("2026-10-15"), d("2026-10-16")}
    assert d("2026-10-15") not in ms.nyse_early_closes(2027)
    o, c = ms.regular_session_utc(d("2026-10-15"))
    assert c == utc("2026-10-15T16:00")                                  # 12:00 EDT
    assert ms.us_market_session(utc("2026-10-15T15:59:59")) == "regular"
    assert ms.us_market_session(utc("2026-10-15T16:00:00")) == "post"
    assert oq.session_of(utc("2026-10-15T15:59:59")) == "RTH" and oq.session_of(utc("2026-10-15T16:00")) == "POST"
    assert rth_marks.capture_window(utc("2026-10-15T15:50")) == (True, "IN_WINDOW")
    assert rth_marks.capture_window(utc("2026-10-15T15:51")) == (False, "AFTER_WINDOW")


def test_overrides_kurztag_an_schliesstag_wird_ignoriert(tmp_path, monkeypatch):
    kalender_datei(monkeypatch, tmp_path, 'closures: ["2026-10-14"]\nearly_closes: {"2026-10-14": "12:00"}\n')
    assert not ms.is_trading_day(d("2026-10-14")) and d("2026-10-14") not in ms.nyse_early_closes(2026)
    assert oq.session_of(utc("2026-10-14T15:00")) == "CLOSED"


@pytest.mark.parametrize("inhalt", [
    None,                                         # Datei fehlt
    "closures: [kein-datum\n",                    # YAML-Syntaxfehler
    "- a\n- b\n",                                 # falscher Typ auf oberster Ebene
    "closures: 5\n",
    "closures: ['2026-13-45']\n",                 # unmögliches Datum
    "early_closes: {'2026-10-15': 'abc'}\n",      # unlesbare Uhrzeit
    "early_closes: ['2026-10-15']\n",
    b"\xff\xfe\x00\x01",                          # kein UTF-8
])
def test_overrides_fehlende_oder_defekte_datei_fuehrt_nie_zu_exception(tmp_path, monkeypatch, inhalt):
    if inhalt is None:
        p = tmp_path / "gibt_es_nicht.yaml"
        monkeypatch.setattr(ms, "CALENDAR_OVERRIDES", p)
        ms._calendar_overrides.cache_clear()
    else:
        p = kalender_datei(monkeypatch, tmp_path, inhalt)
    assert ms._calendar_overrides(str(p)) == (frozenset(), {})
    # Regelkalender bleibt vollständig erhalten (nur die Sonderschließung 2025-01-09 fehlt dann erwartungsgemäß)
    assert werktage(ms.nyse_holidays(2026)) == mdset(2026, OFFIZIELL[2026][0])
    assert ms.nyse_early_closes(2026) == mdset(2026, OFFIZIELL[2026][1])
    assert ms.us_market_session(utc("2026-12-25T15:00")) == "closed"
    assert ms.regular_session_utc(d("2026-10-14")) == (utc("2026-10-14T13:30"), utc("2026-10-14T20:00"))


def test_overrides_leere_datei_ist_gueltig(tmp_path, monkeypatch):
    p = kalender_datei(monkeypatch, tmp_path, "")
    assert ms._calendar_overrides(str(p)) == (frozenset(), {})


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 2. option_quotes: Session, NY-Datum, verwertbare RTH-Quote, Point-in-Time-Historie
# ══════════════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("ts, erwartet", [
    # Sommer (EDT)
    ("2026-10-14T07:59:59", "CLOSED"), ("2026-10-14T08:00:00", "PRE"), ("2026-10-14T13:29:59", "PRE"),
    ("2026-10-14T13:30:00", "RTH"), ("2026-10-14T19:59:59", "RTH"), ("2026-10-14T20:00:00", "POST"),
    ("2026-10-14T23:59:59", "POST"), ("2026-10-15T00:00:00", "CLOSED"),
    # Winter (EST)
    ("2026-11-04T08:59:59", "CLOSED"), ("2026-11-04T09:00:00", "PRE"), ("2026-11-04T14:29:59", "PRE"),
    ("2026-11-04T14:30:00", "RTH"), ("2026-11-04T20:59:59", "RTH"), ("2026-11-04T21:00:00", "POST"),
    ("2026-11-05T00:59:59", "POST"), ("2026-11-05T01:00:00", "CLOSED"),
    # DST-Wochen: Fr vor / Mo nach der Umstellung
    ("2026-03-06T14:29:59", "PRE"), ("2026-03-06T14:30:00", "RTH"), ("2026-03-09T13:29:59", "PRE"),
    ("2026-03-09T13:30:00", "RTH"), ("2026-10-30T19:59:59", "RTH"), ("2026-10-30T20:00:00", "POST"),
    ("2026-11-02T20:59:59", "RTH"), ("2026-11-02T21:00:00", "POST"),
    ("2027-03-12T14:30:00", "RTH"), ("2027-03-15T13:30:00", "RTH"), ("2027-03-15T13:29:59", "PRE"),
    ("2027-11-05T19:59:59", "RTH"), ("2027-11-05T20:00:00", "POST"), ("2027-11-08T20:59:59", "RTH"),
    # verkürzte Sitzungen: Schluss 13:00 ET, danach POST bis 20:00
    ("2026-11-27T14:29:59", "PRE"), ("2026-11-27T14:30:00", "RTH"), ("2026-11-27T17:59:59", "RTH"),
    ("2026-11-27T18:00:00", "POST"), ("2026-11-28T00:59:59", "POST"), ("2026-11-28T01:00:00", "CLOSED"),
    ("2025-07-03T16:59:59", "RTH"), ("2025-07-03T17:00:00", "POST"),
    ("2026-12-24T17:59:59", "RTH"), ("2026-12-24T18:00:00", "POST"),
    # Feiertage / Sonderschließung / Wochenende: ganztägig CLOSED (auch zu RTH-Uhrzeiten)
    ("2026-11-26T15:00:00", "CLOSED"), ("2026-11-26T12:00:00", "CLOSED"), ("2026-07-03T15:00:00", "CLOSED"),
    ("2025-01-09T15:00:00", "CLOSED"), ("2026-12-25T15:00:00", "CLOSED"), ("2027-12-24T15:00:00", "CLOSED"),
    ("2027-06-18T15:00:00", "CLOSED"), ("2026-10-17T15:00:00", "CLOSED"), ("2026-10-18T15:00:00", "CLOSED"),
    ("2026-11-27T03:00:00", "CLOSED"),          # UTC-Datum = Kurztag, in New York noch Thanksgiving-Abend
    # Tag nach Feiertag ist wieder normal
    ("2025-01-10T15:00:00", "RTH"), ("2026-07-02T15:00:00", "RTH"),
])
def test_session_of_nach_nyse_kalender(ts, erwartet):
    assert oq.session_of(utc(ts)) == erwartet


def test_session_of_akzeptiert_alle_zeitformate_und_liefert_unknown_ohne_zeit():
    assert oq.session_of("2026-10-14T13:30:00Z") == "RTH"
    assert oq.session_of(datetime(2026, 10, 14, 13, 30)) == "RTH"                    # naiv = UTC
    assert oq.session_of(epoch_ms(utc("2026-10-14T13:30:00"))) == "RTH"              # Epoch ms
    assert oq.session_of(datetime(2026, 10, 14, 9, 30, tzinfo=ms.NY_TZ)) == "RTH"    # fremde Zeitzone
    for kaputt in (None, "", "kein Datum", True, [], {}):
        assert oq.session_of(kaputt) == "UNKNOWN"


@pytest.mark.parametrize("ts, tag", [
    ("2026-10-13T02:00:00Z", "2026-10-12"), ("2026-10-15T03:59:59Z", "2026-10-14"), ("2026-10-15T04:00:00Z", "2026-10-15"),
    ("2026-11-01T05:30:00Z", "2026-11-01"), ("2026-11-02T04:59:59Z", "2026-11-01"), ("2026-11-02T05:00:00Z", "2026-11-02"),
    ("2026-03-08T04:59:59Z", "2026-03-07"), ("2026-03-08T05:00:00Z", "2026-03-08"),
])
def test_ny_date(ts, tag):
    assert oq.ny_date(ts) == tag


def test_ny_date_ohne_zeit_ist_none():
    assert oq.ny_date(None) is None and oq.ny_date("kaputt") is None


# ── usable_rth_quote ──────────────────────────────────────────────────────────
RTH = "2026-10-14T15:00"        # Mi 11:00 EDT


def test_usable_rth_quote_gute_quote():
    assert oq.usable_rth_quote(norm({"symbol": LONG, "bid": 1.0, "ask": 1.1, "bid_date": epoch_ms(utc(RTH))}, RTH))


@pytest.mark.parametrize("name, raw, at", [
    ("Abruf nach Schluss", {"bid": 1.0, "ask": 1.1}, "2026-10-14T20:30"),
    ("Abruf vor Eröffnung", {"bid": 1.0, "ask": 1.1}, "2026-10-14T13:00"),
    ("Abruf am Feiertag zu RTH-Uhrzeit", {"bid": 1.0, "ask": 1.1}, "2026-11-26T15:00"),
    ("Abruf am Wochenende", {"bid": 1.0, "ask": 1.1}, "2026-10-17T15:00"),
    ("Abruf an Kurztag nach 13:00 ET", {"bid": 1.0, "ask": 1.1}, "2026-11-27T19:00"),
    ("Bid 0", {"bid": 0.0, "ask": 1.1}, RTH),
    ("Ask 0", {"bid": 1.0, "ask": 0.0}, RTH),
    ("Bid fehlt", {"ask": 1.1}, RTH),
    ("Ask fehlt", {"bid": 1.0}, RTH),
    ("gekreuzt (Bid > Ask)", {"bid": 1.2, "ask": 1.1}, RTH),
    ("negativer Preis", {"bid": -1.0, "ask": 1.1}, RTH),
])
def test_usable_rth_quote_lehnt_ab(name, raw, at):
    assert not oq.usable_rth_quote(norm({"symbol": LONG, **raw}, at)), name


def test_usable_rth_quote_boersenzeit_ausserhalb_rth_lehnt_ab():
    ra = utc(RTH)
    pre = norm({"symbol": LONG, "bid": 1.0, "ask": 1.1, "bid_date": epoch_ms(utc("2026-10-14T12:00"))}, ra)   # 08:00 ET
    post = norm({"symbol": LONG, "bid": 1.0, "ask": 1.1, "bid_date": epoch_ms(utc("2026-10-13T21:00"))}, ra)  # Vortag POST
    assert oq.quote_session(pre) == "PRE" and not oq.usable_rth_quote(pre)
    assert oq.quote_session(post) == "POST" and not oq.usable_rth_quote(post)


def test_usable_rth_quote_stale_lehnt_ab_auch_wenn_beide_zeiten_rth_sind():
    rec = norm({"symbol": LONG, "bid": 1.0, "ask": 1.1, "bid_date": epoch_ms(utc("2026-10-12T15:00"))}, RTH)  # 48 h alt
    assert "STALE" in rec["quality_flags"] and oq.session_of(rec["quote_time"]) == "RTH"
    assert not oq.usable_rth_quote(rec)


def test_usable_rth_quote_ohne_boersenzeit_zaehlt_die_abrufzeit():
    rec = norm({"symbol": LONG, "bid": 1.0, "ask": 1.1}, RTH)
    assert rec["quote_time"] is None and oq.quote_session(rec) == "RTH" and oq.usable_rth_quote(rec)


def test_usable_rth_quote_quote_time_wird_vor_abrufzeit_bewertet():
    """quote_time (Börsenzeit) ist maßgeblich: Abruf RTH, Quote aber nachbörslich -> nicht verwertbar."""
    rec = norm({"symbol": LONG, "bid": 1.0, "ask": 1.1, "quote_time": "2026-10-13T22:00:00Z"}, RTH)
    assert oq.quote_session(rec) == "POST" and not oq.usable_rth_quote(rec)


def test_usable_rth_quote_wide_spread_und_locked_bleiben_verwertbar_nur_unusable_und_stale_nicht():
    wide = norm({"symbol": LONG, "bid": 0.1, "ask": 2.0, "bid_date": epoch_ms(utc(RTH))}, RTH)
    assert "WIDE_SPREAD" in wide["quality_flags"] and oq.usable_rth_quote(wide)
    locked = norm({"symbol": LONG, "bid": 1.0, "ask": 1.0, "bid_date": epoch_ms(utc(RTH))}, RTH)
    assert "LOCKED" in locked["quality_flags"] and oq.usable_rth_quote(locked)


def test_usable_rth_quote_none_und_leer():
    assert not oq.usable_rth_quote(None) and not oq.usable_rth_quote({})
    assert not oq.usable_rth_quote({"bid": 1.0, "ask": 1.1})                 # ohne retrieved_at -> UNKNOWN


# ── rth_quote_history ─────────────────────────────────────────────────────────
def test_rth_history_nur_verwertbare_rth_quotes_aufsteigend_mit_ny_datum(archiv):
    archiviere(archiv, LONG, "2026-10-13T19:30", 2.2, 2.3)                       # später am Tag, zuerst geschrieben
    archiviere(archiv, LONG, "2026-10-13T14:00", 2.0, 2.1)                       # früher am Tag
    archiviere(archiv, LONG, "2026-10-12T15:00", 1.9, 2.0)                       # Vortag
    archiviere(archiv, LONG, "2026-10-13T21:30", 9.0, 9.1)                       # POST -> raus
    archiviere(archiv, LONG, "2026-10-13T12:00", 9.0, 9.1)                       # PRE -> raus
    archiviere(archiv, LONG, "2026-10-13T15:00", 0.0, 2.1, context="bid0")       # Bid 0 -> raus
    archiviere(archiv, LONG, "2026-10-13T16:00", 3.0, 2.0, context="gekreuzt")   # gekreuzt -> raus
    archiviere(archiv, SHORT, "2026-10-13T15:00", 0.9, 1.0)                      # anderes Symbol
    h = oq.rth_quote_history([LONG], utc("2026-10-13T23:00"), root=archiv)
    assert list(h) == [LONG]
    assert [(day, r["bid"]) for day, r in h[LONG]] == [("2026-10-12", 1.9), ("2026-10-13", 2.0), ("2026-10-13", 2.2)]
    assert [r["retrieved_at"] for _, r in h[LONG]] == sorted(r["retrieved_at"] for _, r in h[LONG])


def test_rth_history_point_in_time_nie_nach_as_of(archiv):
    archiviere(archiv, LONG, "2026-10-13T15:00", 2.0, 2.1)
    archiviere(archiv, LONG, "2026-10-13T16:00", 2.5, 2.6)
    archiviere(archiv, LONG, "2026-10-14T15:00", 3.0, 3.1)
    bis = lambda s: [r["bid"] for _, r in oq.rth_quote_history([LONG], utc(s), root=archiv).get(LONG, [])]  # noqa: E731
    assert bis("2026-10-13T14:59:59") == []
    assert bis("2026-10-13T15:00:00") == [2.0]                     # as_of ist inklusive
    assert bis("2026-10-13T15:59:59") == [2.0]
    assert bis("2026-10-13T16:00:00") == [2.0, 2.5]
    assert bis("2026-10-14T14:59:59") == [2.0, 2.5]
    assert bis("2026-10-14T15:00:00") == [2.0, 2.5, 3.0]


def test_rth_history_lookback_und_symbolnormalisierung(archiv):
    archiviere(archiv, LONG, "2026-10-01T15:00", 1.0, 1.1)         # 13 Tage vor as_of
    archiviere(archiv, LONG, "2026-10-06T15:00", 1.5, 1.6)         # 8 Tage vor as_of
    archiviere(archiv, LONG, "2026-10-14T15:00", 2.0, 2.1)
    as_of = utc("2026-10-14T20:00")
    assert [r["bid"] for _, r in oq.rth_quote_history([LONG], as_of, root=archiv)[LONG]] == [1.5, 2.0]
    assert [r["bid"] for _, r in oq.rth_quote_history([LONG], as_of, root=archiv, lookback_days=30)[LONG]] == [1.0, 1.5, 2.0]
    assert [r["bid"] for _, r in oq.rth_quote_history([LONG], as_of, root=archiv, lookback_days=0)[LONG]] == [2.0]
    assert LONG in oq.rth_quote_history([f" {LONG.lower()} "], as_of, root=archiv)


def test_rth_history_wirft_nie_und_liefert_leer(archiv, tmp_path):
    assert oq.rth_quote_history([LONG], utc("2026-10-14T20:00"), root=tmp_path / "gibt_es_nicht") == {}
    assert oq.rth_quote_history([], utc("2026-10-14T20:00"), root=archiv) == {}
    assert oq.rth_quote_history([LONG], "kein Datum", root=archiv) == {}
    assert oq.rth_quote_history([LONG], None, root=archiv) == {}
    archiv.mkdir(parents=True, exist_ok=True)
    (archiv / "2026-10-14.jsonl").write_text("{kaputt\n[1,2]\n", encoding="utf-8")
    assert oq.rth_quote_history([LONG], utc("2026-10-14T20:00"), root=archiv) == {}


def test_rth_history_nutzt_modul_root_wenn_kein_root_gegeben(archiv):
    archiviere(archiv, LONG, "2026-10-14T15:00", 2.0, 2.1)
    assert LONG in oq.rth_quote_history([LONG], utc("2026-10-14T20:00"))          # oq.ROOT = tmp (autouse)


# ── Bug: Epoch 0 als Börsenzeit ───────────────────────────────────────────────
def test_nullwert_in_bid_date_ist_keine_boersenzeit():
    ra = utc(RTH)
    rec = norm({"symbol": LONG, "bid": 1.0, "ask": 1.1, "bid_date": 0, "ask_date": epoch_ms(ra)}, ra)
    assert rec["quote_time"] == oq._iso(ra)
    assert "STALE" not in rec["quality_flags"] and oq.usable_rth_quote(rec)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 3. candidate_ledger: Quellenwahl, Archiv-Marks, Settlement, due_mark_symbols
# ══════════════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("ts", [                 # Mark-Fenster: Eröffnung + 30 min bis Schluss − 10 min
    "2026-10-14T14:00:00", "2026-10-14T19:50:00", "2026-11-04T15:00:00", "2026-11-04T20:50:00",
    "2026-11-27T17:50:00", "2025-07-03T16:50:00"])
def test_mark_quotes_live_nur_in_regulaerer_sitzung(monkeypatch, ts):
    jetzt(monkeypatch, ts)
    live = LiveQuotes({LONG: {"bid": 1.0, "ask": 1.1, "mid": 1.05, "ts": None}})
    monkeypatch.setattr(ms, "fetch_option_quotes", live)
    modus, quotes = cl._mark_quotes([LONG])
    assert modus == "live" and quotes == live.quotes and live.calls == [([LONG], "candidate_mark")]


@pytest.mark.parametrize("ts", [
    "2026-10-14T13:29:59", "2026-10-14T20:00:00", "2026-11-04T14:29:59", "2026-11-04T21:00:00",
    "2026-10-14T13:30:00", "2026-10-14T13:59:59", "2026-10-14T19:50:01",   # Puffer nach Eröffnung / vor Schluss
    "2026-11-04T14:30:00", "2026-11-04T20:50:01",
    "2026-11-27T18:00:00",                      # Kurztag nach 13:00 ET
    "2026-11-26T15:00:00", "2026-07-03T15:00:00", "2025-01-09T15:00:00",     # Feiertage/Sonderschließung mitten am Tag
    "2026-10-17T15:00:00", "2026-10-18T15:00:00"])
def test_mark_quotes_archiv_ausserhalb_der_sitzung_ohne_live_abruf(monkeypatch, archiv, kein_live, ts):
    """Auch mit Quotes im Archiv: außerhalb der Sitzung nie ein Live-Abruf; die Quellenangabe lautet 'archive'."""
    jetzt(monkeypatch, ts)
    tag = (utc(ts) - timedelta(days=1)).date().isoformat()          # Vortag, falls Handelstag: dort liegt eine RTH-Quote
    handelstag = ms.is_trading_day(d(tag))
    if handelstag:
        archiviere(archiv, LONG, f"{tag}T15:00", 2.0, 2.1)
    modus, quotes = cl._mark_quotes([LONG])
    assert modus == "archive" and isinstance(quotes, dict)
    assert ([day for day, _ in quotes[LONG]] == [tag]) if handelstag else (quotes == {})


def test_mark_quotes_archiv_ist_point_in_time(monkeypatch, archiv, kein_live):
    archiviere(archiv, LONG, "2026-10-14T15:00", 2.0, 2.1)
    archiviere(archiv, LONG, "2026-10-15T15:00", 3.0, 3.1)          # liegt NACH jetzt
    jetzt(monkeypatch, "2026-10-14T21:00")
    modus, quotes = cl._mark_quotes([LONG])
    assert modus == "archive" and [r["bid"] for _, r in quotes[LONG]] == [2.0]


def test_mark_quotes_reicht_das_lookback_an_das_archiv_weiter(monkeypatch):
    jetzt(monkeypatch, "2026-10-14T21:00")
    gesehen = {}

    def fake(symbols, as_of, *, root=None, lookback_days=0):
        gesehen.update(symbols=list(symbols), as_of=as_of, lookback=lookback_days)
        return {}
    monkeypatch.setattr(oq, "rth_quote_history", fake)
    modus, quotes = cl._mark_quotes([LONG, SHORT])
    assert (modus, quotes) == ("archive", {})
    assert gesehen == {"symbols": [LONG, SHORT], "as_of": utc("2026-10-14T21:00"), "lookback": cl.MARK_ARCHIVE_LOOKBACK_DAYS}
    assert cl.MARK_ARCHIVE_LOOKBACK_DAYS >= 7


def _rec(bid: float) -> dict:
    return {"bid": bid, "ask": bid + 0.1, "mid": round(bid + 0.05, 4)}


def test_archived_mark_fruehester_tag_ab_faelligkeit_mit_allen_symbolen():
    hist = {LONG: [("2026-10-09", _rec(1)), ("2026-10-12", _rec(2)), ("2026-10-12", _rec(2.5)), ("2026-10-13", _rec(3))],
            SHORT: [("2026-10-13", _rec(0.5)), ("2026-10-13", _rec(0.6)), ("2026-10-14", _rec(0.7))]}
    assert cl._archived_mark(hist, [LONG], "2026-10-10") == ("2026-10-12", [hist[LONG][1][1]])
    tag, recs = cl._archived_mark(hist, [LONG, SHORT], "2026-10-10")
    assert tag == "2026-10-13" and recs == [hist[LONG][3][1], hist[SHORT][0][1]]          # erste Quote je Symbol
    assert cl._archived_mark(hist, [LONG, SHORT], "2026-10-14") is None                   # LONG hat am 14. nichts
    assert cl._archived_mark(hist, [LONG], "2026-10-13")[0] == "2026-10-13"               # Fälligkeitstag zählt mit
    assert cl._archived_mark(hist, [LONG], "2026-10-14") is None
    assert cl._archived_mark(hist, [LONG, "FEHLT"], "2026-10-01") is None                 # ein Symbol ohne Historie
    assert cl._archived_mark(hist, [], "2026-10-01") is None
    assert cl._archived_mark({}, [LONG], "2026-10-01") is None


def test_option_mark_ausserhalb_sitzung_frueheste_rth_quote_ab_faelligkeit(monkeypatch, archiv, kein_live):
    jetzt(monkeypatch, "2026-10-13T22:00")                                    # Di nach Schluss
    archiviere(archiv, LONG, "2026-10-09T15:00", 5.0, 5.1)                    # RTH, aber vor Fälligkeit (10-10) -> ignoriert
    archiviere(archiv, LONG, "2026-10-12T12:00", 9.0, 9.1)                    # PRE -> ignoriert
    archiviere(archiv, LONG, "2026-10-12T13:35", 0.0, 9.1, context="bid0")    # RTH, aber Bid 0 -> ignoriert
    archiviere(archiv, LONG, "2026-10-12T14:00", 2.4, 2.5)                    # erste verwertbare Quote ab Fälligkeit
    archiviere(archiv, LONG, "2026-10-12T19:30", 2.8, 2.9)                    # später am selben Tag -> nicht gewählt
    archiviere(archiv, LONG, "2026-10-12T21:00", 9.0, 9.1, context="post")    # POST -> ignoriert
    archiviere(archiv, LONG, "2026-10-13T15:00", 3.0, 3.1)                    # späterer Tag -> nicht gewählt
    row = opt_row("2026-10-05")
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 13)) is True
    oc = row["outcomes"]
    assert oc["real_opt_ret_5d"] == pytest.approx(2.4 / 2.0 - 1, abs=1e-4)           # Verkauf zum Bid, Kauf zum Ask
    assert oc["real_opt_ret_mid_5d"] == pytest.approx(2.45 / 1.9 - 1, abs=1e-4)      # Mid/Mid
    assert oc["real_opt_mark_date_5d"] == "2026-10-12" and oc["real_opt_mark_source_5d"] == "archive_rth"
    assert "real_opt_late_5d" not in oc and "late_mark" not in row
    assert sorted(oc) == ["real_opt_mark_date_5d", "real_opt_mark_source_5d", "real_opt_ret_5d", "real_opt_ret_mid_5d"]


def test_option_mark_bleibt_offen_ohne_verwertbare_quote_und_erfindet_nichts(monkeypatch, archiv, kein_live):
    jetzt(monkeypatch, "2026-10-13T22:00")
    archiviere(archiv, LONG, "2026-10-12T21:30", 2.4, 2.5)                     # POST
    archiviere(archiv, LONG, "2026-10-12T15:00", 0.0, 2.5, context="bid0")     # Bid 0
    archiviere(archiv, LONG, "2026-10-12T16:00", 2.6, 2.5, context="cross")    # gekreuzt
    archiviere(archiv, LONG, "2026-10-09T15:00", 2.0, 2.1)                     # vor Fälligkeit
    row = opt_row("2026-10-05")
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 13)) is False
    assert row["outcomes"] == {} and "late_mark" not in row


def test_option_mark_ohne_jede_quote_oder_archiv_bleibt_offen(monkeypatch, kein_live):
    jetzt(monkeypatch, "2026-10-13T22:00")
    row = opt_row("2026-10-05")
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 13)) is False and row["outcomes"] == {}


@pytest.mark.parametrize("quote_tag, spaet", [("2026-09-21", False), ("2026-09-22", True)])
def test_option_late_flag_erst_ab_mehr_als_7_tagen_nach_faelligkeit(monkeypatch, archiv, kein_live, quote_tag, spaet):
    """Eintritt Mi 2026-09-09, h=5 -> fällig Mo 2026-09-14. Mark am 21.09. = 7 Tage danach (nicht late), am 22.09. = 8 (late)."""
    jetzt(monkeypatch, f"{quote_tag}T22:00")
    archiviere(archiv, LONG, f"{quote_tag}T15:00", 2.4, 2.5)
    row = opt_row("2026-09-09")
    assert cl._fill_real_option_marks([row], datetime.fromisoformat(quote_tag)) is True
    oc = row["outcomes"]
    assert oc["real_opt_mark_date_5d"] == quote_tag
    assert oc.get("real_opt_late_5d", False) is spaet and row.get("late_mark", False) is spaet


def test_option_late_flag_zaehlt_ab_dem_mark_tag_nicht_ab_dem_lauftag(monkeypatch, archiv, kein_live):
    """Quote lag pünktlich vor, Lauf erst 8 Tage später: Mark-Tag (nicht Lauf-Tag) entscheidet -> nicht late."""
    jetzt(monkeypatch, "2026-10-20T22:00")
    archiviere(archiv, LONG, "2026-10-12T15:00", 2.4, 2.5)
    archiviere(archiv, LONG, "2026-10-20T15:00", 4.0, 4.1)
    row = opt_row("2026-10-05")                    # 5d fällig 10-10; Lauf 10-20 (10 Tage danach)
    cl._fill_real_option_marks([row], datetime(2026, 10, 20))
    oc = row["outcomes"]
    assert oc["real_opt_mark_date_5d"] == "2026-10-12" and "real_opt_late_5d" not in oc
    assert oc["real_opt_ret_5d"] == pytest.approx(2.4 / 2.0 - 1, abs=1e-4)


def test_option_marks_mehrere_horizonte_je_eigener_faelligkeitstag(monkeypatch, archiv, kein_live):
    jetzt(monkeypatch, "2026-10-28T22:00")
    archiviere(archiv, LONG, "2026-10-19T15:00", 2.2, 2.3)          # erster RTH-Tag ab 5d-Fälligkeit (10-19 = Mo)
    archiviere(archiv, LONG, "2026-10-26T15:00", 2.6, 2.7)          # Mo; 20d fällig 10-25 (So)
    row = opt_row("2026-10-14")                                      # 5d -> 10-19, 20d -> 11-03 (noch nicht fällig)
    cl._fill_real_option_marks([row], datetime(2026, 10, 28))
    assert row["outcomes"]["real_opt_mark_date_5d"] == "2026-10-19" and "real_opt_ret_20d" not in row["outcomes"]
    row2 = opt_row("2026-10-05")                                     # 5d -> 10-10, 20d -> 10-25
    cl._fill_real_option_marks([row2], datetime(2026, 10, 28))
    assert row2["outcomes"]["real_opt_mark_date_5d"] == "2026-10-19"
    assert row2["outcomes"]["real_opt_mark_date_20d"] == "2026-10-26"
    assert row2["outcomes"]["real_opt_ret_20d"] == pytest.approx(2.6 / 2.0 - 1, abs=1e-4)


def test_option_marks_sind_idempotent_und_ueberschreiben_nichts(monkeypatch, archiv, kein_live):
    jetzt(monkeypatch, "2026-10-13T22:00")
    archiviere(archiv, LONG, "2026-10-12T14:00", 2.4, 2.5)
    row = opt_row("2026-10-05")
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 13)) is True
    erste = json.dumps(row, sort_keys=True)
    archiviere(archiv, LONG, "2026-10-12T13:40", 9.9, 10.0, context="rueckwirkend")     # neue, "frühere" Quote später archiviert
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 13)) is False
    assert json.dumps(row, sort_keys=True) == erste


def test_option_marks_ueberspringen_pending_und_fehlende_entry_preise(monkeypatch, archiv, kein_live):
    jetzt(monkeypatch, "2026-10-13T22:00")
    archiviere(archiv, LONG, "2026-10-12T14:00", 2.4, 2.5)
    rows = [opt_row("2026-10-05", pending=True), opt_row("2026-10-05", ask=None), opt_row("2026-10-05", mid=0)]
    assert cl._fill_real_option_marks(rows, datetime(2026, 10, 13)) is False
    assert all(r["outcomes"] == {} for r in rows)


def test_option_mark_live_in_der_sitzung_nutzt_tradier_und_markiert_heute(monkeypatch):
    jetzt(monkeypatch, "2026-10-14T15:00")                                         # Mi 11:00 ET
    live = LiveQuotes({LONG: {"bid": 3.0, "ask": 3.1, "mid": 3.05, "ts": None}})
    monkeypatch.setattr(ms, "fetch_option_quotes", live)
    row = opt_row("2026-09-20")                                                    # 5d fällig 09-25 (19 Tage danach), 20d fällig 10-10
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 14)) is True
    oc = row["outcomes"]
    assert live.calls == [([LONG], "candidate_mark")]
    assert oc["real_opt_ret_5d"] == pytest.approx(3.0 / 2.0 - 1, abs=1e-4)
    assert oc["real_opt_ret_mid_5d"] == pytest.approx(3.05 / 1.9 - 1, abs=1e-4)
    assert oc["real_opt_mark_date_5d"] == "2026-10-14" and oc["real_opt_mark_date_20d"] == "2026-10-14"
    assert "real_opt_mark_source_5d" not in oc                                      # live: keine Archiv-Herkunft
    assert oc["real_opt_late_5d"] is True and "real_opt_late_20d" not in oc and row["late_mark"] is True


def test_option_mark_live_ohne_quote_bleibt_offen_und_greift_nicht_aufs_archiv_zurueck(monkeypatch, archiv):
    jetzt(monkeypatch, "2026-10-14T15:00")
    archiviere(archiv, LONG, "2026-10-12T14:00", 2.4, 2.5)
    monkeypatch.setattr(ms, "fetch_option_quotes", LiveQuotes({}))
    row = opt_row("2026-10-05")
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 14)) is False and row["outcomes"] == {}


# ── Spreads ───────────────────────────────────────────────────────────────────
def test_spread_mark_braucht_beide_legs_am_selben_tag(monkeypatch, archiv, kein_live):
    jetzt(monkeypatch, "2026-10-13T22:00")
    archiviere(archiv, LONG, "2026-10-12T15:00", 2.0, 2.1)          # 10-12: nur Long-Leg
    archiviere(archiv, LONG, "2026-10-13T15:00", 2.5, 2.6)
    archiviere(archiv, SHORT, "2026-10-13T15:05", 0.9, 1.0)         # 10-13: beide Legs
    row = spread_row("2026-10-05")
    assert cl._fill_real_strategy_marks([row], datetime(2026, 10, 13)) is True
    oc = row["outcomes"]
    assert oc["real_strat_mark_date_5d"] == "2026-10-13" and oc["real_strat_mark_source_5d"] == "archive_rth"
    assert oc["real_strat_ret_5d"] == pytest.approx((2.5 - 1.0) / 1.2 - 1, abs=1e-4)         # Long-Bid − Short-Ask
    assert oc["real_strat_ret_mid_5d"] == pytest.approx((2.55 - 0.95) / 1.1 - 1, abs=1e-4)  # Mid − Mid
    assert "real_strat_late_5d" not in oc


def test_spread_mark_bleibt_offen_wenn_legs_nie_am_selben_tag_quotiert_wurden(monkeypatch, archiv, kein_live):
    jetzt(monkeypatch, "2026-10-13T22:00")
    archiviere(archiv, LONG, "2026-10-12T15:00", 2.0, 2.1)
    archiviere(archiv, SHORT, "2026-10-13T15:00", 0.9, 1.0)
    row = spread_row("2026-10-05")
    assert cl._fill_real_strategy_marks([row], datetime(2026, 10, 13)) is False and row["outcomes"] == {}


def test_spread_mark_exit_wird_bei_null_gefloort(monkeypatch, archiv, kein_live):
    jetzt(monkeypatch, "2026-10-13T22:00")
    archiviere(archiv, LONG, "2026-10-12T15:00", 0.5, 0.6)
    archiviere(archiv, SHORT, "2026-10-12T15:00", 0.9, 1.0)         # Long-Bid 0.5 − Short-Ask 1.0 < 0
    row = spread_row("2026-10-05")
    cl._fill_real_strategy_marks([row], datetime(2026, 10, 13))
    assert row["outcomes"]["real_strat_ret_5d"] == -1.0
    assert row["outcomes"]["real_strat_ret_mid_5d"] == pytest.approx((0.55 - 0.95) / 1.1 - 1, abs=1e-4)


@pytest.mark.parametrize("quote_tag, spaet", [("2026-09-21", False), ("2026-09-22", True)])
def test_spread_late_flag_erst_ab_mehr_als_7_tagen_nach_faelligkeit(monkeypatch, archiv, kein_live, quote_tag, spaet):
    """Eintritt Mi 2026-09-09 -> 5d fällig Mo 2026-09-14; Mark am 21.09. = 7 Tage danach (nicht late), am 22.09. = 8 (late)."""
    jetzt(monkeypatch, f"{quote_tag}T22:00")
    archiviere(archiv, LONG, f"{quote_tag}T15:00", 2.5, 2.6)
    archiviere(archiv, SHORT, f"{quote_tag}T15:00", 0.9, 1.0)
    row = spread_row("2026-09-09")
    assert cl._fill_real_strategy_marks([row], datetime.fromisoformat(quote_tag)) is True
    oc = row["outcomes"]
    assert oc["real_strat_mark_date_5d"] == quote_tag
    assert oc.get("real_strat_late_5d", False) is spaet and row.get("late_mark", False) is spaet


def test_spread_late_flag_zaehlt_ab_dem_mark_tag_nicht_ab_dem_lauftag(monkeypatch, archiv, kein_live):
    """Beide Legs lagen pünktlich am 12.10. vor (fällig 10.10.), der Lauf kommt erst am 20.10. -> Mark-Tag zählt, nicht late."""
    jetzt(monkeypatch, "2026-10-20T22:00")
    archiviere(archiv, LONG, "2026-10-12T15:00", 2.5, 2.6)
    archiviere(archiv, SHORT, "2026-10-12T15:00", 0.9, 1.0)
    archiviere(archiv, LONG, "2026-10-20T15:00", 5.0, 5.1)
    archiviere(archiv, SHORT, "2026-10-20T15:00", 0.1, 0.2)
    row = spread_row("2026-10-05")
    cl._fill_real_strategy_marks([row], datetime(2026, 10, 20))
    oc = row["outcomes"]
    assert oc["real_strat_mark_date_5d"] == "2026-10-12" and "real_strat_late_5d" not in oc and "late_mark" not in row
    assert oc["real_strat_ret_5d"] == pytest.approx((2.5 - 1.0) / 1.2 - 1, abs=1e-4)


def test_single_leg_strategie_wird_aus_archiv_gemarkt(monkeypatch, archiv, kein_live):
    jetzt(monkeypatch, "2026-10-13T22:00")
    archiviere(archiv, LONG, "2026-10-12T15:00", 2.4, 2.5)
    row = single_leg_row("2026-10-05")
    assert cl._fill_real_strategy_marks([row], datetime(2026, 10, 13)) is True
    oc = row["outcomes"]
    assert oc["real_strat_ret_5d"] == pytest.approx(2.4 / 2.0 - 1, abs=1e-4)
    assert oc["real_strat_ret_mid_5d"] == pytest.approx(2.45 / 1.9 - 1, abs=1e-4)
    assert oc["real_strat_mark_source_5d"] == "archive_rth"


def test_spread_mark_live_in_der_sitzung_beide_legs(monkeypatch):
    jetzt(monkeypatch, "2026-10-14T15:00")
    live = LiveQuotes({LONG: {"bid": 2.5, "ask": 2.6, "mid": 2.55, "ts": None},
                       SHORT: {"bid": 0.9, "ask": 1.0, "mid": 0.95, "ts": None}})
    monkeypatch.setattr(ms, "fetch_option_quotes", live)
    row = spread_row("2026-10-05")
    assert cl._fill_real_strategy_marks([row], datetime(2026, 10, 14)) is True
    oc = row["outcomes"]
    assert sorted(live.calls[0][0]) == [LONG, SHORT]
    assert oc["real_strat_ret_5d"] == pytest.approx((2.5 - 1.0) / 1.2 - 1, abs=1e-4)
    assert oc["real_strat_mark_date_5d"] == "2026-10-14" and "real_strat_mark_source_5d" not in oc


def test_update_outcomes_ende_zu_ende_marks_aus_dem_archiv_und_idempotent(monkeypatch, tmp_path, archiv, kein_live):
    """Dateiebene (update_outcomes): Marks außerhalb der Sitzung nur aus dem Archiv; zweiter Lauf ändert keine Zeile."""
    jetzt(monkeypatch, "2026-10-13T22:00")
    for name in ("_fetch_history_batch", "_fetch_history_open_batch", "_fetch_prices_batch"):
        monkeypatch.setattr(cl, name, lambda *a, **k: {})                      # kein yfinance
    led = tmp_path / "ledger" / "2026-10.jsonl"
    _schreibe_ledger(led, [opt_row("2026-10-05"), spread_row("2026-10-05")])
    archiviere(archiv, LONG, "2026-10-12T15:00", 2.4, 2.5)
    archiviere(archiv, SHORT, "2026-10-12T15:00", 0.9, 1.0)
    cl.update_outcomes("2026-10-13", root=led.parent)
    zeilen = [json.loads(z) for z in led.read_text(encoding="utf-8").splitlines()]
    o, sp = zeilen[0]["outcomes"], zeilen[1]["outcomes"]
    assert o["real_opt_ret_5d"] == pytest.approx(2.4 / 2.0 - 1, abs=1e-4) and o["real_opt_mark_source_5d"] == "archive_rth"
    assert sp["real_strat_ret_5d"] == pytest.approx((2.4 - 1.0) / 1.2 - 1, abs=1e-4)
    assert o["real_opt_mark_date_5d"] == sp["real_strat_mark_date_5d"] == "2026-10-12"
    nach = led.read_bytes()
    cl.update_outcomes("2026-10-13", root=led.parent)
    assert led.read_bytes() == nach


# ── Settlement verfallener Kontrakte ──────────────────────────────────────────
def settlement_patch(monkeypatch, spot):
    """unadjusted_close_on -> spot (oder None); protokolliert (ticker, Tag)."""
    calls = []

    def fake(ticker, day):
        calls.append((ticker, day.date() if isinstance(day, datetime) else day))
        return spot
    monkeypatch.setattr(ms, "unadjusted_close_on", fake)
    return calls


@pytest.mark.parametrize("ts", ["2026-10-14T15:00", "2026-10-14T21:00", "2026-10-17T15:00"],
                         ids=["in-Sitzung", "nach-Schluss", "Wochenende"])
def test_verfallene_option_wird_aus_settlement_bewertet_unabhaengig_von_der_sitzung(monkeypatch, archiv, ts):
    jetzt(monkeypatch, ts)
    live = LiveQuotes({LONG: {"bid": 99.0, "ask": 99.1, "mid": 99.05, "ts": None}})   # darf NICHT in den Mark eingehen
    monkeypatch.setattr(ms, "fetch_option_quotes", live)
    archiviere(archiv, LONG, "2026-10-12T15:00", 77.0, 77.1)                           # auch Archiv darf nicht eingehen
    calls = settlement_patch(monkeypatch, 210.0)
    row = opt_row("2026-09-01", expiry="2026-10-09")                                    # 5d, 20d (und ab 10-16 auch 45d) fällig
    today = datetime.fromisoformat(ts[:10])
    assert cl._fill_real_option_marks([row], today) is True
    oc = row["outcomes"]
    faellig = [h for h in cl.HORIZONS if (today - datetime(2026, 9, 1)).days >= h]
    assert faellig[:2] == [5, 20] and 120 not in faellig
    for h in faellig:
        assert oc[f"real_opt_ret_{h}d"] == pytest.approx(10.0 / 2.0 - 1, abs=1e-4)      # Intrinsic 210−200
        assert oc[f"real_opt_ret_mid_{h}d"] == pytest.approx(10.0 / 1.9 - 1, abs=1e-4)
        assert oc[f"real_opt_mark_date_{h}d"] == ts[:10]
    assert "real_opt_ret_120d" not in oc
    assert {c for c in calls} == {("AAPL", d("2026-10-09"))}                             # genau der Verfallstag


@pytest.mark.parametrize("richtung, strike, spot, intrinsic", [
    ("BULLISH", 200, 210.0, 10.0), ("BULLISH", 200, 190.0, 0.0), ("BULLISH", 200, 200.0, 0.0),
    ("BEARISH", 200, 190.0, 10.0), ("BEARISH", 200, 210.0, 0.0)])
def test_settlement_intrinsic_call_und_put(monkeypatch, richtung, strike, spot, intrinsic):
    jetzt(monkeypatch, "2026-10-14T21:00")
    settlement_patch(monkeypatch, spot)
    row = opt_row("2026-09-01", expiry="2026-10-09", direction=richtung, strike=strike)
    cl._fill_real_option_marks([row], datetime(2026, 10, 14))
    assert row["outcomes"]["real_opt_ret_5d"] == pytest.approx(intrinsic / 2.0 - 1, abs=1e-4)


def test_verfallen_ohne_settlement_kurs_bleibt_offen_nie_modellpreis(monkeypatch, archiv):
    jetzt(monkeypatch, "2026-10-14T21:00")
    archiviere(archiv, LONG, "2026-10-12T15:00", 77.0, 77.1)                  # Quote vor Verfall im Archiv
    settlement_patch(monkeypatch, None)
    row = opt_row("2026-09-01", expiry="2026-10-09")
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 14)) is False and row["outcomes"] == {}
    srow = spread_row("2026-09-01", expiry="2026-10-09")
    assert cl._fill_real_strategy_marks([srow], datetime(2026, 10, 14)) is False and srow["outcomes"] == {}


def test_am_verfallstag_selbst_gilt_noch_nicht_als_verfallen(monkeypatch, archiv, kein_live):
    """heute == expiry -> kein Settlement (erst heute > expiry); Mark aus der RTH-Quote des Verfallstags."""
    jetzt(monkeypatch, "2026-10-09T21:00")
    archiviere(archiv, LONG, "2026-10-09T15:00", 1.5, 1.6)
    row = opt_row("2026-10-02", expiry="2026-10-09")                         # 5d fällig 10-07
    assert cl._fill_real_option_marks([row], datetime(2026, 10, 9)) is True
    oc = row["outcomes"]
    assert oc["real_opt_mark_date_5d"] == "2026-10-09" and oc["real_opt_mark_source_5d"] == "archive_rth"
    assert oc["real_opt_ret_5d"] == pytest.approx(1.5 / 2.0 - 1, abs=1e-4)


@pytest.mark.parametrize("spot, long_i, short_i", [(207.0, 7.0, 0.0), (215.0, 15.0, 5.0), (190.0, 0.0, 0.0)])
def test_spread_settlement_intrinsic_beider_legs(monkeypatch, spot, long_i, short_i):
    jetzt(monkeypatch, "2026-10-14T21:00")
    settlement_patch(monkeypatch, spot)
    row = spread_row("2026-09-01", expiry="2026-10-09")
    assert cl._fill_real_strategy_marks([row], datetime(2026, 10, 14)) is True
    oc = row["outcomes"]
    assert oc["real_strat_ret_5d"] == pytest.approx(max(long_i - short_i, 0.0) / 1.2 - 1, abs=1e-4)
    assert oc["real_strat_ret_mid_5d"] == pytest.approx((long_i - short_i) / 1.1 - 1, abs=1e-4)
    assert oc["real_strat_mark_date_5d"] == "2026-10-14"


def test_bear_put_spread_settlement(monkeypatch):
    jetzt(monkeypatch, "2026-10-14T21:00")
    settlement_patch(monkeypatch, 195.0)
    row = spread_row("2026-09-01", expiry="2026-10-09", direction="BEARISH", long_strike=200, short_strike=190)
    cl._fill_real_strategy_marks([row], datetime(2026, 10, 14))
    assert row["outcomes"]["real_strat_ret_5d"] == pytest.approx(5.0 / 1.2 - 1, abs=1e-4)


def test_settlement_mark_traegt_keine_archiv_herkunft_option(monkeypatch):
    jetzt(monkeypatch, "2026-10-14T21:00")                                      # nach Schluss -> mode 'archive'
    settlement_patch(monkeypatch, 210.0)
    row = opt_row("2026-09-01", expiry="2026-10-09")
    cl._fill_real_option_marks([row], datetime(2026, 10, 14))
    assert "real_opt_ret_5d" in row["outcomes"]                                 # Settlement-Mark wurde berechnet
    assert row["outcomes"].get("real_opt_mark_source_5d") != "archive_rth"


def test_settlement_mark_traegt_keine_archiv_herkunft_strategie(monkeypatch):
    jetzt(monkeypatch, "2026-10-14T21:00")
    settlement_patch(monkeypatch, 207.0)
    row = spread_row("2026-09-01", expiry="2026-10-09")
    cl._fill_real_strategy_marks([row], datetime(2026, 10, 14))
    assert "real_strat_ret_5d" in row["outcomes"]
    assert row["outcomes"].get("real_strat_mark_source_5d") != "archive_rth"


# ── due_mark_symbols: nur lesend ──────────────────────────────────────────────
def _schreibe_ledger(pfad: Path, zeilen: list, *, extra: str = "") -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text("".join(json.dumps(z) + "\n" for z in zeilen) + extra, encoding="utf-8")


def _snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()}


def test_due_mark_symbols_liefert_faellige_offene_nicht_verfallene_und_ist_nur_lesend(tmp_path):
    sym = lambda n: f"AAPL270115C{n:08d}"  # noqa: E731
    zeilen = [
        opt_row("2026-10-05", symbol=sym(100000)),                                  # 7 Tage -> 5d fällig -> JA
        opt_row("2026-10-10", symbol=sym(101000)),                                  # 2 Tage -> nichts fällig
        opt_row("2026-10-05", symbol=sym(102000), pending=True),                    # Entry noch offen
        opt_row("2026-10-05", symbol=sym(103000), expiry="2026-10-09"),             # verfallen (heute > expiry)
        opt_row("2026-10-05", symbol=sym(104000), expiry="2026-10-12"),             # Verfallstag == heute -> JA
        opt_row("2026-10-05", symbol=sym(105000), outcomes={"real_opt_ret_5d": 0.1}),   # alle fälligen Horizonte erledigt
        opt_row("2026-09-01", symbol=sym(106000), outcomes={"real_opt_ret_5d": 0.1}),   # 20d fällig + offen -> JA
        spread_row("2026-10-05", long_sym=sym(107000), short_sym=sym(108000)),      # beide Legs -> JA
        single_leg_row("2026-10-05", symbol=sym(109000)),                           # Single-Leg -> JA
        spread_row("2026-10-05", long_sym=sym(110000), short_sym=sym(111000), outcomes={"real_strat_ret_5d": 0.2}),
        {"signal_id": "ohne-option", "ticker": "AAPL", "outcomes": {}},             # keine Real-Option
    ]
    led = tmp_path / "ledger"
    _schreibe_ledger(led / "2026-10.jsonl", zeilen, extra="{kein json\n\n")
    _schreibe_ledger(led / "2026-04.jsonl", [opt_row("2026-04-01", symbol=sym(120000))])   # älter als 5 Monate -> ignoriert
    vorher = _snapshot(tmp_path)
    erwartet = sorted(sym(n) for n in (100000, 104000, 106000, 107000, 108000, 109000))
    assert cl.due_mark_symbols("2026-10-12", root=led) == erwartet
    assert _snapshot(tmp_path) == vorher                                            # Bytes, mtime, Dateiliste unverändert


def test_due_mark_symbols_fehlendes_verzeichnis_und_kaputtes_datum(tmp_path):
    assert cl.due_mark_symbols("2026-10-12", root=tmp_path / "gibt_es_nicht") == []
    assert cl.due_mark_symbols("kein-datum", root=tmp_path) == []


def test_due_mark_symbols_stimmt_mit_den_symbolen_der_mark_funktionen_ueberein(monkeypatch, tmp_path):
    """Kreuzprüfung der Fälligkeitsregel: Symbole, für die _fill_* Quotes anfordert (ohne bereits Verfallene), sind genau
    die von due_mark_symbols gelieferten – über alle Kombinationen aus Alter, Verfall und bereits erledigten Horizonten."""
    heute = "2026-10-12"
    zeilen, verfallen, n = [], set(), 0
    erledigt = [{}, {"5d": 0.1}, {"5d": 0.1, "20d": 0.1, "45d": 0.1, "120d": 0.1}]
    for entry in ("2026-10-12", "2026-10-08", "2026-10-07", "2026-09-22", "2026-09-21", "2026-08-01", "2026-06-01"):
        for expiry in ("2026-10-09", "2026-10-12", "2026-10-13", "2027-01-15"):
            for fertig in erledigt:
                n += 1
                lang, kurz = f"AAPL270115C{100000 + n:08d}", f"AAPL270115C{200000 + n:08d}"
                zeilen.append(opt_row(entry, expiry=expiry, symbol=lang, outcomes={f"real_opt_ret_{k}": v for k, v in fertig.items()}))
                zeilen.append(spread_row(entry, expiry=expiry, long_sym=lang, short_sym=kurz,
                                         outcomes={f"real_strat_ret_{k}": v for k, v in fertig.items()}))
                if expiry < heute:
                    verfallen |= {lang, kurz}
    angefordert: set[str] = set()

    def merke(symbols):
        angefordert.update(symbols)
        return "archive", {}
    monkeypatch.setattr(cl, "_mark_quotes", merke)
    heute_dt = datetime.fromisoformat(heute)
    cl._fill_real_option_marks(json.loads(json.dumps(zeilen)), heute_dt)
    cl._fill_real_strategy_marks(json.loads(json.dumps(zeilen)), heute_dt)
    led = tmp_path / "ledger"
    _schreibe_ledger(led / "2026-10.jsonl", zeilen)
    faellig = set(cl.due_mark_symbols(heute, root=led))
    assert faellig and angefordert - verfallen
    assert faellig == angefordert - verfallen


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 4. scripts/rth_marks.py
# ══════════════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("ts, status", [
    # Sommer (13:30–20:00 UTC): Fenster 14:00–19:50 UTC
    ("2026-10-14T13:30:00", "BEFORE_WINDOW"), ("2026-10-14T13:59:59", "BEFORE_WINDOW"), ("2026-10-14T14:00:00", "IN_WINDOW"),
    ("2026-10-14T19:50:00", "IN_WINDOW"), ("2026-10-14T19:50:01", "AFTER_WINDOW"), ("2026-10-14T21:00:00", "AFTER_WINDOW"),
    # Winter (14:30–21:00 UTC): Fenster 15:00–20:50 UTC
    ("2026-11-04T14:30:00", "BEFORE_WINDOW"), ("2026-11-04T14:59:59", "BEFORE_WINDOW"), ("2026-11-04T15:00:00", "IN_WINDOW"),
    ("2026-11-04T20:50:00", "IN_WINDOW"), ("2026-11-04T20:50:01", "AFTER_WINDOW"),
    # Wochen der Zeitumstellung
    ("2026-03-06T14:59:59", "BEFORE_WINDOW"), ("2026-03-06T15:00:00", "IN_WINDOW"),      # Fr, EST
    ("2026-03-09T13:59:59", "BEFORE_WINDOW"), ("2026-03-09T14:00:00", "IN_WINDOW"),      # Mo, EDT
    ("2026-11-02T14:59:59", "BEFORE_WINDOW"), ("2026-11-02T15:00:00", "IN_WINDOW"),      # Mo, EST
    ("2027-03-15T14:00:00", "IN_WINDOW"), ("2027-11-08T15:00:00", "IN_WINDOW"),
    # verkürzte Sitzungen: Schluss 13:00 ET -> Fenster bis 12:50 ET
    ("2025-07-03T16:50:00", "IN_WINDOW"), ("2025-07-03T16:50:01", "AFTER_WINDOW"),
    ("2026-11-27T17:50:00", "IN_WINDOW"), ("2026-11-27T17:50:01", "AFTER_WINDOW"),
    ("2026-12-24T17:50:00", "IN_WINDOW"), ("2026-12-24T17:51:00", "AFTER_WINDOW"),
    # geschlossen
    ("2026-11-26T17:00:00", "MARKET_CLOSED"), ("2026-07-03T17:00:00", "MARKET_CLOSED"), ("2025-01-09T17:00:00", "MARKET_CLOSED"),
    ("2026-12-25T17:00:00", "MARKET_CLOSED"), ("2027-12-24T17:00:00", "MARKET_CLOSED"),
    ("2026-10-17T17:00:00", "MARKET_CLOSED"), ("2026-10-18T17:00:00", "MARKET_CLOSED"),
    # UTC-Mitternacht: in New York noch (Handels-)Abend bzw. schon Wochenende
    ("2026-10-15T02:00:00", "AFTER_WINDOW"), ("2026-10-17T02:00:00", "AFTER_WINDOW"), ("2026-10-18T02:00:00", "MARKET_CLOSED"),
])
def test_capture_window(ts, status):
    assert rth_marks.capture_window(utc(ts)) == (status == "IN_WINDOW", status)


def test_capture_window_naive_zeit_und_fremde_zeitzone():
    assert rth_marks.capture_window(datetime(2026, 10, 14, 14, 0)) == (True, "IN_WINDOW")           # naiv = UTC
    assert rth_marks.capture_window(datetime(2026, 10, 14, 10, 0, tzinfo=ms.NY_TZ)) == (True, "IN_WINDOW")
    assert rth_marks.capture_window(datetime(2026, 10, 14, 9, 59, tzinfo=ms.NY_TZ)) == (False, "BEFORE_WINDOW")


def test_capture_window_konstanten():
    assert rth_marks.MIN_AFTER_OPEN == timedelta(minutes=30) and rth_marks.MIN_BEFORE_CLOSE == timedelta(minutes=10)


class FakeTradier:
    """Ersatz für market_snapshot.fetch_option_quotes: archiviert wie der echte Hook, aber zur simulierten Abrufzeit."""

    def __init__(self, root: Path):
        self.root = root
        self.now: datetime | None = None
        self.calls: list[tuple[str, list[str]]] = []
        self.ohne_quote: set[str] = set()
        self.preise: dict[str, tuple[float, float]] = {}

    def __call__(self, symbols, context="candidate_mark"):
        syms = [s for s in dict.fromkeys(symbols) if s]
        self.calls.append((context, syms))
        out = {}
        for s in syms:
            if s in self.ohne_quote:
                continue
            bid, ask = self.preise.get(s, (1.0, 1.1))
            archiviere(self.root, s, self.now, bid, ask, context=context)
            out[s] = {"bid": bid or None, "ask": ask, "mid": round((bid + ask) / 2, 4), "ts": None}
        return out


@pytest.fixture
def capture_env(tmp_path, monkeypatch, archiv):
    """tmp als Arbeitsverzeichnis: Ledger mit zwei fälligen Kandidaten (einer teilt das Symbol mit dem aktiven Trade),
    history.json mit einem aktiven Spread (Short-Leg-Symbol wird aus Strike/Verfall rekonstruiert)."""
    monkeypatch.chdir(tmp_path)
    led = tmp_path / "outputs" / "candidate_ledger" / "2026-10.jsonl"
    _schreibe_ledger(led, [opt_row("2026-10-05", symbol=CAND, ticker="MSFT"), opt_row("2026-10-05", symbol=LONG)])
    hp = tmp_path / "history.json"
    hp.write_text(json.dumps({"active_trades": [{
        "ticker": "AAPL", "strategy": "BULL_CALL_SPREAD",
        "option": {"option_symbol": LONG, "strike": 200, "expiry": "2027-01-15",
                   "spread_leg": {"strike": 210, "expiry": "2027-01-15"}}}]}), encoding="utf-8")
    fake = FakeTradier(archiv)
    monkeypatch.setattr(ms, "fetch_option_quotes", fake)
    monkeypatch.setattr(cl, "update_outcomes", lambda *a, **k: pytest.fail("capture-only darf das Ledger nie schreiben"))
    monkeypatch.setattr(cl, "_atomic_write_jsonl", lambda *a, **k: pytest.fail("capture-only darf das Ledger nie schreiben"))
    return tmp_path, hp, fake


def _baum_ohne_archiv(root: Path) -> dict:
    return {k: v for k, v in _snapshot(root).items() if not k.startswith("options_archive")}


def test_capture_nur_im_fenster_ausserhalb_keinerlei_abruf_und_keine_datei(capture_env, monkeypatch):
    tmp, hp, fake = capture_env
    vorher = _snapshot(tmp)
    for ts, fenster in [("2026-10-14T13:59", "BEFORE_WINDOW"), ("2026-10-14T19:51", "AFTER_WINDOW"),
                        ("2026-11-04T14:59", "BEFORE_WINDOW"), ("2026-11-26T17:00", "MARKET_CLOSED"),
                        ("2026-10-17T17:00", "MARKET_CLOSED"), ("2026-11-27T17:51", "AFTER_WINDOW")]:
        for capture_only in (True, False):
            res = rth_marks.run(utc(ts), hp, capture_only=capture_only)
            assert res["skipped"] is True and res["window"] == fenster, (ts, res)
    assert fake.calls == [] and _snapshot(tmp) == vorher


def test_capture_im_fenster_kontexte_nie_ledger_nie_history(capture_env):
    tmp, hp, fake = capture_env
    fake.now = utc("2026-10-14T15:00")
    vorher = _baum_ohne_archiv(tmp)
    res = rth_marks.run(fake.now, hp, capture_only=True)
    assert res["mode"] == "capture" and res["window"] == "IN_WINDOW" and res["skipped"] is False and res["session"] == "regular"
    assert fake.calls == [("trade_mark_rth", [LONG, SHORT]), ("candidate_mark_rth", [CAND])]   # LONG nur als Trade
    assert res["targets"] == 3 and res["requested"] == 3 and res["quotes"] == 3 and res["missing"] == []
    assert _baum_ohne_archiv(tmp) == vorher                                         # Ledger, history.json, Config unverändert
    neu = sorted(p.name for p in (tmp / "options_archive" / "quotes").glob("*.jsonl"))
    assert neu == ["2026-10-14.jsonl"]


def test_capture_zweiter_lauf_am_selben_tag_fordert_nichts_mehr_an(capture_env):
    tmp, hp, fake = capture_env
    fake.now = utc("2026-10-14T15:00")
    rth_marks.run(fake.now, hp, capture_only=True)
    zeilen_vorher = (tmp / "options_archive" / "quotes" / "2026-10-14.jsonl").read_text()
    n_calls = len(fake.calls)
    fake.now = utc("2026-10-14T18:30")
    res = rth_marks.run(fake.now, hp, capture_only=True)
    assert res["requested"] == 0 and res["already_captured"] == 3 and res["quotes"] == 0 and res["missing"] == []
    assert len(fake.calls) == n_calls                                               # gar kein Abruf (auch kein leerer)
    assert (tmp / "options_archive" / "quotes" / "2026-10-14.jsonl").read_text() == zeilen_vorher


def test_capture_fordert_nur_fehlende_symbole_nach(capture_env):
    tmp, hp, fake = capture_env
    fake.now = utc("2026-10-14T15:00")
    fake.ohne_quote = {SHORT, CAND}
    erste = rth_marks.run(fake.now, hp, capture_only=True)
    assert erste["quotes"] == 1 and erste["missing"] == sorted([SHORT, CAND])
    fake.ohne_quote = set()
    fake.calls.clear()
    fake.now = utc("2026-10-14T17:00")
    zweite = rth_marks.run(fake.now, hp, capture_only=True)
    assert fake.calls == [("trade_mark_rth", [SHORT]), ("candidate_mark_rth", [CAND])]
    assert zweite["already_captured"] == 1 and zweite["requested"] == 2


def test_capture_unbrauchbare_quote_zaehlt_nicht_als_erfasst(capture_env):
    tmp, hp, fake = capture_env
    fake.now = utc("2026-10-14T15:00")
    fake.preise[CAND] = (0.0, 1.1)                                                  # Bid 0 -> nicht verwertbar
    rth_marks.run(fake.now, hp, capture_only=True)
    fake.calls.clear()
    fake.preise.clear()
    fake.now = utc("2026-10-14T17:00")
    res = rth_marks.run(fake.now, hp, capture_only=True)
    assert fake.calls == [("candidate_mark_rth", [CAND])] and res["already_captured"] == 2


def test_capture_am_naechsten_handelstag_wieder_voll(capture_env):
    tmp, hp, fake = capture_env
    fake.now = utc("2026-10-14T15:00")
    rth_marks.run(fake.now, hp, capture_only=True)
    fake.calls.clear()
    fake.now = utc("2026-10-15T15:00")
    res = rth_marks.run(fake.now, hp, capture_only=True)
    assert res["requested"] == 3 and res["already_captured"] == 0 and len(fake.calls) == 2


def test_capture_vorabend_quote_nach_schluss_zaehlt_nicht(capture_env, archiv):
    """Eine POST-Quote vom selben Tag (z. B. von einem späten Scanner-Lauf) macht den Tag nicht 'erfasst'."""
    tmp, hp, fake = capture_env
    for s in (LONG, SHORT, CAND):
        archiviere(archiv, s, "2026-10-14T13:00", 1.0, 1.1, context="vorab")         # PRE
    fake.now = utc("2026-10-14T15:00")
    res = rth_marks.run(fake.now, hp, capture_only=True)
    assert res["already_captured"] == 0 and res["requested"] == 3


def test_capture_ohne_history_und_ohne_ledger_ruft_nichts_ab(tmp_path, monkeypatch, archiv):
    monkeypatch.chdir(tmp_path)
    fake = FakeTradier(archiv)
    monkeypatch.setattr(ms, "fetch_option_quotes", fake)
    res = rth_marks.run(utc("2026-10-14T15:00"), tmp_path / "keine_history.json", capture_only=True)
    assert res["targets"] == 0 and res["requested"] == 0 and fake.calls == []


def test_active_trade_symbols_rekonstruktion_aus_strike_und_verfall():
    hist = {"active_trades": [
        {"ticker": "XOM", "strategy": "LONG_PUT", "option": {"strike": 100, "expiry": "2026-12-18"}},
        {"ticker": "AAPL", "strategy": "BEAR_PUT_SPREAD",
         "option": {"strike": 200, "expiry": "2027-01-15", "spread_leg": {"strike": 97.5}}},
        {"ticker": "MSFT", "strategy": "LONG_CALL", "option": {"option_symbol": CAND}},
        {"ticker": "KAPUTT", "strategy": "LONG_CALL", "option": {"strike": None, "expiry": "2027-01-15"}}]}
    assert rth_marks.active_trade_symbols(hist) == sorted(
        {"XOM261218P00100000", "AAPL270115P00200000", "AAPL270115P00097500", CAND})


def test_main_reicht_capture_flag_weiter_und_meldet_fehler_sichtbar(monkeypatch, capsys):
    gesehen = {}

    def fake_run(**kw):
        gesehen.update(kw)
        return {"skipped": True, "window": "BEFORE_WINDOW"}
    monkeypatch.setattr(rth_marks, "run", fake_run)
    assert rth_marks.main(["--capture-only"]) == 0 and gesehen["capture_only"] is True
    assert rth_marks.main([]) == 0 and gesehen["capture_only"] is False

    def kaputt(**kw):
        raise RuntimeError("kaputt")
    monkeypatch.setattr(rth_marks, "run", kaputt)
    assert rth_marks.main(["--capture-only"]) == 1                # Fehler sichtbar (Job rot; im Scanner continue-on-error)
    assert "::error::" in capsys.readouterr().out


def test_capture_laeuft_nur_mit_den_im_workflow_installierten_paketen(tmp_path):
    """options_capture.yml installiert nur requests + pyyaml: Import- und Lauf-Pfad von rth_marks --capture-only darf
    kein weiteres Drittpaket (numpy, pandas, yfinance, scipy, …) laden – sonst bricht der Job auf dem Runner still ab."""
    schritte = _alle_schritte(_lade_wf("options_capture.yml"))
    pip = [s["run"] for s in schritte if "pip install" in s.get("run", "")]
    assert len(pip) == 1
    pakete = set(shlex.split(pip[0])[2:])
    assert pakete == {"requests", "pyyaml"}, pakete
    erlaubt = {"requests", "yaml", "urllib3", "certifi", "idna", "charset_normalizer", "chardet"}
    code = textwrap.dedent("""
        import importlib.abc, json, sys
        from datetime import datetime, timezone
        from pathlib import Path
        ERLAUBT = set(sys.argv[1].split(","))
        REPO = sys.argv[2]
        class Sperre(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                top = name.split(".")[0]
                if top in sys.stdlib_module_names or top in ERLAUBT or top in ("modules", "rth_marks") or top.startswith("_"):
                    return None
                raise ImportError("im Capture-Workflow nicht installiert: " + name)
        sys.meta_path.insert(0, Sperre())
        sys.path[:0] = [REPO, REPO + "/scripts"]
        import rth_marks
        res = rth_marks.run(datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc), Path("keine_history.json"), capture_only=True)
        print(json.dumps(res))
    """)
    r = subprocess.run([sys.executable, "-c", code, ",".join(sorted(erlaubt)), str(REPO)], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    assert json.loads(r.stdout.strip().splitlines()[-1])["window"] == "IN_WINDOW"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 5. workflow_health: Scanner-Guard in New-York-Zeit, Börsenfeiertage, Watchdog
# ══════════════════════════════════════════════════════════════════════════════════════════════
def scanner_cfg(*, artifact: bool = True, requires=()) -> dict:
    wf = {"file": "scanner.yml", "cron_utc": "13:30", "days": [0, 1, 2, 3, 4], "trading_days_only": True,
          "max_delay_hours": 9}
    if artifact:
        wf["artifact"] = {"kind": "file", "path": "outputs/daily_reports/{date}.json"}
    gate = {"not_before_ny": "09:30"}
    if requires:
        gate["requires"] = list(requires)
    return {"workflows": {"scanner": wf}, "gates": {"scanner": gate}}


@pytest.mark.parametrize("ts, status", [
    # Sommer: 09:30 ET = 13:30 UTC
    ("2026-10-07T13:29:59", "SKIP_BEFORE_WINDOW"), ("2026-10-07T13:30:00", "RUN"), ("2026-10-07T14:30:00", "RUN"),
    # Winter: 09:30 ET = 14:30 UTC (13:31 UTC am 04.11. ist 08:31 ET)
    ("2026-11-04T13:30:00", "SKIP_BEFORE_WINDOW"), ("2026-11-04T13:31:00", "SKIP_BEFORE_WINDOW"),
    ("2026-11-04T14:29:59", "SKIP_BEFORE_WINDOW"), ("2026-11-04T14:30:00", "RUN"), ("2026-11-04T14:31:00", "RUN"),
    # Wochen der Zeitumstellung 2026 und 2027
    ("2026-03-06T13:31:00", "SKIP_BEFORE_WINDOW"), ("2026-03-06T14:30:00", "RUN"),       # Fr, noch EST
    ("2026-03-09T13:29:59", "SKIP_BEFORE_WINDOW"), ("2026-03-09T13:30:00", "RUN"),       # Mo, schon EDT
    ("2026-10-30T13:30:00", "RUN"),                                                      # Fr, noch EDT
    ("2026-11-02T13:30:00", "SKIP_BEFORE_WINDOW"), ("2026-11-02T14:30:00", "RUN"),       # Mo, schon EST
    ("2027-03-12T13:30:00", "SKIP_BEFORE_WINDOW"), ("2027-03-12T14:30:00", "RUN"),
    ("2027-03-15T13:30:00", "RUN"), ("2027-11-05T13:30:00", "RUN"),
    ("2027-11-08T13:30:00", "SKIP_BEFORE_WINDOW"), ("2027-11-08T14:30:00", "RUN"),
    # Börsenfeiertage und Sonderschließung (auch zu später Stunde, Priorität vor dem Zeitfenster)
    ("2026-11-26T15:00:00", "SKIP_MARKET_CLOSED"), ("2026-11-26T13:31:00", "SKIP_MARKET_CLOSED"),
    ("2026-07-03T15:00:00", "SKIP_MARKET_CLOSED"), ("2025-01-09T15:00:00", "SKIP_MARKET_CLOSED"),
    ("2026-12-25T15:00:00", "SKIP_MARKET_CLOSED"), ("2027-12-24T15:00:00", "SKIP_MARKET_CLOSED"),
    ("2027-06-18T15:00:00", "SKIP_MARKET_CLOSED"), ("2026-04-03T15:00:00", "SKIP_MARKET_CLOSED"),
    # verkürzte Sitzungen sind Handelstage
    ("2026-11-27T14:31:00", "RUN"), ("2026-12-24T14:31:00", "RUN"),
    # Wochenende
    ("2026-11-28T15:00:00", "SKIP_NOT_TRADING_DAY"), ("2026-11-29T15:00:00", "SKIP_NOT_TRADING_DAY"),
])
def test_scanner_guard_in_new_york_zeit_und_kalender(tmp_path, ts, status):
    assert wh.upstream_ready("scanner", utc(ts), root=tmp_path, cfg=scanner_cfg())["status"] == status


def test_scanner_guard_laeuft_nie_zweimal_pro_tag(tmp_path):
    (tmp_path / "outputs" / "daily_reports").mkdir(parents=True)
    (tmp_path / "outputs" / "daily_reports" / "2026-10-07.json").write_text("{}")
    cfg = scanner_cfg()
    assert wh.upstream_ready("scanner", utc("2026-10-07T14:30"), root=tmp_path, cfg=cfg)["status"] == "SKIP_ALREADY_RAN"
    assert wh.upstream_ready("scanner", utc("2026-10-08T14:30"), root=tmp_path, cfg=cfg)["status"] == "RUN"


def test_scanner_guard_mit_repo_config_new_york_gate_und_feiertag(tmp_path):
    cfg = wh.load_cfg(REPO / "config" / "workflow_schedule.yaml")
    assert cfg["gates"]["scanner"]["not_before_ny"] == "09:30" and "not_before_utc" not in cfg["gates"]["scanner"]
    assert cfg["workflows"]["scanner"]["trading_days_only"] is True
    assert wh.upstream_ready("scanner", utc("2026-11-04T13:31"), root=tmp_path, cfg=cfg)["status"] == "SKIP_BEFORE_WINDOW"
    assert wh.upstream_ready("scanner", utc("2026-11-26T15:00"), root=tmp_path, cfg=cfg)["status"] == "SKIP_MARKET_CLOSED"
    # im Fenster, aber ohne frischen Source-Health-Snapshot: nie still mit alten Daten laufen
    assert wh.upstream_ready("scanner", utc("2026-11-04T14:31"), root=tmp_path, cfg=cfg)["status"] == "UPSTREAM_NOT_READY"
    snap = tmp_path / "outputs" / "health" / "source_health_snapshot.json"
    snap.parent.mkdir(parents=True)
    snap.write_text(json.dumps({"generated": "2026-11-04T12:50:00+00:00"}))
    assert wh.upstream_ready("scanner", utc("2026-11-04T14:31"), root=tmp_path, cfg=cfg)["status"] == "RUN"


def test_upstream_guard_cli_schreibt_run_und_status_nach_github_output(monkeypatch, tmp_path, capsys):
    """Die Workflow-Verdrahtung (needs.guard.outputs.run == 'true'): Feiertag -> run=false, --force -> run=true (protokolliert)."""
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    skip = {"job": "scanner", "run": False, "status": "SKIP_MARKET_CLOSED", "reason": "Feiertag", "checks": [], "advisory": []}
    monkeypatch.setattr(upstream_guard, "upstream_ready", lambda job, now: dict(skip))
    assert upstream_guard.main(["scanner"]) == 0
    assert out.read_text().splitlines() == ["run=false", "status=SKIP_MARKET_CLOSED"]
    out.unlink()
    assert upstream_guard.main(["scanner", "--force"]) == 0
    assert out.read_text().splitlines() == ["run=true", "status=FORCED"]
    assert "übersteuert" in capsys.readouterr().out


@pytest.mark.parametrize("ts", ["2026-11-26T23:50", "2026-11-26T13:35", "2026-07-03T23:50", "2025-01-09T23:50",
                                "2026-12-25T23:50", "2027-12-24T23:50", "2027-06-18T23:50"])
def test_watchdog_erwartet_an_boersenfeiertagen_keinen_scanner_lauf(ts):
    for runs in ([], None, [{"created_at": "2026-01-01T00:00:00Z", "status": "completed", "conclusion": "failure"}]):
        plan = wh.plan_recovery(utc(ts), {"scanner": runs}, cfg=scanner_cfg())["scanner"]
        assert plan["state"] == "NOT_SCHEDULED" and plan["market_closed"] is True and plan["action"] is None


@pytest.mark.parametrize("ts", ["2026-11-25T23:50", "2026-11-27T23:50", "2026-12-24T23:50", "2026-11-04T23:50"])
def test_watchdog_plant_an_handelstagen_inklusive_kurztagen_recovery(ts):
    plan = wh.plan_recovery(utc(ts), {"scanner": []}, cfg=scanner_cfg())["scanner"]
    assert plan["state"] == "MISSED" and plan["action"] == "DISPATCH" and not plan.get("market_closed")


def test_watchdog_mit_repo_config_an_feiertag_nur_scanner_ist_nicht_geplant():
    cfg = wh.load_cfg(REPO / "config" / "workflow_schedule.yaml")
    runs = {j: [{"created_at": "2026-11-26T14:00:00Z", "status": "completed", "conclusion": "success", "event": "schedule"}]
            for j in cfg["workflows"]}
    plan = wh.plan_recovery(utc("2026-11-26T23:50"), runs, cfg=cfg)
    assert plan["scanner"]["state"] == "NOT_SCHEDULED" and plan["scanner"]["market_closed"] is True
    assert plan["scanner"]["action"] is None
    leer = wh.plan_recovery(utc("2026-11-26T23:50"), {j: [] for j in cfg["workflows"]}, cfg=cfg)
    assert leer["scanner"]["action"] is None and leer["scanner"]["state"] == "NOT_SCHEDULED"


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 6. Workflows und Persistenz
# ══════════════════════════════════════════════════════════════════════════════════════════════
def _lade_wf(name: str) -> dict:
    return yaml.safe_load((REPO / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def _trigger(doc: dict) -> dict:
    """PyYAML parst den Schlüssel `on` als True."""
    return doc[True] if True in doc else doc["on"]


def _alle_schritte(doc: dict) -> list[dict]:
    return [s for job in doc["jobs"].values() for s in job["steps"]]


def _crons(name: str) -> list[tuple[int, int]]:
    out = []
    for c in _trigger(_lade_wf(name))["schedule"]:
        teile = c["cron"].split()
        assert len(teile) == 5 and teile[2:] == ["*", "*", "1-5"], f"{name}: Cron nicht Mo–Fr: {c['cron']}"
        out.append((int(teile[1]), int(teile[0])))
    return out


def _push_pfade(schritt: dict) -> list[str]:
    """Pfad-Argumente des ci_push.sh-Aufrufs (nach der Commit-Nachricht)."""
    zeile = next(z for z in schritt["run"].splitlines() if "ci_push.sh" in z)
    teile = shlex.split(zeile)
    return teile[teile.index("scripts/ci_push.sh") + 2:]


def _schritt(doc: dict, name: str) -> tuple[int, dict]:
    liste = doc["jobs"]["scan"]["steps"] if "scan" in doc["jobs"] else _alle_schritte(doc)
    i = next(i for i, s in enumerate(liste) if s.get("name") == name)
    return i, liste[i]


# ── options_capture.yml ───────────────────────────────────────────────────────
def test_options_capture_eigene_concurrency_gruppe_und_nur_archiv_push():
    doc = _lade_wf("options_capture.yml")
    gruppe = doc["concurrency"]["group"]
    andere = {_lade_wf(n)["concurrency"]["group"] for n in ("scanner.yml", "feedback.yml")}
    assert gruppe and gruppe not in andere and gruppe != "history-write"
    assert doc["concurrency"]["cancel-in-progress"] is False
    schritte = _alle_schritte(doc)
    pushes = [s for s in schritte if "ci_push.sh" in s.get("run", "")]
    assert len(pushes) == 1 and _push_pfade(pushes[0]) == ["outputs/options_archive/"]
    assert doc["permissions"] == {"contents": "write"}


def test_options_capture_ruft_nur_den_capture_modus_auf():
    schritte = _alle_schritte(_lade_wf("options_capture.yml"))
    text = "\n".join(s.get("run", "") for s in schritte)
    pythons = re.findall(r"python3?\s+([^\n;&|]+)", text)
    assert [p.strip() for p in pythons] == ["scripts/rth_marks.py --capture-only"]
    for verboten in ("pipeline.py", "feedback.py", "update_outcomes", "history.json", "modules.promotion", "meta_label"):
        assert verboten not in text
    marks = next(s for s in schritte if "rth_marks.py" in s.get("run", ""))
    assert "TRADIER_API_KEY" in marks["env"]
    for s in schritte:                                             # sonst nur Checkout, Python, Artefakt
        if "uses" in s:
            assert s["uses"].split("@")[0] in {"actions/checkout", "actions/setup-python", "actions/upload-artifact"}


def test_options_capture_fehlerpfad_artefakt_nur_archiv():
    up = [s for s in _alle_schritte(_lade_wf("options_capture.yml")) if str(s.get("uses", "")).startswith("actions/upload-artifact@")]
    assert len(up) == 1 and up[0]["if"] == "failure()" and up[0]["with"]["path"].strip() == "outputs/options_archive/"


def test_options_capture_crons_sechs_stueck_mo_bis_fr():
    crons = _crons("options_capture.yml")
    assert len(crons) == 6 and len(set(crons)) == 6
    assert "workflow_dispatch" in _trigger(_lade_wf("options_capture.yml"))


def _trifft(tag: date, crons, verzoegerung_min: int) -> bool:
    return any(rth_marks.capture_window(datetime(tag.year, tag.month, tag.day, h, m, tzinfo=UTC)
                                        + timedelta(minutes=verzoegerung_min))[0] for h, m in crons)


@pytest.mark.parametrize("tag", ["2026-07-15", "2027-01-13", "2026-03-09", "2026-11-02", "2025-07-03", "2026-11-27",
                                 "2026-12-24", "2026-03-06", "2026-10-30", "2027-03-12", "2027-03-15", "2027-11-08"])
def test_capture_crons_treffen_das_fenster_bei_jeder_verzoegerung_bis_7h(tag):
    crons = _crons("options_capture.yml")
    for verz in range(0, 7 * 60 + 1, 5):
        assert _trifft(d(tag), crons, verz), f"{tag}: kein Cron im Fenster bei {verz} min GitHub-Verzögerung"


def test_capture_crons_alle_handelstage_2025_bis_2027_bei_jeder_verzoegerung_bis_7h():
    crons = _crons("options_capture.yml")
    schlecht = []
    tag = d("2025-06-01")
    while tag <= d("2027-12-31"):
        if ms.is_trading_day(tag):
            schlecht += [(tag.isoformat(), v) for v in range(0, 421, 5) if not _trifft(tag, crons, v)]
        tag += timedelta(days=1)
    assert not schlecht, schlecht[:5]


def test_capture_crons_an_boersenfeiertagen_nie_im_fenster():
    """Feiertag/Sonderschließung: kein Abruf, egal wie die Verzögerung ausfällt."""
    crons = _crons("options_capture.yml")
    for tag in ("2026-11-26", "2026-07-03", "2025-01-09", "2026-12-25", "2027-12-24"):
        assert not any(_trifft(d(tag), crons, v) for v in range(0, 7 * 60 + 1, 5)), tag


# ── scanner.yml ───────────────────────────────────────────────────────────────
def test_scanner_zwei_crons_dst_fest():
    assert sorted(_crons("scanner.yml")) == [(13, 30), (14, 30)]
    doc = _lade_wf("scanner.yml")
    assert doc["concurrency"]["group"] == "history-write" and doc["concurrency"]["cancel-in-progress"] is False
    # 13:30 UTC = 09:30 EDT (Sommer), 14:30 UTC = 09:30 EST (Winter): in jeder Jahreszeit startet mindestens einer RUN
    for tag, cron, erwartet in [("2026-10-07", "13:30", "RUN"), ("2026-11-04", "13:30", "SKIP_BEFORE_WINDOW"),
                                ("2026-11-04", "14:30", "RUN")]:
        assert wh.upstream_ready("scanner", utc(f"{tag}T{cron}"), root=Path("/nonexistent"),
                                 cfg=scanner_cfg(artifact=False))["status"] == erwartet


def test_scanner_marks_schritt_vor_pipeline_und_commit_direkt_danach():
    doc = _lade_wf("scanner.yml")
    namen = [s.get("name") for s in doc["jobs"]["scan"]["steps"]]
    i_marks, i_commit, i_scan = (namen.index(n) for n in ("Options-Marks (RTH)", "Commit Options-Marks", "Run Scanner"))
    assert i_marks < i_commit < i_scan
    assert i_commit == i_marks + 1                               # nichts dazwischen, das outputs/ verändern könnte
    marks, commit = doc["jobs"]["scan"]["steps"][i_marks], doc["jobs"]["scan"]["steps"][i_commit]
    assert marks["run"].strip() == "python scripts/rth_marks.py" and marks.get("continue-on-error") is True
    assert marks["id"] == "rthmarks" and "rthmarks" in commit["if"]
    assert _push_pfade(commit) == ["outputs/"] and commit.get("continue-on-error") is True


def test_scanner_fehlerpfad_sichert_nur_beobachtungen_und_laedt_artefakt_hoch():
    schritte = _lade_wf("scanner.yml")["jobs"]["scan"]["steps"]
    fehler = [s for s in schritte if s.get("if") == "failure()"]
    commit = [s for s in fehler if "ci_push.sh" in s.get("run", "")]
    assert len(commit) == 1 and sorted(_push_pfade(commit[0])) == ["outputs/monitoring/", "outputs/options_archive/"]
    namen = [s.get("name") for s in schritte]
    assert namen.index(commit[0]["name"]) > namen.index("Run Scanner")
    up = [s for s in fehler if str(s.get("uses", "")).startswith("actions/upload-artifact@")]
    assert len(up) == 1 and "outputs/options_archive/" in up[0]["with"]["path"]
    assert namen.index(up[0]["name"]) > namen.index("Run Scanner")


def test_scanner_schrittfolge_bleibt_erhalten():
    namen = [s.get("name") for s in _lade_wf("scanner.yml")["jobs"]["scan"]["steps"]]
    for n in ("Run Scanner", "Run Feedback Loop", "Ops-Monitoring (read-only)", "Commit daily report",
              "UNIVERSE_V2 Shadow-Scan", "Stufen-Bilanz (V2-Shadow-Scan ausgefallen -> Lauf rot, nach dem Commit)"):
        assert n in namen, n
    assert namen.index("Run Scanner") < namen.index("Run Feedback Loop") < namen.index("Commit daily report")


# ── feedback.yml ──────────────────────────────────────────────────────────────
def test_feedback_fehlerpfad_sichert_quote_archiv_und_artefakt():
    doc = _lade_wf("feedback.yml")
    assert doc["concurrency"]["group"] == "history-write"
    schritte = doc["jobs"]["feedback"]["steps"]
    namen = [s.get("name") for s in schritte]
    fehler = [s for s in schritte if s.get("if") == "failure()"]
    commit = [s for s in fehler if "ci_push.sh" in s.get("run", "")]
    assert len(commit) == 1 and _push_pfade(commit[0]) == ["outputs/options_archive/"]      # nie history.json/Ledger
    assert namen.index(commit[0]["name"]) > namen.index("Run Feedback Loop")
    up = [s for s in fehler if str(s.get("uses", "")).startswith("actions/upload-artifact@")]
    assert len(up) == 1 and "outputs/options_archive/" in up[0]["with"]["path"]


# ── .gitattributes / .gitignore ───────────────────────────────────────────────
def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("pfad", ["outputs/options_archive/quotes/2026-10-12.jsonl", "outputs/monitoring/ops_history.jsonl",
                                  "outputs/candidate_ledger/2026-10.jsonl"])
def test_gitattributes_jsonl_union_merge(pfad):
    r = _git("check-attr", "merge", "--", pfad)
    assert r.returncode == 0 and r.stdout.strip().endswith("merge: union"), r.stdout


@pytest.mark.parametrize("pfad", ["outputs/intelligence/promotion_transitions.jsonl", "outputs/intelligence/contract_registry.jsonl"])
def test_gitattributes_hash_verkettete_logs_bleiben_normaler_merge(pfad):
    assert _git("check-attr", "merge", "--", pfad).stdout.strip().endswith("merge: text")


def test_gitattributes_regel_steht_woertlich_drin():
    zeilen = [z.strip() for z in (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()]
    assert "outputs/**/*.jsonl merge=union" in zeilen


@pytest.mark.parametrize("pfad", ["outputs/options_archive/quotes/2026-10-12.jsonl", "outputs/models/meta_label/registry.json",
                                  "outputs/monitoring/ops_history.jsonl", "outputs/candidate_ledger/2026-10.jsonl",
                                  "outputs/history.json", "outputs/daily_reports/2026-10-12.json"])
def test_persistente_daten_sind_nicht_gitignored(pfad):
    r = _git("check-ignore", "-q", "--", pfad)
    assert r.returncode == 1, f"{pfad} ist per .gitignore ausgeschlossen (check-ignore rc={r.returncode}) – ginge nie nach main"


# ── scripts/ci_push.sh gegen ein lokales Bare-Repo (kein Netzwerk): parallele Schreiber ──────────
def _ci_sandbox(tmp_path: Path):
    """Bare-Remote + zwei Klone mit dem echten .gitattributes; die Push-URL von ci_push.sh (github.com) wird per
    url.insteadOf auf das lokale Bare-Repo umgebogen, Protokolle außer file sind gesperrt (kein Netzwerk)."""
    bare = tmp_path / "remote.git"
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(HOME=str(tmp_path), GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_ALLOW_PROTOCOL="file",
               GIT_TERMINAL_PROMPT="0", GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t", GITHUB_TOKEN="tok", GITHUB_REPOSITORY="o/r", GIT_CONFIG_COUNT="1",
               GIT_CONFIG_KEY_0=f"url.{bare}.insteadOf", GIT_CONFIG_VALUE_0="https://x-access-token:tok@github.com/o/r.git")

    def git(*args, cwd):
        r = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (args, r.stderr)
        return r.stdout

    git("init", "-q", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    seed = tmp_path / "seed"
    git("init", "-q", "-b", "main", str(seed), cwd=tmp_path)
    (seed / ".gitattributes").write_text((REPO / ".gitattributes").read_text(encoding="utf-8"), encoding="utf-8")
    tag = seed / "outputs" / "options_archive" / "quotes"
    tag.mkdir(parents=True)
    (tag / "2026-10-14.jsonl").write_text('{"id": 0}\n', encoding="utf-8")
    (seed / "outputs" / "history.json").write_text('{"v": 0}\n', encoding="utf-8")
    (seed / "outputs" / "state").mkdir()
    (seed / "outputs" / "state" / "workflow_status.json").write_text('{"v": 0}\n', encoding="utf-8")
    git("add", "-A", cwd=seed)
    git("commit", "-q", "-m", "seed", cwd=seed)
    git("push", "-q", str(bare), "HEAD:main", cwd=seed)
    c1, c2 = tmp_path / "c1", tmp_path / "c2"
    git("clone", "-q", str(bare), str(c1), cwd=tmp_path)
    git("clone", "-q", str(bare), str(c2), cwd=tmp_path)

    def ci_push(clone: Path, msg: str, *pfade: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(REPO / "scripts" / "ci_push.sh"), msg, *pfade], cwd=clone, env=env,
                              capture_output=True, text=True, timeout=120)

    def remote_datei(pfad: str) -> str:
        return git("--git-dir", str(bare), "show", f"main:{pfad}", cwd=tmp_path)
    return c1, c2, ci_push, remote_datei


def test_ci_push_vereinigt_parallele_appends_am_quote_archiv_ohne_verlust(tmp_path):
    """Capture-Job und Scanner/Feedback schreiben dieselbe Tagesdatei (append-only): beide Seiten bleiben erhalten."""
    c1, c2, ci_push, remote_datei = _ci_sandbox(tmp_path)
    pfad = "outputs/options_archive/quotes/2026-10-14.jsonl"
    with open(c1 / pfad, "a", encoding="utf-8") as f:
        f.write('{"id": "capture"}\n')
    with open(c2 / pfad, "a", encoding="utf-8") as f:                       # c2 kennt den Push von c1 noch nicht
        f.write('{"id": "scanner"}\n')
    with open(c2 / "outputs" / "candidate_ledger.jsonl", "a", encoding="utf-8") as f:
        f.write('{"id": "ledger"}\n')
    r1 = ci_push(c1, "chore: options capture", "outputs/options_archive/")
    r2 = ci_push(c2, "chore: options marks", "outputs/")
    assert r1.returncode == 0 and r2.returncode == 0, (r1.stdout + r1.stderr, r2.stdout + r2.stderr)
    zeilen = remote_datei(pfad).splitlines()
    assert sorted(zeilen) == sorted(['{"id": 0}', '{"id": "capture"}', '{"id": "scanner"}'])      # nichts doppelt, nichts weg
    assert remote_datei("outputs/candidate_ledger.jsonl").strip() == '{"id": "ledger"}'


def test_ci_push_nur_archiv_pfad_laesst_fremde_aenderungen_unberuehrt(tmp_path):
    """Der Capture-Push (nur outputs/options_archive/) nimmt eine zufällig veränderte history.json nie mit."""
    c1, _c2, ci_push, remote_datei = _ci_sandbox(tmp_path)
    (c1 / "outputs" / "history.json").write_text('{"v": "halbfertig"}\n', encoding="utf-8")
    with open(c1 / "outputs/options_archive/quotes/2026-10-14.jsonl", "a", encoding="utf-8") as f:
        f.write('{"id": "capture"}\n')
    r = ci_push(c1, "chore: options capture", "outputs/options_archive/")
    assert r.returncode == 0, r.stdout + r.stderr
    assert remote_datei("outputs/history.json") == '{"v": 0}\n'
    assert '{"id": "capture"}' in remote_datei("outputs/options_archive/quotes/2026-10-14.jsonl")


def test_ci_push_konflikt_in_history_json_bricht_ab_ohne_remote_zu_veraendern(tmp_path):
    c1, c2, ci_push, remote_datei = _ci_sandbox(tmp_path)
    (c1 / "outputs" / "history.json").write_text('{"v": "scanner"}\n', encoding="utf-8")
    (c2 / "outputs" / "history.json").write_text('{"v": "feedback"}\n', encoding="utf-8")
    assert ci_push(c1, "A", "outputs/").returncode == 0
    r = ci_push(c2, "B", "outputs/")
    assert r.returncode != 0 and "Push-Konflikt" in r.stdout + r.stderr          # fail-closed: Lauf rot, nie still eine Seite verworfen
    assert remote_datei("outputs/history.json") == '{"v": "scanner"}\n'


def test_ci_push_konflikt_in_abgeleiteter_state_datei_gilt_die_version_dieses_laufs(tmp_path):
    c1, c2, ci_push, remote_datei = _ci_sandbox(tmp_path)
    (c1 / "outputs/state/workflow_status.json").write_text('{"v": "lauf1"}\n', encoding="utf-8")
    (c2 / "outputs/state/workflow_status.json").write_text('{"v": "lauf2"}\n', encoding="utf-8")
    assert ci_push(c1, "A", "outputs/").returncode == 0
    r = ci_push(c2, "B", "outputs/")
    assert r.returncode == 0, r.stdout + r.stderr
    assert remote_datei("outputs/state/workflow_status.json") == '{"v": "lauf2"}\n'


# ── Sicherheits-Schalter ──────────────────────────────────────────────────────
def test_kill_switch_und_promotion_policy_unveraendert_restriktiv():
    ks = yaml.safe_load((REPO / "config" / "production_kill_switch.yaml").read_text(encoding="utf-8"))
    assert ks["active"] is False
    pol = yaml.safe_load((REPO / "config" / "promotion_policy.yaml").read_text(encoding="utf-8"))
    assert pol["max_automatic_influence"] == "ABSTENTION_ONLY"


@pytest.mark.parametrize("name", ["options_capture.yml", "scanner.yml", "feedback.yml"])
def test_workflows_enthalten_keine_order_oder_broker_aufrufe(name):
    text = (REPO / ".github" / "workflows" / name).read_text(encoding="utf-8")
    treffer = re.findall(r"(?i)\b(?:order\w*|broker\w*|place_order|submit_order|alpaca|ibkr|interactive\s*brokers)\b", text)
    assert treffer == []


@pytest.mark.parametrize("pfad", ["scripts/rth_marks.py", "modules/option_quotes.py"])
def test_neue_betriebsmodule_enthalten_keine_order_oder_broker_aufrufe(pfad):
    text = (REPO / pfad).read_text(encoding="utf-8")
    assert re.findall(r"(?i)\b(?:broker\w*|place_order|submit_order|alpaca|ibkr)\b", text) == []
