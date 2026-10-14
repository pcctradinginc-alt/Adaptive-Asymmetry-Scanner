"""Tests für modules/option_quotes.py – Archiv beobachteter Optionsquotes (kein Netzwerk)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from modules import option_quotes as oq

UTC = timezone.utc
RA = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
SYM = "AAPL261218C00150000"
ALL_KEYS = {
    "schema_version", "record_id", "occ_symbol", "underlying", "expiry", "strike", "right",
    "multiplier", "multiplier_assumed", "bid", "ask", "mid", "spread", "spread_pct", "last",
    "volume", "open_interest", "iv", "delta", "gamma", "theta", "vega", "quote_time",
    "retrieved_at", "source", "context", "price_type", "quality_flags", "quality_status",
    "payload_hash",
}


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _raw(**kw) -> dict:
    """Saubere Tradier-ähnliche Quote (Greeks verschachtelt wie bei greeks=true)."""
    r = {
        "symbol": SYM, "expiration_date": "2026-12-18", "strike": 150.0, "option_type": "call",
        "bid": 4.9, "ask": 5.1, "last": 5.0, "volume": 120, "open_interest": 900,
        "contract_size": 100, "bid_date": _ms(RA - timedelta(seconds=10)),
        "greeks": {"mid_iv": 0.31, "delta": 0.55, "gamma": 0.04, "theta": -0.06, "vega": 0.17},
    }
    r.update(kw)
    return r


def _norm(raw=None, **kw) -> dict:
    kw.setdefault("retrieved_at", RA)
    rec = oq.normalize(_raw() if raw is None else raw, underlying="AAPL", source="tradier",
                       context="chain_select", **kw)
    assert rec is not None
    return rec


# --------------------------------------------------------------------------- occ_symbol
def test_occ_symbol_call_put_and_fraction():
    assert oq.occ_symbol("AAPL", "2026-11-20", "call", 150.0) == "AAPL261120C00150000"
    assert oq.occ_symbol("AAPL", "2026-11-20", "put", 150.0) == "AAPL261120P00150000"
    assert oq.occ_symbol("AAPL", "2026-11-20", "C", 152.5) == "AAPL261120C00152500"
    assert oq.occ_symbol("SPY", "2027-01-15", "P", 0.5) == "SPY270115P00000500"
    assert oq.occ_symbol("BRK.B", "2026-11-20", "call", 450) == "BRK.B261120C00450000"


@pytest.mark.parametrize("args", [
    ("", "2026-11-20", "call", 150.0),
    ("AAPL", "kein-datum", "call", 150.0),
    ("AAPL", "2026-11-20", "straddle", 150.0),
    ("AAPL", "2026-11-20", "call", 0),
    ("AAPL", "2026-11-20", "call", -5),
    ("AAPL", "2026-11-20", "call", float("nan")),
    ("AAPL", "2026-11-20", "call", None),
    ("AAPL", None, "call", 150.0),
])
def test_occ_symbol_invalid_raises(args):
    with pytest.raises(ValueError):
        oq.occ_symbol(*args)


# --------------------------------------------------------------------------- normalize
def test_normalize_clean_tradier_quote():
    rec = _norm()
    assert set(rec) == ALL_KEYS
    assert rec["schema_version"] == oq.SCHEMA_VERSION
    assert rec["occ_symbol"] == SYM
    assert (rec["underlying"], rec["expiry"], rec["strike"], rec["right"]) == ("AAPL", "2026-12-18", 150.0, "call")
    assert rec["mid"] == pytest.approx(5.0)
    assert rec["spread"] == pytest.approx(0.2)
    assert rec["spread_pct"] == pytest.approx(0.04)
    assert rec["iv"] == pytest.approx(0.31)
    assert rec["delta"] == pytest.approx(0.55)
    assert rec["multiplier"] == 100 and rec["multiplier_assumed"] is False
    assert rec["quote_time"] == "2026-11-20T14:59:50.000+00:00"
    assert rec["retrieved_at"] == "2026-11-20T15:00:00.000+00:00"
    assert rec["quality_flags"] == []
    assert rec["quality_status"] == "OK"
    assert rec["price_type"] == oq.PRICE_TYPE == "OBSERVED_QUOTE"
    assert len(rec["record_id"]) == 20 and len(rec["payload_hash"]) == 64


def test_normalize_flat_aliases_and_symbol_parsing():
    raw = {"occ_symbol": "aapl261218p00152500", "bid": 1.0, "ask": 1.1, "impliedVolatility": 0.4,
           "ts": "2026-11-20T14:59:00Z"}
    rec = _norm(raw)
    assert (rec["occ_symbol"], rec["expiry"], rec["strike"], rec["right"]) == (
        "AAPL261218P00152500", "2026-12-18", 152.5, "put")
    assert rec["iv"] == 0.4
    assert rec["quote_time"] == "2026-11-20T14:59:00.000+00:00"


def test_normalize_builds_symbol_from_components():
    raw = {"symbol": "AAPL", "expiry": "2026-12-18", "right": "P", "strike": 152.5, "bid": 1, "ask": 1.2}
    assert _norm(raw)["occ_symbol"] == "AAPL261218P00152500"   # "AAPL" ist kein Optionssymbol


def test_normalize_no_symbol_returns_none():
    for raw in ({}, {"bid": 1, "ask": 2}, {"strike": 150, "right": "call"}, {"symbol": "AAPL"}, None, "x", 5):
        assert oq.normalize(raw, underlying="AAPL", source="t", context="c", retrieved_at=RA) is None
    assert oq.normalize(_raw(), underlying="AAPL", source="t", context="c", retrieved_at="kein-datum") is None


def test_missing_fields_stay_none_never_zero():
    rec = _norm({"symbol": SYM})
    for k in ("bid", "ask", "mid", "spread", "spread_pct", "last", "volume", "open_interest",
              "iv", "delta", "gamma", "theta", "vega", "quote_time"):
        assert rec[k] is None, k
    assert {"NO_BID", "NO_ASK", "IV_MISSING", "NO_QUOTE_TIME"} <= set(rec["quality_flags"])
    assert "OI_ZERO" not in rec["quality_flags"]      # fehlend != 0
    assert rec["quality_status"] == "UNUSABLE"


def test_non_finite_and_garbage_numbers_become_none():
    rec = _norm(_raw(bid=float("nan"), ask=float("inf"), last="abc", volume=None, delta=float("-inf")))
    assert rec["bid"] is None and rec["ask"] is None and rec["last"] is None
    assert rec["volume"] is None and rec["delta"] is None
    assert "INVALID_VALUE" not in rec["quality_flags"]
    assert rec["mid"] is None and rec["spread"] is None


def test_crossed():
    rec = _norm(_raw(bid=5.2, ask=5.0))
    assert "CROSSED" in rec["quality_flags"] and "LOCKED" not in rec["quality_flags"]
    assert rec["mid"] is None and rec["spread"] == pytest.approx(-0.2)
    assert rec["quality_status"] == "UNUSABLE"


def test_locked():
    rec = _norm(_raw(bid=5.0, ask=5.0))
    assert "LOCKED" in rec["quality_flags"] and "CROSSED" not in rec["quality_flags"]
    assert rec["mid"] == pytest.approx(5.0) and rec["spread"] == 0
    assert rec["quality_status"] == "SUSPECT"


def test_zero_bid():
    rec = _norm(_raw(bid=0, ask=0.05))
    assert rec["bid"] == 0                              # echter Wert 0 bleibt 0
    assert "ZERO_BID" in rec["quality_flags"] and "NO_BID" not in rec["quality_flags"]
    assert rec["mid"] is None and rec["spread"] == pytest.approx(0.05) and rec["spread_pct"] is None
    assert rec["quality_status"] == "SUSPECT"


def test_no_bid_is_suspect_no_ask_is_unusable():
    rec = _norm(_raw(bid=None))
    assert "NO_BID" in rec["quality_flags"] and rec["quality_status"] == "SUSPECT"
    for ask in (None, 0):
        rec = _norm(_raw(ask=ask))
        assert "NO_ASK" in rec["quality_flags"] and rec["quality_status"] == "UNUSABLE"
        assert rec["mid"] is None


def test_wide_spread():
    rec = _norm(_raw(bid=1.0, ask=2.0))                 # spread/mid = 1.0/1.5
    assert rec["mid"] == pytest.approx(1.5) and rec["spread_pct"] == pytest.approx(1.0 / 1.5, rel=1e-5)
    assert "WIDE_SPREAD" in rec["quality_flags"] and rec["quality_status"] == "SUSPECT"
    assert "WIDE_SPREAD" not in _norm(_raw(bid=1.0, ask=1.1))["quality_flags"]


def test_stale_and_no_quote_time():
    old = _ms(RA - timedelta(hours=oq.STALE_HOURS + 1))
    rec = _norm(_raw(bid_date=old))
    assert "STALE" in rec["quality_flags"] and "NO_QUOTE_TIME" not in rec["quality_flags"]
    assert rec["quality_status"] == "SUSPECT"
    fresh = _norm(_raw(bid_date=_ms(RA - timedelta(hours=oq.STALE_HOURS - 1))))
    assert "STALE" not in fresh["quality_flags"]
    rec = _norm(_raw(bid_date=None))
    assert rec["quote_time"] is None and "NO_QUOTE_TIME" in rec["quality_flags"]
    assert "STALE" not in rec["quality_flags"]          # ohne Zeit nicht als stale erfunden
    assert rec["quality_status"] == "OK"


@pytest.mark.parametrize("field", ["bid", "ask", "last", "volume", "open_interest"])
def test_negative_value_is_invalid_and_unusable(field):
    rec = _norm(_raw(**{field: -1}))
    assert rec[field] is None
    assert "INVALID_VALUE" in rec["quality_flags"] and rec["quality_status"] == "UNUSABLE"


def test_iv_zero_is_missing_iv_negative_invalid():
    rec = _norm(_raw(greeks={"mid_iv": 0.0, "delta": 0.5}))
    assert rec["iv"] is None and "IV_MISSING" in rec["quality_flags"] and "INVALID_VALUE" not in rec["quality_flags"]
    rec = _norm(_raw(greeks={"mid_iv": -0.2}))
    assert rec["iv"] is None and "INVALID_VALUE" in rec["quality_flags"]


def test_oi_zero_flag():
    rec = _norm(_raw(open_interest=0))
    assert rec["open_interest"] == 0 and "OI_ZERO" in rec["quality_flags"]
    assert rec["quality_status"] == "OK"


def test_multiplier_assumed_vs_given():
    raw = _raw()
    del raw["contract_size"]
    rec = _norm(raw)
    assert rec["multiplier"] == 100 and rec["multiplier_assumed"] is True
    assert rec["quality_flags"] == ["MULTIPLIER_ASSUMED"] and rec["quality_status"] == "OK"
    rec = _norm(_raw(multiplier=10))
    assert rec["multiplier"] == 10 and rec["multiplier_assumed"] is False
    assert "MULTIPLIER_ASSUMED" not in rec["quality_flags"]
    rec = _norm(_raw(multiplier=-100))
    assert rec["multiplier_assumed"] is True and "INVALID_VALUE" in rec["quality_flags"]


def test_quote_time_epoch_ms_and_iso():
    dt = datetime(2026, 11, 20, 14, 30, 15, 250000, tzinfo=UTC)
    assert _norm(_raw(quote_time=_ms(dt)))["quote_time"] == "2026-11-20T14:30:15.250+00:00"
    assert _norm(_raw(quote_time=str(_ms(dt))))["quote_time"] == "2026-11-20T14:30:15.250+00:00"
    assert _norm(_raw(quote_time=int(dt.timestamp())))["quote_time"] == "2026-11-20T14:30:15.000+00:00"
    assert _norm(_raw(quote_time="2026-11-20T09:30:15-05:00"))["quote_time"] == "2026-11-20T14:30:15.000+00:00"
    assert _norm(_raw(quote_time="2026-11-20T14:30:15Z"))["quote_time"] == "2026-11-20T14:30:15.000+00:00"
    assert _norm(_raw(quote_time="2026-11-20T14:30:15"))["quote_time"] == "2026-11-20T14:30:15.000+00:00"
    assert _norm(_raw(quote_time="Mist", bid_date=None))["quote_time"] is None


def test_retrieved_at_forms():
    assert _norm(retrieved_at="2026-11-20T10:00:00-05:00")["retrieved_at"] == "2026-11-20T15:00:00.000+00:00"
    assert _norm(retrieved_at=datetime(2026, 11, 20, 15, 0))["retrieved_at"] == "2026-11-20T15:00:00.000+00:00"
    rec = oq.normalize(_raw(), underlying="AAPL", source="t", context="c")   # Default: jetzt (UTC)
    now = datetime.now(UTC)
    assert abs(now - datetime.fromisoformat(rec["retrieved_at"])) < timedelta(minutes=1)


def test_record_id_and_payload_hash():
    a, b = _norm(), _norm()
    assert a["record_id"] == b["record_id"] and a["payload_hash"] == b["payload_hash"]
    assert _norm(_raw(bid=4.8))["payload_hash"] != a["payload_hash"]
    assert _norm(_raw(bid=4.8))["record_id"] == a["record_id"]                 # gleiche Börsenzeit/Quelle/Kontext
    assert _norm(_raw(bid_date=_ms(RA)))["record_id"] != a["record_id"]
    other = oq.normalize(_raw(), underlying="AAPL", source="tradier", context="exit", retrieved_at=RA)
    assert other["record_id"] != a["record_id"]
    no_qt = _raw(bid_date=None)
    assert _norm(no_qt)["record_id"] != _norm(no_qt, retrieved_at=RA + timedelta(seconds=1))["record_id"]


# --------------------------------------------------------------------------- archive / record / read
def test_archive_idempotent(tmp_path):
    recs = [_norm(), _norm(_raw(symbol="AAPL261218P00150000", option_type="put"))]
    assert oq.archive(recs, root=tmp_path) == 2
    assert oq.archive(recs, root=tmp_path) == 0
    lines = (tmp_path / "2026-11-20.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert {json.loads(x)["price_type"] for x in lines} == {"OBSERVED_QUOTE"}
    # Duplikate innerhalb eines Batches werden ebenfalls nur einmal geschrieben
    assert oq.archive([_norm(_raw(bid_date=_ms(RA)))] * 3, root=tmp_path) == 1


def test_archive_groups_by_retrieved_at_date(tmp_path):
    def at(ra: datetime) -> dict:                 # je Abruf eine eigene Börsenquote (kurz vor dem Abruf)
        return _norm(_raw(bid_date=_ms(ra - timedelta(seconds=10))), retrieved_at=ra)
    r1 = at(datetime(2026, 11, 19, 23, 59, 59, tzinfo=UTC))
    r2 = at(datetime(2026, 11, 20, 0, 0, 1, tzinfo=UTC))
    r3 = at(datetime(2026, 11, 21, 1, 0, tzinfo=UTC))
    r3["retrieved_at"] = "2026-11-20T20:00:00-05:00"                    # = 21.11. 01:00 UTC
    assert oq.archive([r1, r2, r3], root=tmp_path) == 3
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == [
        "2026-11-19.jsonl", "2026-11-20.jsonl", "2026-11-21.jsonl"]


def test_archive_same_exchange_quote_after_utc_midnight_is_not_written_again(tmp_path):
    """Regression 2026-10-14: die Schlussquote (Börsenzeit 19:59:59 UTC) wurde um 22:32 UTC und erneut um 01:30 UTC
    (nächster UTC-Tag) abgerufen und landete mit derselben record_id in zwei Tagesdateien."""
    qt = datetime(2026, 11, 20, 19, 59, 59, tzinfo=UTC)
    first = _norm(_raw(bid_date=_ms(qt)), retrieved_at=datetime(2026, 11, 20, 22, 32, tzinfo=UTC))
    again = _norm(_raw(bid_date=_ms(qt)), retrieved_at=datetime(2026, 11, 21, 1, 30, tzinfo=UTC))
    assert first["record_id"] == again["record_id"]
    assert oq.archive([first], root=tmp_path) == 1
    assert oq.archive([again], root=tmp_path) == 0
    assert not (tmp_path / "2026-11-21.jsonl").exists() or (tmp_path / "2026-11-21.jsonl").read_text() == ""
    # Wochenende: Freitagsquote am Montag erneut abgerufen -> ebenfalls kein zweiter Eintrag
    fri = datetime(2026, 11, 20, 20, 30, tzinfo=UTC)
    mon = _norm(_raw(bid_date=_ms(fri - timedelta(seconds=1))), retrieved_at=datetime(2026, 11, 23, 15, 0, tzinfo=UTC))
    fri_rec = _norm(_raw(bid_date=_ms(fri - timedelta(seconds=1))), retrieved_at=fri)
    assert oq.archive([fri_rec], root=tmp_path) == 1
    assert oq.archive([mon], root=tmp_path) == 0
    # eine NEUE Börsenquote am Folgetag wird natürlich geschrieben
    new = _norm(_raw(bid_date=_ms(datetime(2026, 11, 23, 15, 0, tzinfo=UTC) - timedelta(seconds=5))),
                retrieved_at=datetime(2026, 11, 23, 15, 0, tzinfo=UTC))
    assert oq.archive([new], root=tmp_path) == 1


def test_archive_cross_day_dedupe_is_bounded(tmp_path):
    """Nur Tagesdateien zwischen Börsenzeit und Abruf (höchstens DEDUP_LOOKBACK_DAYS zurück) werden geprüft."""
    qt = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
    old = _norm(_raw(bid_date=_ms(qt)), retrieved_at=qt + timedelta(minutes=5))
    late = _norm(_raw(bid_date=_ms(qt)), retrieved_at=qt + timedelta(days=oq.DEDUP_LOOKBACK_DAYS + 2))
    assert oq.archive([old], root=tmp_path) == 1
    assert oq.archive([late], root=tmp_path) == 1          # außerhalb des Fensters: geschrieben (STALE-Kopie)
    assert "STALE" in late["quality_flags"]


def test_by_ids_prefers_first_observation_of_duplicate_record(tmp_path):
    """Liegt dieselbe record_id in zwei Tagesdateien (Altbestand vor dem Dedupe), gilt die früheste Beobachtung –
    eine spätere, nachbörsliche Kopie ersetzt nie die Quote aus dem Mark-Fenster."""
    qt = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
    in_window = _norm(_raw(bid_date=_ms(qt)), retrieved_at=qt + timedelta(minutes=2))
    later = _norm(_raw(bid_date=_ms(qt)), retrieved_at=datetime(2026, 11, 21, 1, 30, tzinfo=UTC))
    (tmp_path / "2026-11-20.jsonl").write_text(json.dumps(in_window) + "\n")
    (tmp_path / "2026-11-21.jsonl").write_text(json.dumps(later) + "\n")
    got = oq.by_ids([in_window["record_id"]], "2026-11-19", "2026-11-22", root=tmp_path)
    assert got[in_window["record_id"]]["retrieved_at"] == in_window["retrieved_at"]
    # Reihenfolge der Dateien spielt keine Rolle: auch umgekehrt geschrieben gewinnt die früheste Abrufzeit
    (tmp_path / "2026-11-20.jsonl").write_text(json.dumps(later | {"retrieved_at": "2026-11-20T23:00:00.000+00:00"})
                                              + "\n" + json.dumps(in_window) + "\n")
    got = oq.by_ids([in_window["record_id"]], "2026-11-19", "2026-11-22", root=tmp_path)
    assert got[in_window["record_id"]]["retrieved_at"] == in_window["retrieved_at"]


def test_archive_rejects_non_observed_or_malformed_records(tmp_path):
    modeled = dict(_norm(), price_type="MODEL_BS")
    wrong_schema = dict(_norm(), schema_version="v0")
    no_id = {k: v for k, v in _norm().items() if k != "record_id"}
    bad_time = dict(_norm(), retrieved_at="gestern")
    assert oq.archive([modeled, wrong_schema, no_id, bad_time, None, "x", {}], root=tmp_path) == 0
    assert list(tmp_path.glob("*.jsonl")) == []
    assert oq.archive(None, root=tmp_path) == 0


def test_archive_never_raises_on_unwritable_root(tmp_path):
    blocker = tmp_path / "datei"
    blocker.write_text("x")
    assert oq.archive([_norm()], root=blocker / "sub") == 0


def test_archive_corrupt_day_file_still_appends(tmp_path):
    path = tmp_path / "2026-11-20.jsonl"
    path.write_text('{"kaputt\n{"auch kaputt\n{"record_id": "x"}\n')
    assert oq.archive([_norm()], root=tmp_path) == 1


def test_record_roundtrip_and_read(tmp_path):
    n = oq.record([_raw(), _raw(symbol="AAPL261218C00155000", strike=155.0)],
                  underlying="AAPL", source="tradier", context="entry", retrieved_at=RA, root=tmp_path)
    assert n == 2
    assert oq.record([_raw()], underlying="AAPL", source="tradier", context="entry",
                     retrieved_at=RA, root=tmp_path) == 0           # idempotent
    rows = oq.read("2026-11-20", root=tmp_path)
    assert len(rows) == 2
    assert all(r["price_type"] == "OBSERVED_QUOTE" and r["context"] == "entry" for r in rows)


def test_record_never_raises_on_garbage(tmp_path):
    kw = dict(underlying="AAPL", source="t", context="c", retrieved_at=RA, root=tmp_path)
    assert oq.record([None, {}, {"strike": "x"}], **kw) == 0
    assert oq.record([None, {}, {"strike": "x"}, _raw()], **kw) == 1
    assert oq.record(None, **kw) == 0
    assert oq.record(42, **kw) == 0
    assert oq.record([_raw()], underlying=None, source="andere", context="c", retrieved_at=RA, root=tmp_path) == 1
    assert oq.record([_raw(symbol="AAPL")], underlying=None, source="t", context="c",
                     retrieved_at=RA, root=tmp_path) == 0
    assert oq.record([_raw()], underlying="AAPL", source="t", context="c", retrieved_at="Mist", root=tmp_path) == 0


def test_read_date_range(tmp_path):
    for d in ("2026-11-18", "2026-11-19", "2026-11-20", "2026-11-23"):
        ra = datetime.fromisoformat(f"{d}T15:00:00+00:00")
        oq.record([_raw(bid_date=_ms(ra - timedelta(seconds=10)))], underlying="AAPL", source="t", context="c",
                  retrieved_at=ra, root=tmp_path)
    days = lambda rows: [r["retrieved_at"][:10] for r in rows]   # noqa: E731
    assert days(oq.read("2026-11-19", "2026-11-20", root=tmp_path)) == ["2026-11-19", "2026-11-20"]
    assert days(oq.read("2026-11-19", "2026-11-19", root=tmp_path)) == ["2026-11-19"]
    assert days(oq.read("2026-11-20", root=tmp_path)) == ["2026-11-20", "2026-11-23"]
    assert oq.read("2026-12-01", root=tmp_path) == []
    assert oq.read("kein-datum", root=tmp_path) == []
    assert oq.read("2026-11-19", root=tmp_path / "gibt-es-nicht") == []


# --------------------------------------------------------------------------- latest_quote (PIT)
def _put(tmp_path, bid, ra, **kw):
    assert oq.record([_raw(bid=bid, ask=bid + 0.2, **kw)], underlying="AAPL", source="tradier",
                     context="mark", retrieved_at=ra, root=tmp_path) == 1


def test_latest_quote_point_in_time_never_returns_later_quote(tmp_path):
    _put(tmp_path, 4.0, "2026-11-19T15:00:00+00:00", bid_date=None)
    _put(tmp_path, 5.0, "2026-11-20T15:00:00+00:00", bid_date=None)
    _put(tmp_path, 6.0, "2026-11-20T18:00:00+00:00", bid_date=None)
    _put(tmp_path, 7.0, "2026-11-23T15:00:00+00:00", bid_date=None)

    def bid(as_of):
        q = oq.latest_quote(SYM, as_of, root=tmp_path)
        return q["bid"] if q else None

    assert bid("2026-11-19T14:59:59+00:00") is None            # vor der ersten Quote
    assert bid("2026-11-19T15:00:00+00:00") == 4.0              # as_of inklusive
    assert bid("2026-11-20T14:59:59+00:00") == 4.0              # spätere Quote desselben Tages nicht sichtbar
    assert bid("2026-11-20T15:00:00+00:00") == 5.0
    assert bid(datetime(2026, 11, 20, 17, 0, tzinfo=UTC)) == 5.0
    assert bid("2026-11-20T18:00:00+00:00") == 6.0
    assert bid("2026-11-22T12:00:00+00:00") == 6.0              # Wochenende: Lookback über Tage
    assert bid("2026-11-23T15:00:00+00:00") == 7.0
    assert bid("2026-11-20T10:00:00-05:00") == 5.0              # as_of in anderer Zeitzone = 15:00 UTC


def test_latest_quote_filters_symbol_lookback_and_garbage(tmp_path):
    _put(tmp_path, 4.0, "2026-11-10T15:00:00+00:00")
    assert oq.latest_quote(SYM, "2026-11-20T15:00:00+00:00", root=tmp_path)["bid"] == 4.0   # genau 10 Tage
    assert oq.latest_quote(SYM, "2026-11-21T15:00:00+00:00", root=tmp_path) is None         # 11 Tage > Lookback
    assert oq.latest_quote(SYM, "2026-11-21T15:00:00+00:00", root=tmp_path, lookback_days=11)["bid"] == 4.0
    assert oq.latest_quote("AAPL261218P00150000", "2026-11-20T15:00:00+00:00", root=tmp_path) is None
    assert oq.latest_quote(SYM.lower(), "2026-11-20T15:00:00+00:00", root=tmp_path)["bid"] == 4.0
    assert oq.latest_quote(SYM, "Mist", root=tmp_path) is None
    assert oq.latest_quote("", "2026-11-20T15:00:00+00:00", root=tmp_path) is None
    assert oq.latest_quote(SYM, None, root=tmp_path) is None
    assert oq.latest_quote(SYM, "2026-11-20T15:00:00+00:00", root=tmp_path, lookback_days="x") is None


def test_latest_quote_returns_unusable_with_status(tmp_path):
    _put(tmp_path, 5.2, "2026-11-20T15:00:00+00:00")
    oq.record([_raw(bid=5.2, ask=5.0, bid_date=None)], underlying="AAPL", source="x", context="y",
              retrieved_at="2026-11-20T16:00:00+00:00", root=tmp_path)
    q = oq.latest_quote(SYM, "2026-11-20T17:00:00+00:00", root=tmp_path)
    assert q["quality_status"] == "UNUSABLE" and "CROSSED" in q["quality_flags"]


def test_every_archived_record_is_observed_quote(tmp_path):
    oq.record([_raw(), _raw(bid=-1, bid_date=_ms(RA - timedelta(seconds=5))), {"symbol": SYM},
               _raw(symbol="AAPL261218P00150000", option_type="put")],
              underlying="AAPL", source="t", context="c", retrieved_at=RA, root=tmp_path)
    rows = oq.read("2026-11-20", root=tmp_path)
    assert len(rows) == 4
    assert {r["price_type"] for r in rows} == {"OBSERVED_QUOTE"}
    assert all(set(r) == ALL_KEYS for r in rows)
