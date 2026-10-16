"""Tests für den Nasdaq-100-Parser (modules/universe._fetch_nasdaq100) – nur SYNTHETISCHES HTML, requests gemockt.

Anlass (Live 2026-10-15): "Nasdaq-100: Keine Tabelle gefunden → statischer Fallback" – die Produktion scannte eine
veraltete statische Liste. Geprüft: alte Spalte "Ticker", neue Spalte "Symbol", Tabellen-ID, mehrzeilige Kopfzeilen,
Änderungstabellen werden nicht als Liste genommen, Fallback mit Diagnose. UNIVERSE_V1-Code-Hash bleibt unverändert.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from modules import universe as uv

N = 101


def _sym(i: int) -> str:
    """Synthetische, rein alphabetische Ticker (get_universe/_clean verwirft Ziffern)."""
    return f"Q{chr(65 + i // 26)}{chr(65 + i % 26)}"


def _rows(n: int, col: str = "Ticker") -> str:
    head = f"<tr><th>Company</th><th>{col}</th><th>GICS Sector</th></tr>"
    body = "".join(f"<tr><td>Firma {i}</td><td>{_sym(i)}</td><td>Tech</td></tr>" for i in range(n - 1))
    return head + body + "<tr><td>Berkshire</td><td>BRK.B</td><td>Fin</td></tr>"


def _changes_table() -> str:
    """Änderungstabelle mit zweizeiligem Kopf ("Added"/"Removed" je Ticker) – nur wenige Zeilen."""
    return ("<table class='wikitable'><tr><th rowspan=2>Date</th><th colspan=2>Added</th><th colspan=2>Removed</th></tr>"
            "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>"
            "<tr><td>2026-09-22</td><td>NEW1</td><td>New One</td><td>OLD1</td><td>Old One</td></tr>"
            "<tr><td>2026-06-23</td><td>NEW2</td><td>New Two</td><td>OLD2</td><td>Old Two</td></tr></table>")


def _page(*tables: str) -> str:
    return "<html><body>" + "".join(tables) + "</body></html>"


@pytest.fixture
def serve(monkeypatch):
    def _serve(html: str):
        monkeypatch.setattr(uv.requests, "get", lambda *a, **k: SimpleNamespace(text=html, raise_for_status=lambda: None))
    return _serve


def test_legacy_ticker_column(serve):
    serve(_page(f"<table class='wikitable'>{_rows(N)}</table>"))
    t = uv._fetch_nasdaq100()
    assert len(t) == N and "BRK-B" in t and t[0] == "QAA"


def test_symbol_column_in_constituents_table(serve):
    serve(_page(_changes_table(), f"<table id='constituents' class='wikitable'>{_rows(N, 'Symbol')}</table>"))
    t = uv._fetch_nasdaq100()
    assert len(t) == N and "NEW1" not in t


def test_changes_table_before_components_is_skipped(serve):
    """Änderungstabelle zuerst (mehrzeiliger Kopf mit "Ticker"), Komponenten ohne ID danach."""
    serve(_page(_changes_table(), f"<table class='wikitable'>{_rows(N, 'Symbol')}</table>"))
    t = uv._fetch_nasdaq100()
    assert len(t) == N and not {"NEW1", "OLD1"} & set(t)


def test_multirow_header_on_components(serve):
    head = "<tr><th colspan=3>Nasdaq-100 components</th></tr>"
    serve(_page(f"<table class='wikitable'>{head}{_rows(N, 'Ticker')}</table>"))
    assert len(uv._fetch_nasdaq100()) == N


def test_no_plausible_table_falls_back_with_diagnosis(serve, caplog):
    serve(_page(_changes_table(), f"<table class='wikitable'>{_rows(20)}</table>"))
    with caplog.at_level(logging.WARNING, logger=uv.log.name):
        assert uv._fetch_nasdaq100() == []
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "statischer Fallback" in msg and "Ticker" in msg            # gefundene Spalten stehen im Log


def test_get_universe_uses_parsed_list_instead_of_static(serve, monkeypatch):
    serve(_page(f"<table id='constituents' class='wikitable'>{_rows(N, 'Symbol')}</table>"))
    monkeypatch.setattr(uv, "_fetch_sp500", lambda: [])
    uv.get_universe.cache_clear()
    try:
        u = uv.get_universe("nasdaq100")
    finally:
        uv.get_universe.cache_clear()
    assert _sym(50) in u and "EA" not in u and "BRK-B" in u and len(u) == N


def test_universe_v1_code_hash_unchanged():
    from modules import universe_v2
    assert universe_v2.v1_unchanged() is True
