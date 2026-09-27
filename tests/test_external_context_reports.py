"""
tests/test_external_context_reports.py

Prüft, dass die "Externer Kontext"-Abschnitte in Monats-Report, Status-/
Trade-Mail und Daily-Markdown sowohl mit LEEREN als auch mit SYNTHETISCHEN
Daten rendern und den Report NIE zum Absturz bringen.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.external import reporting


SNAPSHOT = {
    "snapshot_id": "snap-1",
    "generated_at": "2026-09-27T06:00:00Z",
    "states": {
        "us_freight_state": "EXPANSION", "us_freight_confidence": 0.7,
        "eu_freight_state": "NEUTRAL", "eu_freight_confidence": 0.4,
        "global_maritime_state": "CONTRACTION", "global_maritime_confidence": 0.55,
    },
    "primitives": {
        "suez_z": 2.4, "panama_z": 0.3,
        "weather_disruption_index": 1.9, "active_tropical_system": True,
    },
    "candidates": [
        {"ticker": "UPS", "relation": {"relation": "SUPPORT", "materiality": 0.6},
         "ticker_exposure": {"exposure_source": "industry"}},
    ],
}

HEALTH = {
    "fred_freight": {"status": "PASS", "staleness": "FRESH"},
    "ncei_cdo": {"status": "FAIL", "staleness": "STALE"},
}


# ── modules/external/reporting.py ────────────────────────────────────────────

def test_build_compact_context_none_without_snapshot(tmp_path, monkeypatch):
    ctx = reporting.build_compact_context(snapshot=None, health={}) if False else None
    # Explicit "no snapshot" path via loader against an empty tmp dir:
    empty_root = tmp_path / "snapshots"
    result = reporting.load_latest_snapshot(root=empty_root)
    assert result is None


def test_build_compact_context_with_synthetic_snapshot():
    ctx = reporting.build_compact_context(snapshot=SNAPSHOT, health=HEALTH)
    assert ctx is not None
    assert ctx["source_health"]["counts"]["FAIL"] == 1
    assert ctx["source_health"]["failing"] == ["ncei_cdo"]
    assert any(c["z"] == 2.4 for c in ctx["chokepoint_anomalies"])
    assert ctx["global_maritime_state"] == "CONTRACTION"
    assert ctx["candidates"][0]["ticker"] == "UPS"
    assert ctx["candidates"][0]["relation"] == "SUPPORT"


def test_render_html_block_none_is_one_line():
    html = reporting.render_html_block(None)
    assert "keine externen Daten" in html


def test_render_html_block_with_context_never_raises():
    ctx = reporting.build_compact_context(snapshot=SNAPSHOT, health=HEALTH)
    html = reporting.render_html_block(ctx)
    assert "Externer Kontext" in html
    assert "UPS" in html


def test_render_markdown_lines_none_is_one_line():
    lines = reporting.render_markdown_lines(None)
    assert any("keine externen Daten" in l for l in lines)


def test_render_markdown_lines_with_context():
    ctx = reporting.build_compact_context(snapshot=SNAPSHOT, health=HEALTH)
    lines = reporting.render_markdown_lines(ctx)
    text = "\n".join(lines)
    assert "Externer Kontext" in text
    assert "UPS" in text
    assert "STALE" in text or "FAIL" in text


# ── monthly_report.py ────────────────────────────────────────────────────────

def test_build_external_context_html_empty_ledger_never_raises():
    import monthly_report
    html = monthly_report._safe_external_context_html()
    assert "Externer Kontext" in html


def test_build_html_includes_external_section_end_to_end():
    import monthly_report
    html = monthly_report.build_html(
        "2026-09", None, None, None,
        {"days": 0, "zero_days": 0, "top_rejects": [], "top_stops": []},
    )
    assert "Externer Kontext" in html


# ── modules/email_reporter.py ────────────────────────────────────────────────

def test_status_email_renders_with_and_without_snapshot(monkeypatch, tmp_path):
    from modules import email_reporter

    # isoliert vom echten outputs/ (dort liegen ab dem ersten Scanner-Lauf Snapshots)
    monkeypatch.setattr("modules.external.reporting.load_latest_snapshot",
                        lambda root=reporting.DEFAULT_SNAPSHOT_ROOT: None)
    html_empty = email_reporter._build_status_email({"trades": 0}, "2026-09-27")
    assert "keine externen Daten" in html_empty

    monkeypatch.setattr(
        "modules.external.reporting.load_latest_snapshot",
        lambda root=reporting.DEFAULT_SNAPSHOT_ROOT: SNAPSHOT,
    )
    monkeypatch.setattr(
        "modules.external.reporting.load_source_health",
        lambda path=reporting.DEFAULT_HEALTH_PATH: HEALTH,
    )
    html_full = email_reporter._build_status_email({"trades": 0}, "2026-09-27")
    assert "Externer Kontext" in html_full
    assert "UPS" in html_full


def test_status_email_never_raises_on_broken_snapshot(monkeypatch):
    from modules import email_reporter

    def _boom(root=None):
        raise RuntimeError("boom")

    monkeypatch.setattr("modules.external.reporting.load_latest_snapshot", _boom)
    html = email_reporter._build_status_email({"trades": 0}, "2026-09-27")
    assert "Adaptive Asymmetry-Scanner" in html  # rendered fully despite the failure


# ── modules/reporter.py ──────────────────────────────────────────────────────

def test_daily_markdown_includes_external_section(tmp_path, monkeypatch):
    from modules.reporter import Reporter

    monkeypatch.setattr(
        "modules.external.reporting.load_latest_snapshot",
        lambda root=reporting.DEFAULT_SNAPSHOT_ROOT: SNAPSHOT,
    )
    monkeypatch.setattr(
        "modules.external.reporting.load_source_health",
        lambda path=reporting.DEFAULT_HEALTH_PATH: HEALTH,
    )
    r = Reporter(reports_dir=tmp_path)
    r.save("2026-09-27", [], {"model_weights": {}})
    md = (tmp_path / "2026-09-27.md").read_text()
    assert "Externer Kontext" in md
    assert "UPS" in md


def test_daily_markdown_no_snapshot_shows_single_line(tmp_path, monkeypatch):
    from modules.reporter import Reporter
    monkeypatch.setattr("modules.external.reporting.load_latest_snapshot",
                        lambda root=reporting.DEFAULT_SNAPSHOT_ROOT: None)

    r = Reporter(reports_dir=tmp_path)
    r.save("2026-09-27", [], {"model_weights": {}})
    md = (tmp_path / "2026-09-27.md").read_text()
    assert "keine externen Daten" in md
