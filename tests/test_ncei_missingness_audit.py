"""Offline analysis and safe handling of NCEI missing data."""
from datetime import date
from types import SimpleNamespace

from modules import ncei_missingness_audit as audit

ASOF = date(2026, 11, 14)


def row(station, metric, value, source_id="ncei_normals"):
    return SimpleNamespace(entity_id=station, metric=metric, value=value, source_id=source_id)


def test_null_breakdown_never_fills_missing_values():
    rows = [row("ST01", "tavg", 22.0), row("ST01", "tavg", None),
            row("ST01", "precip", float("nan")), row("ST02", "tavg", -8.0),
            row("ST02", "precip", None), row("ST02", "precip", float("inf")),
            row("ST03", "tavg", 10, "other")]
    original = [r.value for r in rows]
    r = audit.diagnose(rows, ASOF)
    assert r["status"] == "MEASURED" and r["observations"] == 6
    assert r["numeric"] == 2 and r["missing"] == 4
    assert r["missing_fraction"] == 0.6667
    assert r["unique_stations"] == 2 and r["unique_metrics"] == 2
    assert r["stations_by_missing_count"][0]["key"] == "ST02"
    assert all((x is y) or (x != x and y != y) for x, y in zip(
        [r.value for r in rows], original))


def test_empty_archive_is_not_misrepresented_as_zero_missing():
    r = audit.diagnose([], ASOF)
    assert r["status"] == "NO_ARCHIVED_OBSERVATIONS"
    assert r["missing_fraction"] is None and r["observations"] == 0


def test_local_audit_writes_only_research_report(tmp_path):
    class FakeArchive:
        def as_of(self, source, timestamp):
            assert source == "ncei_normals"
            assert timestamp.date() == ASOF
            return [row("ST01", "tavg", 1.0), row("ST01", "tavg", None)]

    path, r = audit.run(ASOF, archive=FakeArchive(), out=tmp_path)
    assert path.exists() and r["missing"] == 1
    assert "ncei_missingness" in path.name
