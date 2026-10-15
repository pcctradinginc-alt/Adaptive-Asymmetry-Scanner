"""Verlustfreie gzip-Kompaktierung kalter normalisierter Partitionen (A5, 2026-10-15).

Synthetische Observations in tmp_path. Geprüft: nur abgeschlossene UND seit N Tagen unberührte Partitionen werden
komprimiert; Inhalt byte-identisch (Zeilen, Reihenfolge, payload_hash); Leser (load, Dedup beim Schreiben,
Byte-Messung, Erstimport-Erkennung) sehen .jsonl und .jsonl.gz transparent; Nachträge nach der Kompaktierung
landen in einer neuen .jsonl und werden später angehängt; Fehler stellen den Vorzustand wieder her.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone

from modules.external import archive as ea
from modules.external.archive import ExternalArchive
from tests.test_external_storage import mk_obs

UTC = timezone.utc
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


def _store(a: ExternalArchive, *obs, partition: str = "monthly") -> dict:
    return a.store_observations(list(obs), max_new_normalized_bytes_per_source_per_run=10**9, partition=partition)


def _files(a: ExternalArchive, source: str = "src_a") -> list[str]:
    d = a.root / "normalized" / source
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


def _old(month: str, i: int, value: float = 1.0, retrieved: str = "2026-08-05T00:00:00+00:00"):
    return mk_obs(series_id=f"S{i}", value=value, obs_time=f"{month}-0{1 + i % 9}T00:00:00+00:00",
                  available_at=retrieved, retrieved_at=retrieved)


def test_partition_end_monthly_weekly_and_unknown():
    assert ea._partition_end("2026-09") == datetime(2026, 10, 1, tzinfo=UTC)
    assert ea._partition_end("2026-12") == datetime(2027, 1, 1, tzinfo=UTC)
    assert ea._partition_end("2026-10-w2") == datetime(2026, 10, 15, tzinfo=UTC)
    assert ea._partition_end("2026-02-w5") == datetime(2026, 3, 1, tzinfo=UTC)       # gedeckelt aufs Monatsende
    assert ea._partition_end("legacy") is None and ea._partition_end("2026-10-x1") is None


def test_cold_partition_is_compressed_losslessly_and_readable(tmp_path):
    a = ExternalArchive(root=tmp_path)
    _store(a, *[_old("2026-08", i) for i in range(5)])
    plain = tmp_path / "normalized" / "src_a" / "2026-08.jsonl"
    raw_before = plain.read_bytes()
    loaded_before = [o.to_dict() for o in a.load("src_a")]
    res = a.compact_cold_partitions(now=NOW, cold_after_days=21)
    assert res["compacted"] == ["normalized/src_a/2026-08.jsonl"] and res["failures"] == {}
    assert _files(a) == ["2026-08.jsonl.gz"]
    gz = tmp_path / "normalized" / "src_a" / "2026-08.jsonl.gz"
    assert gzip.decompress(gz.read_bytes()) == raw_before                        # byte-identisch
    assert res["bytes_before"] == len(raw_before) and res["bytes_after"] == gz.stat().st_size
    assert [o.to_dict() for o in ExternalArchive(root=tmp_path).load("src_a")] == loaded_before
    assert a.normalized_bytes_written_estimate("src_a") == gz.stat().st_size
    # deterministisch: zweiter Lauf ändert nichts
    assert a.compact_cold_partitions(now=NOW, cold_after_days=21)["compacted"] == []


def test_open_or_recently_written_partitions_stay_plain(tmp_path):
    a = ExternalArchive(root=tmp_path)
    _store(a, _old("2026-10", 1, retrieved="2026-10-14T00:00:00+00:00"),         # laufender Monat
           _old("2026-09", 2, retrieved="2026-10-01T00:00:00+00:00"),            # abgeschlossen, aber frisch
           _old("2026-09", 3, retrieved="2026-08-01T00:00:00+00:00"))
    res = a.compact_cold_partitions(now=NOW, cold_after_days=21)
    assert res["compacted"] == [] and _files(a) == ["2026-09.jsonl", "2026-10.jsonl"]


def test_disabled_and_config_default(tmp_path, monkeypatch):
    a = ExternalArchive(root=tmp_path)
    _store(a, _old("2026-08", 1))
    assert a.compact_cold_partitions(now=NOW, cold_after_days=0) == {
        "enabled": False, "cold_after_days": 0, "compacted": [], "bytes_before": 0, "bytes_after": 0, "failures": {}}
    monkeypatch.setattr(ea, "_config_archive_defaults", lambda: {"compress_cold_after_days": 21})
    assert a.compact_cold_partitions(now=NOW)["compacted"] == ["normalized/src_a/2026-08.jsonl"]


def test_dedup_and_revisions_after_compaction(tmp_path):
    a = ExternalArchive(root=tmp_path)
    _store(a, _old("2026-08", 1, value=1.0))
    a.compact_cold_partitions(now=NOW, cold_after_days=21)
    # gleicher Wert erneut abgerufen -> Duplikat (Dedup liest die gz-Partition)
    assert _store(a, _old("2026-08", 1, value=1.0, retrieved="2026-10-15T00:00:00+00:00"))["duplicate"] == 1
    assert _files(a) == ["2026-08.jsonl.gz"]
    # Revision -> neue Zeile in einer frischen .jsonl neben der .gz; Leser sehen beide Vintages
    assert _store(a, _old("2026-08", 1, value=2.0, retrieved="2026-10-15T00:00:00+00:00"))["revision"] == 1
    assert _files(a) == ["2026-08.jsonl", "2026-08.jsonl.gz"]
    assert sorted(o.value for o in a.load("src_a")) == [1.0, 2.0]
    assert a.compact_cold_partitions(now=NOW, cold_after_days=21)["compacted"] == []      # Nachtrag noch frisch
    later = datetime(2026, 11, 20, tzinfo=UTC)
    res = a.compact_cold_partitions(now=later, cold_after_days=21)
    assert res["compacted"] == ["normalized/src_a/2026-08.jsonl"] and _files(a) == ["2026-08.jsonl.gz"]
    rows = [json.loads(x) for x in gzip.decompress((tmp_path / "normalized/src_a/2026-08.jsonl.gz").read_bytes())
            .decode().splitlines()]
    assert [r["value"] for r in rows] == [1.0, 2.0]                                          # Reihenfolge erhalten
    assert all(ea.verify_row(r) for r in rows)                                               # payload_hash intakt


def test_weekly_partitions_and_legacy_month_file(tmp_path):
    a = ExternalArchive(root=tmp_path)
    _store(a, mk_obs(series_id="W", obs_time="2026-08-03T00:00:00+00:00", retrieved_at="2026-08-30T00:00:00+00:00",
                     available_at="2026-08-30T00:00:00+00:00"), partition="weekly")
    _store(a, mk_obs(series_id="L", obs_time="2026-08-20T00:00:00+00:00", retrieved_at="2026-08-30T00:00:00+00:00",
                     available_at="2026-08-30T00:00:00+00:00"))                              # Legacy-Monatsdatei
    res = a.compact_cold_partitions(now=NOW, cold_after_days=21)
    assert sorted(res["compacted"]) == ["normalized/src_a/2026-08-w1.jsonl", "normalized/src_a/2026-08.jsonl"]
    assert _files(a) == ["2026-08-w1.jsonl.gz", "2026-08.jsonl.gz"]
    # Dedup einer Wochenpartition berücksichtigt auch die komprimierte Legacy-Monatsdatei
    dup = mk_obs(series_id="L", obs_time="2026-08-20T00:00:00+00:00", retrieved_at="2026-10-15T00:00:00+00:00",
                 available_at="2026-10-15T00:00:00+00:00")
    assert _store(a, dup, partition="weekly")["duplicate"] == 1
    assert {o.series_id for o in a.load("src_a")} == {"W", "L"}


def test_failed_verification_restores_previous_state(tmp_path, monkeypatch):
    a = ExternalArchive(root=tmp_path)
    _store(a, _old("2026-08", 1))
    plain = tmp_path / "normalized" / "src_a" / "2026-08.jsonl"
    before = plain.read_bytes()
    real = gzip.decompress
    calls = {"n": 0}

    def flaky(data):
        calls["n"] += 1
        out = real(data)
        return out + b"x" if calls["n"] == 2 else out                 # Prüfung der geschriebenen Datei scheitert

    monkeypatch.setattr(ea.gzip, "decompress", flaky)
    res = a.compact_cold_partitions(now=NOW, cold_after_days=21)
    assert res["compacted"] == [] and "normalized/src_a/2026-08.jsonl" in res["failures"]
    assert _files(a) == ["2026-08.jsonl"] and plain.read_bytes() == before                  # nichts doppelt


def test_first_import_detection_sees_compressed_partitions(tmp_path):
    a = ExternalArchive(root=tmp_path)
    _store(a, _old("2026-08", 1))
    a.compact_cold_partitions(now=NOW, cold_after_days=21)
    big = [_old("2026-10", i, retrieved="2026-10-15T00:00:00+00:00") for i in range(50)]
    a.store_observations(big, max_new_normalized_bytes_per_source_per_run=100, max_backfill_bytes=10**9)
    assert a.last_guard_blocked["src_a"]["first_import"] is False                        # kein zweiter Erstimport
