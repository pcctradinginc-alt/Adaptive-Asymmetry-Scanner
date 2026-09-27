"""Tests für das optionale Object-Storage-Backend des externen PIT-Archivs
(modules/external/storage.py, Offload-/Retention-Logik in archive.py).

Nutzt ausschließlich ein In-Memory-Fake-Backend (kein boto3, kein Netzwerk).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone

import pytest

from modules.external.archive import ExternalArchive
from modules.external.pit import AvailabilityPrecision, Observation
from modules.external.storage import LocalBackend, StorageBackend, sha256_hex

UTC = timezone.utc


def mk_obs(source_id="src_a", dataset="daily", series_id="S1", entity_id="E1",
           metric="m", value=1.0, obs_time="2025-01-01T00:00:00+00:00",
           available_at="2025-01-01T00:00:00+00:00",
           retrieved_at="2025-01-01T00:00:00+00:00", **kw) -> Observation:
    return Observation(
        source_id=source_id, dataset=dataset, series_id=series_id, entity_id=entity_id,
        metric=metric, value=value, unit="idx",
        observation_time=obs_time, available_at=available_at, retrieved_at=retrieved_at,
        availability_precision=AvailabilityPrecision.EXACT_TIMESTAMP, parser_version="1", **kw,
    )


class FakeS3Backend(StorageBackend):
    """In-Memory-Fake für Tests – kein boto3/Netzwerk nötig. Verhält sich wie
    ein echtes S3-kompatibles Backend inkl. Metadaten, aber optional
    steuerbar (fail_puts) um Upload-Fehler zu simulieren."""

    def __init__(self, fail_puts: bool = False, fail_keys: set[str] | None = None):
        self._store: dict[str, bytes] = {}
        self._metadata: dict[str, dict[str, str]] = {}
        self.fail_puts = fail_puts
        self.fail_keys = fail_keys or set()
        self.put_calls: list[str] = []

    def put_bytes(self, key: str, data: bytes, metadata: dict[str, str] | None = None) -> dict:
        self.put_calls.append(key)
        if self.fail_puts or key in self.fail_keys:
            raise RuntimeError(f"simulated upload failure for {key}")
        self._store[key] = data
        self._metadata[key] = dict(metadata or {})
        return {"sha256": sha256_hex(data), "bytes": len(data)}

    def get_bytes(self, key: str) -> bytes:
        if key not in self._store:
            raise KeyError(key)
        return self._store[key]

    def exists(self, key: str) -> bool:
        return key in self._store

    def list(self, prefix: str = "") -> list[str]:
        return [k for k in self._store if k.startswith(prefix)]


# ── LocalBackend (Referenzimplementierung) ──────────────────────────────

def test_local_backend_roundtrip(tmp_path):
    backend = LocalBackend(tmp_path)
    info = backend.put_bytes("a/b.txt", b"hello")
    assert info["sha256"] == sha256_hex(b"hello")
    assert backend.exists("a/b.txt")
    assert backend.get_bytes("a/b.txt") == b"hello"
    assert "a/b.txt" in backend.list()


def test_local_backend_missing_key_not_exists(tmp_path):
    backend = LocalBackend(tmp_path)
    assert not backend.exists("nope")


# ── offload_normalized_months(): no-op when backend=git ─────────────────

def test_offload_noop_when_backend_git(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "git"})
    a = ExternalArchive(root=tmp_path, storage_backend=FakeS3Backend())
    a.store_observations([mk_obs(value=1.0)])
    result = a.offload_normalized_months(now=datetime(2025, 6, 1, tzinfo=UTC))
    assert result["enabled"] is False
    assert result["uploaded"] == [] and result["offloaded"] == []
    # Datei bleibt unangetastet
    assert (tmp_path / "normalized" / "src_a" / "2025-01.jsonl").exists()
    assert not a.offload_manifest_path().exists()


# ── upload + verify + offload (retention) ────────────────────────────────

def test_offload_uploads_verifies_and_deletes_old_closed_month(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 3})
    backend = FakeS3Backend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    a.store_observations([mk_obs(value=1.0, obs_time="2025-01-01T00:00:00+00:00",
                                  available_at="2025-01-01T00:00:00+00:00")])
    local_path = tmp_path / "normalized" / "src_a" / "2025-01.jsonl"
    assert local_path.exists()
    original_bytes = local_path.read_bytes()

    # "now" ist weit genug in der Zukunft, dass 2025-01 abgeschlossen und
    # älter als die 3-Monats-Retention ist.
    now = datetime(2025, 6, 1, tzinfo=UTC)
    result = a.offload_normalized_months(now=now)

    rel = "normalized/src_a/2025-01.jsonl"
    assert rel in result["uploaded"]
    assert rel in result["verified"]
    assert rel in result["offloaded"]
    assert not local_path.exists()  # lokal gelöscht

    # Objekt liegt gzip-komprimiert im Backend, sha256 passt zum Original.
    key = rel + ".gz"
    assert backend.exists(key)
    assert gzip.decompress(backend.get_bytes(key)) == original_bytes
    assert backend._metadata[key]["sha256"] == sha256_hex(original_bytes)

    # Pointer-Manifest enthält die erwarteten Felder.
    manifest = json.loads(a.offload_manifest_path().read_text())
    entry = manifest[rel]
    assert entry["s3_key"] == key
    assert entry["sha256"] == sha256_hex(original_bytes)
    assert entry["bytes"] == len(original_bytes)
    assert entry["offloaded_at"] is not None


def test_offload_transparent_readback_after_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 3})
    backend = FakeS3Backend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    old_obs = mk_obs(value=42.0, obs_time="2025-01-01T00:00:00+00:00",
                      available_at="2025-01-01T00:00:00+00:00")
    a.store_observations([old_obs])
    now = datetime(2025, 6, 1, tzinfo=UTC)
    a.offload_normalized_months(now=now)
    assert not (tmp_path / "normalized" / "src_a" / "2025-01.jsonl").exists()

    # load()/as_of() liefern die ausgelagerten Werte weiterhin -- PIT bleibt
    # unverändert, transparent über /tmp-Cache aus dem Backend gelesen.
    rows = a.load("src_a")
    assert len(rows) == 1 and rows[0].value == 42.0

    as_of_rows = a.as_of("src_a", datetime(2025, 6, 1, tzinfo=UTC))
    assert len(as_of_rows) == 1 and as_of_rows[0].value == 42.0

    # Zweite Instanz (z.B. neuer Prozess) mit demselben Backend liest genauso.
    a2 = ExternalArchive(root=tmp_path, storage_backend=backend)
    rows2 = a2.load("src_a")
    assert len(rows2) == 1 and rows2[0].value == 42.0


def test_offload_readback_uses_tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 3})
    backend = FakeS3Backend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    a.store_observations([mk_obs(value=7.0, obs_time="2025-01-01T00:00:00+00:00",
                                  available_at="2025-01-01T00:00:00+00:00")])
    now = datetime(2025, 6, 1, tzinfo=UTC)
    a.offload_normalized_months(now=now)

    a.load("src_a")  # füllt den Cache
    cache_path = a._offload_cache_dir() / "normalized/src_a/2025-01.jsonl"
    assert cache_path.exists()

    # Backend "kaputt machen" -- load() muss trotzdem aus dem Cache lesen.
    key = "normalized/src_a/2025-01.jsonl.gz"
    del backend._store[key]
    rows = a.load("src_a")
    assert len(rows) == 1 and rows[0].value == 7.0


# ── no deletion when upload fails ────────────────────────────────────────

def test_offload_never_deletes_when_upload_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 3})
    backend = FakeS3Backend(fail_puts=True)
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    a.store_observations([mk_obs(value=1.0, obs_time="2025-01-01T00:00:00+00:00",
                                  available_at="2025-01-01T00:00:00+00:00")])
    local_path = tmp_path / "normalized" / "src_a" / "2025-01.jsonl"
    now = datetime(2025, 6, 1, tzinfo=UTC)
    result = a.offload_normalized_months(now=now)

    assert local_path.exists()  # NIE gelöscht bei fehlgeschlagenem Upload
    assert result["offloaded"] == []
    assert "src_a" in result["failures"]
    assert not a.offload_manifest_path().exists() or \
        "normalized/src_a/2025-01.jsonl" not in json.loads(a.offload_manifest_path().read_text())


def test_offload_never_deletes_when_verification_fails(tmp_path, monkeypatch):
    """Simuliert eine Backend-Implementierung, deren Re-Download nicht zum
    Original passt (Bit-Rot/Transport-Fehler) -- Löschung darf trotz
    erfolgreichem put_bytes() nicht stattfinden."""
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 3})

    class CorruptingBackend(FakeS3Backend):
        def get_bytes(self, key: str) -> bytes:
            data = super().get_bytes(key)
            # Gültiges Gzip, aber mit anderem Inhalt -> sha256-Mismatch beim
            # Re-Download-Verify, statt eines Dekompressions-Fehlers.
            return gzip.compress(gzip.decompress(data) + b"tampered")

    backend = CorruptingBackend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    a.store_observations([mk_obs(value=1.0, obs_time="2025-01-01T00:00:00+00:00",
                                  available_at="2025-01-01T00:00:00+00:00")])
    local_path = tmp_path / "normalized" / "src_a" / "2025-01.jsonl"
    now = datetime(2025, 6, 1, tzinfo=UTC)
    result = a.offload_normalized_months(now=now)

    assert local_path.exists()
    assert result["offloaded"] == []
    assert "src_a" in result["failures"]
    assert any("Verifikation" in m for m in result["failures"]["src_a"])
    assert "normalized/src_a/2025-01.jsonl" not in result["verified"]


# ── retention month logic ────────────────────────────────────────────────

def test_offload_keeps_current_open_month(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 0})
    backend = FakeS3Backend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    now = datetime(2025, 6, 15, tzinfo=UTC)
    a.store_observations([mk_obs(value=1.0, obs_time="2025-06-01T00:00:00+00:00",
                                  available_at="2025-06-01T00:00:00+00:00")])
    result = a.offload_normalized_months(now=now)
    # Auch mit retention_months=0 wird der LAUFENDE Monat nie gelöscht.
    assert (tmp_path / "normalized" / "src_a" / "2025-06.jsonl").exists()
    assert result["offloaded"] == []
    # Aber sehr wohl hochgeladen (Backup).
    assert "normalized/src_a/2025-06.jsonl" in result["uploaded"]


def test_offload_keeps_closed_month_within_retention(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 3})
    backend = FakeS3Backend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    now = datetime(2025, 6, 15, tzinfo=UTC)
    # April ist abgeschlossen, aber innerhalb der 3-Monats-Retention (Apr/Mai/Jun).
    a.store_observations([mk_obs(value=1.0, obs_time="2025-04-15T00:00:00+00:00",
                                  available_at="2025-04-15T00:00:00+00:00")])
    result = a.offload_normalized_months(now=now)
    assert (tmp_path / "normalized" / "src_a" / "2025-04.jsonl").exists()
    assert result["offloaded"] == []
    assert "normalized/src_a/2025-04.jsonl" in result["uploaded"]  # aber gesichert


def test_offload_deletes_closed_month_beyond_retention(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 3})
    backend = FakeS3Backend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    now = datetime(2025, 6, 15, tzinfo=UTC)
    # Januar liegt außerhalb Apr/Mai/Jun -> darf gelöscht werden.
    a.store_observations([mk_obs(value=1.0, obs_time="2025-01-15T00:00:00+00:00",
                                  available_at="2025-01-15T00:00:00+00:00")])
    result = a.offload_normalized_months(now=now)
    assert not (tmp_path / "normalized" / "src_a" / "2025-01.jsonl").exists()
    assert "normalized/src_a/2025-01.jsonl" in result["offloaded"]


def test_is_closed_month_and_cutoff_helpers():
    now = datetime(2025, 6, 15, tzinfo=UTC)
    assert ExternalArchive._is_closed_month("2025-05", now) is True
    assert ExternalArchive._is_closed_month("2025-06", now) is False
    assert ExternalArchive._is_closed_month("2025-07", now) is False
    assert ExternalArchive._months_before_cutoff("2025-02", now, 3) is True   # Apr/Mai/Jun kept
    assert ExternalArchive._months_before_cutoff("2025-03", now, 3) is False


# ── idempotence: second offload run does not re-upload unchanged files ──

def test_offload_idempotent_no_reupload_when_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 12})
    backend = FakeS3Backend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    now = datetime(2025, 6, 15, tzinfo=UTC)
    # retention 12 Monate -> Januar bleibt lokal (nur Backup-Upload testen).
    a.store_observations([mk_obs(value=1.0, obs_time="2025-01-15T00:00:00+00:00",
                                  available_at="2025-01-15T00:00:00+00:00")])
    r1 = a.offload_normalized_months(now=now)
    assert "normalized/src_a/2025-01.jsonl" in r1["uploaded"]
    put_calls_after_first = len(backend.put_calls)

    r2 = a.offload_normalized_months(now=now)
    assert r2["uploaded"] == []  # unverändert -> kein erneuter Upload
    assert len(backend.put_calls) == put_calls_after_first


# ── raw files are uploaded but never deleted ─────────────────────────────

def test_raw_files_uploaded_but_never_deleted(tmp_path, monkeypatch):
    from modules.external.sources.base import RawRecord

    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 0})
    backend = FakeS3Backend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    rec = RawRecord(source_id="src_a", dataset="d", url="u", fingerprint="fp",
                     retrieved_at=datetime(2025, 1, 1, tzinfo=UTC), status_code=200,
                     content_type="application/json", content_hash="h1", bytes=5,
                     content=b"hello")
    a.store_raw(rec, policy="gzip", raw_max_bytes=1000)
    now = datetime(2025, 6, 15, tzinfo=UTC)
    result = a.offload_normalized_months(now=now)
    raw_files = list((tmp_path / "raw" / "src_a").rglob("*"))
    assert any(f.is_file() for f in raw_files)  # nichts gelöscht
    assert any(u.startswith("raw/src_a/") for u in result["uploaded"])


# ── credentials never leak into logs/manifests ───────────────────────────

def test_offload_manifest_never_contains_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 0})
    backend = FakeS3Backend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    now = datetime(2025, 6, 15, tzinfo=UTC)
    a.store_observations([mk_obs(value=1.0, obs_time="2025-01-01T00:00:00+00:00",
                                  available_at="2025-01-01T00:00:00+00:00")])
    a.offload_normalized_months(now=now)
    manifest_text = a.offload_manifest_path().read_text()
    for secret_marker in ("AWS_SECRET_ACCESS_KEY", "aws_secret", "Authorization", "sk-ant"):
        assert secret_marker not in manifest_text
    # Objekt-Metadaten im Backend enthalten nur sha256, keine Credential-Felder.
    for meta in backend._metadata.values():
        assert set(meta.keys()) <= {"sha256"}


def test_build_s3_backend_from_env_no_bucket_returns_none(monkeypatch):
    from modules.external.storage import build_s3_backend_from_env
    monkeypatch.delenv("EXTERNAL_ARCHIVE_S3_BUCKET", raising=False)
    assert build_s3_backend_from_env() is None


def test_s3_backend_describe_never_exposes_credentials(monkeypatch):
    pytest.importorskip("boto3")
    from modules.external.storage import S3Backend
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST_SECRET")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "supersecretvalue")
    backend = S3Backend(bucket="my-bucket", prefix="p")
    desc = json.dumps(backend.describe())
    assert "AKIA_TEST_SECRET" not in desc
    assert "supersecretvalue" not in desc


# ── storage_telemetry(): new fields ──────────────────────────────────────

def test_storage_telemetry_includes_backend_and_offload_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.external.archive._config_archive_defaults",
                         lambda: {"backend": "s3", "git_retention_months": 3,
                                  "storage_warn_mb_1y": 200})
    backend = FakeS3Backend()
    a = ExternalArchive(root=tmp_path, storage_backend=backend)
    a.store_observations([mk_obs(value=1.0, obs_time="2025-01-01T00:00:00+00:00",
                                  available_at="2025-01-01T00:00:00+00:00")])
    now = datetime(2025, 6, 15, tzinfo=UTC)
    a.offload_normalized_months(now=now)
    telemetry = a.storage_telemetry()
    assert telemetry["backend"] == "s3"
    assert telemetry["bytes_offloaded"] > 0
    assert telemetry["bytes_in_git"] >= 0
