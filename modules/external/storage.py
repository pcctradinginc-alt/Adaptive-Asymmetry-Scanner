"""
modules/external/storage.py – austauschbares Storage-Backend für das externe
PIT-Archiv (siehe archive.py).

Zwei Backends:
  LocalBackend – heutiges Verhalten (Dateisystem unter dem Archiv-Root, i.d.R.
                 von git versioniert). Immer verfügbar, keine Abhängigkeiten.
  S3Backend    – beliebiger S3-kompatibler Object-Store (AWS S3, Cloudflare
                 R2, Backblaze B2, MinIO, …) über boto3 (lazy import – boto3
                 ist NIE eine Hard-Dependency).

Konfiguration AUSSCHLIESSLICH über Env-Variablen (nie in config.yaml, damit
nie versehentlich Zugangsdaten committed werden):
  EXTERNAL_ARCHIVE_S3_BUCKET     – Bucket-Name (erforderlich für S3Backend)
  EXTERNAL_ARCHIVE_S3_ENDPOINT   – optionaler Custom-Endpoint (R2/B2/MinIO)
  EXTERNAL_ARCHIVE_S3_PREFIX     – Objekt-Prefix (Default siehe unten)
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION – Standard-AWS-Env,
    von boto3 selbst gelesen.

WICHTIG: Zugangsdaten werden NIE geloggt, NIE in Exceptions/Manifesten
serialisiert. `S3Backend.describe()` gibt ausschließlich unkritische
Metadaten zurück (bucket/endpoint/prefix), nie Keys/Secrets.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

DEFAULT_S3_PREFIX = "adaptive-asymmetry/external_data"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class StorageBackend(ABC):
    """Minimales Objekt-Storage-Interface, das archive.py gegen benutzt.

    `key` ist ein posix-artiger relativer Pfad (z.B.
    "normalized/src_a/2025-01.jsonl.gz"). Implementierungen dürfen nie
    Zugangsdaten in Exceptions oder Rückgabewerten offenlegen."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes, metadata: dict[str, str] | None = None) -> dict:
        """Schreibt `data` unter `key`. Rückgabe mind. {"sha256": ...,
        "bytes": len(data)} – nie Zugangsdaten."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Liest die Bytes unter `key`. Wirft bei fehlendem Key."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """True, wenn `key` existiert."""

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """Alle Keys unter `prefix` (rekursiv)."""


class LocalBackend(StorageBackend):
    """Dateisystem-Backend – bildet das heutige (Git-)Verhalten ab: Objekte
    werden als normale Dateien unter `root` abgelegt. Wird von archive.py
    NICHT für die normale Ablage benutzt (die schreibt direkt mit
    Path/open), sondern dient als austauschbare Referenzimplementierung und
    Test-/Fallback-Backend."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def put_bytes(self, key: str, data: bytes, metadata: dict[str, str] | None = None) -> dict:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"sha256": sha256_hex(data), "bytes": len(data)}

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str = "") -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        out = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(str(p.relative_to(self.root)).replace(os.sep, "/"))
        return out


class S3Backend(StorageBackend):
    """S3-kompatibles Backend (AWS S3, Cloudflare R2, Backblaze B2, MinIO, …).

    boto3 wird NUR bei Instanziierung importiert (lazy) – ein Checkout ohne
    boto3-Installation bleibt für den git-Backend-Pfad voll funktionsfähig."""

    def __init__(self, bucket: str, prefix: str = DEFAULT_S3_PREFIX,
                 endpoint_url: str | None = None, region: str | None = None):
        if not bucket:
            raise ValueError("S3Backend benötigt einen Bucket-Namen.")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url or None
        self._client = self._build_client(endpoint_url, region)

    @staticmethod
    def _build_client(endpoint_url: str | None, region: str | None):
        try:
            import boto3  # lazy – optionale Abhängigkeit
        except ImportError as e:  # pragma: no cover - Abhängigkeit fehlt
            raise RuntimeError(
                "boto3 ist nicht installiert – für S3-Backend `pip install -r "
                "requirements-storage.txt` (nie geloggt: keine Zugangsdaten in "
                "dieser Meldung)."
            ) from e
        # boto3 liest AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_REGION selbst
        # aus der Umgebung – hier werden nie Credential-Werte gelesen/geloggt.
        kwargs: dict[str, Any] = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if region:
            kwargs["region_name"] = region
        return boto3.client("s3", **kwargs)

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes, metadata: dict[str, str] | None = None) -> dict:
        sha = sha256_hex(data)
        meta = dict(metadata or {})
        meta.setdefault("sha256", sha)
        self._client.put_object(
            Bucket=self.bucket, Key=self._full_key(key), Body=data, Metadata=meta,
        )
        return {"sha256": sha, "bytes": len(data)}

    def get_bytes(self, key: str) -> bytes:
        resp = self._client.get_object(Bucket=self.bucket, Key=self._full_key(key))
        return resp["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except Exception:
            return False

    def list(self, prefix: str = "") -> list[str]:
        full_prefix = self._full_key(prefix) if prefix else self.prefix
        out = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []) or []:
                k = obj["Key"]
                if self.prefix and k.startswith(self.prefix + "/"):
                    k = k[len(self.prefix) + 1:]
                out.append(k)
        return out

    def describe(self) -> dict:
        """Unkritische Metadaten fürs Manifest/Logging – NIE Zugangsdaten."""
        return {"bucket": self.bucket, "endpoint": self.endpoint_url, "prefix": self.prefix}


def build_s3_backend_from_env(prefix: str | None = None) -> S3Backend | None:
    """Baut ein S3Backend ausschließlich aus Env-Variablen. Gibt None zurück,
    wenn EXTERNAL_ARCHIVE_S3_BUCKET nicht gesetzt ist (kein Fehler – das
    Aufrufer-Backend "git" ist dann einfach nicht konfiguriert)."""
    bucket = os.environ.get("EXTERNAL_ARCHIVE_S3_BUCKET", "").strip()
    if not bucket:
        return None
    endpoint = os.environ.get("EXTERNAL_ARCHIVE_S3_ENDPOINT", "").strip() or None
    resolved_prefix = prefix or os.environ.get("EXTERNAL_ARCHIVE_S3_PREFIX", "").strip() \
        or DEFAULT_S3_PREFIX
    region = os.environ.get("AWS_REGION", "").strip() or None
    return S3Backend(bucket=bucket, prefix=resolved_prefix, endpoint_url=endpoint, region=region)
