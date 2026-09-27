"""
modules/external/archive.py – Point-in-Time-Archiv für externe Rohdaten und
normalisierte Beobachtungen.

Layout (root, Default "outputs/external_data"):
  raw/<source_id>/<YYYY-MM>/...            – Rohdaten-Metadaten (+ optional gzip Payload)
  normalized/<source_id>/<YYYY-MM>.jsonl   – normalisierte Observations (append-only)
  manifests/<YYYY-MM-DD>/<run_id>.json     – ein Manifest pro Ingestion-Run
  health/source_health.json                – SourceHealth je source_id (siehe registry.py)
  health/storage_telemetry.json            – Speicher-Messung + Projektion

Grundregeln:
  - Never store credentials/headers – RawRecord kennt ohnehin keine Header/Keys.
  - Dedupe von Rohdaten NACH content_hash, pro Quelle.
  - Observations werden NIE überschrieben: gleiche Identität + neuer Wert wird
    als neue Vintage-Zeile angehängt (siehe pit.Observation.identity_key()).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from modules.external.pit import Observation, ensure_utc, utc_now
from modules.external.sources.base import RawRecord
from modules.external.storage import (
    StorageBackend, build_s3_backend_from_env, sha256_hex,
)

DEFAULT_ROOT = "outputs/external_data"
DEFAULT_RAW_MAX_BYTES = 2_000_000
DEFAULT_RAW_POLICY = "hash_only"
DEFAULT_STORAGE_WARN_MB_1Y = 200
DEFAULT_MAX_BACKFILL_BYTES_PER_SOURCE = 30_000_000
MAX_BACKFILL_BYTES_HARD_CAP = 80_000_000
DEFAULT_MAX_NEW_NORMALIZED_BYTES_PER_SOURCE_PER_RUN = 2_000_000
DEFAULT_ARCHIVE_BACKEND = "git"
DEFAULT_GIT_RETENTION_MONTHS = 3
OFFLOAD_MANIFEST_NAME = "offloaded.json"


def _config_archive_defaults() -> dict:
    """Liest external_context.archive aus config.yaml – Fehler → Code-Defaults."""
    try:
        from modules.config import cfg
        arc = getattr(getattr(cfg, "external_context", None), "archive", None)
        if arc is None:
            return {}
        return dict(arc)
    except Exception:
        return {}


def _config_hash() -> str:
    """Wiederverwendet candidate_ledger._compute_config_hash falls importierbar,
    sonst eigener sha256(config.yaml)[:12]-Fallback."""
    try:
        from modules.candidate_ledger import _compute_config_hash
        return _compute_config_hash()
    except Exception:
        pass
    try:
        import hashlib
        cfg_path = Path(__file__).resolve().parents[2] / "config.yaml"
        return hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:12]
    except Exception:
        return "unknown"


def _git_sha() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt is not None else None


class ExternalArchive:
    def __init__(self, root: str | os.PathLike = DEFAULT_ROOT,
                 storage_backend: StorageBackend | None = None):
        self.root = Path(root)
        # Von store_observations() bei jedem Aufruf neu gesetzt: source_id ->
        # {"estimated_bytes", "max_bytes"} für Quellen, deren neue
        # normalisierte Zeilen in DIESEM Aufruf wegen der Volumen-Guard NICHT
        # geschrieben wurden (siehe Punkt 4 der Storage-Fix-Aufgabe).
        self.last_guard_blocked: dict[str, dict] = {}
        # Optionales Object-Storage-Backend für die Offload-/Retention-Logik
        # (siehe offload_normalized_months()). Explizit übergeben in Tests
        # (Fake-Backend); sonst lazy aus config.yaml + Env aufgelöst.
        self._storage_backend_override = storage_backend
        self._storage_backend_resolved = False
        self._storage_backend: StorageBackend | None = None

    # ── Pfade ────────────────────────────────────────────────────────────

    def _raw_dir(self, source_id: str, month: str) -> Path:
        return self.root / "raw" / source_id / month

    def _raw_hash_index_path(self, source_id: str) -> Path:
        return self.root / "raw" / source_id / "_hashes.json"

    def _normalized_dir(self, source_id: str) -> Path:
        return self.root / "normalized" / source_id

    def _normalized_path(self, source_id: str, month: str) -> Path:
        return self._normalized_dir(source_id) / f"{month}.jsonl"

    def _health_dir(self) -> Path:
        return self.root / "health"

    def source_health_path(self) -> Path:
        return self._health_dir() / "source_health.json"

    def storage_telemetry_path(self) -> Path:
        return self._health_dir() / "storage_telemetry.json"

    def _manifest_dir(self, day: str) -> Path:
        return self.root / "manifests" / day

    def offload_manifest_path(self) -> Path:
        return self.root / "manifests" / OFFLOAD_MANIFEST_NAME

    # ── Object-Storage-Backend (optional) ───────────────────────────────

    def storage_backend(self) -> StorageBackend | None:
        """Liefert das konfigurierte Object-Storage-Backend, oder None wenn
        `external_context.archive.backend` nicht "s3" ist oder
        EXTERNAL_ARCHIVE_S3_BUCKET fehlt. Wird genau einmal pro Instanz
        aufgelöst (Ausnahme: explizit im Konstruktor übergebenes Backend)."""
        if self._storage_backend_override is not None:
            return self._storage_backend_override
        if self._storage_backend_resolved:
            return self._storage_backend
        self._storage_backend_resolved = True
        defaults = _config_archive_defaults()
        backend_name = str(defaults.get("backend", DEFAULT_ARCHIVE_BACKEND) or DEFAULT_ARCHIVE_BACKEND)
        if backend_name != "s3":
            self._storage_backend = None
            return None
        try:
            self._storage_backend = build_s3_backend_from_env()
        except Exception:
            # Nie Zugangsdaten in der Exception/im Log – Backend bleibt None,
            # Aufrufer behandelt das wie "nicht konfiguriert" (WARN, keine
            # Löschung).
            self._storage_backend = None
        return self._storage_backend

    # ── Raw-Payload ──────────────────────────────────────────────────────

    def store_raw(self, record: RawRecord, policy: str | None = None,
                   raw_max_bytes: int | None = None) -> dict:
        """Speichert Metadaten (+ optional gzip-Payload) eines Abrufs. Dedupe
        über content_hash pro Quelle. Speichert NIE Credentials/Headers."""
        defaults = _config_archive_defaults()
        policy = policy or defaults.get("raw_payload_policy", DEFAULT_RAW_POLICY)
        raw_max_bytes = raw_max_bytes if raw_max_bytes is not None else int(
            defaults.get("raw_max_bytes", DEFAULT_RAW_MAX_BYTES))

        index_path = self._raw_hash_index_path(record.source_id)
        seen = self._load_json(index_path, default=[])
        if record.content_hash in seen:
            return {"stored": False, "duplicate": True, "content_hash": record.content_hash}

        retrieved_at = ensure_utc(record.retrieved_at) or utc_now()
        month = retrieved_at.strftime("%Y-%m")
        out_dir = self._raw_dir(record.source_id, month)
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = retrieved_at.strftime("%Y%m%dT%H%M%SZ")
        base_name = f"{stamp}_{record.content_hash[:16]}"

        meta = {
            "source_id": record.source_id,
            "dataset": record.dataset,
            "url": record.url,
            "fingerprint": record.fingerprint,
            "retrieved_at": _iso(retrieved_at),
            "status_code": record.status_code,
            "content_type": record.content_type,
            "content_hash": record.content_hash,
            "bytes": record.bytes,
        }

        payload_stored = "none"
        if policy == "gzip" and record.content is not None and record.bytes <= raw_max_bytes:
            payload_path = out_dir / f"{base_name}.raw.gz"
            with gzip.open(payload_path, "wb") as fh:
                fh.write(record.content)
            meta["payload_file"] = payload_path.name
            payload_stored = "gzip"
        else:
            payload_stored = "hash_only"
        meta["payload_stored"] = payload_stored

        meta_path = out_dir / f"{base_name}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        seen.append(record.content_hash)
        self._write_json(index_path, seen)
        return {"stored": True, "duplicate": False, "content_hash": record.content_hash,
                "payload_stored": payload_stored, "meta_path": str(meta_path)}

    # ── Observations ─────────────────────────────────────────────────────

    def store_observations(self, observations: Iterable[Observation],
                            max_new_normalized_bytes_per_source_per_run: int | None = None,
                            max_backfill_bytes: int | None = None) -> dict:
        """Idempotente Ablage: gleiche Identität + gleicher Wert = duplicate
        (skip); gleiche Identität + anderer Wert = NEUE Vintage-Zeile (nie
        überschreiben).

        Volumen-Guard: bevor irgendetwas geschrieben wird, werden die neuen
        Zeilen JE source_id (über alle Monats-Buckets hinweg) simuliert und
        ihre serialisierte Byte-Größe geschätzt. Übersteigt eine Quelle
        `max_new_normalized_bytes_per_source_per_run` (Default aus
        config.yaml external_context.archive, sonst
        DEFAULT_MAX_NEW_NORMALIZED_BYTES_PER_SOURCE_PER_RUN), wird für DIESE
        Quelle in diesem Run NICHTS geschrieben (nie stillschweigend
        kürzen) — siehe self.last_guard_blocked, das der Orchestrator dann
        in WARN + Alert übersetzt."""
        defaults = _config_archive_defaults()
        max_bytes_per_source = (
            max_new_normalized_bytes_per_source_per_run
            if max_new_normalized_bytes_per_source_per_run is not None
            else int(defaults.get("max_new_normalized_bytes_per_source_per_run",
                                   DEFAULT_MAX_NEW_NORMALIZED_BYTES_PER_SOURCE_PER_RUN))
        )

        self.last_guard_blocked = {}
        counts = {"new": 0, "duplicate": 0, "revision": 0}
        by_bucket: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        for obs in observations:
            month = obs.observation_time.strftime("%Y-%m")
            by_bucket[(obs.source_id, month)].append(obs)

        buckets_by_source: dict[str, list[tuple[str, list[Observation]]]] = defaultdict(list)
        for (source_id, month), obs_list in by_bucket.items():
            buckets_by_source[source_id].append((month, obs_list))

        for source_id, month_buckets in buckets_by_source.items():
            source_counts = {"new": 0, "duplicate": 0, "revision": 0}
            per_month_new_rows: dict[str, list[Observation]] = {}
            estimated_bytes = 0

            for month, obs_list in month_buckets:
                path = self._normalized_path(source_id, month)
                existing = self._read_jsonl(path)
                existing_by_key: dict[str, list[Observation]] = defaultdict(list)
                for o in existing:
                    existing_by_key[o.identity_key()].append(o)

                new_rows: list[Observation] = []
                for obs in obs_list:
                    key = obs.identity_key()
                    prior = existing_by_key[key]
                    # Duplikat nur, wenn der Wert der JÜNGSTEN Version
                    # entspricht — eine Revision zurück auf einen früheren
                    # Wert (A→B→A) ist eine echte neue Vintage und muss
                    # erhalten bleiben.
                    latest = max(prior, key=lambda p: (p.vintage_time or p.available_at), default=None)
                    if latest is not None and latest.value == obs.value:
                        source_counts["duplicate"] += 1
                        continue
                    # Quellen mit echten Vintages (ALFRED) liefern bei jedem
                    # Abruf ALLE historischen Vintages erneut: dieselbe
                    # (Vintage-Zeit, Wert)-Kombination ist schon archiviert
                    # -> Duplikat, keine neue Revision.
                    if any(p.vintage_time == obs.vintage_time and p.value == obs.value for p in prior):
                        source_counts["duplicate"] += 1
                        continue
                    if prior:
                        source_counts["revision"] += 1
                    else:
                        source_counts["new"] += 1
                    new_rows.append(obs)
                    existing_by_key[key].append(obs)

                per_month_new_rows[month] = new_rows
                estimated_bytes += sum(
                    len((json.dumps(obs.to_dict()) + "\n").encode("utf-8")) for obs in new_rows
                )

            # Erstimport (Quelle hat noch keine normalisierten Daten) ist ein
            # einmaliger historischer Backfill -> eigene, höhere Grenze
            # (max_backfill_bytes_per_source). Die tägliche Sperre gilt für den
            # inkrementellen Zuwachs; beides schreibt nie gekürzt.
            is_first_import = not self._normalized_dir(source_id).exists() or not any(
                self._normalized_dir(source_id).glob("*.jsonl"))
            # max_backfill_bytes: Quellen-Override (Registry) für einmalige
            # große Referenz-/Historienimporte, hart gedeckelt.
            backfill_limit = int(defaults.get("max_backfill_bytes_per_source",
                                              DEFAULT_MAX_BACKFILL_BYTES_PER_SOURCE))
            if max_backfill_bytes:
                backfill_limit = min(int(max_backfill_bytes), MAX_BACKFILL_BYTES_HARD_CAP)
            limit = (max(max_bytes_per_source, backfill_limit)
                     if is_first_import else max_bytes_per_source)
            if estimated_bytes > limit:
                self.last_guard_blocked[source_id] = {
                    "estimated_bytes": estimated_bytes, "max_bytes": limit,
                    "first_import": is_first_import,
                }
                continue  # NICHTS für diese Quelle in diesem Run schreiben — nie kürzen

            for month, new_rows in per_month_new_rows.items():
                if new_rows:
                    self._normalized_dir(source_id).mkdir(parents=True, exist_ok=True)
                    with open(self._normalized_path(source_id, month), "a", encoding="utf-8") as fh:
                        for obs in new_rows:
                            fh.write(json.dumps(obs.to_dict()) + "\n")

            counts["new"] += source_counts["new"]
            counts["duplicate"] += source_counts["duplicate"]
            counts["revision"] += source_counts["revision"]
        return counts

    def normalized_bytes_written_estimate(self, source_id: str) -> int:
        """Aktuelle Gesamtgröße der normalisierten JSONL-Dateien einer Quelle
        (Bytes) — der Orchestrator misst dies vor/nach store_observations(),
        um bytes_written fürs Manifest zu bestimmen (siehe orchestrator.py)."""
        d = self._normalized_dir(source_id)
        if not d.exists():
            return 0
        return sum(p.stat().st_size for p in d.glob("*.jsonl") if p.is_file())

    def load(self, source_id: str, since: datetime | None = None) -> list[Observation]:
        """Alle Observations einer Quelle (über alle Monate), optional gefiltert
        auf observation_time >= since. Liest transparent auch nach Object
        Storage ausgelagerte (lokal gelöschte) Monate zurück (siehe
        offload_normalized_months()) – PIT-Semantik bleibt unverändert:
        der Aufrufer sieht dieselben Observations wie vor dem Offload."""
        since = ensure_utc(since)
        out: list[Observation] = []
        d = self._normalized_dir(source_id)
        local_months: set[str] = set()
        if d.exists():
            for path in sorted(d.glob("*.jsonl")):
                local_months.add(path.name)
                for obs in self._read_jsonl(path):
                    if since is not None and obs.observation_time < since:
                        continue
                    out.append(obs)

        prefix = f"normalized/{source_id}/"
        offloaded = self._load_offload_manifest()
        offloaded_months = []
        for rel_path, info in offloaded.items():
            if not rel_path.startswith(prefix):
                continue
            month_file = Path(rel_path).name
            if month_file in local_months:
                continue  # lokale Datei hat Vorrang (sollte nach Offload nie vorkommen)
            offloaded_months.append((rel_path, info))
        for rel_path, info in sorted(offloaded_months):
            for obs in self._read_offloaded_jsonl(rel_path, info):
                if since is not None and obs.observation_time < since:
                    continue
                out.append(obs)
        return out

    def as_of(self, source_id: str, t: datetime, filters: dict | None = None) -> list[Observation]:
        """PIT-Snapshot: was zum Zeitpunkt t verfügbar/bekannt war (pit.available_as_of)."""
        from modules.external.pit import available_as_of
        obs = self.load(source_id)
        obs = available_as_of(obs, t)
        return self._apply_filters(obs, filters)

    def latest_vintages(self, source_id: str, filters: dict | None = None) -> list[Observation]:
        """Neueste bekannte Vintage je Identität (unabhängig von einem Stichtag)."""
        return self.as_of(source_id, utc_now(), filters)

    def vintage_history(self, source_id: str, series_id: str, entity_id: str, metric: str,
                         observation_time: datetime, dataset: str | None = None,
                         forecast_issue_time: datetime | None = None) -> list[Observation]:
        """Alle Versionen (Revisionen) EINES Datenpunkts, chronologisch nach
        vintage_time – Basis für "welcher Wert war am Datum X bekannt"."""
        observation_time = ensure_utc(observation_time)
        forecast_issue_time = ensure_utc(forecast_issue_time)
        rows = [
            o for o in self.load(source_id)
            if o.series_id == series_id and o.entity_id == entity_id and o.metric == metric
            and o.observation_time == observation_time
            and (dataset is None or o.dataset == dataset)
            and (forecast_issue_time is None or o.forecast_issue_time == forecast_issue_time)
        ]
        rows.sort(key=lambda o: o.vintage_time or o.available_at)
        return rows

    @staticmethod
    def _apply_filters(obs: list[Observation], filters: dict | None) -> list[Observation]:
        if not filters:
            return obs
        out = []
        for o in obs:
            ok = True
            for k, v in filters.items():
                if getattr(o, k, None) != v:
                    ok = False
                    break
            if ok:
                out.append(o)
        return out

    # ── Manifeste ────────────────────────────────────────────────────────

    def write_manifest(self, run_id: str, entries: list[dict], now: datetime | None = None) -> Path:
        now = ensure_utc(now) or utc_now()
        day_dir = self._manifest_dir(now.strftime("%Y-%m-%d"))
        day_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "generated_at": _iso(now),
            "git_sha": _git_sha(),
            "config_hash": _config_hash(),
            "entries": entries,
        }
        path = day_dir / f"{run_id}.json"
        path.write_text(json.dumps(manifest, indent=2, default=str))
        return path

    # ── Object-Storage-Offload / Retention ──────────────────────────────

    def _load_offload_manifest(self) -> dict:
        return self._load_json(self.offload_manifest_path(), default={})

    def _save_offload_manifest(self, data: dict) -> None:
        self._write_json(self.offload_manifest_path(), data)

    def _offload_cache_dir(self) -> Path:
        return Path(tempfile.gettempdir()) / "aas_external_archive_cache"

    def _read_offloaded_jsonl(self, rel_path: str, info: dict) -> list[Observation]:
        """Liest eine nach Object Storage ausgelagerte Monatsdatei zurück,
        mit lokalem /tmp-Cache (verifiziert per sha256 gegen den Manifest-
        Eintrag, damit ein veralteter Cache nie stillschweigend benutzt
        wird)."""
        cache_path = self._offload_cache_dir() / rel_path
        expected_sha = info.get("sha256")
        if cache_path.exists():
            try:
                if hashlib.sha256(cache_path.read_bytes()).hexdigest() == expected_sha:
                    return self._read_jsonl(cache_path)
            except Exception:
                pass
        backend = self.storage_backend()
        if backend is None:
            # Kein Backend konfiguriert (z.B. Checkout ohne S3-Secrets) –
            # ausgelagerte Monate sind dann schlicht nicht lesbar; PIT-
            # Aufrufer bekommen die lokal vorhandenen Daten, keinen Crash.
            return []
        try:
            raw = backend.get_bytes(info["s3_key"])
            data = gzip.decompress(raw)
            if expected_sha and hashlib.sha256(data).hexdigest() != expected_sha:
                return []
        except Exception:
            return []
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
        return self._read_jsonl(cache_path)

    @staticmethod
    def _is_closed_month(month: str, now: datetime) -> bool:
        """Ein Monat gilt als "geschlossen", wenn er vor dem aktuellen
        Kalendermonat liegt (nie der laufende Monat, der noch geschrieben
        werden kann)."""
        try:
            y, m = (int(x) for x in month.split("-"))
        except Exception:
            return False
        return (y, m) < (now.year, now.month)

    @staticmethod
    def _months_before_cutoff(month: str, now: datetime, retention_months: int) -> bool:
        y, m = (int(x) for x in month.split("-"))
        idx = y * 12 + (m - 1)
        cutoff_total = now.year * 12 + (now.month - 1) - int(retention_months)
        return idx < cutoff_total

    def offload_normalized_months(self, now: datetime | None = None,
                                   retention_months: int | None = None) -> dict:
        """Lädt normalisierte/rohe Dateien ins konfigurierte Object-Storage-
        Backend hoch (gzip, sha256 in Metadaten, idempotent) und entfernt
        anschließend NUR normalisierte Monatsdateien, die (a) abgeschlossen
        sind (nicht der laufende Monat) UND (b) älter als
        `git_retention_months` sind, aus dem Git-Arbeitsverzeichnis — erst
        nach verifiziertem Re-Download (sha256-Vergleich). Ist kein
        S3-Backend konfiguriert (`backend: git` oder fehlende Secrets),
        passiert NICHTS (kein Löschen, wie heute).

        Rückgabe: {"enabled", "uploaded": [...], "verified": [...],
        "offloaded": [...], "failures": {source_id: [messages]}}."""
        now = ensure_utc(now) or utc_now()
        defaults = _config_archive_defaults()
        backend_name = str(defaults.get("backend", DEFAULT_ARCHIVE_BACKEND) or DEFAULT_ARCHIVE_BACKEND)
        retention_months = (retention_months if retention_months is not None
                            else int(defaults.get("git_retention_months", DEFAULT_GIT_RETENTION_MONTHS)))

        result = {"enabled": False, "backend": backend_name, "uploaded": [], "verified": [],
                  "offloaded": [], "failures": {}}
        if backend_name != "s3":
            return result

        backend = self.storage_backend()
        if backend is None:
            result["failures"]["_backend"] = [
                "backend=s3 konfiguriert, aber kein Object-Storage erreichbar "
                "(EXTERNAL_ARCHIVE_S3_BUCKET fehlt oder boto3 nicht installiert)."
            ]
            return result

        result["enabled"] = True
        offload_manifest = self._load_offload_manifest()
        failed_rel_paths: set[str] = set()

        # 1) Upload aller normalisierten + rohen Dateien (Backup, immer –
        #    Löschung folgt separat und nur für normalisierte Monatsdateien).
        for area in ("normalized", "raw"):
            area_root = self.root / area
            if not area_root.exists():
                continue
            for path in sorted(area_root.rglob("*")):
                if not path.is_file():
                    continue
                if path.name.startswith("_"):  # z.B. _hashes.json – kein Datenfile
                    continue
                rel_path = str(path.relative_to(self.root)).replace(os.sep, "/")
                # source_id ist das erste Pfadsegment unter normalized/raw/
                parts = Path(rel_path).parts
                source_id = parts[1] if len(parts) > 1 else "unknown"
                try:
                    local_bytes = path.read_bytes()
                    local_sha = sha256_hex(local_bytes)
                    key = rel_path if rel_path.endswith(".gz") else rel_path + ".gz"

                    existing = offload_manifest.get(rel_path)
                    if existing and existing.get("sha256") == local_sha and backend.exists(key):
                        continue  # idempotent: unverändert & bereits hochgeladen

                    payload = local_bytes if rel_path.endswith(".gz") else gzip.compress(local_bytes)
                    backend.put_bytes(key, payload, metadata={"sha256": local_sha})
                    result["uploaded"].append(rel_path)

                    # Safety: NIE löschen vor verifiziertem Re-Download.
                    verify_raw = backend.get_bytes(key)
                    verify_bytes = gzip.decompress(verify_raw) if not rel_path.endswith(".gz") else verify_raw
                    if hashlib.sha256(verify_bytes).hexdigest() != local_sha:
                        result["failures"].setdefault(source_id, []).append(
                            f"Verifikation fehlgeschlagen für {rel_path} (sha256-Mismatch nach Re-Download)."
                        )
                        failed_rel_paths.add(rel_path)
                        continue
                    result["verified"].append(rel_path)
                    offload_manifest[rel_path] = {
                        "s3_key": key,
                        "sha256": local_sha,
                        "bytes": len(local_bytes),
                        "offloaded_at": None,  # erst bei tatsächlicher lokaler Löschung gesetzt
                    }
                except Exception as e:  # noqa: BLE001 - nie fatal für den Ingestion-Run
                    result["failures"].setdefault(source_id, []).append(
                        f"Upload fehlgeschlagen für {rel_path}: {e!r}"
                    )
                    failed_rel_paths.add(rel_path)

        # 2) Retention: nur normalisierte, abgeschlossene Monatsdateien
        #    älter als git_retention_months werden lokal gelöscht – erst
        #    nachdem sie oben erfolgreich verifiziert wurden.
        norm_root = self.root / "normalized"
        if norm_root.exists():
            for source_dir in sorted(p for p in norm_root.iterdir() if p.is_dir()):
                source_id = source_dir.name
                for path in sorted(source_dir.glob("*.jsonl")):
                    month = path.stem
                    if not self._is_closed_month(month, now):
                        continue
                    if not self._months_before_cutoff(month, now, retention_months):
                        continue
                    rel_path = str(path.relative_to(self.root)).replace(os.sep, "/")
                    manifest_entry = offload_manifest.get(rel_path)
                    if manifest_entry is None or manifest_entry.get("offloaded_at") is not None:
                        continue  # nicht (neu) verifiziert in diesem Lauf -> nie löschen
                    if rel_path in failed_rel_paths:
                        continue
                    try:
                        path.unlink()
                        manifest_entry["offloaded_at"] = _iso(now)
                        offload_manifest[rel_path] = manifest_entry
                        result["offloaded"].append(rel_path)
                    except Exception as e:  # noqa: BLE001
                        result["failures"].setdefault(source_id, []).append(
                            f"Löschen fehlgeschlagen für {rel_path}: {e!r}"
                        )

        self._save_offload_manifest(offload_manifest)
        return result

    # ── Storage-Telemetrie ───────────────────────────────────────────────

    def _manifest_entries_by_source_day(self) -> dict[str, dict[str, dict[str, int]]]:
        """source_id -> retrieval_day (YYYY-MM-DD) -> {raw_bytes,
        normalized_bytes}, ausschließlich aus den Ingestion-Run-Manifesten
        (manifests/<day>/<run>.json) gelesen — die Manifest-Tage SIND die
        tatsächlichen Abruf-Tage, im Unterschied zur beobachteten
        observation_time-Spanne (die bei z.B. Eurostat Jahrzehnte zurück-
        reicht, obwohl wir erst seit kurzem abrufen)."""
        out: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"raw_bytes": 0, "normalized_bytes": 0}))
        manifests_root = self.root / "manifests"
        if not manifests_root.exists():
            return out
        for day_dir in sorted(manifests_root.iterdir()):
            if not day_dir.is_dir():
                continue
            day = day_dir.name
            for run_path in sorted(day_dir.glob("*.json")):
                try:
                    manifest = json.loads(run_path.read_text())
                except Exception:
                    continue
                for entry in manifest.get("entries", []) or []:
                    sid = entry.get("source_id")
                    if not sid:
                        continue
                    out[sid][day]["raw_bytes"] += int(entry.get("bytes") or 0)
                    out[sid][day]["normalized_bytes"] += int(entry.get("bytes_written") or 0)
        return out

    def storage_telemetry(self, warn_mb_1y: float | None = None) -> dict:
        """Misst das ECHTE Wachstum je Quelle über die Ingestion-Run-
        Manifeste (retrieval days), NICHT über die beobachtete
        observation_time-Spanne der Daten selbst (das war der Bug: eine
        Quelle mit Jahrzehnten an historischen Beobachtungen, aber erst
        einem einzigen tatsächlichen Abruf-Tag, wurde so behandelt, als sei
        ihr Byte-Volumen über all diese Jahre gewachsen).

        Je Quelle wird der ERSTE beobachtete Retrieval-Tag als einmaliger
        Backfill (`one_off_backfill_bytes`) separat ausgewiesen; die
        Wachstumsrate (bytes/Tag) ist der Median der bytes_written/bytes der
        FOLGENDEN Retrieval-Tage. Mit nur einem einzigen Retrieval-Tag
        (genau der Fall des ersten Live-Runs) ist die Wachstumsrate 0 —
        NIE wird ein einmaliger Batch als Dauer-Wachstumsrate hochgerechnet.
        Projektionen 30d/1y/5y aus dieser Rate; Warnung, wenn die
        1y-Projektion die Schwelle übersteigt (Default 200 MB gesamt).
        Persistiert nach health/storage_telemetry.json."""
        defaults = _config_archive_defaults()
        warn_mb_1y = warn_mb_1y if warn_mb_1y is not None else float(
            defaults.get("storage_warn_mb_1y", DEFAULT_STORAGE_WARN_MB_1Y))

        by_source_day = self._manifest_entries_by_source_day()
        norm_root = self.root / "normalized"
        raw_root = self.root / "raw"

        source_ids = set(by_source_day)
        if norm_root.exists():
            source_ids.update(p.name for p in norm_root.iterdir() if p.is_dir())
        if raw_root.exists():
            source_ids.update(p.name for p in raw_root.iterdir() if p.is_dir())

        per_source: dict[str, dict] = {}
        total_bytes_per_day = 0.0
        total_one_off_backfill_bytes = 0

        for source_id in sorted(source_ids):
            days_map = by_source_day.get(source_id, {})
            days_sorted = sorted(days_map)
            rows = len(self.load(source_id)) if (norm_root / source_id).exists() else 0

            raw_one_off = norm_one_off = 0
            raw_growth_vals: list[int] = []
            norm_growth_vals: list[int] = []
            if days_sorted:
                first_day = days_sorted[0]
                raw_one_off = days_map[first_day]["raw_bytes"]
                norm_one_off = days_map[first_day]["normalized_bytes"]
                for d in days_sorted[1:]:
                    raw_growth_vals.append(days_map[d]["raw_bytes"])
                    norm_growth_vals.append(days_map[d]["normalized_bytes"])

            raw_bpd = statistics.median(raw_growth_vals) if raw_growth_vals else 0.0
            norm_bpd = statistics.median(norm_growth_vals) if norm_growth_vals else 0.0
            one_off = raw_one_off + norm_one_off

            per_source[source_id] = {
                "retrieval_days": len(days_sorted),
                "rows": rows,
                "one_off_backfill_bytes": one_off,
                "raw_bytes_per_day": raw_bpd,
                "normalized_bytes_per_day": norm_bpd,
            }
            total_bytes_per_day += raw_bpd + norm_bpd
            total_one_off_backfill_bytes += one_off

        projections = {
            "30d": total_bytes_per_day * 30,
            "1y": total_bytes_per_day * 365,
            "5y": total_bytes_per_day * 365 * 5,
        }
        flagged = projections["1y"] > warn_mb_1y * 1_000_000

        defaults = _config_archive_defaults()
        backend_name = str(defaults.get("backend", DEFAULT_ARCHIVE_BACKEND) or DEFAULT_ARCHIVE_BACKEND)
        bytes_in_git = 0
        for area in ("normalized", "raw"):
            area_root = self.root / area
            if area_root.exists():
                bytes_in_git += sum(p.stat().st_size for p in area_root.rglob("*") if p.is_file())
        offload_manifest = self._load_offload_manifest()
        bytes_offloaded = sum(
            int(v.get("bytes") or 0) for v in offload_manifest.values()
            if v.get("offloaded_at") is not None
        )

        telemetry = {
            "measured_at": _iso(utc_now()),
            "total_bytes_per_day": total_bytes_per_day,
            "one_off_backfill_bytes": total_one_off_backfill_bytes,
            "projections_bytes": projections,
            "warn_mb_1y": warn_mb_1y,
            "flagged": flagged,
            "backend": backend_name,
            "bytes_in_git": bytes_in_git,
            "bytes_offloaded": bytes_offloaded,
            "per_source": per_source,
        }
        if flagged:
            telemetry["migration_note"] = (
                "1y-Projektion überschreitet die konfigurierte Schwelle. GitHub-Actions- "
                "Artefakte/Checkout sind KEIN dauerhafter Speicher — bei anhaltendem "
                "Wachstum auf Object Storage (z.B. S3/GCS/Backblaze B2) migrieren, "
                "outputs/external_data als Git-Verlauf entlasten (z.B. Kompaktierung "
                "alter raw/-Monate oder externe Ablage der normalisierten JSONL-Dateien)."
            )
        self._health_dir().mkdir(parents=True, exist_ok=True)
        self._write_json(self.storage_telemetry_path(), telemetry)
        return telemetry

    # ── Hilfsfunktionen ──────────────────────────────────────────────────

    @staticmethod
    def _read_jsonl(path: Path) -> list[Observation]:
        if not path.exists():
            return []
        out = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(Observation.from_dict(json.loads(line)))
        return out

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except Exception:
            return default

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
