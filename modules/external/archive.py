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
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from modules.external.pit import Observation, ensure_utc, utc_now
from modules.external.sources.base import RawRecord

DEFAULT_ROOT = "outputs/external_data"
DEFAULT_RAW_MAX_BYTES = 2_000_000
DEFAULT_RAW_POLICY = "hash_only"
DEFAULT_STORAGE_WARN_MB_1Y = 200


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
    def __init__(self, root: str | os.PathLike = DEFAULT_ROOT):
        self.root = Path(root)

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

    def store_observations(self, observations: Iterable[Observation]) -> dict:
        """Idempotente Ablage: gleiche Identität + gleicher Wert = duplicate
        (skip); gleiche Identität + anderer Wert = NEUE Vintage-Zeile (nie
        überschreiben)."""
        counts = {"new": 0, "duplicate": 0, "revision": 0}
        by_bucket: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        for obs in observations:
            month = obs.observation_time.strftime("%Y-%m")
            by_bucket[(obs.source_id, month)].append(obs)

        for (source_id, month), obs_list in by_bucket.items():
            path = self._normalized_path(source_id, month)
            existing = self._read_jsonl(path)
            existing_by_key: dict[str, list[Observation]] = defaultdict(list)
            for o in existing:
                existing_by_key[o.identity_key()].append(o)

            new_rows: list[Observation] = []
            for obs in obs_list:
                key = obs.identity_key()
                prior = existing_by_key[key]
                # Duplikat nur, wenn der Wert der JÜNGSTEN Version entspricht —
                # eine Revision zurück auf einen früheren Wert (A→B→A) ist eine
                # echte neue Vintage und muss erhalten bleiben.
                latest = max(prior, key=lambda p: (p.vintage_time or p.available_at), default=None)
                if latest is not None and latest.value == obs.value:
                    counts["duplicate"] += 1
                    continue
                if prior:
                    counts["revision"] += 1
                else:
                    counts["new"] += 1
                new_rows.append(obs)
                existing_by_key[key].append(obs)

            if new_rows:
                self._normalized_dir(source_id).mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as fh:
                    for obs in new_rows:
                        fh.write(json.dumps(obs.to_dict()) + "\n")
        return counts

    def load(self, source_id: str, since: datetime | None = None) -> list[Observation]:
        """Alle Observations einer Quelle (über alle Monate), optional gefiltert
        auf observation_time >= since."""
        since = ensure_utc(since)
        out: list[Observation] = []
        d = self._normalized_dir(source_id)
        if not d.exists():
            return out
        for path in sorted(d.glob("*.jsonl")):
            for obs in self._read_jsonl(path):
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

    # ── Storage-Telemetrie ───────────────────────────────────────────────

    def storage_telemetry(self, warn_mb_1y: float | None = None) -> dict:
        """Misst Bytes/Zeilen je Quelle aus den Dateien, projiziert 30d/1y/5y
        und markiert eine Warnung wenn die 1y-Projektion die Schwelle
        übersteigt (Default 200 MB gesamt). Persistiert nach
        health/storage_telemetry.json."""
        defaults = _config_archive_defaults()
        warn_mb_1y = warn_mb_1y if warn_mb_1y is not None else float(
            defaults.get("storage_warn_mb_1y", DEFAULT_STORAGE_WARN_MB_1Y))

        per_source: dict[str, dict] = {}
        raw_root = self.root / "raw"
        norm_root = self.root / "normalized"

        source_ids = set()
        if raw_root.exists():
            source_ids.update(p.name for p in raw_root.iterdir() if p.is_dir())
        if norm_root.exists():
            source_ids.update(p.name for p in norm_root.iterdir() if p.is_dir())

        total_bytes_per_day = 0.0
        for source_id in sorted(source_ids):
            raw_bytes, raw_days = self._dir_bytes_and_days(raw_root / source_id)
            norm_bytes, norm_days = self._dir_bytes_and_days(norm_root / source_id)
            rows = len(self.load(source_id)) if (norm_root / source_id).exists() else 0
            days = max(raw_days, norm_days, 1)
            raw_bpd = raw_bytes / days
            norm_bpd = norm_bytes / days
            per_source[source_id] = {
                "raw_bytes": raw_bytes,
                "normalized_bytes": norm_bytes,
                "rows": rows,
                "rows_per_day": rows / days,
                "raw_bytes_per_day": raw_bpd,
                "normalized_bytes_per_day": norm_bpd,
            }
            total_bytes_per_day += raw_bpd + norm_bpd

        projections = {
            "30d": total_bytes_per_day * 30,
            "1y": total_bytes_per_day * 365,
            "5y": total_bytes_per_day * 365 * 5,
        }
        flagged = projections["1y"] > warn_mb_1y * 1_000_000
        telemetry = {
            "measured_at": _iso(utc_now()),
            "total_bytes_per_day": total_bytes_per_day,
            "projections_bytes": projections,
            "warn_mb_1y": warn_mb_1y,
            "flagged": flagged,
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

    @staticmethod
    def _dir_bytes_and_days(path: Path) -> tuple[int, int]:
        if not path.exists():
            return 0, 0
        total = 0
        months = set()
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
                # YYYY-MM Monatsordner oder .jsonl-Dateiname als Zeit-Proxy
                for part in (p.parent.name, p.stem):
                    if len(part) == 7 and part[4] == "-":
                        months.add(part)
        days = max(len(months) * 30, 1)
        return total, days

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
