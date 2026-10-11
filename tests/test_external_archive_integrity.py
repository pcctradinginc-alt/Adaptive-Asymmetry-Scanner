"""Zeilen-Integrität des externen PIT-Archivs (Audit-Befund 14).

Neue Zeilen tragen `payload_hash` (sha256 über den kanonischen Inhalt) und
`quality_status`; Altzeilen (ohne diese Felder) bleiben unverändert und gelten
als LEGACY_UNVERIFIED (row_quality) bzw. nicht prüfbar (verify_row -> None).
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone

import pytest

from modules.external.archive import (
    HASHED_FIELDS, QUALITY_STATUSES, ExternalArchive, canonical_payload, compute_payload_hash,
    derive_quality_status, row_quality, stamp_observation, verify_row,
)
from modules.external.pit import AvailabilityPrecision, Observation

UTC = timezone.utc
SRC = "fred_test"


def _obs(value=105.6, obs_time=datetime(2026, 8, 1, tzinfo=UTC), retrieved=datetime(2026, 9, 1, 12, tzinfo=UTC),
         **kw) -> Observation:
    base = dict(
        source_id=SRC, dataset="ds", series_id="S1", entity_id="US", metric="m", value=value, unit="idx",
        observation_time=obs_time, available_at=datetime(2026, 8, 20, tzinfo=UTC), retrieved_at=retrieved,
        availability_precision=AvailabilityPrecision.EXACT_DATE, parser_version="1",
        attrs={"realtime_end": "9999-12-31"},
    )
    base.update(kw)
    return Observation(**base)


def _lines(arc: ExternalArchive, month: str = "2026-08") -> list[dict]:
    path = arc._normalized_path(SRC, month)
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ── neue Zeilen: Hash + Status ──────────────────────────────────────────────

def test_new_row_gets_hash_and_status(tmp_path):
    arc = ExternalArchive(tmp_path)
    counts = arc.store_observations([_obs()])
    assert counts["new"] == 1
    (line,) = _lines(arc)
    assert len(line["payload_hash"]) == 64 and int(line["payload_hash"], 16) >= 0
    assert line["quality_status"] == "OK" and line["quality_status"] in QUALITY_STATUSES
    assert verify_row(line) is True
    assert row_quality(line) == "OK"
    loaded = arc.load(SRC)[0]                      # Rückweg über Observation
    assert loaded.payload_hash == line["payload_hash"] and loaded.quality_status == "OK"
    assert verify_row(loaded) is True and row_quality(loaded) == "OK"


def test_hash_follows_documented_canonical_spec(tmp_path):
    arc = ExternalArchive(tmp_path)
    arc.store_observations([_obs()])
    (line,) = _lines(arc)
    content = {k: line[k] for k in HASHED_FIELDS}
    expected = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=False).encode("utf-8")).hexdigest()
    assert line["payload_hash"] == expected
    assert "retrieved_at" not in canonical_payload(line) and "quality_status" not in canonical_payload(line)


def test_hash_is_deterministic_and_value_sensitive():
    a, b = _obs(), _obs()
    assert compute_payload_hash(a) == compute_payload_hash(b) == compute_payload_hash(a.to_dict())
    # erneuter Abruf desselben Inhalts (anderes retrieved_at) -> gleicher Hash
    assert compute_payload_hash(_obs(retrieved=datetime(2026, 10, 1, tzinfo=UTC))) == compute_payload_hash(a)
    # anderer Wert / andere Vintage / andere attrs -> anderer Hash
    assert compute_payload_hash(_obs(value=105.7)) != compute_payload_hash(a)
    assert compute_payload_hash(_obs(vintage_time=datetime(2026, 9, 1, tzinfo=UTC))) != compute_payload_hash(a)
    assert compute_payload_hash(_obs(attrs={"realtime_end": "2026-09-01"})) != compute_payload_hash(a)
    # Key-Reihenfolge in attrs ist egal (kanonisches JSON)
    assert compute_payload_hash(_obs(attrs={"a": 1, "b": 2})) == compute_payload_hash(_obs(attrs={"b": 2, "a": 1}))


def test_revision_row_gets_its_own_hash(tmp_path):
    arc = ExternalArchive(tmp_path)
    arc.store_observations([_obs(value=1.0)])
    counts = arc.store_observations([_obs(value=2.0, vintage_time=datetime(2026, 9, 5, tzinfo=UTC))])
    assert counts["revision"] == 1
    first, second = _lines(arc)
    assert first["payload_hash"] != second["payload_hash"]
    assert verify_row(first) is True and verify_row(second) is True


def test_store_is_idempotent_and_does_not_mutate_input(tmp_path):
    arc = ExternalArchive(tmp_path)
    o = _obs()
    arc.store_observations([o])
    assert o.payload_hash == "" and o.quality_status == ""        # Original unverändert
    counts = arc.store_observations([_obs()])
    assert counts == {"new": 0, "duplicate": 1, "revision": 0}
    assert len(_lines(arc)) == 1


# ── quality_status-Ableitung ────────────────────────────────────────────────

def test_quality_status_derivation(tmp_path):
    assert derive_quality_status(_obs()) == "OK"
    assert derive_quality_status(_obs(value=None)) == "MISSING"
    assert derive_quality_status(_obs(value=float("nan"))) == "INVALID"
    assert derive_quality_status(_obs(value=float("inf"))) == "INVALID"
    assert derive_quality_status(_obs(value="abc")) == "INVALID"
    assert derive_quality_status(_obs(quality_status="STALE")) == "STALE"
    # Wert-Befund schlägt Konnektor-Hinweis; LEGACY wird nie für neue Zeilen vergeben
    assert derive_quality_status(_obs(value=None, quality_status="STALE")) == "MISSING"
    assert derive_quality_status(_obs(quality_status="LEGACY_UNVERIFIED")) == "OK"

    arc = ExternalArchive(tmp_path)
    arc.store_observations([
        _obs(value=None, obs_time=datetime(2026, 8, 2, tzinfo=UTC)),
        _obs(value=3.0, obs_time=datetime(2026, 8, 3, tzinfo=UTC), quality_status="STALE"),
    ])
    by_day = {ln["observation_time"][:10]: ln for ln in _lines(arc)}
    assert by_day["2026-08-02"]["quality_status"] == "MISSING" and verify_row(by_day["2026-08-02"]) is True
    assert by_day["2026-08-03"]["quality_status"] == "STALE" and verify_row(by_day["2026-08-03"]) is True


def test_connector_supplied_source_hash_is_preserved_in_attrs():
    o = _obs(payload_hash="deadbeef" * 8)
    s = stamp_observation(o)
    assert s.attrs["source_payload_hash"] == "deadbeef" * 8
    assert s.payload_hash != o.payload_hash and verify_row(s) is True
    again = stamp_observation(s)                                   # Re-Stamping ist stabil
    assert again.payload_hash == s.payload_hash and again.attrs == s.attrs


# ── Legacy-Zeilen ───────────────────────────────────────────────────────────

def _legacy_line(value=7.0, day="2026-07-01") -> str:
    d = _obs(value=value, obs_time=datetime.fromisoformat(day + "T00:00:00+00:00")).to_dict()
    d.pop("quality_status")                    # So sahen Zeilen vor dem Fix auf der Platte aus:
    d["payload_hash"] = ""                     # leerer Hash, kein quality_status
    return json.dumps(d)


def test_legacy_row_is_legacy_unverified(tmp_path):
    arc = ExternalArchive(tmp_path)
    arc._normalized_dir(SRC).mkdir(parents=True)
    line = _legacy_line()
    arc._normalized_path(SRC, "2026-07").write_text(line + "\n", encoding="utf-8")

    raw = json.loads(line)
    assert "quality_status" not in raw
    assert row_quality(raw) == "LEGACY_UNVERIFIED" and verify_row(raw) is None
    (loaded,) = arc.load(SRC)
    assert loaded.quality_status == "" and loaded.payload_hash == ""
    assert row_quality(loaded) == "LEGACY_UNVERIFIED" and verify_row(loaded) is None
    # ein Hash allein (z. B. Quell-Hash eines Konnektors, ohne Status) ist ebenfalls nicht prüfbar
    assert row_quality(dict(raw, payload_hash="abc")) == "LEGACY_UNVERIFIED"
    assert verify_row(dict(raw, payload_hash="abc")) is None


def test_existing_rows_are_never_rewritten(tmp_path):
    arc = ExternalArchive(tmp_path)
    arc._normalized_dir(SRC).mkdir(parents=True)
    legacy = _legacy_line(day="2026-08-01") + "\n"
    path = arc._normalized_path(SRC, "2026-08")
    path.write_text(legacy, encoding="utf-8")

    counts = arc.store_observations([_obs(value=1.5, obs_time=datetime(2026, 8, 5, tzinfo=UTC))])
    assert counts["new"] == 1
    text = path.read_text(encoding="utf-8")
    assert text.startswith(legacy)             # Altzeile byte-identisch, neue nur angehängt
    old, new = [json.loads(ln) for ln in text.splitlines()]
    assert row_quality(old) == "LEGACY_UNVERIFIED" and verify_row(old) is None
    assert row_quality(new) == "OK" and verify_row(new) is True
    assert sorted(row_quality(o) for o in arc.load(SRC)) == ["LEGACY_UNVERIFIED", "OK"]


# ── Manipulation ────────────────────────────────────────────────────────────

def test_tampered_row_fails_verification(tmp_path):
    arc = ExternalArchive(tmp_path)
    arc.store_observations([_obs(value=100.0)])
    path = arc._normalized_path(SRC, "2026-08")
    (line,) = _lines(arc)
    assert verify_row(line) is True

    for field, bad in (("value", 999.0), ("available_at", "2020-01-01T00:00:00+00:00"),
                       ("series_id", "OTHER"), ("attrs", {"realtime_end": "x"}), ("payload_hash", "0" * 64)):
        assert verify_row(dict(line, **{field: bad})) is False, field

    # Manipulation direkt in der Datei -> beim Laden erkannt
    path.write_text(path.read_text(encoding="utf-8").replace('"value": 100.0', '"value": 101.0'), encoding="utf-8")
    (loaded,) = arc.load(SRC)
    assert loaded.value == 101.0 and verify_row(loaded) is False
    assert row_quality(loaded) == "OK"          # row_quality prüft nicht; verify_row ist der Integritätstest


def test_status_inconsistent_with_value_fails_verification(tmp_path):
    arc = ExternalArchive(tmp_path)
    arc.store_observations([_obs(value=None)])
    (line,) = _lines(arc)
    assert line["quality_status"] == "MISSING" and verify_row(line) is True
    assert verify_row(dict(line, quality_status="OK")) is False        # fehlenden Wert nicht als OK ausgeben
    assert verify_row(dict(line, quality_status="BOGUS")) is False
    assert verify_row(dict(line, quality_status="LEGACY_UNVERIFIED")) is None


def test_non_finite_value_is_invalid_and_verifiable(tmp_path):
    arc = ExternalArchive(tmp_path)
    arc.store_observations([_obs(value=float("nan"))])
    (line,) = _lines(arc)                       # NaN überlebt den JSON-Roundtrip
    assert math.isnan(line["value"]) and line["quality_status"] == "INVALID"
    assert verify_row(line) is True and verify_row(arc.load(SRC)[0]) is True
    assert verify_row(dict(line, quality_status="OK")) is False


@pytest.mark.parametrize("partition", ["monthly", "weekly"])
def test_weekly_partition_rows_are_stamped(tmp_path, partition):
    arc = ExternalArchive(tmp_path)
    arc.store_observations([_obs()], partition=partition)
    rows = arc.load(SRC)
    assert len(rows) == 1 and verify_row(rows[0]) is True
