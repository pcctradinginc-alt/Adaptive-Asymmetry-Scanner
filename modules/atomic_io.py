"""modules/atomic_io.py – crash-sichere Datei-Schreibvorgänge (Audit 2026-10-04).

Vorher schrieben pipeline.py/feedback.py history.json und der PromotionController
promotion_state.json direkt mit open("w"): ein Abbruch mitten im Schreiben hinterließ eine
halbe Datei. Jetzt:

  atomic_write_text / atomic_write_json  temporäre Datei im selben Verzeichnis + fsync +
                                          os.replace -> Leser sehen nie einen halben Zustand
  append_jsonl                            ein write()-Aufruf je Batch; endet die Datei nach einem
                                          Abbruch ohne Zeilenumbruch, wird zuerst einer ergänzt
                                          (die abgebrochene Zeile verschmilzt nie mit der nächsten)
  read_jsonl                              überspringt NUR eine abgeschnittene LETZTE Zeile (Abbruch
                                          beim Anhängen, nie vollständig geschrieben -> idempotente
                                          Jobs schreiben sie beim Neustart erneut); defekte Zeilen
                                          mitten in der Datei -> Fehler (nie still verwerfen)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path | str, obj, **dump_kwargs) -> None:
    dump_kwargs.setdefault("default", str)
    atomic_write_text(path, json.dumps(obj, **dump_kwargs))


def _ends_without_newline(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size == 0:
        return False
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) != b"\n"


def append_jsonl(path: Path | str, rows: list[dict], **dump_kwargs) -> int:
    """Hängt Zeilen in EINEM write() an (Abbruch -> höchstens die letzte Zeile unvollständig)."""
    path = Path(path)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_kwargs.setdefault("default", str)
    payload = "".join(json.dumps(r, **dump_kwargs) + "\n" for r in rows)
    if _ends_without_newline(path):
        log.warning(f"{path}: letzte Zeile ohne Zeilenumbruch (Abbruch?) – wird abgeschlossen")
        payload = "\n" + payload
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    return len(rows)


class CorruptLedgerError(ValueError):
    """Defekte Zeile MITTEN in einer append-only Datei – nie still verwerfen."""


def read_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out, bad = [], []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            bad.append(i)
    if not bad:
        return out
    last_content = max(i for i, x in enumerate(lines) if x.strip())
    if bad == [last_content]:
        log.warning(f"{path}: abgeschnittene letzte Zeile ignoriert (Abbruch beim Anhängen)")
        return out
    # Abgeschnittene Zeile, an die nach einem Neustart weiter angehängt wurde (vor append_jsonl):
    # eine einzelne defekte Zeile, deren Rest nicht parsebar ist, gilt als Abbruch-Rest -> melden
    raise CorruptLedgerError(f"{path}: {len(bad)} defekte Zeile(n) (erste: {bad[0] + 1}) – manuelle Prüfung nötig")
