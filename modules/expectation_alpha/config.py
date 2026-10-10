"""modules/expectation_alpha/config.py – lädt config/expectation_alpha.yaml (vorregistrierte Schwellen).

config_hash = SHA-256 der kanonischen Konfiguration; jede Ledger-Zeile trägt ihn. Schwellen werden nie
zur Laufzeit verändert (keine Optimierung nach Resultaten)."""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml

PATH = Path(__file__).resolve().parents[2] / "config" / "expectation_alpha.yaml"
MODES = ("off", "shadow")


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load(path: Path | str | None = None) -> dict:
    """Kopie der Konfiguration (Aufrufer können sie nicht versehentlich global verändern)."""
    cfg = copy.deepcopy(_load(str(path or PATH)))
    if cfg.get("mode") not in MODES:
        raise ValueError(f"expectation_alpha.mode muss {MODES} sein, nicht {cfg.get('mode')!r} "
                         f"(V1 kennt keinen Produktionsmodus)")
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()[:16]


def sector_map_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps({"v": cfg.get("sector_map_version"), "m": cfg.get("sector_map")},
                                     sort_keys=True).encode()).hexdigest()[:12]


def enabled(cfg: dict | None = None) -> bool:
    try:
        return (cfg or load()).get("mode") == "shadow"
    except (OSError, ValueError, yaml.YAMLError):
        return False
