"""Research-only cap extension for genuine observed option snapshots.

Never change production eligibility or the pinned Champion configuration. No
external provider quota may be silently assumed from the fact an API key exists.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
import yaml

log = logging.getLogger(__name__)
CONFIG = Path(__file__).resolve().parents[1] / "config/research_snapshot_expansion.yaml"
MAX_EXTRA = 40
MAX_TOTAL = 80


def policy(path: Path = CONFIG) -> dict:
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict) or set(cfg) != {
            "enabled", "max_extra_snapshots", "max_total_snapshots", "require_tradier_plan_approval"}:
            raise ValueError("Unexpected research snapshot policy")
        if type(cfg["enabled"]) is not bool or type(cfg["require_tradier_plan_approval"]) is not bool:
            raise ValueError("Boolean policy fields invalid")
        if type(cfg["max_extra_snapshots"]) is not int or not 0 <= cfg["max_extra_snapshots"] <= MAX_EXTRA:
            raise ValueError("Research extra cap invalid")
        if type(cfg["max_total_snapshots"]) is not int or not 40 <= cfg["max_total_snapshots"] <= MAX_TOTAL:
            raise ValueError("Research total cap invalid")
        if not cfg["require_tradier_plan_approval"]:
            raise ValueError("Provider approval requirement cannot be disabled")
        return cfg
    except (OSError, yaml.YAMLError, ValueError) as e:
        log.error("Research snapshot expansion invalid, disabled: %s", e)
        return {"enabled": False, "reason": "invalid_config"}


def additional_slots(champion_slots: int, *, cfg: dict | None = None,
                     environ: dict | None = None) -> tuple[int, str]:
    """Only unused *research* capacity; no effect on champion or on its priority."""
    cfg = policy() if cfg is None else cfg
    env = os.environ if environ is None else environ
    if not cfg.get("enabled"):
        return 0, "disabled"
    if env.get("TRADIER_RESEARCH_80_APPROVED") != "1":
        return 0, "provider_not_approved"
    if not env.get("TRADIER_API_KEY", "").strip():
        return 0, "no_api_key"
    # Hard ceilings guard accidental config or environment changes.
    try:
        extra = int(cfg["max_extra_snapshots"])
        total = int(cfg["max_total_snapshots"])
    except (TypeError, ValueError, KeyError):
        return 0, "invalid_config"
    if not 0 <= extra <= MAX_EXTRA or not 40 <= total <= MAX_TOTAL:
        return 0, "invalid_caps"
    return min(extra, max(0, total - max(0, champion_slots))), "approved"
