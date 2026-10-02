"""modules/entity_resolution – Point-in-Time Entity Resolution (Infrastruktur, kein Alpha).

Ticker ↕ Name ↕ CIK ↕ LEI ↕ Legal Entity ↕ Parent ↕ Ultimate Parent ↕ Land.
Siehe docs/ENTITY_RESOLUTION.md.
"""
from modules.entity_resolution.store import EntityRecord, EntityStore, normalize_name  # noqa: F401
