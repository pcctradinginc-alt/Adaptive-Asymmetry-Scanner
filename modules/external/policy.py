"""
modules/external/policy.py – Entscheidungs-Policy für externen Kontext.

Modi (config external_context.mode):
  off         – kein Kontext, keine Policy-Auswirkung.
  shadow      – Kontext wird berechnet/angehängt, NIE auf die Entscheidung
                angewendet: score_delta IMMER 0, veto IMMER False.
  challenger  – wie shadow, zusätzlich wird ein hypothetischer score_delta
                aus den (noch nicht promoteten) Challenger-Regeln berechnet
                und NUR zur Beobachtung zurückgegeben — nie angewendet.
  production  – NUR explizit promotete Regeln (promoted_rules) dürfen einen
                score_delta liefern, hart begrenzt auf
                external_context.production.max_score_delta (Default 0).

WICHTIG: Nichts in pipeline.py darf score_delta/veto konsumieren, außer
mode == "production" UND max_score_delta > 0 (siehe Kommentar in pipeline.py
an der Stelle, an der das geprüft wird — Guard ist dort im Code, nicht hier).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyRule:
    """Eine einzelne (Challenger- oder promotete) Regel. `condition` und
    `delta` sind rein deklarativ (kein beliebiger Code) – condition ist ein
    Callable(primitives: dict) -> bool, delta ein fester score_delta."""
    name: str
    condition: "callable"
    delta: float
    mechanism: str = ""


@dataclass
class ExternalContextPolicy:
    mode: str = "shadow"                  # off | shadow | challenger | production
    promoted_rules: list[PolicyRule] = field(default_factory=list)
    challenger_rules: list[PolicyRule] = field(default_factory=list)
    max_score_delta: float = 0.0

    def evaluate(self, candidate_ctx: dict) -> dict:
        """candidate_ctx: candidate["external_context"] (primitives/states/
        divergences/etc. — read-only, nie mutiert).

        Rückgabe: {relation, materiality, confidence, score_delta, veto,
                   hypothetical_score_delta}
        relation/materiality/confidence werden aus candidate_ctx["relation"]
        übernommen (LLM-Shadow-Ergebnis), falls vorhanden — reine
        Durchreichung, die Policy erfindet keine eigene Einschätzung."""
        rel = (candidate_ctx or {}).get("relation") or {}
        out = {
            "relation":    rel.get("relation", "NEUTRAL"),
            "materiality": rel.get("materiality", "NONE"),
            "confidence":  rel.get("confidence", 0.0),
            "score_delta": 0.0,
            "veto": False,
            "hypothetical_score_delta": 0.0,
        }

        if self.mode in ("off", "shadow"):
            # score_delta/veto bleiben IMMER 0/False — reine Beobachtung.
            return out

        primitives = (candidate_ctx or {}).get("primitives") or {}

        if self.mode == "challenger":
            # Nur zur Beobachtung — wird NIE angewendet (score_delta bleibt 0).
            out["hypothetical_score_delta"] = self._apply_rules(
                self.challenger_rules, primitives
            )
            return out

        if self.mode == "production":
            # NUR promotete Regeln, hart begrenzt auf max_score_delta (Default 0).
            raw = self._apply_rules(self.promoted_rules, primitives)
            bounded = max(-abs(self.max_score_delta), min(abs(self.max_score_delta), raw))
            out["score_delta"] = bounded if self.max_score_delta > 0 else 0.0
            out["veto"] = False   # externer Kontext löst in dieser Version niemals ein Veto aus
            return out

        return out

    @staticmethod
    def _apply_rules(rules: list[PolicyRule], primitives: dict) -> float:
        total = 0.0
        for rule in rules or []:
            try:
                if rule.condition(primitives):
                    total += rule.delta
            except Exception:
                continue
        return total


def load_policy_from_config() -> ExternalContextPolicy:
    """Baut eine ExternalContextPolicy aus config.yaml (external_context.mode,
    external_context.production.max_score_delta). `promoted_rules`/
    `challenger_rules` sind für diese Version leer (Regel-DSL ist noch nicht
    Teil des Configs) — Erweiterung ist rein additiv."""
    try:
        from modules.config import cfg
        ext = getattr(cfg, "external_context", None)
        mode = str(getattr(ext, "mode", "off") or "off")
        prod = getattr(ext, "production", None)
        max_delta = float(getattr(prod, "max_score_delta", 0.0) or 0.0)
    except Exception:
        mode, max_delta = "off", 0.0
    return ExternalContextPolicy(mode=mode, promoted_rules=[], challenger_rules=[],
                                  max_score_delta=max_delta)
