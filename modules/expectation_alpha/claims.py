"""modules/expectation_alpha/claims.py – maschinell prüfbare Aussagen aus News/Katalysator-Text (SHADOW).

Das LLM darf Text interpretieren und Aussagen strukturieren (claim_extraction.py, separater Call, nie der
Produktions-Prompt). Ob eine faktische Aussage stimmt, entscheidet NUR diese deterministische Prüfung gegen
Roh-/Extrakt-Evidenz, die zum Entscheidungszeitpunkt verfügbar war:
  * SEC XBRL companyfacts (available_at = Filing-Tag + 1, jüngstes Filing je Periode zum Stichtag)
  * SEC-Submissions (8-K-Items, acceptanceDateTime exakt)
Status je Claim: VERIFIED | UNVERIFIED | CONTRADICTED.
  * keine Selbstverifikation: LLM-Text, -Konfidenz oder -Begründung zählen nie als Evidenz
  * keine erfundenen Werte: value_if_known nur, wenn die Zahl wörtlich im Quelltext steht
  * unprüfbare Claim-Typen (z. B. guidance_raised ohne strukturierte Guidance-Daten) bleiben UNVERIFIED
  * Widersprüche werden explizit als CONTRADICTED markiert
V1: nur Research/Logging. verified_claim_fraction ist KEIN Score und fließt in keine Entscheidung.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from modules.expectation_alpha.schemas import rnd

log = logging.getLogger(__name__)

VERIFIED, UNVERIFIED, CONTRADICTED = "VERIFIED", "UNVERIFIED", "CONTRADICTED"
CLAIM_STATUSES = (VERIFIED, UNVERIFIED, CONTRADICTED)
DIRECTIONS = ("up", "down", "none")

# Claim-Typ -> Prüfmethode (fest, versioniert). evidence_family für die Familien-Aggregation.
#   xbrl_yoy:   jüngstes Quartal vs. Vorjahresquartal (bzw. Instant ~1 Jahr zuvor)
#   xbrl_accel: YoY-Wachstum des jüngsten Quartals vs. YoY-Wachstum des Vorquartals
#   xbrl_margin: Nettomarge (net_income / revenue) jüngstes Quartal vs. Vorjahresquartal
#   filing_item: 8-K mit passendem Item im Fenster vor der Entscheidung (Abwesenheit widerlegt nie)
#   None:       keine maschinenlesbare PIT-Evidenz im Bestand -> immer UNVERIFIED
TAXONOMY_VERSION = "claims-v1"
CLAIM_TYPES: dict[str, dict] = {
    "revenue_up":            {"method": "xbrl_yoy", "metric": "revenue", "direction": "up",
                              "description": "Umsatz des jüngsten Quartals über Vorjahresquartal"},
    "revenue_down":          {"method": "xbrl_yoy", "metric": "revenue", "direction": "down",
                              "description": "Umsatz des jüngsten Quartals unter Vorjahresquartal"},
    "revenue_growth_up":     {"method": "xbrl_accel", "metric": "revenue", "direction": "up",
                              "description": "Umsatzwachstum (YoY) beschleunigt sich gegenüber dem Vorquartal"},
    "revenue_growth_down":   {"method": "xbrl_accel", "metric": "revenue", "direction": "down",
                              "description": "Umsatzwachstum (YoY) verlangsamt sich gegenüber dem Vorquartal"},
    "eps_up":                {"method": "xbrl_yoy", "metric": "eps_diluted", "direction": "up",
                              "description": "Verwässertes EPS über Vorjahresquartal"},
    "eps_down":              {"method": "xbrl_yoy", "metric": "eps_diluted", "direction": "down",
                              "description": "Verwässertes EPS unter Vorjahresquartal"},
    "margin_up":             {"method": "xbrl_margin", "metric": "net_margin", "direction": "up",
                              "description": "Nettomarge über Vorjahresquartal"},
    "margin_down":           {"method": "xbrl_margin", "metric": "net_margin", "direction": "down",
                              "description": "Nettomarge unter Vorjahresquartal"},
    "operating_cash_flow_up":   {"method": "xbrl_yoy", "metric": "cfo", "direction": "up",
                                 "description": "Operativer Cashflow über Vorjahr"},
    "operating_cash_flow_down": {"method": "xbrl_yoy", "metric": "cfo", "direction": "down",
                                 "description": "Operativer Cashflow unter Vorjahr"},
    "share_count_down":      {"method": "xbrl_yoy", "metric": "shares", "direction": "down",
                              "description": "Aktienanzahl gesunken (Rückkauf)"},
    "share_count_up":        {"method": "xbrl_yoy", "metric": "shares", "direction": "up",
                              "description": "Aktienanzahl gestiegen (Verwässerung)"},
    "earnings_released":     {"method": "filing_item", "items": ["2.02"], "direction": "none",
                              "description": "Quartalsergebnis veröffentlicht (8-K Item 2.02)"},
    "material_agreement":    {"method": "filing_item", "items": ["1.01"], "direction": "none",
                              "description": "Wesentlicher Vertrag/Deal (8-K Item 1.01)"},
    "acquisition_completed": {"method": "filing_item", "items": ["2.01"], "direction": "none",
                              "description": "Übernahme/Verkauf abgeschlossen (8-K Item 2.01)"},
    "management_change":     {"method": "filing_item", "items": ["5.02"], "direction": "none",
                              "description": "Wechsel im Management/Vorstand (8-K Item 5.02)"},
    "guidance_raised":       {"method": None, "direction": "up",
                              "description": "Prognose angehoben (keine strukturierte PIT-Guidance im Bestand)"},
    "guidance_lowered":      {"method": None, "direction": "down",
                              "description": "Prognose gesenkt (keine strukturierte PIT-Guidance im Bestand)"},
    "orders_accelerating":   {"method": None, "direction": "up",
                              "description": "Auftragseingang beschleunigt (keine PIT-Auftragsdaten im Bestand)"},
    "capex_increased":       {"method": None, "direction": "up",
                              "description": "Investitionen erhöht (CapEx-Konzept nicht im XBRL-Bestand)"},
    "other":                 {"method": None, "direction": "none",
                              "description": "sonstige Aussage ohne definierte Prüfmethode"},
}
EVIDENCE_FAMILY = {"xbrl_yoy": "company_filing", "xbrl_accel": "company_filing", "xbrl_margin": "company_filing",
                   "filing_item": "company_filing", None: None}
SOURCE_FIELDS = ("catalyst", "asymmetry_reasoning", "bear_case", "macro_assessment")
DEADBAND = 0.01                       # |relative Änderung| <= 1 % -> weder bestätigt noch widerlegt
FILING_WINDOW_DAYS = 14               # 8-K-Items: Fenster vor der Entscheidung


@dataclass
class Claim:
    claim_type: str
    entity: str | None
    metric: str | None
    direction: str
    value_if_known: float | None
    source_reference: str | None
    confidence: float | None

    def to_dict(self) -> dict:
        return {"claim_type": self.claim_type, "entity": self.entity, "metric": self.metric,
                "direction": self.direction, "value_if_known": self.value_if_known,
                "source_reference": self.source_reference, "confidence": self.confidence}


def _num_in_text(v: float, text: str) -> bool:
    """Steht die Zahl wörtlich im Quelltext? (Dezimal-Komma/-Punkt, %-Angaben, Tausender-Trennzeichen)."""
    t = text.replace(" ", " ").replace("\xa0", " ")
    nums = re.findall(r"[-+]?\d[\d.,']*", t)
    for raw in nums:
        cands = {raw.replace(",", "").replace("'", ""), raw.replace(".", "").replace(",", ".").replace("'", "")}
        for c in cands:
            try:
                if math.isclose(float(c.rstrip(".")), float(v), rel_tol=1e-9, abs_tol=1e-9):
                    return True
            except ValueError:
                continue
    return False


def normalize(raw: list | None, source_texts: dict[str, str], ticker: str | None) -> tuple[list[Claim], list[dict]]:
    """Schema-Prüfung der LLM-Ausgabe. -> (gültige Claims, verworfene mit Grund). Werte, die nicht wörtlich
    im Quelltext stehen, werden entfernt (nie erfunden). Unbekannte Claim-Typen -> 'other'."""
    ok, dropped = [], []
    text_all = " ".join(str(source_texts.get(f) or "") for f in SOURCE_FIELDS)
    for i, c in enumerate(raw or []):
        if not isinstance(c, dict):
            dropped.append({"index": i, "reason": "kein Objekt"})
            continue
        ct = str(c.get("claim_type") or "").strip().lower()
        if ct not in CLAIM_TYPES:
            ct = "other"
        spec = CLAIM_TYPES[ct]
        d = str(c.get("direction") or spec["direction"]).strip().lower()
        if d not in DIRECTIONS:
            d = spec["direction"]
        if spec["direction"] != "none" and d != spec["direction"]:
            dropped.append({"index": i, "reason": f"Richtung {d} widerspricht Claim-Typ {ct}"})
            continue
        v = c.get("value_if_known")
        val = None
        notes = []
        if v is not None:
            try:
                fv = float(v)
                if math.isfinite(fv) and _num_in_text(fv, text_all):
                    val = fv
                else:
                    notes.append("value_not_in_source")
            except (TypeError, ValueError):
                notes.append("value_not_numeric")
        ref = str(c.get("source_reference") or "").strip() or None
        if ref and ref.split(":")[0] not in SOURCE_FIELDS:
            notes.append("unknown_source_reference")
        conf = c.get("confidence")
        try:
            conf = min(1.0, max(0.0, float(conf))) if conf is not None else None
        except (TypeError, ValueError):
            conf = None
        cl = Claim(ct, ticker, spec.get("metric") or (str(c.get("metric"))[:60] if c.get("metric") else None),
                   d, val, ref, conf)
        ok.append(cl)
        if notes:
            dropped.append({"index": i, "reason": ",".join(notes), "kept": True})
    return ok, dropped


# ── Evidenz (PIT) ───────────────────────────────────────────────────────────
def load_stores() -> dict:
    """SEC-Speicher und Ticker->CIK-Map einmal je Lauf (lazy beim ersten Claim). Fehler -> leer + Grund."""
    from modules.entity_resolution.store import DEFAULT_PATH, EntityStore
    from modules.external.sources import sec_ingest as si
    out: dict = {"xbrl": pd.DataFrame(), "filings": pd.DataFrame(), "ciks": {}, "errors": []}
    try:
        out["ciks"] = {r.ticker: str(int(r.cik)).zfill(10) for r in EntityStore(DEFAULT_PATH).records
                       if r.usage == "research_ticker_identity" and r.valid_to is None and r.cik}
    except Exception as e:  # noqa: BLE001 – ohne Entity-Map bleibt jeder Claim UNVERIFIED (nie Default)
        out["errors"].append(f"entity_map: {type(e).__name__}: {e}")
    for key, f in (("xbrl", "xbrl.csv.gz"), ("filings", "filings.csv.gz")):
        try:
            out[key] = si.read_store(si.DIR / f)
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"{key}: {type(e).__name__}: {e}")
    return out


def load_evidence(ticker: str, decision_time: datetime, *, stores: dict | None = None,
                  xbrl_rows: pd.DataFrame | None = None, filing_rows: pd.DataFrame | None = None,
                  cik: str | None = None) -> dict:
    """Roh-Evidenz eines Titels mit available_at < decision_time (strikt). Fehlende Quelle -> leer (nie Default)."""
    t = decision_time if decision_time.tzinfo else decision_time.replace(tzinfo=timezone.utc)
    stores = stores if stores is not None else (load_stores() if xbrl_rows is None and filing_rows is None else {})
    cik = cik or (stores.get("ciks") or {}).get(ticker)
    out = {"ticker": ticker, "cik": cik, "decision_time": t.isoformat(), "xbrl": pd.DataFrame(),
           "filings": pd.DataFrame(), "sources": []}
    if not cik:
        out["error"] = "kein CIK (Entity-Map)"
        return out
    ent = f"cik:{cik}"
    xr = xbrl_rows if xbrl_rows is not None else stores.get("xbrl", pd.DataFrame())
    fr = filing_rows if filing_rows is not None else stores.get("filings", pd.DataFrame())
    if len(xr):
        x = xr[xr["entity_id"] == ent].reset_index(drop=True)
        if len(x):
            at = x["attrs"].map(lambda a: json.loads(a) if isinstance(a, str) else (a or {}))
            x = pd.DataFrame({"metric": x["metric"], "value": pd.to_numeric(x["value"], errors="coerce"),
                              "avail": pd.to_datetime(x["available_at"], utc=True),
                              "start": pd.to_datetime(at.map(lambda a: a.get("start")), utc=True),
                              "end": pd.to_datetime(at.map(lambda a: a.get("end")), utc=True),
                              "accn": at.map(lambda a: a.get("accn"))}).dropna(subset=["value"])
            x = x[(x["avail"] < pd.Timestamp(t)) & ~(x["end"] > x["avail"])]
            out["xbrl"] = x
            out["sources"].append("sec_companyfacts")
    if len(fr):
        f = fr[fr["entity_id"] == ent].copy()
        if len(f):
            f["avail"] = pd.to_datetime(f["available_at"], utc=True)
            f = f[f["avail"] < pd.Timestamp(t)]
            f["attrs_d"] = f["attrs"].map(lambda a: json.loads(a) if isinstance(a, str) else (a or {}))
            out["filings"] = f
            out["sources"].append("sec_submissions")
    return out


# ── Prüfung (deterministisch) ───────────────────────────────────────────────
def _status_from_change(rel: float | None, direction: str) -> str:
    if rel is None or not math.isfinite(rel):
        return UNVERIFIED
    if abs(rel) <= DEADBAND:
        return UNVERIFIED
    up = rel > 0
    return VERIFIED if (up == (direction == "up")) else CONTRADICTED


def _series(x: pd.DataFrame, metric: str) -> pd.Series:
    from modules.external.sources import sec_xbrl as sx
    k = sx._known(x, x["avail"].max()) if len(x) else x
    if metric == "shares":
        return sx._instants(k, "shares")
    if metric == "cfo":                                   # Jahreswerte (Duration ~1 Jahr)
        d = k[k["metric"] == "cfo"]
        d = d[((d["end"] - d["start"]).dt.days >= 350) & ((d["end"] - d["start"]).dt.days <= 380)]
        return d.set_index("end")["value"].sort_index()
    return sx._quarters(k, metric)


def _prior_year(s: pd.Series, end) -> float | None:
    prev = s[(s.index <= end - pd.Timedelta(days=340)) & (s.index >= end - pd.Timedelta(days=390))]
    return float(prev.iloc[-1]) if len(prev) else None


def _check_xbrl(spec: dict, ev: dict, max_stale_days: int = 200) -> dict:
    x = ev.get("xbrl")
    if x is None or not len(x):
        return {"status": UNVERIFIED, "reason": "keine XBRL-Evidenz vor dem Entscheidungszeitpunkt"}
    t = pd.Timestamp(ev["decision_time"])
    m = spec["metric"]
    if spec["method"] == "xbrl_margin":
        rev, ni = _series(x, "revenue"), _series(x, "net_income")
        common = rev.index.intersection(ni.index)
        if not len(common):
            return {"status": UNVERIFIED, "reason": "Umsatz/Nettogewinn nicht für dieselbe Periode bekannt"}
        s = pd.Series({e: ni[e] / rev[e] for e in common if rev[e] and rev[e] > 0}).sort_index()
    else:
        s = _series(x, m)
    if not len(s):
        return {"status": UNVERIFIED, "reason": f"Metrik {m} nicht im XBRL-Bestand"}
    end = s.index[-1]
    if (t - end).days > max_stale_days:
        return {"status": UNVERIFIED, "reason": f"jüngste Periode {end.date()} zu alt"}
    cur, prev = float(s.iloc[-1]), _prior_year(s, end)
    evid = {"metric": m, "period_end": end.date().isoformat(), "value": rnd(cur, 6), "prior_year_value": rnd(prev, 6)}
    if spec["method"] == "xbrl_accel":
        prior_end = s.index[-2] if len(s) > 1 else None
        p2 = _prior_year(s, prior_end) if prior_end is not None else None
        if prev in (None, 0) or p2 in (None, 0) or prior_end is None:
            return {"status": UNVERIFIED, "reason": "zu wenig Quartale für Wachstumsvergleich", "evidence": evid}
        g_now, g_prev = cur / prev - 1.0, float(s.iloc[-2]) / p2 - 1.0
        evid.update(growth_latest=rnd(g_now, 6), growth_previous=rnd(g_prev, 6))
        st = _status_from_change(g_now - g_prev, spec["direction"])
    else:
        if prev is None or prev == 0:
            return {"status": UNVERIFIED, "reason": "kein Vorjahreswert", "evidence": evid}
        rel = (cur - prev) / abs(prev) if m != "net_margin" else cur - prev
        evid["change"] = rnd(rel, 6)
        st = _status_from_change(rel, spec["direction"])
    out = {"status": st, "evidence": evid,
           "reason": {VERIFIED: "Richtung durch XBRL bestätigt", CONTRADICTED: "XBRL zeigt Gegenrichtung",
                      UNVERIFIED: f"Änderung innerhalb ±{DEADBAND:.0%} (nicht entscheidbar)"}[st]}
    acc = x[(x["end"] == end)]["accn"].dropna()
    out["evidence_id"] = f"xbrl:{ev.get('cik')}:{m}:{end.date().isoformat()}:{acc.iloc[-1] if len(acc) else 'n/a'}"
    return out


def _check_filing(spec: dict, ev: dict) -> dict:
    f = ev.get("filings")
    if f is None or not len(f):
        return {"status": UNVERIFIED, "reason": "keine Filing-Evidenz vor dem Entscheidungszeitpunkt"}
    t = pd.Timestamp(ev["decision_time"])
    win = f[(f["avail"] >= t - pd.Timedelta(days=FILING_WINDOW_DAYS))]
    hits = win[win["attrs_d"].map(lambda a: a.get("form") in ("8-K", "8-K/A")
                                  and bool(set(a.get("items") or []) & set(spec["items"])))]
    if len(hits):
        h = hits.sort_values("avail").iloc[-1]
        return {"status": VERIFIED, "reason": f"8-K Item {','.join(spec['items'])} am {h['avail'].date()}",
                "evidence_id": f"sec_filing:{h['series_id']}",
                "evidence": {"form": h["attrs_d"].get("form"), "items": h["attrs_d"].get("items"),
                             "accepted": h["avail"].isoformat()}}
    # Abwesenheit eines 8-K widerlegt nichts (Meldung kann anders erfolgt sein) -> UNVERIFIED
    return {"status": UNVERIFIED, "reason": f"kein 8-K Item {','.join(spec['items'])} in {FILING_WINDOW_DAYS} T"}


def verify(claims: list[Claim], evidence: dict) -> list[dict]:
    """Deterministische Prüfung jedes Claims ausschließlich gegen `evidence` (PIT). Nie gegen LLM-Text."""
    out = []
    for c in claims:
        spec = CLAIM_TYPES.get(c.claim_type) or CLAIM_TYPES["other"]
        method = spec.get("method")
        if method is None:
            res = {"status": UNVERIFIED, "reason": "Claim-Typ ohne maschinenlesbare PIT-Evidenz im Bestand"}
        elif method.startswith("xbrl"):
            res = _check_xbrl(spec, evidence)
        elif method == "filing_item":
            res = _check_filing(spec, evidence)
        else:
            res = {"status": UNVERIFIED, "reason": f"unbekannte Methode {method}"}
        out.append({**c.to_dict(), "verification_method": method, "evidence_family": EVIDENCE_FAMILY.get(method),
                    "status": res["status"], "reason": res.get("reason"), "evidence_id": res.get("evidence_id"),
                    "evidence": res.get("evidence")})
    return out


def summarize(verified: list[dict] | None, *, extraction_status: str) -> dict:
    """Kennzahlen je Kandidat. Ohne erfolgreiche Extraktion: Zählwerte None (nie 0 -> keine Scheinsicherheit)."""
    if extraction_status != "OK" or verified is None:
        return {"extraction_status": extraction_status, "n_claims": None, "n_verified": None, "n_unverified": None,
                "n_contradicted": None, "verified_claim_fraction": None, "taxonomy_version": TAXONOMY_VERSION}
    n = len(verified)
    cnt = {s: sum(1 for v in verified if v["status"] == s) for s in CLAIM_STATUSES}
    return {"extraction_status": "OK", "n_claims": n, "n_verified": cnt[VERIFIED], "n_unverified": cnt[UNVERIFIED],
            "n_contradicted": cnt[CONTRADICTED],
            "verified_claim_fraction": rnd(cnt[VERIFIED] / n, 4) if n else None,
            "taxonomy_version": TAXONOMY_VERSION}


def source_texts(analysis: dict) -> dict[str, str]:
    da = analysis.get("deep_analysis") or {}
    return {f: str(da.get(f)) for f in SOURCE_FIELDS if da.get(f)}

