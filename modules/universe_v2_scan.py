"""modules/universe_v2_scan.py – täglicher UNIVERSE_V2-Shadow-Scan (LLM-frei, keine Produktionswirkung).

Signal = die nicht-LLM Stufen des V1-Champions, unverändert übernommen:
  1. Relative Volume >= 0.6 × clip(VIX/20, 0.5, 1.5); unter 0.25 nie; dazwischen nur mit >= 3 News
  2. News-Präsenz (DataIngestion._fetch_news: Finnhub -> NewsAPI -> yfinance)
  3. Pre-MC sigma-only (MirofishSimulation, 45 T, neutraler Impact) >= pre_mc_threshold
Danach frische Options-/Liquiditätsbewertung (universe_v2.assess) und Aufzeichnung im V2-Ledger.
Die LLM-Stufen (Haiku/Sonnet) laufen für V2 bewusst NICHT (Kostenbudget) – V2-Evidenz betrifft
daher die Quant-Stufen; ein LLM-V2-Test wäre ein eigener Vertrag.

Trade-Empfehlungen: nur über universe_v2_ledger.recommendations() für Segmente mit
TRADE_RECOMMENDATION_ENABLED (verifizierter PromotionController-Zustand). Standard: keine.
    python -m modules.universe_v2_scan
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from modules import universe_v2 as uv
from modules import universe_v2_ledger as v2l

log = logging.getLogger(__name__)

RV_BASE, RV_NEWS_MIN = 0.6, 0.25          # identisch zu modules/data_ingestion.py (V1-Champion)


def rv_threshold(vix) -> float:
    v = float(vix) if vix else 20.0
    return round(RV_BASE * max(0.5, min(1.5, v / 20.0)), 2)


def scan(today: date, snapshot: dict, *, vix, quotes_fn, news_fn, premc_fn, chain_fn,
         cfg: dict | None = None) -> tuple[list[dict], dict]:
    cfg = cfg or uv.load_cfg()
    sc = cfg["shadow_scan"]
    members = uv.membership(snapshot, "research")
    stats = {"research_universe": len(members), "rv_pass": 0, "news_pass": 0, "premc_pass": 0,
             "option_ok": 0, "candidates": 0}
    quotes = quotes_fn(sorted(members))
    thr = rv_threshold(vix)
    pre = []
    for t, q in quotes.items():
        avg, vol = q.get("avg_volume"), q.get("volume")
        rv = vol / avg if avg and vol else None
        if rv is None or (rv < thr and rv < RV_NEWS_MIN):
            continue
        pre.append((rv, t, q))
    pre.sort(key=lambda x: -x[0])
    out = []
    for rv, t, q in pre[: int(sc["max_option_checks_per_day"]) * 3]:
        stats["rv_pass"] += 1
        news = news_fn(t) or []
        if (rv < thr and not (len(news) >= 3 and rv >= RV_NEWS_MIN)) or not news:
            continue
        stats["news_pass"] += 1
        hit = premc_fn(t, q.get("price"))
        if hit is None or hit < float(sc["pre_mc_threshold"]):
            continue
        stats["premc_pass"] += 1
        if stats["premc_pass"] > int(sc["max_option_checks_per_day"]):
            break
        chain, src = chain_fn(t, today, cfg)
        rec = members[t]
        a = uv.assess(t, q, chain, rec.get("market_cap"), today, cfg, rec.get("market_cap_source"))
        a["option_source"] = src
        if not a["research_ok"]:
            continue
        stats["option_ok"] += 1
        out.append({"ticker": t, "assessment": a, "hit_rate": round(hit, 4), "rel_volume": round(rv, 3),
                    "news_count": len(news), "direction": "BULLISH"})
        if len(out) >= int(sc["max_candidates_per_day"]):
            break
    stats["candidates"] = len(out)
    return out, stats


# ── CI-Lauf ─────────────────────────────────────────────────────────────────
def _news(t: str) -> list:
    from modules.data_ingestion import DataIngestion
    return DataIngestion()._fetch_news(t, {})


def _premc(t: str, price) -> float | None:
    from modules.mirofish_simulation import MirofishSimulation
    sc = uv.load_cfg()["shadow_scan"]
    r = MirofishSimulation().run_for_dte({"ticker": t, "current_price": price or 0,
                                          "deep_analysis": {"impact": 5, "surprise": 5,
                                                            "time_to_materialization": "2-3 Monate"}},
                                         days_to_expiry=int(sc["pre_mc_days"]), min_hit_rate=float(sc["pre_mc_threshold"]))
    return (r or {}).get("simulation", {}).get("hit_rate")


def _vix() -> float | None:
    try:
        import yfinance as yf
        return float(yf.Ticker("^VIX").fast_info.last_price)
    except Exception:  # noqa: BLE001 – ohne VIX: neutrale Schwelle (wie V1)
        return None


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    today = datetime.now(timezone.utc).date()
    snap = uv.latest_snapshot()
    if not snap:
        print("Kein UNIVERSE_V2-Snapshot – Discovery (python -m modules.universe_v2) zuerst.")
        return 0
    try:
        from modules.universe import get_universe
        v1 = set(get_universe())
    except Exception as e:  # noqa: BLE001
        log.warning(f"V1-Mitglieder nicht bestimmbar ({e}) – in_universe_v1 bleibt 0")
        v1 = set()
    vix = _vix()
    cands, stats = scan(today, snap, vix=vix, quotes_fn=uv.fetch_quotes, news_fn=_news, premc_fn=_premc,
                        chain_fn=uv.fetch_chain)
    rows = v2l.record_candidates(cands, today=today.isoformat(), vix=vix, v1_members=v1)
    from modules.production_intelligence_adapter import v2_segment_levels
    recs = v2l.recommendations(rows, v2_segment_levels(), v2l.v2_contracts())
    v2l.RECOMMENDATIONS.mkdir(parents=True, exist_ok=True)
    (v2l.RECOMMENDATIONS / f"{today.isoformat()}.json").write_text(json.dumps(
        {"date": today.isoformat(), "stats": stats, "recorded": len(rows),
         "recommendations": [{k: r.get(k) for k in ("ticker", "market_cap_bucket", "execution_quality",
                                                     "reference_option", "enabled_by")} for r in recs]}, default=str))
    print(f"V2-Shadow-Scan: {stats} · aufgezeichnet {len(rows)} · freigegebene Empfehlungen {len(recs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
