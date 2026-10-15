"""modules/news_novelty.py – Neuheit der Schlagzeilen per lokalem Embedding (C1, SHADOW, 2026-10-15).

Frage: Ist die heutige Nachrichtenlage eines Tickers NEU oder wiederholt sie, was in den letzten Tagen schon
berichtet wurde? Messung ohne LLM-Kosten: lokales Satz-Embedding (sentence-transformers/all-MiniLM-L6-v2 über das
vorhandene transformers-Paket, Mean-Pooling, L2-normiert; keine neue Abhängigkeit, kein bezahlter Dienst).

  novelty_max   1 − kleinste maximale Kosinus-Ähnlichkeit: neueste Schlagzeile gegen die Historie
  novelty_mean  Mittel über die heutigen Schlagzeilen von (1 − max. Ähnlichkeit zur Historie)
  n_history     Zahl der Vergleichs-Schlagzeilen (Fenster HISTORY_DAYS, nur frühere Tage -> PIT)

Gedächtnis: outputs/research/news_memory/<UTC-Datum>.jsonl (Ticker + Schlagzeilen-Texte, kein Binär-Embedding im
Repo; Embeddings werden je Lauf neu berechnet). Ohne Historie oder ohne Modell -> None (nie 0, nie geraten).
Nur Shadow-Feld im Kandidaten-Ledger; keine Champion-, Gate- oder Promotion-Wirkung.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(os.environ.get("FINBERT_CACHE", "/tmp/finbert_cache"))
MEMORY = Path("outputs/research/news_memory")
HISTORY_DAYS = 14
MAX_HEADLINES = 8
VERSION = "news-novelty-v1"
_model = None
_tok = None
_load_failed = False
_emb_cache: dict[str, list[float]] = {}


def _load() -> bool:
    global _model, _tok, _load_failed
    if _model is not None:
        return True
    if _load_failed:                                       # nur ein Ladeversuch je Lauf
        return False
    try:
        from transformers import AutoModel, AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
        _model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
        _model.eval()
        return True
    except Exception as e:  # noqa: BLE001
        _load_failed = True
        log.warning(f"News-Neuheit: Modell nicht ladbar ({e}) -> Werte None")
        return False


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed(texts: list[str]) -> list[list[float]] | None:
    """L2-normierte Satz-Embeddings (Mean-Pooling); None ohne Modell."""
    todo = [t for t in dict.fromkeys(texts) if _key(t) not in _emb_cache]
    if todo:
        if not _load():
            return None
        import torch
        for i in range(0, len(todo), 64):
            batch = todo[i:i + 64]
            enc = _tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            with torch.no_grad():
                out = _model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            for t, v in zip(batch, vec.tolist()):
                _emb_cache[_key(t)] = v
    return [_emb_cache[_key(t)] for t in texts]


def _clean(headlines) -> list[str]:
    out = []
    for h in headlines or []:
        t = " ".join(str(h or "").split())[:256]
        if t and t not in out:
            out.append(t)
    return out[:MAX_HEADLINES]


def history(ticker: str, today: date, root: Path | None = None, days: int = HISTORY_DAYS) -> list[str]:
    """Schlagzeilen des Tickers aus den VORHERIGEN `days` Kalendertagen (nie heute -> PIT)."""
    out: list[str] = []
    base = Path(root or MEMORY)
    for k in range(1, days + 1):
        f = base / f"{(today - timedelta(days=k)).isoformat()}.jsonl"
        if not f.exists():
            continue
        for ln in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("ticker") == ticker:
                out += [h for h in r.get("headlines") or [] if h not in out]
    return out


def novelty(today_headlines: list[str], past: list[str], embed_fn=None) -> dict:
    cur = _clean(today_headlines)
    base = {"novelty_max": None, "novelty_mean": None, "n_today": len(cur), "n_history": len(past),
            "novelty_version": VERSION}
    if not cur or not past:
        return {**base, "novelty_status": "no_history" if cur else "no_headlines"}
    vecs = (embed_fn or embed)(cur + past)
    if vecs is None:
        return {**base, "novelty_status": "model_unavailable"}
    a, b = vecs[:len(cur)], vecs[len(cur):]
    sims = [max(sum(x * y for x, y in zip(u, v)) for v in b) for u in a]
    nov = [round(1.0 - max(-1.0, min(1.0, s)), 4) for s in sims]
    return {**base, "novelty_max": max(nov), "novelty_mean": round(sum(nov) / len(nov), 4), "novelty_status": "ok"}


def remember(ticker: str, headlines: list[str], now: datetime | None = None, root: Path | None = None) -> None:
    """Heutige Schlagzeilen ins Gedächtnis (einmal je Ticker und Tag). Nie werfend."""
    try:
        cur = _clean(headlines)
        if not cur:
            return
        now = now or datetime.now(timezone.utc)
        base = Path(root or MEMORY)
        base.mkdir(parents=True, exist_ok=True)
        f = base / f"{now.date().isoformat()}.jsonl"
        if f.exists() and any(json.loads(ln).get("ticker") == ticker for ln in f.read_text().splitlines() if ln):
            return
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ticker": ticker, "headlines": cur, "at": now.isoformat(timespec="seconds")},
                                ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        log.debug(f"news_novelty.remember [{ticker}] ignoriert: {e}")


def score_and_remember(ticker: str, headlines: list[str], now: datetime | None = None,
                       root: Path | None = None) -> dict:
    """Neuheit gegen die Historie (nur frühere Tage), danach heutige Schlagzeilen merken. Nie werfend."""
    if root is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return {"novelty_status": "disabled_in_tests", "novelty_version": VERSION}   # nie Modell-Download/Repo-Schreiben
    now = now or datetime.now(timezone.utc)
    try:
        res = novelty(headlines, history(ticker, now.date(), root))
    except Exception as e:  # noqa: BLE001
        res = {"novelty_status": f"error:{type(e).__name__}", "novelty_version": VERSION}
    remember(ticker, headlines, now, root)
    return res
