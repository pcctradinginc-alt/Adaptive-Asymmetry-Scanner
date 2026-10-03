"""
modules/rl_robust_shadow.py – robuster PPO-Challenger (SHADOW ONLY).

Warum: Das Produktions-PPO (Belohnung = Rendite) ist degeneriert (erst 100 %
SKIP, dann 100 % BOOST; Audit 2026-09-27). Der Mittelwert der Optionsrenditen
wird von wenigen Ausreißer-Gewinnern getragen (Median −42 %), daher lohnt
"immer BOOST" im Erwartungswert.

Dieser Challenger:
  - Belohnung = Log-Depotwachstum bei realem Positionsanteil
    (rl_environment.robust_reward, f = portfolio.max_position_pct) statt Rendite
  - Trainingsdaten chronologisch nach entry_date sortiert, fester Seed,
    Neu-Training je Lauf (kein kumulatives Weitertrainieren derselben Trades)
  - Walk-forward-Diagnose: Training auf den ersten 80 % (chronologisch),
    Bewertung der Aktionen auf den späteren 20 %; Kollaps (eine Aktion
    >= 95 %) wird in den Metadaten markiert
  - Inferenz nur als Shadow-Aktion im Candidate Ledger
    (features.rl_robust_action) -- beeinflusst NIE Ranking, Gates oder Trades.
    Bewertung ausschließlich prospektiv über challengers.yaml (ppo_robust_shadow).
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

ROBUST_MODEL_PATH = Path("outputs/models/ppo_robust_shadow.zip")
ROBUST_META_PATH = Path("outputs/models/ppo_robust_shadow_meta.json")
ACTION_NAMES = {0: "SKIP", 1: "NORMAL", 2: "BOOST"}
COLLAPSE_SHARE = 0.95
MIN_TRADES = 20
TRAIN_FRACTION = 0.8
SEED = 7
TIMESTEPS_PER_TRADE = 40      # Gesamtschritte = 40 × Anzahl Trades (feste Regel)


def _chronological(history: dict) -> list[dict]:
    from modules.outcomes import is_reliable_outcome
    closed = [t for t in history.get("closed_trades", []) if t.get("outcome") is not None and is_reliable_outcome(t)]
    return sorted(closed, key=lambda t: (str(t.get("entry_date", "")), str(t.get("ticker", ""))))


def _new_model(env):
    from stable_baselines3 import PPO
    return PPO("MlpPolicy", env, learning_rate=3e-4,
               n_steps=min(64, len(env.trade_data)), batch_size=min(32, len(env.trade_data)),
               n_epochs=10, gamma=0.95, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
               policy_kwargs={"net_arch": [64, 64]}, seed=SEED, verbose=0)


def _fit(trades: list[dict]):
    from modules.rl_environment import OptionsRLEnv
    env = OptionsRLEnv(trade_data=trades, reward_mode="robust")
    model = _new_model(env)
    model.learn(total_timesteps=TIMESTEPS_PER_TRADE * len(trades), progress_bar=False)
    return model


def evaluate_actions(model, trades: list[dict]) -> dict:
    """Deterministische Aktionen + Ökonomie der gewählten Trades."""
    from modules.rl_environment import features_to_obs
    actions, outcomes = [], []
    for t in trades:
        dte = int((t.get("option") or {}).get("dte", 0) or (t.get("roi_analysis") or {}).get("dte", 90))
        obs = features_to_obs(features=t.get("features", {}), simulation=t.get("simulation", {}),
                              deep_analysis=t.get("deep_analysis", {}), dte=dte)
        a, _ = model.predict(obs, deterministic=True)
        actions.append(ACTION_NAMES.get(int(a), str(int(a))))
        outcomes.append(float(t["outcome"]))
    counts = {k: actions.count(k) for k in ACTION_NAMES.values()}
    n = len(actions)
    taken = [o for a, o in zip(actions, outcomes) if a != "SKIP"]
    skipped = [o for a, o in zip(actions, outcomes) if a == "SKIP"]

    def _s(v):
        return {"n": len(v), "mean": round(statistics.fmean(v), 4) if v else None,
                "median": round(statistics.median(v), 4) if v else None,
                "win_rate": round(sum(x > 0 for x in v) / len(v), 3) if v else None}
    return {
        "n": n, "action_counts": counts,
        "max_share": round(max(counts.values()) / n, 3) if n else None,
        "collapsed": bool(n and max(counts.values()) / n >= COLLAPSE_SHARE),
        "all": _s(outcomes), "taken": _s(taken), "skipped": _s(skipped),
        "avoided_losers": sum(1 for o in skipped if o < 0),
        "lost_winners": sum(1 for o in skipped if o > 0),
        "lost_big_winners": sum(1 for o in skipped if o >= 1.0),
    }


def train_robust_shadow(history: dict) -> dict | None:
    """Walk-forward-Diagnose + finales Modell auf allen Trades. Gibt die
    Metadaten zurück (auch in ROBUST_META_PATH), None bei zu wenig Daten."""
    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        log.info("stable-baselines3 fehlt -> robuster PPO-Shadow übersprungen")
        return None
    trades = _chronological(history)
    if len(trades) < MIN_TRADES:
        return None
    cut = int(len(trades) * TRAIN_FRACTION)
    train, test = trades[:cut], trades[cut:]
    wf_model = _fit(train)
    walk_forward = {
        "train_n": len(train), "test_n": len(test),
        "train_until": train[-1].get("entry_date"), "test_from": test[0].get("entry_date"),
        "test": evaluate_actions(wf_model, test),
    }
    final = _fit(trades)
    ROBUST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.save(str(ROBUST_MODEL_PATH))
    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reward": "log(1 + f*s*r)/f, f = portfolio.max_position_pct; SKIP=0, BOOST s=1.5",
        "n_trades": len(trades), "seed": SEED,
        "timesteps": TIMESTEPS_PER_TRADE * len(trades),
        "walk_forward": walk_forward,
        "in_sample": evaluate_actions(final, trades),
        "model_sha256": hashlib.sha256(ROBUST_MODEL_PATH.read_bytes()).hexdigest(),
        "status": "SHADOW_ONLY",
    }
    ROBUST_META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


def shadow_actions(candidates: list[dict]) -> dict[str, str]:
    """{ticker: SKIP|NORMAL|BOOST} des robusten Modells. Nie filternd."""
    if not ROBUST_MODEL_PATH.exists():
        return {}
    try:
        from stable_baselines3 import PPO
        from modules.rl_environment import OBS_DIM, features_to_obs
        model = PPO.load(str(ROBUST_MODEL_PATH))
        if tuple(model.observation_space.shape) != (OBS_DIM,):
            log.error("robustes PPO: observation_space passt nicht -> keine Shadow-Aktionen")
            return {}
        out = {}
        for s in candidates:
            dte = int((s.get("option") or {}).get("dte", 0) or (s.get("roi_analysis") or {}).get("dte", 90))
            obs = features_to_obs(features=s.get("features", {}), simulation=s.get("simulation", {}),
                                  deep_analysis=s.get("deep_analysis", {}), dte=dte)
            a, _ = model.predict(obs, deterministic=True)
            out[s.get("ticker")] = ACTION_NAMES.get(int(a), str(int(a)))
        return out
    except Exception as e:  # noqa: BLE001 - Shadow darf die Pipeline nie stören
        log.debug(f"robustes PPO Shadow-Inferenz fehlgeschlagen: {e}")
        return {}
