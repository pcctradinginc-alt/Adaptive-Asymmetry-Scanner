"""
modules/rl_environment.py – Gymnasium-Environment für Options-Scoring

Ersetzt QuasiML (Stufe 6). Der RL-Agent lernt aus closed_trades in
history.json welche Feature-Kombinationen zu positivem Options-P&L führen.

Design nach hab5510/finrl_project-Prinzipien, aber minimal und
ohne externe Broker-API:

Observation Space (9 Dimensionen, alle normalisiert auf [0, 1]):
    0: impact          (0–10 → /10)
    1: surprise        (0–10 → /10)
    2: mismatch        (0–20 → /20, geclippt)
    3: z_score         (0–5 → /5, geclippt)
    4: eps_drift       (-0.5–0.5 → +0.5 /1.0)
    5: hit_rate        (0–1, aus MiroFish)
    6: iv_rank_norm    (0–100 → /100)
    7: sentiment_score (-1–1 → +1 /2)
    8: bear_severity   (0–10 → /10, invertiert: 1 - x)

Action Space (Diskret, 3 Aktionen):
    0: SKIP   → Trade nicht empfehlen (final_score = 0)
    1: NORMAL → Trade empfehlen      (final_score = raw_score)
    2: BOOST  → Stark empfehlen      (final_score = raw_score × 1.5)

Reward:
    Direkt der realisierte Options-P&L des Trade-Vorschlags.
    Bei SKIP: reward = 0 (kein Verlust, aber kein Gewinn).
    Wird in feedback.py nach Trade-Close eingetragen.

Training:
    Offline auf closed_trades aus history.json.
    Online: Jeder neue closed_trade triggert einen PPO-Update-Step.
    Kein Echtzeit-Trading, kein Broker nötig.
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

log = logging.getLogger(__name__)

# Observation-Dimensionen (muss mit _build_obs() übereinstimmen)
OBS_DIM = 9

# Aktionen
ACTION_SKIP   = 0
ACTION_NORMAL = 1
ACTION_BOOST  = 2


def features_to_obs(features: dict, simulation: dict, deep_analysis: dict) -> np.ndarray:
    """
    Konvertiert einen Pipeline-Signal-Dict in einen normierten Observation-Vektor.

    Wird sowohl beim Training (aus history.json) als auch
    beim Inference (in pipeline.py Stufe 6) genutzt.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # 0: impact (0–10 → 0–1)
    obs[0] = np.clip(features.get("impact", 0) / 10.0, 0.0, 1.0)

    # 1: surprise (0–10 → 0–1)
    obs[1] = np.clip(features.get("surprise", 0) / 10.0, 0.0, 1.0)

    # 2: mismatch (0–20+ → 0–1, clip bei 20)
    obs[2] = np.clip(features.get("mismatch", 0) / 20.0, 0.0, 1.0)

    # 3: z_score (0–5+ → 0–1, clip bei 5)
    obs[3] = np.clip(features.get("z_score", 0) / 5.0, 0.0, 1.0)

    # 4: eps_drift (-0.5–0.5 → 0–1)
    obs[4] = np.clip((features.get("eps_drift", 0) + 0.5) / 1.0, 0.0, 1.0)

    # 5: hit_rate aus MiroFish (0–1, direkt)
    obs[5] = np.clip(simulation.get("hit_rate", 0.7), 0.0, 1.0)

    # 6: iv_rank (0–100 → 0–1)
    iv_rank = features.get("iv_rank", 30.0)
    obs[6] = np.clip(iv_rank / 100.0, 0.0, 1.0)

    # 7: FinBERT sentiment_score (-1–1 → 0–1)
    sentiment = features.get("sentiment_score", 0.0)
    obs[7] = np.clip((sentiment + 1.0) / 2.0, 0.0, 1.0)

    # 8: bear_severity (0–10 → 0–1, invertiert: niedriger Bear = besser)
    bear_sev = deep_analysis.get("bear_case_severity", 5)
    obs[8] = np.clip(1.0 - (bear_sev / 10.0), 0.0, 1.0)

    return obs


class OptionsRLEnv(gym.Env):
    """
    Offline-Trainings-Environment.
    Iteriert über einen Datensatz von (obs, outcome)-Paaren aus history.json.

    Bei jedem step() bekommt der Agent eine Observation und wählt eine Aktion.
    Der Reward ist der tatsächlich realisierte Options-P&L.

    Ein Episode = ein Durchlauf durch alle closed_trades.
    """

    metadata = {"render_modes": []}

    def __init__(self, trade_data: list[dict]):
        """
        Args:
            trade_data: Liste von closed_trades aus history.json.
                        Jeder Trade muss 'features', 'simulation',
                        'deep_analysis' und 'outcome' haben.
        """
        super().__init__()

        self.trade_data = [t for t in trade_data if t.get("outcome") is not None]
        if not self.trade_data:
            raise ValueError(
                "Keine abgeschlossenen Trades in trade_data. "
                "Mindestens 1 closed_trade mit 'outcome' nötig."
            )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # SKIP, NORMAL, BOOST

        self._idx        = 0
        self._current_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        trade   = self.trade_data[self._idx]
        outcome = float(trade.get("outcome", 0.0))

        # Reward-Funktion
        if action == ACTION_SKIP:
            reward = 0.0          # Neutral: kein Trade, kein P&L
        elif action == ACTION_NORMAL:
            reward = outcome      # Direkt der realisierte P&L
        else:  # BOOST
            # Boost ist nur gut wenn outcome positiv, doppelt schlecht wenn negativ
            reward = outcome * 1.5

        self._idx += 1
        done      = self._idx >= len(self.trade_data)
        truncated = False

        if done:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, reward, done, truncated, {}

    def _get_obs(self) -> np.ndarray:
        trade = self.trade_data[self._idx]
        return features_to_obs(
            features     = trade.get("features", {}),
            simulation   = trade.get("simulation", {}),
            deep_analysis = trade.get("deep_analysis", {}),
        )

    def render(self):
        pass


def build_env_from_history(history: dict) -> Optional[OptionsRLEnv]:
    """
    Factory: Erstellt das Environment direkt aus history.json-Dict.
    Gibt None zurück wenn zu wenig Daten vorhanden.
    """
    closed = history.get("closed_trades", [])
    valid  = [t for t in closed if t.get("outcome") is not None]

    if len(valid) < 5:
        log.info(
            f"Nur {len(valid)} abgeschlossene Trades → "
            f"RL-Environment nicht erstellt (Minimum: 5)."
        )
        return None

    log.info(f"RL-Environment mit {len(valid)} Trades erstellt.")
    return OptionsRLEnv(trade_data=valid)
