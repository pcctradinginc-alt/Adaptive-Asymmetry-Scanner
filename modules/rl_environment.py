"""
modules/rl_environment.py – Gymnasium-Environment für Options-Scoring v9.0

Änderungen v9.0:
    #11 Observation Space erweitert von 9 auf 11 Dimensionen:
        - Dim 9:  dte_normalized (DTE/365) — Agent lernt kurze Laufzeit = höheres Risiko
        - Dim 10: catalyst_confidence (0-10 → /10) — separater Catalyst-Score aus v9.0

        Reward-Shaping: Kurz-DTE-Verluste werden stärker bestraft, da der Agent
        diese hätte meiden sollen. Positiv-Reward bei Short-DTE-Gewinn unveränder.

        Cold-Start-Verbesserung: Minimum von 5 → 3 abgeschlossene Trades.
        Unter 3: RL-Env wird nicht erstellt, PPO-Agent greift auf Heuristik zurück.

Ersetzt QuasiML (Stufe 6). Der RL-Agent lernt aus closed_trades in
history.json welche Feature-Kombinationen zu positivem Options-P&L führen.

Observation Space (11 Dimensionen, alle normalisiert auf [0, 1]):
    0: impact              (0–10 → /10)
    1: surprise            (0–10 → /10)
    2: mismatch            (0–20 → /20, geclippt)
    3: z_score             (0–5 → /5, geclippt)
    4: eps_drift           (-0.5–0.5 → +0.5 /1.0)
    5: hit_rate            (0–1, aus MiroFish)
    6: iv_rank_norm        (0–100 → /100)
    7: sentiment_score     (-1–1 → +1 /2)
    8: bear_severity       (0–10 → /10, invertiert: 1 - x)
    9: dte_normalized      (0–365 → /365, kurz=riskant)  [NEU v9.0]
    10: catalyst_confidence (0–10 → /10)                  [NEU v9.0]

Action Space (Diskret, 3 Aktionen):
    0: SKIP   → Trade nicht empfehlen (final_score = 0)
    1: NORMAL → Trade empfehlen      (final_score = raw_score)
    2: BOOST  → Stark empfehlen      (final_score = raw_score × 1.5)

Reward:
    Direkt der realisierte Options-P&L des Trade-Vorschlags.
    Bei SKIP: reward = 0.
    Bei SHORT-DTE-Verlust (DTE < 30 bei NORMAL/BOOST): Penalty × 1.3.
    Wird in feedback.py nach Trade-Close eingetragen.
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

log = logging.getLogger(__name__)

OBS_DIM = 11  # v9.0: 9 → 11

ACTION_SKIP   = 0
ACTION_NORMAL = 1
ACTION_BOOST  = 2

SHORT_DTE_THRESHOLD = 30   # DTE < 30 → erhöhter Penalty bei Verlust
SHORT_DTE_PENALTY   = 1.3  # Verlust-Multiplikator für Short-DTE-Entscheidungen


def features_to_obs(
    features:      dict,
    simulation:    dict,
    deep_analysis: dict,
    dte:           int = 90,
) -> np.ndarray:
    """
    Konvertiert einen Pipeline-Signal-Dict in einen normierten Observation-Vektor.

    v9.0: dte und catalyst_confidence als neue Dimensionen (9, 10).
    dte-Parameter: tatsächliche Option-Laufzeit in Tagen.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    obs[0] = np.clip(features.get("impact", 0) / 10.0, 0.0, 1.0)
    obs[1] = np.clip(features.get("surprise", 0) / 10.0, 0.0, 1.0)
    obs[2] = np.clip(features.get("mismatch", 0) / 20.0, 0.0, 1.0)
    obs[3] = np.clip(features.get("z_score", 0) / 5.0, 0.0, 1.0)
    obs[4] = np.clip((features.get("eps_drift", 0) + 0.5) / 1.0, 0.0, 1.0)
    obs[5] = np.clip(simulation.get("hit_rate", 0.7), 0.0, 1.0)

    iv_rank = features.get("iv_rank", 30.0)
    obs[6] = np.clip(iv_rank / 100.0, 0.0, 1.0)

    sentiment = features.get("sentiment_score", 0.0)
    obs[7] = np.clip((sentiment + 1.0) / 2.0, 0.0, 1.0)

    bear_sev = deep_analysis.get("bear_case_severity", 5)
    obs[8] = np.clip(1.0 - (bear_sev / 10.0), 0.0, 1.0)

    # v9.0 dim 9: DTE normalisiert (kurze DTE → niedrigerer Wert → riskanter)
    obs[9] = np.clip(dte / 365.0, 0.0, 1.0)

    # v9.0 dim 10: catalyst_confidence (0-10 → 0-1)
    cat_conf = deep_analysis.get("catalyst_confidence", 5)
    obs[10] = np.clip(float(cat_conf) / 10.0, 0.0, 1.0)

    return obs


class OptionsRLEnv(gym.Env):
    """
    Offline-Trainings-Environment.
    Iteriert über einen Datensatz von (obs, outcome)-Paaren aus history.json.

    v9.0: Erweiterte Observation (11 dims), DTE-aware Reward-Shaping.
    """

    metadata = {"render_modes": []}

    def __init__(self, trade_data: list[dict]):
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
        self.action_space = spaces.Discrete(3)

        self._idx         = 0
        self._current_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        trade   = self.trade_data[self._idx]
        outcome = float(trade.get("outcome", 0.0))

        # DTE aus gespeichertem Trade lesen (falls vorhanden)
        trade_dte = int(
            trade.get("option", {}).get("dte", 0) or
            trade.get("roi_analysis", {}).get("dte", 90)
        )
        is_short_dte = trade_dte > 0 and trade_dte < SHORT_DTE_THRESHOLD

        if action == ACTION_SKIP:
            reward = 0.0
        elif action == ACTION_NORMAL:
            # v9.0: Short-DTE-Verlust stärker bestrafen
            if is_short_dte and outcome < 0:
                reward = outcome * SHORT_DTE_PENALTY
            else:
                reward = outcome
        else:  # BOOST
            if is_short_dte and outcome < 0:
                reward = outcome * SHORT_DTE_PENALTY * 1.5
            else:
                reward = outcome * 1.5

        self._idx += 1
        done      = self._idx >= len(self.trade_data)
        truncated = False

        obs = np.zeros(OBS_DIM, dtype=np.float32) if done else self._get_obs()

        return obs, reward, done, truncated, {}

    def _get_obs(self) -> np.ndarray:
        trade = self.trade_data[self._idx]
        dte   = int(
            trade.get("option", {}).get("dte", 0) or
            trade.get("roi_analysis", {}).get("dte", 90)
        )
        return features_to_obs(
            features      = trade.get("features", {}),
            simulation    = trade.get("simulation", {}),
            deep_analysis = trade.get("deep_analysis", {}),
            dte           = dte,
        )

    def render(self):
        pass


def build_env_from_history(history: dict) -> Optional[OptionsRLEnv]:
    """
    Factory: Erstellt das Environment direkt aus history.json-Dict.
    v9.0: Minimum von 5 → 3 abgeschlossene Trades.
    """
    MIN_TRADES = 3

    closed = history.get("closed_trades", [])
    valid  = [t for t in closed if t.get("outcome") is not None]

    if len(valid) < MIN_TRADES:
        log.info(
            f"Nur {len(valid)} abgeschlossene Trades (Minimum: {MIN_TRADES}) → "
            f"RL-Environment nicht erstellt. Sobald {MIN_TRADES} Trades mit "
            f"'outcome' in history.json eingetragen sind, wird das RL-Modell aktiv."
        )
        return None

    log.info(f"RL-Environment v9.0 mit {len(valid)} Trades erstellt (OBS_DIM={OBS_DIM}).")
    return OptionsRLEnv(trade_data=valid)
