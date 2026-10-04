"""PPO-Loader: ein Modell mit anderem Beobachtungsraum darf nie geladen
werden (Audit 2026-09-27: vorher nur String-Guard auf die Config)."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("stable_baselines3")

from modules import rl_agent  # noqa: E402
from modules.rl_environment import OBS_DIM  # noqa: E402


def _fake_model(dim):
    return SimpleNamespace(observation_space=SimpleNamespace(shape=(dim,)))


def test_model_with_wrong_observation_dim_is_rejected():
    with patch("stable_baselines3.PPO.load", return_value=_fake_model(OBS_DIM + 1)):
        scorer = rl_agent.RLScorer({}, veto_enabled=True)
    assert scorer._model is None


def test_model_with_matching_observation_dim_is_loaded():
    with patch("stable_baselines3.PPO.load", return_value=_fake_model(OBS_DIM)):
        scorer = rl_agent.RLScorer({}, veto_enabled=True)
    assert scorer._model is not None


def test_ppo_collapse_detector_warns_on_single_action():
    from modules import engine_monitor as em

    class AlwaysBoost:
        def predict(self, obs, deterministic=True):
            return 2, None

    class Mixed:
        def __init__(self):
            self.i = 0

        def predict(self, obs, deterministic=True):
            self.i += 1
            return self.i % 3, None

    feats = {"impact": 6, "surprise": 5, "mismatch": 3.0, "z_score": 0.5, "eps_drift": 0.0}
    history = {"closed_trades": [{"ticker": f"T{i}", "outcome": (-1) ** i * 0.3, "features": feats,
                                  "option": {"dte": 60}, "outcome_method": "option_quote"} for i in range(10)]}
    w = []
    dist = em._check_ppo_policy_collapse(history, w, model=AlwaysBoost())
    assert dist["max_share"] == 1.0 and w and "BOOST" in w[0]
    w2 = []
    em._check_ppo_policy_collapse(history, w2, model=Mixed())
    assert not w2
