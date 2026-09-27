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
