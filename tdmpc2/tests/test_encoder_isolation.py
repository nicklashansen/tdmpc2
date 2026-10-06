from types import SimpleNamespace

import pytest
import torch
from torch import nn

from common import layers
from common.world_model import WorldModel


def config(observation):
    shape = (4,) if observation == "state" else (3, 64, 64)
    return SimpleNamespace(
        obs=observation, obs_shape={observation: shape}, latent_dim=16,
        enc_dim=16, mlp_dim=16, num_enc_layers=2, simnorm_dim=8,
        num_channels=1, action_dim=3, task_dim=0, multitask=False,
        episodic=False, num_bins=11, num_q=2, dropout=0.0,
        log_std_min=-5.0, log_std_max=-1.0,
    )


@pytest.mark.parametrize("first", ["state", "rgb"])
@pytest.mark.parametrize("second", ["state", "rgb"])
def test_world_models_only_register_their_configured_observation_encoders(first, second):
    first_model = WorldModel(config(first))
    second_model = WorldModel(config(second))
    assert set(first_model._encoder) == {first}
    assert set(second_model._encoder) == {second}
    first_parameters = {id(p) for p in first_model._encoder.parameters()}
    second_parameters = {id(p) for p in second_model._encoder.parameters()}
    assert first_parameters.isdisjoint(second_parameters)


def test_creating_rgb_world_model_does_not_reinitialize_existing_state_model():
    torch.manual_seed(71)
    model = WorldModel(config("state"))
    observation = torch.tensor([[0.2, -0.1, 0.5, 1.0]])
    parameters = {name: value.detach().clone()
                  for name, value in model._encoder.named_parameters()}
    prediction = model.encode(observation, None).detach().clone()
    torch.manual_seed(72)
    WorldModel(config("rgb"))
    for name, value in model._encoder.named_parameters():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
    torch.testing.assert_close(model.encode(observation, None), prediction, rtol=0, atol=0)


@pytest.mark.parametrize("first,second", [("state", "rgb"), ("rgb", "state")])
def test_moving_new_world_model_preserves_existing_model_dtype(first, second):
    original = WorldModel(config(first))
    parameters = dict(original._encoder.named_parameters())
    newer = WorldModel(config(second)).double()
    assert all(value.dtype == torch.float64 for value in newer._encoder.parameters())
    assert all(value.dtype == torch.float32 for value in parameters.values())


def test_explicit_encoder_mapping_is_still_extended():
    retained = nn.Identity()
    mapping = {"retained": retained}
    result = layers.enc(config("state"), out=mapping)
    assert mapping["retained"] is retained
    assert result["retained"] is retained
    assert mapping["state"] is result["state"]
