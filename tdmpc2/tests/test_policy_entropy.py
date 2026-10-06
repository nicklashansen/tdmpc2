import math
from types import SimpleNamespace

import pytest
import torch
from torch.distributions import Normal, TanhTransform

from common.world_model import WorldModel
from tdmpc2 import TDMPC2


def make_policy(action_dim, multitask, dtype):
    torch.manual_seed(19)
    cfg = SimpleNamespace(
        obs="state", obs_shape={"state": (4,)}, latent_dim=8, enc_dim=16,
        mlp_dim=16, num_enc_layers=2, simnorm_dim=8, action_dim=action_dim,
        task_dim=2 if multitask else 0, tasks=["one", "three"],
        action_dims=[1, 3], multitask=multitask, episodic=False,
        num_bins=11, num_q=2, dropout=0.0, log_std_min=-5.0, log_std_max=-1.0,
    )
    # Construct the complete production WorldModel, including its real
    # TensorDict ensemble and detached/target Q modules.
    model = WorldModel(cfg).to(dtype=dtype)
    assert TDMPC2.__module__ == "tdmpc2"
    with torch.no_grad():
        model._pi[-1].bias[:action_dim].copy_(torch.linspace(0.25, 0.75, action_dim, dtype=dtype))
    return model


def sample_policy(model, time_batch=False):
    torch.manual_seed(27)
    shape = (2, 3, model.cfg.latent_dim) if time_batch else (3, model.cfg.latent_dim)
    z = torch.randn(shape, dtype=model.log_std_min.dtype)
    task = torch.tensor([0, 1, 0]) if model.cfg.multitask else None
    action, info = model.pi(z, task)
    return action, info, task


def distribution_reference(action, info, model, task):
    mean = torch.atanh(info["mean"])
    std = info["log_std"].exp()
    # Hold the realized base noise fixed, as required by reparameterization.
    noise = (torch.atanh(action.detach()) - mean.detach()) / std.detach()
    raw_action = mean + std * noise
    gaussian_log_prob = Normal(mean, std).log_prob(raw_action).sum(-1, keepdim=True)
    jacobian = TanhTransform().log_abs_det_jacobian(raw_action, action).sum(-1, keepdim=True)
    dim = action.shape[-1]
    size = (torch.tensor(model.cfg.action_dims, dtype=action.dtype)[task].unsqueeze(-1)
            if model.cfg.multitask else dim)
    # Historical TD-MPC2 scales the Gaussian residual, with a single Gaussian
    # normalization constant, then applies the tanh Jacobian without scaling.
    scaled_gaussian = (gaussian_log_prob + (dim - 1) * math.log(2 * math.pi) / 2) * size
    return -scaled_gaussian + jacobian, -gaussian_log_prob + jacobian


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("action_dim,multitask", [(1, False), (3, False), (6, False), (3, True)])
@pytest.mark.parametrize("time_batch", [False, True])
def test_scaled_entropy_matches_original_squashed_policy_objective(dtype, action_dim, multitask, time_batch):
    model = make_policy(action_dim, multitask, dtype)
    action, info, task = sample_policy(model, time_batch)
    expected, _ = distribution_reference(action, info, model, task)
    # The production tanh correction adds 1e-6 inside its logarithm; the
    # independent distribution transform uses the exact Jacobian.
    torch.testing.assert_close(info["scaled_entropy"], expected, rtol=2e-5, atol=6e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("action_dim,multitask", [(1, False), (3, False), (6, False), (3, True)])
def test_scaled_entropy_retains_policy_mean_and_variance_gradients(dtype, action_dim, multitask):
    model = make_policy(action_dim, multitask, dtype)
    action, info, task = sample_policy(model, time_batch=True)
    expected, _ = distribution_reference(action, info, model, task)
    params = tuple(model._pi.parameters())
    actual_grads = torch.autograd.grad(info["scaled_entropy"].sum(), params, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), params)
    assert expected_grads[-1][:action_dim].abs().max() > 0.1
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("action_dim,multitask", [(1, False), (3, False), (6, False), (3, True)])
def test_unscaled_entropy_and_action_mask_remain_unchanged(dtype, action_dim, multitask):
    model = make_policy(action_dim, multitask, dtype)
    action, info, task = sample_policy(model, time_batch=True)
    _, expected = distribution_reference(action, info, model, task)
    torch.testing.assert_close(info["entropy"], expected, rtol=2e-5, atol=6e-5)
    assert (action.abs() < 1).all()
    if multitask:
        assert torch.count_nonzero(action[:, task == 0, 1:]) == 0
