"""Protect TD-MPC2's scalar reward/Q regression modes."""

from types import SimpleNamespace
import numpy as np
import pytest
import torch
from common import math


def config(num_bins):
    return SimpleNamespace(
        num_bins=num_bins, vmin=-10.0, vmax=10.0, bin_size=20.0 / max(num_bins - 1, 1)
    )


@pytest.mark.parametrize("num_bins", [0, 1])
def test_scalar_loss_matches_squared_error_and_has_learning_gradient(num_bins):
    targets = torch.tensor([[-3.0], [0.5], [5.0]], dtype=torch.float64)
    predictions = torch.tensor(
        [[0.2], [-0.8], [1.0]], dtype=torch.float64, requires_grad=True
    )
    values = targets.numpy()
    if num_bins == 1:
        values = np.sign(values) * np.log1p(np.abs(values))
    residual = predictions.detach().numpy() - values
    loss = math.soft_ce(predictions, targets, config(num_bins))
    np.testing.assert_allclose(
        loss.detach().numpy(), residual**2, rtol=1e-12, atol=1e-12
    )
    loss.sum().backward()
    np.testing.assert_allclose(
        predictions.grad.numpy(), 2 * residual, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("num_bins", [0, 1])
def test_scalar_reward_head_can_learn_and_roundtrip(num_bins):
    x = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    targets = torch.tensor([[-1.5], [0.2], [1.7]], dtype=torch.float64)
    head = torch.nn.Linear(1, 1, dtype=torch.float64)
    with torch.no_grad():
        head.weight.fill_(0.1)
        head.bias.fill_(0.1)
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    cfg = config(num_bins)
    initial = math.soft_ce(head(x), targets, cfg).mean()
    opt.zero_grad()
    initial.backward()
    opt.step()
    final = math.soft_ce(head(x), targets, cfg).mean()
    assert final.item() < initial.item()
    encoded = math.two_hot(targets, cfg)
    torch.testing.assert_close(math.two_hot_inv(encoded, cfg), targets)


def test_categorical_loss_and_gradient_are_unchanged():
    predictions = torch.tensor(
        [[0.3, -0.4, 1.2], [-0.1, 0.2, 0.5]], dtype=torch.float64, requires_grad=True
    )
    cfg = SimpleNamespace(num_bins=3, vmin=-1.0, vmax=1.0, bin_size=1.0)
    # Targets live on the middle and upper grid atoms after symlog.
    targets = torch.tensor([[0.0], [np.expm1(1.0)]], dtype=torch.float64)
    loss = math.soft_ce(predictions, targets, cfg)
    ref = torch.nn.functional.cross_entropy(
        predictions, torch.tensor([1, 2]), reduction="none"
    ).unsqueeze(-1)
    torch.testing.assert_close(loss, ref)
    actual_grad = torch.autograd.grad(loss.sum(), predictions, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(ref.sum(), predictions)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
