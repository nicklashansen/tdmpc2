"""Independent numerical and gradient tests for signed value transforms."""
import math
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('name', ['symlog', 'symexp'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_unit_derivative_at_zero(tdmpc_math, name, dtype):
    x = torch.tensor([-0., 0.], dtype=dtype, requires_grad=True)
    y = getattr(tdmpc_math, name)(x)
    torch.testing.assert_close(y, torch.zeros_like(x), atol=0, rtol=0)
    grad, = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(grad, torch.ones_like(x), atol=0, rtol=0)


@pytest.mark.parametrize('name', ['symlog', 'symexp'])
@pytest.mark.parametrize('dtype,small', [(torch.float32, 1e-8), (torch.float64, 1e-18), (torch.float16, 1e-4), (torch.bfloat16, 1e-4)])
def test_tiny_values_against_scalar_reference(tdmpc_math, name, dtype, small):
    x = torch.tensor([-small, 0., small], dtype=dtype)
    scalar = math.log1p if name == 'symlog' else math.expm1
    reference = torch.tensor([math.copysign(scalar(abs(v)), v) for v in x.tolist()], dtype=dtype)
    result = getattr(tdmpc_math, name)(x)
    assert result.dtype == dtype
    torch.testing.assert_close(result, reference, rtol=0.02, atol=0)


@pytest.mark.parametrize('name', ['symlog', 'symexp'])
def test_gradcheck_including_zero(tdmpc_math, name):
    x = torch.tensor([-0.8, -1e-3, 0., 1e-3, 0.8], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(getattr(tdmpc_math, name), (x,), atol=2e-6, rtol=1e-4)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_inverse_values_and_gradients(tdmpc_math, dtype):
    x = torch.tensor([-1e3, -0.8, -1e-8, 0., 1e-8, 0.8, 1e3], dtype=dtype, requires_grad=True)
    result = tdmpc_math.symexp(tdmpc_math.symlog(x))
    torch.testing.assert_close(result, x, rtol=2e-6, atol=1e-12)
    grad, = torch.autograd.grad(result.sum(), x)
    torch.testing.assert_close(grad, torch.ones_like(x), rtol=2e-6, atol=2e-6)


def test_decoded_zero_value_has_logit_gradient(tdmpc_math):
    cfg = SimpleNamespace(num_bins=3, vmin=-1., vmax=1.)
    logits = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    value = tdmpc_math.two_hot_inv(logits, cfg)
    torch.testing.assert_close(value, torch.zeros(1, 1, dtype=torch.float64), atol=0, rtol=0)
    grad, = torch.autograd.grad(value.sum(), logits)
    # Derivative of softmax-weighted [-1, 0, 1] at zero logits; symexp'(0) = 1.
    expected = torch.tensor([[-1/3, 0., 1/3]], dtype=torch.float64)
    torch.testing.assert_close(grad, expected, atol=1e-14, rtol=1e-14)


def test_single_bin_decoding_at_zero_has_gradient(tdmpc_math):
    x = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    value = tdmpc_math.two_hot_inv(x, SimpleNamespace(num_bins=1))
    grad, = torch.autograd.grad(value.sum(), x)
    torch.testing.assert_close(grad, torch.ones_like(x), atol=0, rtol=0)


def test_toy_value_fit_can_leave_zero(tdmpc_math):
    logits = torch.nn.Parameter(torch.zeros(1, 3, dtype=torch.float64))
    optimizer = torch.optim.SGD([logits], lr=0.1)
    cfg = SimpleNamespace(num_bins=3, vmin=-1., vmax=1.)
    before = (tdmpc_math.two_hot_inv(logits, cfg) - 1).square().item()
    loss = (tdmpc_math.two_hot_inv(logits, cfg) - 1).square().sum()
    loss.backward()
    optimizer.step()
    after = (tdmpc_math.two_hot_inv(logits, cfg) - 1).square().item()
    assert after < before


@pytest.mark.parametrize('name', ['symlog', 'symexp'])
def test_regular_values_oddness_nonmutation_and_layout(tdmpc_math, name):
    x = torch.tensor([[-2., -0.5, 0.3], [0.7, 1.3, 2.5]], dtype=torch.float64).T
    before = x.clone()
    fn = getattr(tdmpc_math, name)
    scalar = math.log1p if name == 'symlog' else math.expm1
    expected = torch.tensor([[math.copysign(scalar(abs(v)), v) for v in row] for row in x.tolist()], dtype=x.dtype)
    torch.testing.assert_close(fn(x), expected)
    torch.testing.assert_close(fn(-x), -fn(x))
    torch.testing.assert_close(x, before, atol=0, rtol=0)
    assert fn(torch.empty(0, dtype=x.dtype)).shape == (0,)


@pytest.fixture(scope="module")
def tdmpc_math():
    import importlib.util
    from pathlib import Path

    root = next(p for p in Path(__file__).resolve().parents if (p / "tdmpc2/common/math.py").is_file())
    path = root / "tdmpc2/common/math.py"
    spec = importlib.util.spec_from_file_location("regression_tdmpc_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
