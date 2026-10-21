"""Regression coverage: tdmpc gumbel."""

import pytest
import torch

@pytest.mark.parametrize('shape,dim', [((3, 4), 0), ((2, 3, 4), 0), ((2, 3, 4), 1), ((2, 3, 4), -2)])
def test_requested_category_axis(load_case, shape, dim):
    m = load_case('tdmpc_gumbel')
    axis = dim % len(shape)
    result_shape = shape[:axis] + shape[axis + 1:]
    indices = torch.arange(torch.tensor(result_shape).prod().item()).reshape(result_shape) % shape[axis]
    p = torch.zeros(shape)
    p.scatter_(axis, indices.unsqueeze(axis), 1.0)
    actual = m.gumbel_softmax_sample(p, dim=dim)
    assert actual.shape == indices.shape
    assert torch.equal(actual, indices)

@pytest.mark.parametrize('shape,dim', [((5,), 0), ((3, 5), -1), ((2, 3, 5), -1)])
def test_last_axis_and_one_dimensional_inputs_unchanged(load_case, shape, dim):
    m = load_case('tdmpc_gumbel')
    p = torch.arange(1, torch.tensor(shape).prod().item() + 1, dtype=torch.float32).reshape(shape)
    with torch.random.fork_rng():
        torch.manual_seed(123)
        noise = -torch.empty_like(p, memory_format=torch.legacy_contiguous_format).exponential_().log()
        expected = (p.log() + noise).argmax(dim)
        torch.manual_seed(123)
        got = m.gumbel_softmax_sample(p, dim=dim)
    assert torch.equal(got, expected)
import importlib.util
from pathlib import Path

@pytest.fixture
def load_case():
    source = Path(__file__).resolve().parents[1] / 'tdmpc2/common/math.py'
    spec = importlib.util.spec_from_file_location('_regression_target', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda _: module
