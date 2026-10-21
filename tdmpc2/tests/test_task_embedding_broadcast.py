from types import SimpleNamespace

import pytest
import torch

from common.world_model import WorldModel


@pytest.fixture(params=[torch.float32, torch.float64])
def model(request):
    torch.manual_seed(31)
    cfg = SimpleNamespace(
        obs="state", obs_shape={"state": (4,)}, latent_dim=8,
        enc_dim=16, mlp_dim=16, num_enc_layers=2, simnorm_dim=8,
        action_dim=3, task_dim=2, tasks=["one", "three"],
        action_dims=[1, 3], multitask=True, episodic=False,
        num_bins=11, num_q=2, dropout=0.0, log_std_min=-5.0, log_std_max=-1.0,
    )
    return WorldModel(cfg).to(dtype=request.param)


def task_id(representation):
    return {"integer": 1, "scalar_tensor": torch.tensor(1),
            "one_element_tensor": torch.tensor([1])}[representation]


@pytest.mark.parametrize("time_batch", [False, True])
@pytest.mark.parametrize("representation", ["integer", "scalar_tensor", "one_element_tensor"])
def test_scalar_tasks_match_per_batch_ids_through_world_model_entries(model, time_batch, representation):
    torch.manual_seed(32)
    leading = (2, 3) if time_batch else (3,)
    dtype = model.log_std_min.dtype
    observation = torch.randn(*leading, 4, dtype=dtype)
    latent = torch.randn(*leading, 8, dtype=dtype)
    action = torch.randn(*leading, 3, dtype=dtype)
    scalar = task_id(representation)
    repeated = torch.full((3,), 1, dtype=torch.long)
    entries = (
        lambda task: model.encode(observation, task),
        lambda task: model.next(latent, action, task),
        lambda task: model.reward(latent, action, task),
        lambda task: model.Q(latent, action, task, return_type="all"),
    )
    for entry in entries:
        torch.testing.assert_close(entry(scalar), entry(repeated))
    torch.manual_seed(33)
    actual_action, actual_info = model.pi(latent, scalar)
    torch.manual_seed(33)
    expected_action, expected_info = model.pi(latent, repeated)
    torch.testing.assert_close(actual_action, expected_action)
    for key in actual_info.keys():
        torch.testing.assert_close(actual_info[key], expected_info[key])


@pytest.mark.parametrize("time_batch", [False, True])
@pytest.mark.parametrize("representation", ["integer", "scalar_tensor", "one_element_tensor"])
def test_broadcast_embedding_gradients_accumulate_over_actual_batch(model, time_batch, representation):
    leading = (2, 3) if time_batch else (3,)
    inputs = torch.zeros(*leading, 4, dtype=model.log_std_min.dtype, requires_grad=True)
    embedded = model.task_emb(inputs, task_id(representation))
    assert embedded.shape == (*leading, 6)
    input_grad, weight_grad = torch.autograd.grad(
        embedded.sum(), (inputs, model._task_emb.weight))
    torch.testing.assert_close(input_grad, torch.ones_like(inputs))
    expected = torch.zeros_like(model._task_emb.weight)
    expected[1] = inputs.numel() / inputs.shape[-1]
    torch.testing.assert_close(weight_grad, expected, rtol=0, atol=0)


@pytest.mark.parametrize("time_batch", [False, True])
def test_mixed_per_batch_tasks_preserve_distinct_embeddings(model, time_batch):
    leading = (2, 3) if time_batch else (3,)
    inputs = torch.zeros(*leading, 4, dtype=model.log_std_min.dtype)
    tasks = torch.tensor([1, 0, 1])
    embedded = model.task_emb(inputs, tasks)
    expected = model._task_emb.weight[tasks]
    if time_batch:
        expected = expected.expand(2, -1, -1)
    torch.testing.assert_close(embedded[..., -2:], expected)
