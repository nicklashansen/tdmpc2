"""
Shared helpers for the tests: build an agent from `config.yaml` without a simulator.

The simulators (DMControl, Meta-World, ManiSkill, MyoSuite) are never imported.
The values that `make_env` fills in at run time (`obs_shape`, `action_dim`,
`episode_length`, ...) are given directly, and `parse_cfg` is replaced by the
few lines of it that do not need a Hydra run directory.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'tdmpc2'
if str(SRC) not in sys.path:
	sys.path.insert(0, str(SRC))

import torch
from omegaconf import OmegaConf

from common import MODEL_SIZE
from common.parser import cfg_to_dataclass


def make_cfg(obs_dim=24, action_dim=6, episode_length=100, model_size=1, device='cpu',
			 tasks=None, task_dim=0, **overrides):
	"""
	State-observation config for a small model. Single-task by default; pass `tasks`
	(and `task_dim`) for a multi-task one.
	"""
	cfg = OmegaConf.load(SRC / 'config.yaml')
	del cfg['defaults']
	for key, value in MODEL_SIZE[model_size].items():
		cfg[key] = value
	cfg.task = 'test-task'
	cfg.model_size = model_size
	cfg.device = device
	cfg.compile = False
	cfg.multitask = tasks is not None
	cfg.task_dim = task_dim
	cfg.tasks = list(tasks) if tasks is not None else ['test-task']
	if cfg.multitask:
		cfg.action_dims = [action_dim] * len(tasks)
		cfg.episode_lengths = [episode_length] * len(tasks)
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
	cfg.obs_shape = {'state': (obs_dim,)}
	cfg.action_dim = action_dim
	cfg.episode_length = episode_length
	cfg.seed_steps = max(1000, 5 * episode_length)
	cfg.work_dir = str(ROOT / 'logs')
	cfg.horizon = 3
	cfg.num_samples = 64
	cfg.num_elites = 16
	cfg.num_pi_trajs = 8
	cfg.iterations = 2
	cfg.batch_size = 16
	cfg.enable_wandb = False
	for key, value in overrides.items():
		cfg[key] = value
	return cfg_to_dataclass(cfg)


def build_agent(cfg):
	from tdmpc2 import TDMPC2
	return TDMPC2(cfg)


class FixedNoise:
	"""
	Replace every random draw of the planner by a draw from a CPU generator.

	The draws are moved to the requested device, so a CPU run and an accelerator run
	that make the same calls in the same order see the same noise. Covers the sources
	used by the world model and the planner: `randn`, `randn_like`, `randperm` and the
	Gumbel sample that picks the final action.
	"""

	def __init__(self, seed=0):
		self.seed = seed

	def __enter__(self):
		from unittest import mock
		from common import math as tdmath
		gen = torch.Generator('cpu').manual_seed(self.seed)
		real_randn, real_randperm = torch.randn, torch.randperm

		def randn(*size, device=None, dtype=None, **kwargs):
			if len(size) == 1 and isinstance(size[0], (tuple, list)):
				size = tuple(size[0])
			return real_randn(*size, generator=gen, dtype=dtype or torch.float32).to(device)

		def randn_like(t, **kwargs):
			return real_randn(*t.shape, generator=gen, dtype=t.dtype).to(t.device)

		def randperm(n, device=None, **kwargs):
			return real_randperm(n, generator=gen).to(device)

		def gumbel_softmax_sample(p, temperature=1.0, dim=0):
			noise = torch.empty(p.shape, dtype=p.dtype).exponential_(generator=gen).to(p.device)
			return ((p.log() - noise.log()) / temperature).softmax(dim).argmax(-1)

		self._patches = [
			mock.patch.object(torch, 'randn', randn),
			mock.patch.object(torch, 'randn_like', randn_like),
			mock.patch.object(torch, 'randperm', randperm),
			mock.patch.object(tdmath, 'gumbel_softmax_sample', gumbel_softmax_sample),
		]
		for patch in self._patches:
			patch.start()
		return self

	def __exit__(self, *exc):
		for patch in reversed(self._patches):
			patch.stop()


@torch.no_grad()
def run_agent(agent, obs, seed=0, task=None):
	"""
	Run the world-model pieces and one planning call under fixed noise.
	Returns a dict of CPU float32 tensors.
	"""
	from common import math as tdmath
	obs = obs.to(agent.device)
	model = agent.model
	task_t = None if task is None else torch.tensor([task], device=agent.device)
	out = {}
	with FixedNoise(seed):
		z = model.encode(obs, task_t)
		a = torch.tanh(torch.arange(z.shape[0] * agent.cfg.action_dim, dtype=torch.float32)
			.reshape(z.shape[0], -1).sin()).to(agent.device)
		out['encode'] = z
		out['next'] = model.next(z, a, task_t)
		out['reward_logits'] = model.reward(z, a, task_t)
		out['reward'] = tdmath.two_hot_inv(out['reward_logits'], agent.cfg)
		out['Q_all'] = model.Q(z, a, task_t, return_type='all')
		out['Q_avg'] = model.Q(z, a, task_t, return_type='avg')
		pi_action, info = model.pi(z, task_t)
		out['pi_action'], out['pi_mean'] = pi_action, info['mean']
		out['pi_log_std'] = info['log_std']
		agent._prev_mean.zero_()
		out['act_eval'] = agent.act(obs[0].cpu(), t0=True, eval_mode=True, task=task)
		out['act_explore'] = agent.act(obs[0].cpu(), t0=False, eval_mode=False, task=task)
	return {k: v.detach().float().cpu() for k, v in out.items()}


def compare(ref, got, atol=1e-3, rtol=1e-3):
	"""Per-tensor MSE, relative MSE, max abs diff and share of elements inside tolerance."""
	rows = {}
	for key in ref:
		r, g = ref[key].double(), got[key].double()
		diff = g - r
		mse = diff.pow(2).mean().item()
		rows[key] = dict(
			mse=mse,
			rel_mse=mse / max(r.pow(2).mean().item(), 1e-30),
			max_abs=diff.abs().max().item(),
			inside=torch.isclose(g, r, atol=atol, rtol=rtol).double().mean().item(),
		)
	return rows


@torch.no_grad()
def randomize_zero_init(agent, std=0.2, seed=0):
	"""
	A fresh model has a zero reward head and zero last Q layers, so every planning
	value ties at zero. Give them weights, otherwise the comparison cannot tell
	two backends apart.
	"""
	gen = torch.Generator('cpu').manual_seed(seed)
	model = agent.model
	weight = model._reward[-1].weight
	weight.copy_(torch.randn(weight.shape, generator=gen) * std)
	qw = model._Qs.params['2', 'weight']
	qw.copy_(torch.randn(qw.shape, generator=gen) * std)
	model._target_Qs_params['2', 'weight'].copy_(qw)
