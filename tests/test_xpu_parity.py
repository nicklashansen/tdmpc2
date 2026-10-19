"""
CPU versus XPU parity of a tiny random-weight TD-MPC2, built from `config.yaml`.

The model is created on the CPU, its weights are copied to the XPU agent, and both run
the same synthetic observations under the same planning noise (drawn on the CPU and
moved). Skipped when no XPU is present. No checkpoint and no simulator is needed; the
released checkpoints are covered by `check_checkpoint_parity.py`.
"""
import unittest

import torch

import helpers

HAS_XPU = hasattr(torch, 'xpu') and torch.xpu.is_available()
ATOL = 1e-3


def paired_agents(**cfg_kwargs):
	"""A CPU agent and an XPU agent with identical weights."""
	torch.manual_seed(0)
	cpu = helpers.build_agent(helpers.make_cfg(device='cpu', **cfg_kwargs))
	helpers.randomize_zero_init(cpu)
	xpu = helpers.build_agent(helpers.make_cfg(device='xpu', **cfg_kwargs))
	xpu.load(cpu.model.state_dict())
	return cpu, xpu


@unittest.skipUnless(HAS_XPU, 'needs an Intel XPU')
class XpuParityTest(unittest.TestCase):
	def assert_close(self, ref, got):
		for key, row in helpers.compare(ref, got, atol=ATOL, rtol=ATOL).items():
			self.assertEqual(row['inside'], 1.0, f'{key}: {row}')

	def test_agent_state_is_on_the_xpu(self):
		cpu, xpu = paired_agents()
		self.assertEqual(xpu.device.type, 'xpu')
		self.assertEqual(xpu.scale.value.device.type, 'xpu')
		self.assertTrue(all(p.device.type == 'xpu' for p in xpu.model.parameters()))

	def test_world_model_and_planner_match_the_cpu(self):
		cpu, xpu = paired_agents()
		obs = torch.randn(4, 24, generator=torch.Generator().manual_seed(1))
		self.assert_close(helpers.run_agent(cpu, obs), helpers.run_agent(xpu, obs))

	def test_multitask_model_matches_the_cpu(self):
		kwargs = dict(tasks=['a', 'b', 'c'], task_dim=16)
		cpu, xpu = paired_agents(**kwargs)
		obs = torch.randn(4, 24, generator=torch.Generator().manual_seed(2))
		self.assert_close(helpers.run_agent(cpu, obs, task=1), helpers.run_agent(xpu, obs, task=1))

	def test_update_step_matches_the_cpu(self):
		cpu, xpu = paired_agents(dropout=0.0)
		gen = torch.Generator().manual_seed(3)
		H, B = cpu.cfg.horizon, cpu.cfg.batch_size
		batch = (torch.randn(H + 1, B, 24, generator=gen), torch.rand(H, B, 6, generator=gen) * 2 - 1,
				 torch.randn(H, B, 1, generator=gen), torch.zeros(H, B, 1))
		with helpers.FixedNoise(0):
			ref = cpu._update(*batch)
		with helpers.FixedNoise(0):
			got = xpu._update(*[t.to('xpu') for t in batch])
		self.assert_close({k: v.reshape(1).cpu() for k, v in ref.items()},
						  {k: v.reshape(1).cpu() for k, v in got.items()})

	def test_replay_buffer_stores_and_samples_on_the_xpu(self):
		from tensordict.tensordict import TensorDict
		from common.buffer import Buffer
		cfg = helpers.make_cfg(device='xpu', buffer_size=1000, steps=1000)
		buffer = Buffer(cfg)
		for _ in range(3):
			buffer.add(TensorDict(obs=torch.randn(51, 24), action=torch.randn(51, 6),
								  reward=torch.randn(51), terminated=torch.zeros(51), batch_size=(51,)))
		self.assertEqual(buffer._storage_device.type, 'xpu')
		obs, action, reward, terminated, _ = buffer.sample()
		self.assertEqual(obs.device.type, 'xpu')
		self.assertEqual(action.shape, (cfg.horizon, cfg.batch_size, 6))

	def test_compiled_agent_acts_and_trains_on_the_xpu(self):
		# Compiled code draws its own noise, so this checks that it runs and stays finite.
		agent = helpers.build_agent(helpers.make_cfg(device='xpu', compile=True))
		for step in range(2):
			action = agent.act(torch.randn(24), t0=step == 0)
		self.assertTrue(torch.isfinite(action).all())
		self.assertLessEqual(action.abs().max().item(), 1.0)
		H, B = agent.cfg.horizon, agent.cfg.batch_size
		info = agent._update(torch.randn(H + 1, B, 24).to('xpu'), (torch.rand(H, B, 6) * 2 - 1).to('xpu'),
							 torch.randn(H, B, 1).to('xpu'), torch.zeros(H, B, 1).to('xpu'))
		self.assertTrue(all(torch.isfinite(v) for v in info.values()))


if __name__ == '__main__':
	unittest.main()
