"""
Device selection of the XPU, CUDA and CPU backends.

Nothing here needs an accelerator, so CUDA-only and CPU-only CI hosts run it too:
availability is faked, and the agent, the scale estimator and the replay buffer
are built for real on the CPU to show that no `cuda:0` literal is left in them.

Run from the repository root with `python -m pytest tests` or
`python -m unittest discover tests`.
"""
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

import helpers
from common import device as dev


def fake_availability(*, xpu: bool, cuda: bool):
	"""Make `is_available` report exactly the given backends."""

	def fake(device_type: str) -> bool:
		return {'xpu': xpu, 'cuda': cuda}.get(device_type, device_type == 'cpu')

	return mock.patch.object(dev, 'is_available', side_effect=fake)


class DetectionTest(unittest.TestCase):
	def test_xpu_preferred_over_cuda(self):
		with fake_availability(xpu=True, cuda=True):
			self.assertEqual(dev.default_device_type(), 'xpu')
			self.assertEqual(dev.resolve_device(None), torch.device('xpu', 0))

	def test_cuda_when_no_xpu(self):
		with fake_availability(xpu=False, cuda=True):
			self.assertEqual(dev.resolve_device(None), torch.device('cuda', 0))

	def test_cpu_when_no_accelerator(self):
		with fake_availability(xpu=False, cuda=False):
			self.assertEqual(dev.resolve_device(None), torch.device('cpu'))

	def test_explicit_device_wins_over_detection(self):
		with fake_availability(xpu=True, cuda=True):
			self.assertEqual(dev.resolve_device('cpu'), torch.device('cpu'))
			self.assertEqual(dev.resolve_device('cuda:1'), torch.device('cuda:1'))
			self.assertEqual(dev.resolve_device(torch.device('xpu:1')), torch.device('xpu:1'))

	def test_device_type_of_strings_devices_and_tensors(self):
		self.assertEqual(dev.device_type('xpu:0'), 'xpu')
		self.assertEqual(dev.device_type(torch.device('cuda')), 'cuda')
		self.assertEqual(dev.device_type(torch.zeros(1)), 'cpu')

	def test_scripts_need_an_accelerator_unless_cpu_is_requested(self):
		with fake_availability(xpu=False, cuda=False):
			self.assertFalse(dev.is_usable(None))
			self.assertTrue(dev.is_usable('cpu'))
			self.assertFalse(dev.is_usable('xpu'))
		with fake_availability(xpu=True, cuda=False):
			self.assertTrue(dev.is_usable(None))
			self.assertTrue(dev.is_usable('xpu:0'))
			self.assertFalse(dev.is_usable('cuda'))

	def test_config_default_is_detection(self):
		cfg = OmegaConf.load(helpers.SRC / 'config.yaml')
		self.assertIn('device', cfg)
		self.assertIsNone(cfg.device)


class DispatchTest(unittest.TestCase):
	def test_mem_get_info_follows_the_device(self):
		with fake_availability(xpu=True, cuda=True):
			with mock.patch.object(torch.xpu, 'mem_get_info', return_value=(3, 4)) as xpu_info, \
					mock.patch.object(torch.cuda, 'mem_get_info', return_value=(1, 2)) as cuda_info:
				self.assertEqual(dev.mem_get_info('xpu:0'), (3, 4))
				self.assertEqual(dev.mem_get_info(torch.device('cuda', 0)), (1, 2))
				self.assertEqual(dev.mem_get_info('cpu'), (0, 0))
		xpu_info.assert_called_once()
		cuda_info.assert_called_once()

	def test_mem_get_info_never_touches_a_missing_backend(self):
		with fake_availability(xpu=False, cuda=False):
			with mock.patch.object(torch.cuda, 'mem_get_info') as cuda_info:
				self.assertEqual(dev.mem_get_info('cuda:0'), (0, 0))
		cuda_info.assert_not_called()

	def test_seed_reaches_every_available_accelerator(self):
		with fake_availability(xpu=True, cuda=True):
			with mock.patch.object(torch.xpu, 'manual_seed_all') as xpu_seed, \
					mock.patch.object(torch.cuda, 'manual_seed_all') as cuda_seed:
				dev.manual_seed_all(7)
		xpu_seed.assert_called_once_with(7)
		cuda_seed.assert_called_once_with(7)

	def test_seed_skips_missing_backends(self):
		with fake_availability(xpu=False, cuda=False):
			with mock.patch.object(torch.xpu, 'manual_seed_all') as xpu_seed, \
					mock.patch.object(torch.cuda, 'manual_seed_all') as cuda_seed:
				dev.manual_seed_all(7)
		xpu_seed.assert_not_called()
		cuda_seed.assert_not_called()

	def test_compile_mode_keeps_cuda_graphs_on_cuda_only(self):
		self.assertEqual(dev.compile_mode('cuda:0'), 'reduce-overhead')
		self.assertIsNone(dev.compile_mode('xpu:0'))
		self.assertIsNone(dev.compile_mode('cpu'))

	def test_compile_fallback_is_logged_once(self):
		dev._warned.discard('compile_mode')
		with self.assertLogs(dev.logger, level='WARNING') as logs:
			dev.compile_mode('xpu:0')
			dev.compile_mode('xpu:0')
		self.assertEqual(len(logs.records), 1)
		self.assertIn('reduce-overhead', logs.records[0].getMessage())

	def test_step_marker_is_cuda_only(self):
		with mock.patch.object(torch.compiler, 'cudagraph_mark_step_begin') as marker:
			dev.mark_step_begin('xpu:0')
			dev.mark_step_begin('cpu')
			marker.assert_not_called()
			dev.mark_step_begin('cuda:0')
			marker.assert_called_once()


class CpuAgentTest(unittest.TestCase):
	"""The agent used to hard-code cuda:0 in four places; it now runs where it is told."""

	def test_agent_state_lives_on_the_requested_device(self):
		agent = helpers.build_agent(helpers.make_cfg(device='cpu'))
		self.assertEqual(agent.device, torch.device('cpu'))
		self.assertEqual(agent.scale.value.device.type, 'cpu')
		self.assertEqual(agent._prev_mean.device.type, 'cpu')
		self.assertTrue(all(p.device.type == 'cpu' for p in agent.model.parameters()))

	def test_agent_without_a_device_setting_uses_detection(self):
		cfg = helpers.make_cfg(device='cpu')
		with fake_availability(xpu=False, cuda=False):
			object.__setattr__(cfg, 'device', None)
			agent = helpers.build_agent(cfg)
		self.assertEqual(agent.device, torch.device('cpu'))

	def test_act_and_update_run_on_cpu(self):
		cfg = helpers.make_cfg(device='cpu')
		agent = helpers.build_agent(cfg)
		action = agent.act(torch.randn(24), t0=True)
		self.assertEqual(action.shape, (6,))
		self.assertTrue(torch.isfinite(action).all())
		H, B = cfg.horizon, cfg.batch_size
		info = agent._update(torch.randn(H + 1, B, 24), torch.rand(H, B, 6) * 2 - 1,
							 torch.randn(H, B, 1), torch.zeros(H, B, 1))
		self.assertTrue(all(torch.isfinite(v) for v in info.values()))

	def test_compile_uses_the_default_mode_off_cuda(self):
		cfg = helpers.make_cfg(device='cpu', compile=True)
		with mock.patch.object(torch, 'compile', side_effect=lambda fn, **kw: fn) as compile_:
			agent = helpers.build_agent(cfg)
			_ = agent.plan
		self.assertEqual(compile_.call_count, 2)
		for call in compile_.call_args_list:
			self.assertIsNone(call.kwargs['mode'])


class BufferTest(unittest.TestCase):
	"""Storage placement used `torch.cuda.mem_get_info()` and the literal 'cuda:0'."""

	def storage_device(self, device, free_bytes):
		from common.buffer import Buffer
		cfg = helpers.make_cfg(device=device, buffer_size=1000, steps=1000)
		buffer = Buffer(cfg)
		episode = TensorDict(obs=torch.randn(51, 24), action=torch.randn(51, 6),
							 reward=torch.randn(51), terminated=torch.zeros(51), batch_size=(51,))
		with mock.patch.object(dev, 'mem_get_info', return_value=(free_bytes, free_bytes)) as info, \
				mock.patch.object(Buffer, '_reserve_buffer'):
			buffer._init(episode)
		info.assert_called_once_with(torch.device(device))
		return buffer._storage_device

	def test_storage_uses_accelerator_memory_when_it_fits(self):
		self.assertEqual(self.storage_device('xpu:0', 10 ** 12), torch.device('xpu:0'))
		self.assertEqual(self.storage_device('cuda:0', 10 ** 12), torch.device('cuda:0'))

	def test_storage_falls_back_to_cpu_memory(self):
		self.assertEqual(self.storage_device('xpu:0', 0), torch.device('cpu'))
		self.assertEqual(self.storage_device('cuda:0', 0), torch.device('cpu'))

	def test_sampled_batches_land_on_the_device(self):
		from common.buffer import Buffer
		cfg = helpers.make_cfg(device='cpu', buffer_size=1000, steps=1000)
		buffer = Buffer(cfg)
		for _ in range(3):
			buffer.add(TensorDict(obs=torch.randn(51, 24), action=torch.randn(51, 6),
								  reward=torch.randn(51), terminated=torch.zeros(51), batch_size=(51,)))
		obs, action, reward, terminated, task = buffer.sample()
		self.assertEqual(obs.shape, (cfg.horizon + 1, cfg.batch_size, 24))
		self.assertEqual(obs.device.type, 'cpu')
		self.assertIsNone(task)


if __name__ == '__main__':
	unittest.main()
