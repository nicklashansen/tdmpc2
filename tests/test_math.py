import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'tdmpc2'))

from common import math


class TestRegressionLoss(unittest.TestCase):
	def test_raw_scalar_regression_has_gradient(self):
		pred = torch.tensor([[0.25], [-0.5]], requires_grad=True)
		target = torch.tensor([[2.0], [-3.0]])
		cfg = SimpleNamespace(num_bins=0)

		loss = math.soft_ce(pred, target, cfg)
		torch.testing.assert_close(loss, (pred - target).square())
		loss.sum().backward()
		torch.testing.assert_close(pred.grad, 2 * (pred.detach() - target))

	def test_symlog_scalar_regression_has_gradient(self):
		pred = torch.tensor([[0.25], [-0.5]], requires_grad=True)
		target = torch.tensor([[2.0], [-3.0]])
		cfg = SimpleNamespace(num_bins=1)

		symlog_target = math.symlog(target)
		loss = math.soft_ce(pred, target, cfg)
		torch.testing.assert_close(loss, (pred - symlog_target).square())
		loss.sum().backward()
		torch.testing.assert_close(pred.grad, 2 * (pred.detach() - symlog_target))

	def test_discrete_regression_still_uses_soft_cross_entropy(self):
		pred = torch.tensor([[0.2, 1.1, -0.4]], requires_grad=True)
		target = torch.tensor([[0.0]])
		cfg = SimpleNamespace(num_bins=3, vmin=-1.0, vmax=1.0, bin_size=1.0)

		loss = math.soft_ce(pred, target, cfg)
		expected = torch.nn.functional.cross_entropy(
			pred, torch.tensor([1]), reduction='none'
		).unsqueeze(-1)
		torch.testing.assert_close(loss, expected)


class TestParser(unittest.TestCase):
	def test_scalar_regression_does_not_compute_a_bin_width(self):
		from omegaconf import OmegaConf
		from common.parser import parse_cfg

		for num_bins in (0, 1):
			with self.subTest(num_bins=num_bins):
				cfg = OmegaConf.create({
					'task': 'dog-run',
					'seed': 1,
					'exp_name': 'test',
					'vmin': -10,
					'vmax': 10,
					'num_bins': num_bins,
				})
				with patch('hydra.utils.get_original_cwd', return_value='.'):
					parsed = parse_cfg(cfg)
				self.assertIsNone(parsed.bin_size)

	def test_distributional_regression_keeps_its_bin_width(self):
		from omegaconf import OmegaConf
		from common.parser import parse_cfg

		cfg = OmegaConf.create({
			'task': 'dog-run',
			'seed': 1,
			'exp_name': 'test',
			'vmin': -10,
			'vmax': 10,
			'num_bins': 101,
		})
		with patch('hydra.utils.get_original_cwd', return_value='.'):
			parsed = parse_cfg(cfg)
		self.assertEqual(parsed.bin_size, 0.2)


if __name__ == '__main__':
	unittest.main()
