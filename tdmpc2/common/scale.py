import torch
from torch.nn import Buffer

class RunningScale(torch.nn.Module):
	"""Running trimmed scale estimator."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._value = Buffer(torch.ones(1, dtype=torch.float32, device=torch.device('cuda')))
		self._percentiles = Buffer(torch.tensor([5, 95], dtype=torch.float32, device=torch.device('cuda')))

	def state_dict(self):
		return dict(value=self._value, percentiles=self._percentiles)

	def load_state_dict(self, state_dict):
		self._value.copy_(state_dict['value'])
		self._percentiles.copy_(state_dict['percentiles'])

	@property
	def value(self):
		return self._value

	def _positions(self, x_shape):
		positions = self._percentiles * (x_shape-1) / 100
		floored = torch.floor(positions)
		ceiled = floored + 1
		ceiled = torch.where(ceiled > x_shape - 1, x_shape - 1, ceiled)
		weight_ceiled = positions-floored
		weight_floored = 1.0 - weight_ceiled
		return floored.long(), ceiled.long(), weight_floored.unsqueeze(1), weight_ceiled.unsqueeze(1)

	def _percentile(self, x):
		x_dtype, x_shape = x.dtype, x.shape
		x = x.flatten(1, x.ndim-1)
		in_sorted = torch.sort(x, dim=0).values
		floored, ceiled, weight_floored, weight_ceiled = self._positions(x.shape[0])
		d0 = in_sorted[floored] * weight_floored
		d1 = in_sorted[ceiled] * weight_ceiled
		return (d0+d1).reshape(-1, *x_shape[1:]).to(x_dtype)

	def update(self, x):
		percentiles = self._percentile(x.detach())
		value = torch.clamp(percentiles[1] - percentiles[0], min=1.)
		self._value.lerp_(value, self.cfg.tau)

	def forward(self, x, update=False):
		if update:
			self.update(x)
		return x / self.value

	def __repr__(self):
		return f'RunningScale(S: {self.value})'
