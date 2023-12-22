from pathlib import Path
import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.envs import RandomCropTensorDict, Transform, Compose

from common.logger import make_dir
from common.samplers import SliceSampler


class Buffer():
	"""
	Create a replay buffer for TD-MPC2 training.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda')
		self._batch_size = self.cfg.batch_size * (self.cfg.horizon+1)
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._num_steps = 0
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity
	
	@property
	def num_steps(self):
		"""Return the number of steps in the buffer."""
		return self._num_steps

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=SliceSampler(
				slice_len=self.cfg.horizon+1,
				end_key='done',
				truncated_key=None,
			),
			pin_memory=True,
			prefetch=2,
			batch_size=self.cfg.batch_size,
		)

	def _init(self, td):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([x.numel()*x.element_size() for x in td[0].values()])
		print(f'Bytes per step: {bytes_per_step:,}')
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		if 2.5*total_bytes > mem_free: # Insufficient CUDA memory
			print('Using CPU memory for storage.')
			return self._reserve_buffer(
				LazyTensorStorage(self._capacity, device=torch.device('cpu'))
			)
		else: # Sufficient CUDA memory
			print('Using CUDA memory for storage.')
			return self._reserve_buffer(
				LazyTensorStorage(self._capacity, device=torch.device('cuda'))
			)

	def add(self, td):
		"""Add a step to the buffer."""
		done = bool(td['done'].any())
		if done:
			self._num_eps +=1
		td['episode'] = torch.ones_like(td['done']) * self._num_eps
		td['step'] = torch.arange(0, len(td))
		if self._num_steps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_steps += 1
		return self._num_steps

	def sample(self):
		"""Sample a batch of sub-trajectories from the buffer."""
		td = self._buffer.sample(batch_size=self._batch_size) \
			.reshape(-1, self.cfg.horizon+1).permute(1, 0)
		obs = td['obs'].to(self._device, non_blocking=True)
		action = td['action'][1:].to(self._device, non_blocking=True)
		reward = td['reward'][1:].unsqueeze(-1).to(self._device, non_blocking=True)
		task = td['task'][0].to(self._device, non_blocking=True) if 'task' in td.keys() else None
		return obs, action, reward, task
		
	def save(self):
		"""Save the buffer to disk. Useful for storing offline datasets."""
		td = self._buffer._storage._storage.cpu()
		fp = make_dir(Path(self.cfg.buffer_dir) / self.cfg.task / str(self.cfg.seed)) / f'{self._num_eps}.pt'
		torch.save(td, fp)
