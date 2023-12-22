from pathlib import Path
import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage

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
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity
	
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
				end_key=None,
				traj_key='episode',
				truncated_key=None,
			),
			pin_memory=True,
			prefetch=2,
			batch_size=self.cfg.batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_ep = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		])
		print(f'Bytes per episode: {bytes_per_ep:,}')
		total_bytes = bytes_per_ep * (self._capacity // self.cfg.episode_length)
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

	def add(self, tds):
		"""Add a step to the buffer."""
		tds['episode'] = torch.ones_like(tds['reward'], dtype=torch.int64) * self._num_eps
		tds['step'] = torch.arange(0, len(tds))
		if self._num_eps == 0:
			self._buffer = self._init(tds)
		self._buffer.extend(tds)
		self._num_eps += 1
		return self._num_eps

	def sample(self):
		"""Sample a batch of sub-trajectories from the buffer."""
		td = self._buffer.sample(batch_size=self._batch_size) \
			.view(-1, self.cfg.horizon+1).permute(1, 0)
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
