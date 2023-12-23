import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.envs import RandomCropTensorDict

from common.samplers import SliceSampler


class Buffer():
	"""
	Base class for TD-MPC2 replay buffers.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda')
		self._capacity = None
		self._max_eps = None
		self._num_eps = 0
		self._sampler = None
		self._transform = None
		self._batch_size = None

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
			sampler=self._sampler,
			pin_memory=True,
			prefetch=1,
			transform=self._transform,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print('Buffer capacity:', self._capacity)
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_ep = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for k,v in tds.items()
		])		
		print(f'Bytes per episode: {bytes_per_ep:,}')
		total_bytes = bytes_per_ep*self._max_eps
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda' if 2.5*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=torch.device(storage_device))
		)

	def _to_device(self, *args, device=None):
		if device is None:
			device = self._device
		return (arg.to(device, non_blocking=True) \
			if arg is not None else None for arg in args)

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		obs = td['obs']
		action = td['action'][1:]
		reward = td['reward'][1:].unsqueeze(-1)
		task = td['task'][0] if 'task' in td.keys() else None
		return self._to_device(obs, action, reward, task)

	def _add(self, td):
		"""Internal function that adds episode to the buffer."""
		pass

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * self._num_eps
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._add(td)
		self._num_eps += 1
		return self._num_eps

	def sample(self):
		"""Sample a batch of sub-trajectories from the buffer."""
		pass


class CropBuffer(Buffer):
	"""
	A replay buffer that first samples trajectories, and then crops to desired length.
	"""

	def __init__(self, cfg):
		super().__init__(cfg)
		self._capacity = min(cfg.buffer_size, cfg.steps)//cfg.episode_length
		self._max_eps = self._capacity
		self._sampler = RandomSampler()
		self._transform = RandomCropTensorDict(cfg.horizon+1, -1)
		self._batch_size = cfg.batch_size
	
	def _add(self, td):
		"""Add an episode to the buffer, with trajectories as the leading dimension."""
		self._buffer.add(td)

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().permute(1,0)
		return self._prepare_batch(td)
	

class SliceBuffer(Buffer):
	"""
	A replay buffer that directly samples subsequences. More efficient than CropBuffer.
	"""

	def __init__(self, cfg):
		super().__init__(cfg)
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._max_eps = self._capacity//cfg.episode_length
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
	
	def _add(self, td):
		"""Add an episode to the buffer, with transitions as the leading dimension."""
		self._buffer.extend(td)

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)
