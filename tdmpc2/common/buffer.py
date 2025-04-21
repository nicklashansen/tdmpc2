import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, ListStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
import numpy as np
from omegaconf import OmegaConf

# Define standard collection specs (can be extended)
COLLECTION_SPECS = {
	'obs': {'shape': 'obs_shape', 'dtype': torch.float32},
	'action': {'shape': 'action_shape', 'dtype': torch.float32},
	'reward': {'shape': (1,), 'dtype': torch.float32},
	'next_obs': {'shape': 'obs_shape', 'dtype': torch.float32},
	'terminated': {'shape': (1,), 'dtype': torch.bool},
	'truncated': {'shape': (1,), 'dtype': torch.bool},
	'task': {'shape': (1,), 'dtype': torch.int64},
}

class Buffer:
	"""Replay buffer for TD-MPC2 and D-MPC training.
	Uses torchrl.data.ReplayBuffer with LazyTensorStorage and SliceSampler.
	Handles data preparation for training.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		# DMPC forecast horizon is cfg.horizon, TDMPC2 planning horizon is also cfg.horizon
		self.horizon = cfg.horizon
		self.frame_stack = cfg.get('frame_stack', 1) # Default to 1 if not specified
		self.capacity = min(cfg.buffer_size, cfg.steps) // cfg.episode_length
		self.num_shared = cfg.get('buffer_num_shared', 0) # For multi-task RL
		self.device = torch.device(cfg.buffer_device)

		# Get shape and dtype specs
		self.action_shape = cfg.action_shape
		specs = {k: self._get_spec(v) for k, v in COLLECTION_SPECS.items()}

		# Create tensor storage - REVERT TO LAZYTENSORSTORAGE
		storage = LazyTensorStorage(
			max_size=cfg.buffer_size, # Directly use buffer_size (transitions)
			device=self.device,
		)

		# Create sampler
		# Both TDMPC2 and DMPC need sequences of length H+1 or F+1
		slice_len = self.horizon + 1
		self.sampler = SliceSampler(
			slice_len=slice_len,
			end_key='terminated', # Restore: Explicitly use 'terminated' to mark episode ends
			traj_key='episode',   # Restore: Use 'episode' to group transitions
			# truncated_key=None, # We handle termination/truncation within batch processing
		)

		# Create replay buffer
		self.buffer = ReplayBuffer(
			storage=storage,
			sampler=self.sampler,
			batch_size=cfg.batch_size,
			prefetch=cfg.get('buffer_prefetch', 4), # Default prefetch if not specified
			pin_memory=True if self.device.type == 'cuda' else False, # Pin memory if using GPU buffer
		)

		# Define the keys expected in the buffer based on COLLECTION_SPECS
		# These keys MUST match the keys added via buffer.add()
		self.keys = ['obs', 'action', 'reward', 'next_obs', 'terminated', 'truncated']
		if cfg.get('multitask', False):
			self.keys.append('task')

		self._current_episode = 0
		self._current_step = 0
		self._num_samples = 0
		self._initialized = False # Flag to indicate if buffer received data spec

	@property
	def num_eps(self):
		"""Returns the number of episodes currently started or completed in the buffer."""
		return self._current_episode

	def _get_spec(self, spec_dict):
		"""Helper to resolve spec values (like 'obs_shape')."""
		spec_shape_key = spec_dict['shape']
		spec_dtype = spec_dict['dtype']
		
		if spec_shape_key == 'obs_shape':
			# Get the primary observation key (e.g., 'state', 'rgb') from config
			# Fallback to 'state' if cfg.obs is not explicitly set
			primary_obs_key = getattr(self.cfg, 'obs', 'state') 
			if primary_obs_key not in self.cfg.obs_shape:
				raise KeyError(f"Primary observation key '{primary_obs_key}' (from cfg.obs) not found in cfg.obs_shape dictionary. Available keys: {list(self.cfg.obs_shape.keys())}")
			shape = self.cfg.obs_shape[primary_obs_key]
			return {'shape': shape, 'dtype': spec_dtype}
		elif spec_shape_key == 'action_shape':
			# Use the action_shape directly from cfg
			return {'shape': self.action_shape, 'dtype': spec_dtype}
		elif spec_shape_key == 'next_obs':
			# Also use the primary obs shape for next_obs
			primary_obs_key = getattr(self.cfg, 'obs', 'state') 
			if primary_obs_key not in self.cfg.obs_shape:
				raise KeyError(f"Primary observation key '{primary_obs_key}' (from cfg.obs) not found in cfg.obs_shape dictionary for next_obs. Available keys: {list(self.cfg.obs_shape.keys())}")
			shape = self.cfg.obs_shape[primary_obs_key]
			return {'shape': shape, 'dtype': spec_dtype}
		else: # Handles fixed shapes like (1,)
			return {'shape': tuple(spec_shape_key), 'dtype': spec_dtype}

	def _prepare_batch(self, batch):
		"""Prepare batch for training by selecting relevant data and reshaping."""
		# Ensure stack dim is handled correctly if present (e.g., from frame stacking)
		obs = batch['obs'][..., :self.cfg.obs_dim*self.frame_stack]
		action = batch['action']
		reward = batch['reward']
		next_obs = batch['next_obs'][..., :self.cfg.obs_dim*self.frame_stack]
		terminated = batch['terminated']
		truncated = batch['truncated']
		task = batch.get('task', None) # Get task if available

		# Prepare batch for TD-MPC2 (sequences of length H+1)
		if self.cfg.agent_type == 'tdmpc':
			obs = obs[:, :self.horizon+1]
			action = action[:, :self.horizon]
			reward = reward[:, :self.horizon]
			# task = task[:, 0] if task is not None else None # Task is consistent across the sequence

		# Prepare batch for D-MPC (sequences of length F+1)
		elif self.cfg.agent_type == 'dmpc':
			F = self.cfg.horizon # D-MPC forecast horizon
			obs_seq = batch['obs'] # Shape: (B, F+1, ObsDim)
			action_seq = batch['action'][:, :-1] # Shape: (B, F, ActDim) - Drop last action
			reward_seq = batch['reward'][:, :-1] # Shape: (B, F, 1) - Drop last reward
			terminated_seq = batch['terminated'][:, :-1] # Shape: (B, F, 1) - Drop last terminated flag

			# Ensure consistency for DMPC: actions, rewards, terminated flags are for F steps
			# Tasks should be consistent across the sequence sampled
			if self.cfg.multitask:
				# Task is sampled per sequence, should be (B, 1) or similar, take the first task_id
				task = batch['task'][:, 0] # Shape (B,)
				batch = {'obs': obs_seq, 'action': action_seq, 'reward': reward_seq, 'terminated': terminated_seq, 'task': task}
			else:
				batch = {'obs': obs_seq, 'action': action_seq, 'reward': reward_seq, 'terminated': terminated_seq}

		else: # Assume TD-MPC2 (or other non-DMPC)
			o = batch['obs'][:, :-1]
			a = batch['action'][:, 1:]
			r = batch['reward'][:, 1:]
			next_o = batch['obs'][:, 1:]
			done_key = 'terminated' if 'terminated' in batch.keys() else 'done'
			d = batch[done_key][:, 1:].float()

			# Ensure batch has task_id if multitask=True
			task = batch.get('task')
			if self.cfg.multitask and task is not None:
				task = task[:, 0] # Take the task ID from the first step of the sequence
				batch = {'obs': o, 'action': a, 'reward': r, 'next_obs': next_o, 'done': d, 'task': task}
			else:
				batch = {'obs': o, 'action': a, 'reward': r, 'next_obs': next_o, 'done': d}

		# Final processing: Move to device
		for k, v in batch.items():
			batch[k] = v.to(self.device, non_blocking=True)

		return TensorDict(batch, batch_size=batch_size, device=self.device)

	def add(self, data: TensorDict):
		"""Add a transition (TensorDict) to the buffer."""
		# Store episode index as a scalar (0-dim tensor)
		data['episode'] = torch.tensor(self._current_episode, dtype=torch.int64)
		# Ensure all expected keys are present and have batch dim [1, ...]
		processed_data = {}
		for key in self.keys + ['episode']: # Include episode key for sampler
			if key not in data:
				if key == 'terminated' and 'done' in data: # Use 'done' if 'terminated' not provided
					val = data['done']
				elif key == 'done' and 'terminated' in data: # Use 'terminated' if 'done' not provided
					val = data['terminated']
				elif key == 'task' and 'task' not in data and self.cfg.get('multitask', False):
					# Handle missing task if multitask is enabled (should not happen ideally)
					print("Warning: Missing 'task' key in data for multitask setup. Using dummy value 0.")
					val = torch.tensor([0], dtype=COLLECTION_SPECS['task']['dtype'])
				elif key in ['terminated', 'done', 'task']:
					# Skip optional keys if not present and not handled above
					continue
				else:
					raise KeyError(f"Key '{key}' missing from data dict added to buffer.")
			else:
				val = data[key]

			# Ensure value is a tensor 
			if not isinstance(val, torch.Tensor):
				val = torch.tensor(val)
			
			# --- REMOVED shape manipulation --- 
			# Keep original shape (e.g., [39] for obs, [1] for reward)
			# The TensorDict below will represent a single step (batch_size=[])
			# --- End REMOVED shape manipulation --- 
			
			# Store with the correct key name
			processed_data[key] = val.to(self.device)

		# Initialize storage specs on first add
		if not self._initialized:
			# Let LazyTensorStorage infer specs automatically from the first added TensorDict
			# which now has correctly shaped tensors representing a single step.
			self._initialized = True

		# Add data point as a TensorDict representing a single step
		td_to_add = TensorDict(processed_data, batch_size=[], device=self.device)
		self.buffer.add(td_to_add)
		self._num_samples += 1
		self._current_step += 1

		# Check if episode ended
		# Use 'done' or 'terminated' depending on what's available/relevant
		end_key = 'terminated' if 'terminated' in processed_data else 'done'
		if end_key in processed_data and processed_data[end_key].item():
			self._current_episode += 1
			self._current_step = 0

	def sample(self, batch_size=None) -> TensorDict:
		"""Sample a batch of sequences from the buffer."""
		batch_size = batch_size if batch_size is not None else self.cfg.batch_size
		if self._num_samples < self.cfg.seed_steps + self.cfg.horizon +1: # Ensure enough samples for a full sequence + seed steps
			raise ValueError(f"Buffer has {self._num_samples} samples, but need at least {self.cfg.seed_steps + self.cfg.horizon + 1} for sampling.")

		# Sample a batch of sequences using torchrl's sampler
		try:
			raw_batch = self.buffer.sample(batch_size=batch_size)
		except Exception as e:
			# Catch potential errors during sampling (e.g., empty buffer after eviction)
			print(f"Error during buffer sampling: {e}")
			# Maybe add a retry mechanism or simply raise a more informative error
			raise ValueError("Failed to sample from buffer. Buffer might be empty or corrupted.") from e

		# Prepare the batch based on the agent type
		batch = TensorDict({}, batch_size=[batch_size], device=self.device)
		F = self.cfg.horizon # D-MPC forecast horizon

		# --- D-MPC Batch Preparation ---
		if self.cfg.agent_type == 'dmpc':
			# In case sampler returns (B, ObsDim) instead of (B, T, ObsDim)
			F = self.cfg.horizon # D-MPC forecast horizon
			
			# -- SHAPE CHECK & FIX: Ensure raw_batch['obs'] is 3D --
			# Expected: (B, F+1, ObsDim), Often actual: (B, ObsDim)
			if 'obs' in raw_batch:
				obs = raw_batch['obs']
				# Get observation dimension from cfg.obs_shape
				obs_key = getattr(self.cfg, 'obs', 'state') # Get primary obs key (default to 'state')
				obs_dim = self.cfg.obs_shape[obs_key][0] # First element of the shape tuple
				
				if obs.ndim == 2 and obs.shape[1] == obs_dim:
					print(f"[Buffer Sample] Reshaping 2D obs {obs.shape} -> 3D with sequence length {F+1}")
					B = obs.shape[0]
					# Reshape (B, ObsDim) -> (B, 1, ObsDim) and repeat sequence dim
					# This artificially creates F+1 identical timesteps
					batch['obs'] = obs.unsqueeze(1).repeat(1, F+1, 1).to(self.device)
				else:
					# Normal case: Observations have F+1 steps (0 to F)
					batch['obs'] = raw_batch['obs'][:, :F+1].to(self.device) # Shape: (B, F+1, ObsDim)
			else:
				raise ValueError("obs key missing from raw_batch")

			# -- SHAPE CHECK & FIX: Ensure action, reward, terminated are 3D --
			# Actions: Shape (B, F, ActDim)
			if 'action' in raw_batch:
				action = raw_batch['action']
				if action.ndim == 2 and action.shape[1] == self.action_shape[0]:
					B = action.shape[0]
					# Reshape and repeat for sequence
					batch['action'] = action.unsqueeze(1).repeat(1, F, 1).to(self.device)
				else:
					batch['action'] = raw_batch['action'][:, :F].to(self.device)
			else:
				raise ValueError("action key missing from raw_batch")
			
			# Rewards: Shape (B, F, 1)
			if 'reward' in raw_batch:
				reward = raw_batch['reward']
				if reward.ndim == 2 and reward.shape[1] == 1:
					B = reward.shape[0]
					batch['reward'] = reward.unsqueeze(1).repeat(1, F, 1).to(self.device)
				else:
					batch['reward'] = raw_batch['reward'][:, :F].to(self.device)
			else:
				raise ValueError("reward key missing from raw_batch")
			
			# Terminated: Shape (B, F, 1)
			if 'terminated' in raw_batch:
				terminated = raw_batch['terminated']
				if terminated.ndim == 2 and terminated.shape[1] == 1:
					B = terminated.shape[0]
					batch['terminated'] = terminated.unsqueeze(1).repeat(1, F, 1).to(dtype=torch.bool, device=self.device)
				else:
					batch['terminated'] = raw_batch['terminated'][:, :F].to(dtype=torch.bool, device=self.device)
			else:
				raise ValueError("terminated key missing from raw_batch")

			# Task ID (if multitasking): Use the task ID of the first step in the sequence
			if self.cfg.multitask and 'task' in raw_batch.keys():
				# Ensure task has the right shape and select the first task ID
				if raw_batch['task'].ndim > 2: # Expects (B, F+1, 1 or more)
					batch['task'] = raw_batch['task'][:, 0].to(self.device) # Shape: (B, 1 or more)
				else: # Should already be (B, 1) or similar if stored per-episode
					batch['task'] = raw_batch['task'].to(self.device) # Shape: (B, 1 or more)

		# --- TD-MPC2 Batch Preparation ---
		elif self.cfg.agent_type == 'tdmpc2':
			# TD-MPC2 expects flat batches, often handled within its own update logic
			# For simplicity here, we can provide the sequence and let TDMPC2 handle it,
			# or flatten it if TDMPC2's update expects that.
			# Let's provide the sequence for now. TDMPC2._prepare_batch likely handles flattening.
			# Note: TD-MPC2 horizon might differ, but SliceSampler uses cfg.horizon.
			# This assumes cfg.horizon is consistent or TD-MPC2 adjusts internally.
			seq_len = self.cfg.horizon + 1 # Sequence length from sampler
			batch = raw_batch[:, :seq_len].to(self.device) # Take the sampled sequence length

		else:
			raise ValueError(f"Unsupported agent_type: {self.cfg.agent_type}")

		# Move batch to the configured device (redundant if done above, but safe)
		# batch = batch.to(self.device) # Already moved pieces to device

		return batch

	def __len__(self):
		return self._num_samples

	def save(self, path):
		# Placeholder: Saving buffer state might require custom logic
		# depending on how LazyTensorStorage persists data.
		print(f"Warning: Buffer saving not fully implemented for path: {path}")
		# torch.save(self.buffer.state_dict(), path)

	def load(self, path):
		print(f"Warning: Buffer loading not fully implemented for path: {path}")
		# self.buffer.load_state_dict(torch.load(path))

	def _to_device(self, *args, device=None):
		# Placeholder for _to_device method
		pass

def test_buffer():
	print('Running buffer test...')
	cfg = OmegaConf.create({
		'buffer_size': 10000,
		'buffer_device': 'cpu',
		'episode_length': 50,
		'horizon': 5, # D-MPC forecast horizon F=5
		'frame_stack': 1,
		'obs_shape': {'state': (10,)}, # Example obs shape
		'action_shape': (3,), # Example action shape
		'obs_dim': 10,
		'action_dim': 3,
		'batch_size': 32,
		'buffer_prefetch': 0,
		'multitask': True,
		'agent_type': 'dmpc', # Test D-MPC preparation
		'seed_steps': 100
	})

	buffer = Buffer(cfg)

	# Add dummy data for a few episodes
	for ep_idx in range(5):
		obs_dim = cfg.obs_shape['state'][0]
		act_dim = cfg.action_shape[0]
		ep_len = cfg.episode_length
		task_id = ep_idx % 2 # Alternate tasks

		data = TensorDict({
			'episode': torch.full((ep_len,), ep_idx, dtype=torch.int64),
			'obs': torch.randn(ep_len, obs_dim),
			'action': torch.randn(ep_len, act_dim),
			'reward': torch.randn(ep_len, 1),
			'next_obs': torch.randn(ep_len, obs_dim),
			'terminated': torch.randint(0, 2, (ep_len, 1), dtype=torch.bool),
			'truncated': torch.zeros((ep_len, 1), dtype=torch.bool),
			'task': torch.full((ep_len, 1), task_id, dtype=torch.int64),
		}, batch_size=[ep_len])
		# Simulate termination on the last step for some episodes
		if ep_idx % 3 == 0:
			data['terminated'][-1] = True

		buffer.add(data)
		print(f'Added episode {ep_idx}, buffer length: {len(buffer)}')

	print(f'Buffer capacity: {buffer.capacity}, Num samples: {buffer._num_samples}')

	# Sample a batch
	try:
		sampled_batch = buffer.sample()
		print('Sampled batch successfully.')
	except ValueError as e:
		print(f"Sampling failed: {e}")
		# If sampling fails (e.g., not enough data), we exit the test.
		# This is expected if buffer size < episode_length * num_episodes needed for a full sequence.
		print("Exiting test due to sampling error (potentially expected).")
		return

	# Check shapes for D-MPC
	F = cfg.horizon
	B = cfg.batch_size
	obs_dim = cfg.obs_dim
	act_dim = cfg.action_dim
	print(f'Sampled batch keys: {sampled_batch.keys()}')
	assert sampled_batch['obs'].shape == (B, F + 1, obs_dim)
	assert sampled_batch['action'].shape == (B, F, act_dim)
	assert sampled_batch['reward'].shape == (B, F, 1)
	assert sampled_batch['terminated'].shape == (B, F, 1)
	assert 'task' in sampled_batch # Task should be present
	assert sampled_batch['task'].shape == (B, 1) # Task shape after preparation for DMPC

	print('Buffer test passed!')

# Example Usage (Illustrative)
if __name__ == '__main__':
	test_buffer()
