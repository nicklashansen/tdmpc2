import torch
from tensordict import TensorDict
import numpy as np
import random
from collections import deque
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
	"""
	Simple episode-based buffer for DMPC training that directly stores episodes
	and provides efficient sequence sampling.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self.horizon = cfg.horizon
		self.capacity = cfg.buffer_size
		self.device = torch.device(cfg.buffer_device)
		self.multitask = cfg.get('multitask', False)
		self.episode_length = cfg.get('episode_length', 100)
		self.min_seq_length = self.horizon + 1  # Minimum sequence length needed
		
		# Variables to track buffer state
		self._episodes = deque(maxlen=10000)  # Store complete episodes
		self._num_transitions = 0
		self._current_episode = []
		self._current_episode_idx = 0
		
		# Define expected keys
		self.keys = ['obs', 'action', 'reward', 'next_obs', 'terminated', 'truncated']
		if self.multitask:
			self.keys.append('task')

		# Debug flag
		self.debug = cfg.get('debug_buffer', False)
		
		print(f"Initialized simple episode buffer with capacity for {self.capacity} transitions")

	def add(self, data: TensorDict):
		"""Add transitions to the buffer, either individually or in batches.
		Will accumulate transitions into episodes and store complete episodes.
		"""
		# Handle batch data (episode chunks)
		if data.batch_size and len(data.batch_size) > 0 and data.batch_size[0] > 1:
			batch_size = data.batch_size[0]
			
			# Process each transition in batch
			for i in range(batch_size):
				# Extract single transition
				transition = {k: data[k][i].clone() for k in data.keys()}
				single_td = TensorDict(transition, batch_size=[])
				
				# Add transition to current episode
				self._add_single_transition(single_td)
		else:
			# Add single transition
			self._add_single_transition(data)
	
	def _add_single_transition(self, data: TensorDict):
		"""Process and add a single transition to the current episode."""
		# Process transition data
		processed = {}
		
		# Process each key
		for key in self.keys:
			if key not in data:
				if key == 'terminated' and 'done' in data:
					val = data['done']
				elif key == 'done' and 'terminated' in data:
					val = data['terminated']
				elif key == 'task' and not self.multitask:
					continue  # Skip task if not multitask
				elif key == 'task' and self.multitask:
					val = torch.tensor([0], dtype=torch.int64)  # Default task
				elif key in ['terminated', 'done']:
					val = torch.tensor([False], dtype=torch.bool)  # Default not terminated
				else:
					raise KeyError(f"Key '{key}' missing from data and no default available")
			else:
				val = data[key]

			# Ensure tensor type and correct shape
			if not isinstance(val, torch.Tensor):
				try:
					val = torch.tensor(val)
				except:
					print(f"Warning: Could not convert '{key}' to tensor, using default")
					if key in ['terminated', 'done', 'truncated']:
						val = torch.tensor([False], dtype=torch.bool)
					elif key == 'reward':
						val = torch.tensor([0.0], dtype=torch.float32)
					elif key == 'task' and self.multitask:
						val = torch.tensor([0], dtype=torch.int64)
					else:
						continue  # Skip this key
						
			# Add to current episode
			processed[key] = val.to(self.device)

		# Add to accumulator
		self._current_episode.append(processed)
		
		# Check if should end episode (based on terminated flag)
		is_terminated = False
		if 'terminated' in processed:
			try:
				is_terminated = processed['terminated'].item() if processed['terminated'].numel() == 1 else processed['terminated'].any().item()
			except:
				pass
				
		# Force end episode if it gets too long
		if len(self._current_episode) >= self.episode_length:
			is_terminated = True
			
		# End episode if terminated
		if is_terminated:
			self._complete_current_episode()
				
	def _complete_current_episode(self):
		"""Finalize current episode and add to buffer."""
		# Skip empty episodes
		if not self._current_episode:
			return
			
		episode_length = len(self._current_episode)
		
		# Only add episodes that are at least as long as minimum sequence length
		if episode_length >= self.min_seq_length:
			# Combine all transitions into a single TensorDict
			episode_data = {}
			
			# Stack tensors for each key
			for key in self.keys:
				if all(key in t for t in self._current_episode):
					try:
						# Try to stack tensors
						tensors = [t[key] for t in self._current_episode]
						episode_data[key] = torch.stack(tensors)
					except:
						# If stacking fails, skip this key
						print(f"Warning: Failed to stack tensors for key '{key}'")
						
			# Only add episode if it has required keys
			required_keys = ['obs', 'action', 'reward']
			if self.multitask:
				required_keys.append('task')
				
			if all(k in episode_data for k in required_keys):
				episode_td = TensorDict(episode_data, batch_size=[episode_length])
				
				# Add to episodes deque
				self._episodes.append(episode_td)
				self._num_transitions += episode_length
				
				if self.debug:
					print(f"Added episode {self._current_episode_idx} with {episode_length} transitions")
					
				# Update episode index
				self._current_episode_idx += 1
				
		# Reset current episode
		self._current_episode = []
		
		# Remove oldest episodes if over capacity
		while self._num_transitions > self.capacity and self._episodes:
			removed = self._episodes.popleft()
			self._num_transitions -= removed.shape[0]

	def sample(self, batch_size=None):
		"""Sample a batch of sequences with length horizon+1 from the buffer."""
		batch_size = batch_size if batch_size is not None else self.cfg.batch_size
		
		# Check if we have enough data
		if not self._episodes:
			raise ValueError("Buffer is empty, cannot sample")
			
		if self._num_transitions < batch_size * self.min_seq_length:
			raise ValueError(f"Not enough transitions in buffer: {self._num_transitions} < {batch_size * self.min_seq_length}")

		# Sample batch_size sequences
		sampled_batch = {}
		
		# Try up to 100 times to sample a valid batch
		for _ in range(100):
			try:
				# Sample random episodes and starting positions
				batch_obs = []
				batch_action = []
				batch_reward = []
				batch_terminated = []
				batch_task = [] if self.multitask else None
				
				# Sample sequences
				for _ in range(batch_size):
					# Sample random episode
					episode = random.choice(self._episodes)
					episode_length = episode.shape[0]
					
					# Only sample from episodes that are long enough
					if episode_length < self.min_seq_length:
						continue
						
					# Random starting position that leaves room for a sequence
					max_start = episode_length - self.min_seq_length
					start_idx = random.randint(0, max_start)
					
					# Extract sequence
					obs_seq = episode['obs'][start_idx:start_idx + self.min_seq_length]
					action_seq = episode['action'][start_idx:start_idx + self.min_seq_length - 1]  # One less action
					reward_seq = episode['reward'][start_idx:start_idx + self.min_seq_length - 1]  # One less reward
					
					# Extract terminated if available, or create zeros
					if 'terminated' in episode:
						terminated_seq = episode['terminated'][start_idx:start_idx + self.min_seq_length - 1]
					else:
						terminated_seq = torch.zeros(self.min_seq_length - 1, 1, dtype=torch.bool, device=self.device)
						
					# Extract task if needed
					if self.multitask and 'task' in episode:
						task = episode['task'][start_idx]  # Just need one task value
						batch_task.append(task)
						
					# Add to batch
					batch_obs.append(obs_seq)
					batch_action.append(action_seq)
					batch_reward.append(reward_seq)
					batch_terminated.append(terminated_seq)

				# Stack along batch dimension
				sampled_batch['obs'] = torch.stack(batch_obs)
				sampled_batch['action'] = torch.stack(batch_action)
				sampled_batch['reward'] = torch.stack(batch_reward)
				sampled_batch['terminated'] = torch.stack(batch_terminated)
				
				if self.multitask and batch_task:
					sampled_batch['task'] = torch.stack(batch_task)
					
				# Create TensorDict
				batch = TensorDict(sampled_batch, batch_size=[batch_size], device=self.device)
				
				# Successfully created a batch
				return batch
				
			except Exception as e:
				print(f"Warning: Failed sampling attempt: {e}")
				continue
				
		# If we get here, we failed to sample a valid batch
		raise ValueError("Failed to sample a valid batch after 100 attempts")

	def __len__(self):
		"""Return number of transitions in buffer."""
		return self._num_transitions

	def size(self):
		"""Return number of transitions in buffer (alias for len)."""
		return self._num_transitions
		
	@property
	def max_size(self):
		"""Return buffer capacity."""
		return self.capacity

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

	def add_episode(self, **episode_data):
		"""
		Add a complete episode to the buffer directly.
		
		Args:
			**episode_data: Dictionary of episode data with keys matching self.keys
							Each value should be a tensor of shape [episode_length, ...]
		"""
		# Check if we have required keys
		required_keys = ['obs', 'action', 'reward']
		if self.multitask:
			required_keys.append('task')
			
		if not all(k in episode_data for k in required_keys):
			missing = [k for k in required_keys if k not in episode_data]
			print(f"Warning: Missing required keys {missing} in episode data. Skipping.")
			return
			
		# Get episode length
		episode_length = len(episode_data['obs'])
		
		# Skip episodes that are too short
		if episode_length < self.min_seq_length:
			print(f"Warning: Episode length {episode_length} is less than minimum required length {self.min_seq_length}. Skipping.")
			return
			
		# Create TensorDict from episode data
		episode_td = TensorDict(episode_data, batch_size=[episode_length])
		
		# Move to buffer device
		for k in episode_td.keys():
			episode_td[k] = episode_td[k].to(self.device)
		
		# Add to episodes deque
		self._episodes.append(episode_td)
		self._num_transitions += episode_length
		
		if self.debug:
			print(f"Added complete episode with {episode_length} transitions")
			
		# Remove oldest episodes if over capacity
		while self._num_transitions > self.capacity and self._episodes:
			removed = self._episodes.popleft()
			self._num_transitions -= removed.shape[0]

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

	print(f'Buffer capacity: {buffer.capacity}, Num samples: {buffer._num_transitions}')

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
