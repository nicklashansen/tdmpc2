from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)
		# Define device for tensor conversions
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, **kwargs):
		# Pass kwargs down to the wrapped environment's reset
		return self._obs_to_tensor(self.env.reset(**kwargs))

	def step(self, action):
		# Check if action is a Tensor, convert to numpy if needed
		if isinstance(action, torch.Tensor):
			action_np = action.cpu().numpy()
		elif isinstance(action, np.ndarray):
			action_np = action # Already numpy
		else:
			# Handle potential unexpected types
			try:
				action_np = np.array(action)
			except Exception as e:
				raise TypeError(f"Action type {type(action)} cannot be converted to NumPy array in TensorWrapper. Error: {e}")

		obs, reward, done, info = self.env.step(action_np)
		# Convert obs, reward, done back to tensors
		info = defaultdict(float, info)
		info['success'] = float(info['success'])
		# _obs_to_tensor will handle conversion using self.device
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info
