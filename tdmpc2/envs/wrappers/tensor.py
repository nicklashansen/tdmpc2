from collections import defaultdict

import gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)
		self._wrapped_vectorized = env.__class__.__name__ == 'Vectorized'
	
	def rand_act(self):
		if self._wrapped_vectorized:
			return self.env.rand_act()
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
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

	def reset(self, task_idx=None, **kwargs):
		if self._wrapped_vectorized:
			obs = self.env.reset(**kwargs)
		else:
			obs = self.env.reset()
		return self._obs_to_tensor(obs)

	def step(self, action, **kwargs):
		if self._wrapped_vectorized:
			obs, reward, done, info = self.env.step(action.numpy(), **kwargs)
		else:
			obs, reward, done, info = self.env.step(action.numpy())
		if isinstance(info, tuple):
			info = {key: torch.stack([torch.tensor(d[key]) for d in info]) for key in info[0].keys()}
			if 'success' not in info.keys():
				info['success'] = torch.zeros(len(done))
		else:
			info = defaultdict(float, info)
			info['success'] = float(info['success'])
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info
