import gym
import torch

from common import math


class DiscreteWrapper(gym.Wrapper):
	"""
	Wrapper for converting continuous action spaces to discrete via binning.
	"""

	def __init__(self, env, bins_per_dim=5):
		super().__init__(env)
		self.bins_per_dim = bins_per_dim
		self.continuous_dims = self.env.action_space.shape[0]
		# Equally spaced bins along each dimension
		self.action_space = gym.spaces.Discrete(bins_per_dim ** self.continuous_dims)
	
	def rand_act(self):
		action = torch.tensor(self.action_space.sample(), dtype=torch.int64)
		return math.int_to_one_hot(action, self.action_space.n)
	
	def _discrete_to_continuous(self, action):
		# Convert a discrete action to a continuous action
		action = torch.argmax(action)
		action = action.item()
		action = [action // self.bins_per_dim ** i % self.bins_per_dim for i in range(self.continuous_dims)]
		action = torch.tensor(action, dtype=torch.float32)
		return (action - 1) / 1

	def step(self, action):
		action = self._discrete_to_continuous(action)
		return self.env.step(action)
