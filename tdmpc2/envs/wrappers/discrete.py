import gym
import numpy as np
import torch

from common import math


class DiscreteWrapper(gym.Wrapper):
	"""
	Wrapper for converting continuous action spaces to discrete via binning.
	"""

	def __init__(self, env):
		super().__init__(env)
		self.continuous_dims = self.env.action_space.shape[0]
		# Bins at [-1, 0, 1] for each dimension
		# Discrete actions include all possible combinations of these bins
		self.action_space = gym.spaces.Discrete(3 ** self.continuous_dims)
	
	def rand_act(self):
		action = torch.tensor(self.action_space.sample(), dtype=torch.int64)
		return math.int_to_one_hot(action, self.action_space.n)
	
	def _discrete_to_continuous(self, action):
		# Convert a discrete action to a continuous action
		# action is a one-hot encoded tensor
		action = torch.argmax(action)
		action = action.item()
		action = [action // 3 ** i % 3 for i in range(self.continuous_dims)]
		action = torch.tensor(action, dtype=torch.float32)
		return (action - 1) / 1

	def step(self, action):
		action = self._discrete_to_continuous(action)
		return self.env.step(action)
