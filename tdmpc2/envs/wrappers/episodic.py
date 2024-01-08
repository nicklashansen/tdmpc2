from collections import deque

import gym
import numpy as np
import torch


class EpisodicWrapper(gym.Wrapper):
	"""
	Wrapper for testing episodic tasks. Only compatible with cartpole-balance-sparse at the moment.
	"""

	def __init__(self, cfg, env):
		super().__init__(env)
		assert cfg.task == 'cartpole-balance-sparse'
		self.cfg = cfg
		self.env = env

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		if self.cfg.episodic and reward == 0:
			done = True
			info['terminated'] = True
		return obs, reward, done, info
