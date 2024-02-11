from copy import deepcopy

from gym.vector import AsyncVectorEnv
import numpy as np
import torch


class Vectorized():
	"""
	Vectorized environment for TD-MPC2 online training.
	"""

	def __init__(self, cfg, env_fn):
		super().__init__()
		self.cfg = cfg

		def make():
			_cfg = deepcopy(cfg)
			_cfg.num_envs = 1
			_cfg.seed = cfg.seed + np.random.randint(1000)
			return env_fn(_cfg)

		print(f'Creating {cfg.num_envs} environments...')
		self.env = AsyncVectorEnv([make for _ in range(cfg.num_envs)])
		env = make()
		self.observation_space = env.observation_space
		self.action_space = env.action_space
		self.max_episode_steps = env.max_episode_steps

	def rand_act(self):
		return torch.rand((self.cfg.num_envs, *self.action_space.shape)) * 2 - 1

	def reset(self):
		return self.env.reset()

	def step(self, action):
		return self.env.step(action)

	def render(self, *args, **kwargs):
		return self.env.render(*args, **kwargs)
