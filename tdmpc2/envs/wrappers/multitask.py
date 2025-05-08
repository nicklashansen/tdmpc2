import gymnasium as gym
import numpy as np
import torch


class MultitaskWrapper(gym.Wrapper):
	"""
	Wrapper for multi-task environments.
	"""

	def __init__(self, cfg, envs):
		super().__init__(envs[0])
		self.cfg = cfg
		self.envs = envs
		self._task_idx = 0
		self._task = cfg.tasks[self._task_idx]
		self.env = self.envs[self._task_idx]

		self._obs_dims = [env.observation_space.shape[0] for env in self.envs]
		self._action_dims = [env.action_space.shape[0] for env in self.envs]
		self._episode_lengths = [getattr(env, 'max_episode_steps', cfg.episode_length) for env in self.envs]
		self._obs_shape = (max(self._obs_dims),)
		self._action_dim = max(self._action_dims)
		
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=np.float32
		)
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(self._action_dim,), dtype=np.float32
		)
	
	@property
	def task(self):
		return self._task
	
	@property
	def task_idx(self):
		return self._task_idx
	
	@property
	def _env(self):
		return self.envs[self.task_idx]

	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _pad_obs(self, obs):
		if isinstance(obs, np.ndarray):
			obs = torch.from_numpy(obs).float()
		
		if obs.shape[0] < self._obs_shape[0]:
			padding_size = self._obs_shape[0] - obs.shape[0]
			padding = torch.zeros(padding_size, dtype=obs.dtype, device=obs.device)
			obs = torch.cat((obs, padding))
		return obs
	
	def reset(self, task=None, task_idx=None, **kwargs):
		if task is not None:
			if task not in self.cfg.tasks:
				raise ValueError(f"Unknown task: {task}. Available tasks: {self.cfg.tasks}")
			self._task_idx = self.cfg.tasks.index(task)
		elif task_idx is not None:
			if task_idx < 0 or task_idx >= len(self.envs):
				raise ValueError(f"Invalid task_idx: {task_idx}. Must be between 0 and {len(self.envs)-1}")
			self._task_idx = task_idx
		else:
			pass
		
		self._task = self.cfg.tasks[self._task_idx]
		self.env = self.envs[self._task_idx]
		print(f"[MultitaskWrapper] Resetting to task: {self._task} (Index: {self._task_idx})")

		reset_obs = self.env.reset(**kwargs)
		return self._pad_obs(reset_obs)

	def step(self, action):
		action_dim = self._action_dims[self.task_idx]
		cropped_action = action[:action_dim]

		obs, reward, done, info = self.env.step(cropped_action)
		return self._pad_obs(obs), reward, done, info
