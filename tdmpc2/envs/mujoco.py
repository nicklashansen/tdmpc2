import numpy as np
import gymnasium as gym
from envs.wrappers.timeout import Timeout


MUJOCO_TASKS = {
	'mujoco-walker': 'Walker2d-v4',
	'mujoco-halfcheetah': 'HalfCheetah-v4',
	'lunarlander-continuous': 'LunarLander-v2',
}

class MuJoCoWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self._cumulative_reward = 0

	def reset(self):
		self._cumulative_reward = 0
		return self.env.reset()[0]

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.copy())
		self._cumulative_reward += reward
		done = terminated or truncated
		info['terminated'] = terminated
		if self.cfg.task == 'lunarlander-continuous':
			info['success'] = self._cumulative_reward > 200
		return obs, reward, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def render(self, **kwargs):
		return self.env.render(**kwargs)


def make_env(cfg):
	"""
	Make classic/MuJoCo environment.
	"""
	if not cfg.task in MUJOCO_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	if cfg.task == 'lunarlander-continuous':
		env = gym.make(MUJOCO_TASKS[cfg.task], continuous=True, render_mode='rgb_array')
	else:
		env = gym.make(MUJOCO_TASKS[cfg.task], render_mode='rgb_array')
	env = MuJoCoWrapper(env, cfg)
	env = Timeout(env, max_episode_steps=500 if cfg.task.startswith('lunarlander') else 1000)
	cfg.discount_max = 0.99 # TODO: temporarily hardcode for these envs, makes comparison to other codebases easier
	cfg.rho = 0.7 # TODO: temporarily increase rho for episodic tasks
	return env
