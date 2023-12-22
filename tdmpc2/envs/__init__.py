from copy import deepcopy
import warnings

import gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper
from envs.dmcontrol import make_env as make_dm_control_env
from envs.maniskill import make_env as make_maniskill_env
from envs.metaworld import make_env as make_metaworld_env
from envs.myosuite import make_env as make_myosuite_env
from envs.exceptions import UnknownTaskError

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise UnknownTaskError(task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env
	

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	if cfg.multitask:
		env = make_multitask_env(cfg)
	else:
		env = None
		for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env]:
			try:
				env = fn(cfg)
			except UnknownTaskError:
				pass
		if env is None:
			raise UnknownTaskError(cfg.task)
		env = TensorWrapper(env)
	if cfg.get('obs', 'state') == 'rgb':
		env = PixelWrapper(cfg, env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
