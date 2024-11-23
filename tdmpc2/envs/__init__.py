from copy import deepcopy
import warnings

import gym

from envs.wrappers.discrete import DiscreteWrapper
from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies
try:
	from envs.maniskill import make_env as make_maniskill_env
except:
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except:
	make_metaworld_env = missing_dependencies
try:
	from envs.myosuite import make_env as make_myosuite_env
except:
	make_myosuite_env = missing_dependencies


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
			raise ValueError('Unknown task:', task)
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
		if cfg.task.startswith('discrete-'):
			discrete = True
			cfg.task = cfg.task.replace('discrete-', '')
		else:
			discrete = False
		for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env]:
			try:
				env = fn(cfg)
				break
			except ValueError:
				pass
		if env is None:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
		env = TensorWrapper(env)
		if discrete:
			env = DiscreteWrapper(env)
	if cfg.get('obs', 'state') == 'rgb':
		env = PixelWrapper(cfg, env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	assert not isinstance(env.action_space, (gym.spaces.Dict, gym.spaces.MultiDiscrete)), \
		'Dict and MultiDiscrete action spaces are not supported.'
	cfg.action_space = 'discrete' if isinstance(env.action_space, gym.spaces.Discrete) else 'continuous'
	cfg.action_dim = env.action_space.n if cfg.action_space == 'discrete' else env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
