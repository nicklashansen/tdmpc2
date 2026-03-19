"""
TurtleBot3 Burger 'Go To' environment for TD-MPC2.

Uses gym.Env as base (standard practice) but step()/reset() return the
TD-MPC2-compatible signatures:
  reset() -> np.ndarray          (not the gymnasium 2-tuple)
  step()  -> (obs, r, done, info) (not the gymnasium 5-tuple)
This matches how DMControlWrapper and MuJoCoWrapper already behave.
"""
import os
from collections import defaultdict

import numpy as np
import mujoco
import gymnasium as gym

from envs.wrappers.timeout import Timeout

_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'robotis_tb3'))
_SCENE_XML  = os.path.join(_ASSETS_DIR, 'scene.xml')



class TB3GoToEnv(gym.Env):
	"""
	TurtleBot3 Burger differential-drive 'Go To' task.

	Ground-truth x/y/yaw provided by the simulator (no LiDAR/SLAM).
	Agent controls left and right wheel velocities directly.

	Observation (6-dim):
	  [x, y, cos(yaw), sin(yaw), dx, dy]
	  dx/dy = target - robot  (tells the agent where to go)

	Action (2-dim):
	  [vel_left, vel_right]  in [-1, 1] (normalised; scaled to ±6.67 rad/s internally)

	Reward:
	  Scaled negative distance + heading bonus + success bonus.
	  Scaled so cumulative returns fit within TD-MPC2's [-10, 10] value range.
	"""

	metadata = {'render_modes': ['rgb_array']}
	_FRAME_SKIP = 5   # 0.002 s × 5 = 0.01 s per policy step
	_GOAL_RADIUS_MIN = 0.3   # metres
	_GOAL_RADIUS_MAX = 0.8   # metres — reachable within episode
	_SUCCESS_THRESH  = 0.10  # metres — goal reached if closer than this
	_SUCCESS_BONUS   = 25.0  # one-time reward on reaching goal

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		self.model = mujoco.MjModel.from_xml_path(_SCENE_XML)
		self.data  = mujoco.MjData(self.model)

		# Capture the model's default reset state (valid free-joint quaternion)
		mujoco.mj_resetData(self.model, self.data)
		mujoco.mj_forward(self.model, self.data)
		self._init_qpos = self.data.qpos.copy()

		self._target = np.zeros(2, dtype=np.float32)
		self._success = False
		self._goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'goal')
		self._renderer = mujoco.Renderer(self.model, height=240, width=320)

		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
		)
		self._action_scale = np.float32(6.67)
		self.action_space = gym.spaces.Box(
			low=np.float32(-1.0), high=np.float32(1.0), shape=(2,), dtype=np.float32
		)

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------

	def _yaw(self):
		"""Extract yaw from the free-joint quaternion (MuJoCo: w, x, y, z)."""
		qw, qx, qy, qz = self.data.qpos[3:7]
		return float(np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy**2 + qz**2)))

	def _get_obs(self):
		x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
		yaw  = self._yaw()
		return np.array(
			[x, y, np.cos(yaw), np.sin(yaw),
			 float(self._target[0]) - x,
			 float(self._target[1]) - y],
			dtype=np.float32
		)

	def _get_reward(self):
		x, y = self.data.qpos[0], self.data.qpos[1]
		dx = self._target[0] - x
		dy = self._target[1] - y
		dist = float(np.hypot(dx, dy))

		# Scaled distance penalty: divide by max goal distance and episode len
		# so cumulative returns stay within [-10, 10]
		dist_reward = -dist / self._GOAL_RADIUS_MAX / 50.0

		# Heading hint: small bonus for facing toward the goal (guides early exploration)
		yaw = self._yaw()
		goal_angle = float(np.arctan2(dy, dx))
		heading_reward = float(np.cos(yaw - goal_angle)) / 500.0

		# Success detection
		self._success = dist < self._SUCCESS_THRESH

		reward = dist_reward + heading_reward
		if self._success:
			reward += self._SUCCESS_BONUS
		return reward

	# ------------------------------------------------------------------
	# TD-MPC2-compatible interface
	# reset() -> obs only  (TensorWrapper expects this, not a 2-tuple)
	# step()  -> (obs, reward, done, info)  (4-tuple, same as DMControlWrapper)
	# ------------------------------------------------------------------

	def reset(self, **kwargs):
		self.data.qpos[:] = self._init_qpos
		self.data.qvel[:] = 0.0
		self.data.ctrl[:] = 0.0
		mujoco.mj_forward(self.model, self.data)

		self._success = False
		# Randomise target: random angle, radius in reachable range
		angle  = np.random.uniform(0.0, 2.0 * np.pi)
		radius = np.random.uniform(self._GOAL_RADIUS_MIN, self._GOAL_RADIUS_MAX)
		self._target = np.array(
			[radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
		)
		# Move the goal sphere to the new target position
		self.model.geom_pos[self._goal_geom_id, :2] = self._target
		return self._get_obs()

	def step(self, action):
		scaled = np.asarray(action, dtype=np.float64) * self._action_scale
		self.data.ctrl[:] = np.clip(scaled, -self._action_scale, self._action_scale)
		for _ in range(self._FRAME_SKIP):
			mujoco.mj_step(self.model, self.data)
		reward = self._get_reward()
		done = self._success
		info = defaultdict(float)
		info['success'] = float(self._success)
		info['terminated'] = float(self._success)
		return self._get_obs(), reward, done, info

	def render(self, **kwargs):
		self._renderer.update_scene(self.data)
		return self._renderer.render()  # (H, W, 3) uint8


# ---------------------------------------------------------------------------

def make_env(cfg):
	"""Entry point used by envs/__init__.py. Task name: 'tb3-goto'."""
	if cfg.task != 'tb3-goto':
		raise ValueError(f'Unknown TB3 task: {cfg.task}')
	assert cfg.obs == 'state', 'TB3 environment only supports state observations.'
	env = TB3GoToEnv(cfg)
	env = Timeout(env, max_episode_steps=500)
	return env
