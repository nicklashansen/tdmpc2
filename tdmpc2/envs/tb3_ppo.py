"""
TurtleBot3 Burger 'Go To' environment for PPO.

Identical to tb3.py in every way (same scene, same observation, same reward)
except the action space uses unicycle parameterisation [linear_vel, angular_vel]
instead of [vel_left, vel_right].

Why: the differential-drive action space couples rotation and translation through
the wheel velocities, causing PPO to get trapped in a rotate-in-place local
optimum driven by the heading reward.  Unicycle parameterisation decouples them:
  action[0] = linear velocity  -> controls forward/backward motion independently
  action[1] = angular velocity -> controls turning independently
The heading reward now trains angular_vel and the distance reward trains
linear_vel without interference.

TD-MPC2 is unaffected (it uses task=tb3-goto / tb3.py).
"""
import os
from collections import defaultdict

import numpy as np
import mujoco
import gymnasium as gym

from envs.wrappers.timeout import Timeout

_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'robotis_tb3'))
_SCENE_XML  = os.path.join(_ASSETS_DIR, 'scene.xml')


class TB3GoToPPOEnv(gym.Env):
	"""
	TurtleBot3 Burger 'Go To' task with unicycle action parameterisation.

	Observation (6-dim)  — identical to TB3GoToEnv:
	  [x, y, cos(yaw), sin(yaw), dx, dy]

	Action (2-dim)  — unicycle parameterisation:
	  [linear_vel, angular_vel]  in [-1, 1]
	  linear_vel  scaled to ±0.22 m/s  (TB3 Burger max linear speed)
	  angular_vel scaled to ±2.75 rad/s (TB3 Burger max angular speed)
	  Internally converted to [vel_left, vel_right] wheel velocities.

	Reward  — identical to TB3GoToEnv:
	  Scaled negative distance + heading bonus + success bonus.
	"""

	metadata = {'render_modes': ['rgb_array']}
	_FRAME_SKIP      = 5      # 0.002 s × 5 = 0.01 s per policy step
	_GOAL_RADIUS_MIN = 0.3    # metres
	_GOAL_RADIUS_MAX = 0.8    # metres
	_SUCCESS_THRESH  = 0.10   # metres
	_SUCCESS_BONUS   = 25.0

	# TB3 Burger kinematics
	_wheel_radius    = 0.033  # metres
	_wheel_base      = 0.160  # metres (distance between wheels)
	_linear_scale    = 0.22   # m/s  (maps action ±1 → ±0.22 m/s)
	_angular_scale   = 2.75   # rad/s (maps action ±1 → ±2.75 rad/s)
	_wheel_vel_max   = 6.67   # rad/s (max wheel angular velocity for clipping)

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		self.model = mujoco.MjModel.from_xml_path(_SCENE_XML)
		self.data  = mujoco.MjData(self.model)

		mujoco.mj_resetData(self.model, self.data)
		mujoco.mj_forward(self.model, self.data)
		self._init_qpos = self.data.qpos.copy()

		self._target      = np.zeros(2, dtype=np.float32)
		self._prev_dist   = 0.0
		self._success     = False
		self._goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'goal')
		self._renderer    = mujoco.Renderer(self.model, height=240, width=320)

		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
		)
		self.action_space = gym.spaces.Box(
			low=np.float32(-1.0), high=np.float32(1.0), shape=(2,), dtype=np.float32
		)

	# ------------------------------------------------------------------
	# Helpers  (identical to TB3GoToEnv)
	# ------------------------------------------------------------------

	def _yaw(self):
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
		dist = float(np.hypot(self._target[0] - x, self._target[1] - y))

		# Progress reward: positive when moving toward goal, negative when moving away.
		# Clear per-step signal that directly rewards forward motion.
		progress = (self._prev_dist - dist) * 5.0
		self._prev_dist = dist

		self._success = dist < self._SUCCESS_THRESH

		# Small time penalty forces the robot to move — staying still is never optimal
		reward = progress - 0.01
		if self._success:
			reward += self._SUCCESS_BONUS
		return reward

	# ------------------------------------------------------------------
	# TD-MPC2-compatible interface
	# ------------------------------------------------------------------

	def reset(self, **kwargs):
		self.data.qpos[:] = self._init_qpos
		self.data.qvel[:] = 0.0
		self.data.ctrl[:] = 0.0
		mujoco.mj_forward(self.model, self.data)

		self._success = False
		angle  = np.random.uniform(0.0, 2.0 * np.pi)
		radius = np.random.uniform(self._GOAL_RADIUS_MIN, self._GOAL_RADIUS_MAX)
		self._target = np.array(
			[radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
		)
		self.model.geom_pos[self._goal_geom_id, :2] = self._target
		self._prev_dist = float(np.hypot(
			self._target[0] - self.data.qpos[0],
			self._target[1] - self.data.qpos[1]
		))
		return self._get_obs()

	def step(self, action):
		# Unicycle → differential drive conversion
		v_ms   = float(action[0]) * self._linear_scale   # m/s
		w_rads = float(action[1]) * self._angular_scale  # rad/s

		v_left_rads  = (v_ms - w_rads * self._wheel_base / 2) / self._wheel_radius
		v_right_rads = (v_ms + w_rads * self._wheel_base / 2) / self._wheel_radius

		ctrl = np.array([v_left_rads, v_right_rads], dtype=np.float64)
		self.data.ctrl[:] = np.clip(ctrl, -self._wheel_vel_max, self._wheel_vel_max)

		for _ in range(self._FRAME_SKIP):
			mujoco.mj_step(self.model, self.data)

		reward = self._get_reward()
		done   = self._success
		info   = defaultdict(float)
		info['success']    = float(self._success)
		info['terminated'] = float(self._success)
		return self._get_obs(), reward, done, info

	def render(self, **kwargs):
		self._renderer.update_scene(self.data)
		return self._renderer.render()


# ---------------------------------------------------------------------------

def make_env(cfg):
	"""Entry point used by envs/__init__.py. Task name: 'tb3-goto-ppo'."""
	if cfg.task != 'tb3-goto-ppo':
		raise ValueError(f'Unknown TB3-PPO task: {cfg.task}')
	assert cfg.obs == 'state', 'TB3 environment only supports state observations.'
	env = TB3GoToPPOEnv(cfg)
	env = Timeout(env, max_episode_steps=500)
	return env
