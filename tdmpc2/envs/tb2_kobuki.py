"""
TurtleBot2 Kobuki 'Go To' environment for TD-MPC2.

Uses gym.Env as base (standard practice) but step()/reset() return the
TD-MPC2-compatible signatures:
  reset() -> np.ndarray          (not the gymnasium 2-tuple)
  step()  -> (obs, r, done, info) (not the gymnasium 5-tuple)
This matches how DMControlWrapper and MuJoCoWrapper already behave.

Reward v2 — bearing-based, five-term compound reward inspired by:
  Batista et al., "A Deep Reinforcement Learning Framework and Methodology
  for Reducing the Sim-to-Real Gap in ASV Navigation" (arXiv 2407.08263)
"""
import os
from collections import defaultdict

import numpy as np
import mujoco
import gymnasium as gym

from envs.wrappers.timeout import Timeout

_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'kobuki_tb2'))
_SCENE_XML  = os.path.join(_ASSETS_DIR, 'scene.xml')


class TB2KobukiGoToEnv(gym.Env):
	"""
	TurtleBot2 Kobuki differential-drive 'Go To' task.

	Ground-truth x/y/yaw provided by the simulator (no LiDAR/SLAM).
	Agent outputs twist commands (v_linear, omega), matching the ROS2
	cmd_vel interface used on the real robot.  The environment converts
	to per-wheel angular velocities internally for MuJoCo.

	Observation (5-dim, BODY FRAME):
	  [surge, yaw_rate, cos(bearing), sin(bearing), distance]
	  - surge: forward velocity in body frame
	  - yaw_rate: angular velocity about z
	  - bearing: angle of goal in body frame (0 = straight ahead)
	  - distance: Euclidean distance to goal
	  Sway is dropped (always ~0 for differential drive on flat ground).

	Action (2-dim):
	  [v_linear, omega]  in [-1, 1] (normalised; scaled to physical twist internally)
	  Converted to wheel velocities via inverse diff-drive kinematics:
	    v_l = (v_linear - omega * w_b / 2) / r_w
	    v_r = (v_linear + omega * w_b / 2) / r_w

	Reward (five-term compound):
	  r = λ1·(d_{t-1} - d_t)                         # distance progress
	    + λ2·(exp(k1·bearing⁴) + exp(k2·bearing²))    # bearing alignment
	    + λ3·(exp(k3·|Δyaw_rate|) - 1)                 # smoothness penalty
	    + λ4                                            # time penalty
	    + λ5  (if dist < threshold)                     # goal bonus
	"""

	metadata = {'render_modes': ['rgb_array']}

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		# ---- Goal sampling ----
		self._goal_radius_min = 0.3   # metres
		self._success_thresh  = 0.15  # metres — goal reached if closer than this

		# ---- Curriculum: progressively harder goals ----
		self._curriculum_episodes = 500   # episodes to go from easy → full difficulty
		self._angle_start  = np.pi / 4   # ±45° in front at the beginning
		self._angle_end    = np.pi       # full 360° at the end
		self._radius_start = 0.3         # close goals at the beginning
		self._radius_end   = 3.0         # full range at the end

		# ---- Reward weights (λ) ----
		self._lambda_dist     = 35.0     # distance progress
		self._lambda_bearing  =  0.02    # bearing alignment
		self._lambda_smooth   =  0.3     # angular smoothness
		self._lambda_time     = -0.04    # time step penalty
		self._lambda_goal     = 40.0     # one-time success bonus
		self._lambda_approach = -0.3     # braking zone speed² penalty

		# ---- Reward shaping constants (k) ----
		self._k1_bearing = -10.0   # sharpness on bearing⁴ (tight peak near 0)
		self._k2_bearing =  -0.1   # sharpness on bearing² (broad guidance)
		self._d_slow     =  0.8    # braking zone radius (metres)
		self._k3_smooth  =  -0.33  # smoothness sensitivity

		# ---- Action repeat (match real Kobuki odom rate: 20 Hz) ----
		self._action_repeat = 25  # mj_steps per policy step → 0.05s per decision

		# ---- Action noise (sim-to-real: wheel slippage) ----
		self._slip_sigma = 0.0  # std of multiplicative per-wheel noise (0 = disabled)

		# ---- Differential-drive kinematics (from URDF/XML) ----
		self._wheel_radius  = 0.035   # metres  (geom size="0.035 ...")
		self._wheelbase     = 0.230   # metres  (wheels at y = ±0.115)
		self._max_wheel_vel = 20.0    # rad/s   (ctrlrange="-20 20")

		# Derived twist limits (reachable when both wheels saturate together)
		self._v_linear_max = self._wheel_radius * self._max_wheel_vel          # 0.7 m/s
		self._omega_max    = 2.0 * self._wheel_radius * self._max_wheel_vel / self._wheelbase  # ≈6.09 rad/s

		# ---- MuJoCo model ----
		self.model = mujoco.MjModel.from_xml_path(_SCENE_XML)
		self.data  = mujoco.MjData(self.model)

		# Capture the model's default reset state (valid free-joint quaternion)
		mujoco.mj_resetData(self.model, self.data)
		mujoco.mj_forward(self.model, self.data)
		self._init_qpos = self.data.qpos.copy()

		self._target = np.zeros(2, dtype=np.float32)
		self._success = False
		self._goal_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'goal')
		self._renderer = mujoco.Renderer(self.model, height=480, width=640)

		# State carried between steps for progress & smoothness rewards
		self._prev_dist     = 0.0
		self._prev_yaw_rate = 0.0
		self._episode_count = 0

		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
		)
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

	def _body_frame_goal(self):
		"""
		Transform goal offset from world frame into the robot's body frame.
		Returns (dx_body, dy_body) — the goal vector as seen from the robot.
		  dx_body > 0  →  goal is ahead
		  dy_body > 0  →  goal is to the left
		"""
		x, y = float(self.data.qpos[0]), float(self.data.qpos[1])
		dx_w = self._target[0] - x
		dy_w = self._target[1] - y
		yaw = self._yaw()
		cos_y, sin_y = np.cos(yaw), np.sin(yaw)
		# Inverse rotation (world → body): rotate by -yaw
		dx_b =  cos_y * dx_w + sin_y * dy_w
		dy_b = -sin_y * dx_w + cos_y * dy_w
		return dx_b, dy_b

	def _body_frame_velocities(self):
		"""Return (surge, yaw_rate) — forward velocity in body frame and angular velocity about z."""
		vx_w, vy_w = float(self.data.qvel[0]), float(self.data.qvel[1])
		yaw = self._yaw()
		surge = np.cos(yaw) * vx_w + np.sin(yaw) * vy_w
		yaw_rate = float(self.data.qvel[5])
		return surge, yaw_rate

	def _get_obs(self):
		"""
		Body-frame observation (5-dim):
		  [surge, yaw_rate, cos(bearing), sin(bearing), distance]
		"""
		surge, yaw_rate = self._body_frame_velocities()
		dx_b, dy_b = self._body_frame_goal()
		dist = float(np.hypot(dx_b, dy_b))
		bearing = float(np.arctan2(dy_b, dx_b))
		return np.array(
			[surge, yaw_rate,
			 np.cos(bearing), np.sin(bearing),
			 dist],
			dtype=np.float32
		)

	def _get_reward(self):
		"""
		Five-term compound reward.
		All terms are designed so cumulative return stays negative until
		the goal is reached, giving the optimizer a clear gradient toward
		success.
		"""
		dx_b, dy_b = self._body_frame_goal()
		dist = float(np.hypot(dx_b, dy_b))
		bearing = float(np.arctan2(dy_b, dx_b))
		_, yaw_rate = self._body_frame_velocities()

		# 1. Distance progress: reward for getting closer
		r_dist = self._lambda_dist * (self._prev_dist - dist)

		# 2. Bearing alignment: dual-exponential with sharp peak near 0
		r_bearing = self._lambda_bearing * (
			np.exp(self._k1_bearing * bearing**4) +
			np.exp(self._k2_bearing * bearing**2)
		)

		# 3. Smoothness: penalise abrupt yaw rate changes
		delta_yaw_rate = abs(yaw_rate - self._prev_yaw_rate)
		r_smooth = self._lambda_smooth * (np.exp(self._k3_smooth * delta_yaw_rate) - 1.0)

		# 4. Time penalty: constant cost per step → encourages speed
		r_time = self._lambda_time

		# 5. Approach braking: penalise speed² inside braking zone
		#    surge² makes total penalty proportional to speed (not constant),
		#    so the robot genuinely learns to decelerate.
		surge, _ = self._body_frame_velocities()
		proximity = max(0.0, 1.0 - dist / self._d_slow)  # 0 outside, 1 at goal
		r_approach = self._lambda_approach * surge**2 * proximity

		# 6. Goal bonus — requires near-zero speed to trigger success
		self._success = dist < self._success_thresh and abs(surge) < 0.01
		r_goal = self._lambda_goal if self._success else 0.0

		# Update state for next step
		self._prev_dist = dist
		self._prev_yaw_rate = yaw_rate

		reward = r_dist + r_bearing + r_smooth + r_time + r_goal + r_approach
		self._reward_components = {
			'r_dist': r_dist, 'r_bearing': r_bearing, 'r_smooth': r_smooth,
			'r_time': r_time, 'r_goal': r_goal, 'r_approach': r_approach,
		}
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
		self._episode_count += 1

		progress = min(self._episode_count / self._curriculum_episodes, 1.0)

		max_angle  = self._angle_start + progress * (self._angle_end - self._angle_start)
		max_radius = self._radius_start + progress * (self._radius_end - self._radius_start)

		# Goal angle centred on robot's forward direction (yaw=0 at reset)
		angle  = np.random.uniform(-max_angle, max_angle)
		radius = np.random.uniform(self._goal_radius_min, max_radius)
		self._target = np.array(
			[radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32
		)
		# Move the goal sphere to the new target position
		self.model.geom_pos[self._goal_geom_id, :2] = self._target

		# Initialise prev-step state for progress & smoothness rewards
		dx_b, dy_b = self._body_frame_goal()
		self._prev_dist = float(np.hypot(dx_b, dy_b))
		self._prev_yaw_rate = 0.0

		return self._get_obs()

	def _twist_to_wheels(self, v_linear, omega):
		"""Inverse diff-drive kinematics: twist → wheel angular velocities."""
		v_l = (v_linear - omega * self._wheelbase / 2.0) / self._wheel_radius
		v_r = (v_linear + omega * self._wheelbase / 2.0) / self._wheel_radius
		return v_l, v_r

	def step(self, action):
		action = np.asarray(action, dtype=np.float64)
		# Policy outputs normalised twist; scale to physical units
		v_linear = action[0] * self._v_linear_max
		omega    = action[1] * self._omega_max
		# Convert to per-wheel angular velocities for MuJoCo actuators
		v_l, v_r = self._twist_to_wheels(v_linear, omega)
		# Multiplicative wheel noise: simulates slippage / uneven traction
		if self._slip_sigma > 0:
			v_l *= 1.0 + np.random.normal(0.0, self._slip_sigma)
			v_r *= 1.0 + np.random.normal(0.0, self._slip_sigma)
		self.data.ctrl[0] = np.clip(v_l, -self._max_wheel_vel, self._max_wheel_vel)
		self.data.ctrl[1] = np.clip(v_r, -self._max_wheel_vel, self._max_wheel_vel)
		# Action repeat: hold command for multiple physics steps (match 20 Hz odom)
		reward = 0.0
		for _ in range(self._action_repeat):
			mujoco.mj_step(self.model, self.data)
			reward += self._get_reward()
			if self._success:
				break
		done = self._success
		info = defaultdict(float)
		info['success'] = float(self._success)
		info['terminated'] = float(self._success)
		info.update(self._reward_components)
		return self._get_obs(), reward, done, info

	def render(self, **kwargs):
		cam_id = self.model.camera('topdown').id
		self._renderer.update_scene(self.data, camera=cam_id)
		return self._renderer.render()  # (H, W, 3) uint8


# ---------------------------------------------------------------------------

def make_env(cfg):
	"""Entry point used by envs/__init__.py. Task name: 'tb2-kobuki-goto'."""
	if cfg.task != 'tb2-kobuki-goto':
		raise ValueError(f'Unknown TB2 Kobuki task: {cfg.task}')
	assert cfg.obs == 'state', 'TB2 Kobuki environment only supports state observations.'
	env = TB2KobukiGoToEnv(cfg)
	env = Timeout(env, max_episode_steps=600)
	return env