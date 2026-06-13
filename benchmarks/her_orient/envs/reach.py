"""Code largely adapted from Gymnasium's robotics Fetch environments."""

import logging
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from array_api_compat import array_namespace, is_numpy_namespace
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from rotations.rotations import matrix_to_quat

logger = logging.getLogger(__name__)


class ReachOrient(MujocoEnv):
    """FR3 robot simulation for reaching a position."""

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    frame_skip = 20

    def __init__(self, render_mode: Literal["human", "rgb_array", "depth_array"] | None = None):
        super().__init__(
            str(Path(__file__).parent / "data/fr3_reach.xml"),
            frame_skip=self.frame_skip,
            observation_space=None,
            render_mode=render_mode,
            width=1920,
            height=1080,
            default_camera_config={"azimuth": 180, "elevation": -30, "lookat": [0, 0, 1.2]},
        )
        self.metadata["render_fps"] = int(1 / (self.frame_skip * self.model.opt.timestep))
        assert self.model.nmocap > 0, "Model does not have mocap bodies"
        # Define action and observation spaces
        self.action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            desired_goal=spaces.Box(-1, 1, (12,), dtype=np.float32),
            achieved_goal=spaces.Box(-1, 1, (12,), dtype=np.float32),
            observation=spaces.Box(-np.inf, np.inf, (18,), dtype=np.float32),  # 3 + 9 + 3 + 3
        )
        # Helper variables
        self._robot_origin = self.data.site("fr3_origin").xpos.copy()
        self._goal = np.concat([np.zeros(3), np.eye(3).flatten()], axis=-1).astype(np.float32)
        self._goal_pos_range = np.ones(3) * 0.15
        self._goal_rot_range = np.pi / 2
        self._goal_pos_tol = 0.05
        self._goal_rot_tol = 0.1 * np.pi
        self._action_pos_scale = 0.05
        self._action_rot_scale = 0.1 * np.pi

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.float64], dict]:
        super().reset(seed=seed, options=options)
        # Set mocap position
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
        mocap_id = self.model.body_mocapid[body_id]
        self.data.mocap_pos[mocap_id] = self.data.site("eef_center").xpos
        self.data.mocap_quat[mocap_id] = np.array([0, 1, 0, 0])
        self.set_state(self.init_qpos, np.zeros(self.model.nv))  # Reset model
        if self.model.eq_data is not None:
            self.model.eq_data[0, :7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        # Resample goal pos
        self._goal = self._sample_goal()  # In robot frame
        self.model.body("goal").pos[:] = self._goal[:3] + self._robot_origin
        if self._goal.shape[-1] > 3:
            self.model.body("goal").quat[:] = R.from_matrix(self._goal[3:].reshape(3, 3)).as_quat(
                scalar_first=True
            )
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self.obs(), {}

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict]:
        assert action in self.action_space, f"Invalid action: {action}"
        # Set mocap position
        self.data.mocap_pos[:] = (
            self.data.site("eef_center").xpos + action[:3] * self._action_pos_scale
        )
        if action.shape[-1] == 7:
            # Scale rotation to at most self.action_rot_scale * pi radians
            des_rot = R.from_quat(action[3:])
            rot = R.from_matrix(self.eef_rot)
            delta = rot.inv() * des_rot
            delta_mag = R.magnitude(delta)
            scale = np.minimum(1, self._action_rot_scale / delta_mag)
            des_quat = (rot * delta**scale).as_quat(scalar_first=True)
        else:
            des_quat = np.array([0, 1, 0, 0])
        self.data.mocap_quat[:] = des_quat
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self.obs(), self.reward(), self.terminated(), self.truncated(), {}

    def obs(self) -> NDArray[np.float64]:
        """Return the current observation of the robot."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_center")
        eef_vel = np.zeros(6)  # 3 for translational, 3 for rotational
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, site_id, eef_vel, 0
        )
        obs = np.concatenate([self.eef_pos, self.eef_rot.flatten(), eef_vel], dtype=np.float32)
        pose = np.concat([self.eef_pos, self.eef_rot.flatten()], axis=-1).astype(np.float32)
        return {"observation": obs, "desired_goal": self._goal, "achieved_goal": pose}

    def reward(self) -> float:
        """Return the reward of the robot."""
        pose = np.concat([self.eef_pos, self.eef_rot.flatten()], axis=-1).astype(np.float32)
        return self.compute_reward(pose, self._goal)

    def terminated(self) -> bool:
        return False

    def truncated(self) -> bool:
        return False

    def compute_reward(
        self, achieved_goal: NDArray | Tensor, desired_goal: NDArray | Tensor
    ) -> NDArray | Tensor:
        """Compute the sparse reward for the given achieved and desired goal."""
        xp = array_namespace(achieved_goal)
        pos = achieved_goal[..., :3]
        quat = matrix_to_quat(achieved_goal[..., 3:].reshape(*pos.shape[:-1], 3, 3))
        des_pos = desired_goal[..., :3]
        des_quat = matrix_to_quat(desired_goal[..., 3:].reshape(*pos.shape[:-1], 3, 3))
        pos_reached = xp.linalg.norm(pos - des_pos, axis=-1) < self._goal_pos_tol
        angle = 2 * xp.arccos(xp.clip(xp.sum(quat * des_quat, axis=-1), -1.0, 1.0))

        if is_numpy_namespace(xp) and pos_reached.shape == ():
            return float(xp.where(pos_reached & (angle < self._goal_rot_tol), 0.0, -1.0))
        return xp.where(pos_reached & (angle < self._goal_rot_tol), 0.0, -1.0)

    def reset_model(self) -> NDArray[np.float64]:
        """Reset the robot degrees of freedom (qpos and qvel)."""
        self.set_state(self.init_qpos, np.zeros(self.model.nv))

    @property
    def eef_pos(self) -> NDArray[np.float64]:
        """Return the current end-effector position in the robot frame."""
        return self.data.site("eef_center").xpos - self._robot_origin

    @property
    def eef_rot(self) -> NDArray[np.float64]:
        """Return the current end-effector rotation in the robot frame."""
        return self.data.site("eef_center").xmat.reshape(3, 3)

    def _initialize_simulation(self) -> tuple:
        model, data = super()._initialize_simulation()
        data.qpos[:] = np.copy(model.key_qpos[0])  # Set start pose to the first keyframe
        mujoco.mj_forward(model, data)
        return model, data

    def _sample_goal(self) -> NDArray[np.float64]:
        """Sample a goal position in the robot frame."""
        pos = self.eef_pos + self.np_random.uniform(-self._goal_pos_range, self._goal_pos_range)
        # Magnitude is in [0, pi], so to limit it to 0.5*pi we need to take R to the power of 0.5,
        # not 0.5*pi.
        random_rot = R.random() ** (self._goal_rot_range / np.pi)
        assert random_rot.magnitude() <= self._goal_rot_range + 1e-6  # Sanity check
        rot = (R.from_matrix(self.eef_rot) * random_rot).as_matrix().flatten()
        return np.concat([pos, rot], axis=-1).astype(np.float32)


class Reach(ReachOrient):
    def __init__(self, render_mode: Literal["human", "rgb_array", "depth_array"] | None = None):
        super().__init__(render_mode)
        self._goal = np.zeros(3, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            }
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.float64], dict]:
        obs, info = super().reset(seed=seed, options=options)
        self.model.body("goal").quat[:] = np.array([0, 1, 0, 0])
        return obs, info

    def obs(self) -> NDArray[np.float64]:
        """Return the current observation of the robot."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_center")
        eef_vel = np.zeros(6)  # 3 for translational, 3 for rotational
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, site_id, eef_vel, 0
        )
        obs = np.concatenate([self.eef_pos, eef_vel[:3]], dtype=np.float32)
        return {
            "observation": obs,
            "desired_goal": self._goal,
            "achieved_goal": self.eef_pos.astype(np.float32),
        }

    def reward(self) -> float:
        """Return the reward of the robot."""
        return self.compute_reward(self.eef_pos, self._goal)

    def compute_reward(self, achieved_goal, desired_goal):
        xp = array_namespace(achieved_goal)
        pos_reached = xp.linalg.norm(achieved_goal - desired_goal, axis=-1) < self._goal_pos_tol
        if is_numpy_namespace(xp) and pos_reached.shape == ():
            return float(xp.where(pos_reached, 0.0, -1.0))
        return xp.where(pos_reached, 0.0, -1.0)

    def _sample_goal(self) -> NDArray[np.float64]:
        """Sample a goal position in the robot frame."""
        pos = self.eef_pos + self.np_random.uniform(-self._goal_pos_range, self._goal_pos_range)
        return pos.astype(np.float32)
