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


class PickAndPlaceOrient(MujocoEnv):
    """FR3 robot simulation for picking up a block and putting it in the correct orientation."""

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}
    frame_skip = 20

    def __init__(self, render_mode: Literal["human", "rgb_array", "depth_array"] | None = None):
        super().__init__(
            str(Path(__file__).parent / "data/fr3_pick_and_place.xml"),
            frame_skip=self.frame_skip,
            observation_space=None,
            render_mode=render_mode,
            width=1920,
            height=1080,
            default_camera_config={"azimuth": 180, "elevation": -30, "lookat": [0, 0, 1.2]},
        )
        self.metadata["render_fps"] = int(1 / (self.frame_skip * self.model.opt.timestep))
        assert self.model.nmocap > 0, "Model does not have mocap bodies"
        if self.data.ctrl.shape != (2,):
            raise ValueError(f"Expected 2 actuators, got {self.data.ctrl.shape[0]}")
        self._left_finger_id = self.model.joint("finger:left_joint").qposadr[0]
        self._right_finger_id = self.model.joint("finger:right_joint").qposadr[0]
        # Define action and observation spaces
        self.action_space = spaces.Box(-1, 1, (8,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            observation=spaces.Box(-np.inf, np.inf, (55,), dtype=np.float32),
            desired_goal=spaces.Box(-1, 1, (12,), dtype=np.float32),
            achieved_goal=spaces.Box(-1, 1, (12,), dtype=np.float32),
        )
        # Helper variables
        self._robot_origin = self.data.site("fr3_origin").xpos.copy()
        self._goal = np.concat([np.zeros(3), np.eye(3).flatten()], axis=-1).astype(np.float32)
        self._goal_pos_range = np.array([[-0.15, -0.15, 0.0], [0.15, 0.15, 0.45]], dtype=np.float32)
        self._goal_rot_range = np.pi / 2
        self._goal_pos_tol = 0.05
        self._goal_rot_tol = 0.1 * np.pi
        self._action_pos_scale = 0.05
        self._action_rot_scale = 0.1 * np.pi
        self._action_grip_scale = 0.04

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[NDArray[np.float64], dict]:
        super().reset(seed=seed, options=options)
        # Set mocap position
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "mocap")
        mocap_id = self.model.body_mocapid[body_id]
        self.data.mocap_pos[mocap_id] = self.data.site("eef_center").xpos
        self.data.mocap_quat[mocap_id] = np.array([0, 1, 0, 0])
        self.data.ctrl[:] = 1.0
        self.set_state(self.init_qpos, np.zeros(self.model.nv))  # Reset model
        if self.model.eq_data is not None:
            self.model.eq_data[0, :7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        # Check if we are using rotation actions
        orient = self.action_space.shape[-1] > 4
        # Resample cube
        cube_pos, cube_quat = self._sample_cube()  # In robot frame
        self.data.qpos[-7:-4] = cube_pos + self._robot_origin
        if orient:  # Rotation actions
            self.data.qpos[-4:] = R.from_quat(cube_quat).as_quat(scalar_first=True)
        # Resample goal
        self._goal = self._sample_goal()  # In robot frame
        if orient:  # Rotation actions
            self.model.body("goal_frame").pos[:] = self._goal[:3] + self._robot_origin
            rot = R.from_matrix(self._goal[3:].reshape(3, 3))
            self.model.body("goal_frame").quat[:] = rot.as_quat(scalar_first=True)
        else:
            self.model.body("goal").pos[:] = self._goal[:3] + self._robot_origin
        # Apply changes
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self.obs(), {}

    def step(
        self, action: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict]:
        assert action in self.action_space, f"Invalid action: {action}"
        # Set mocap position
        des_pos = self.data.site("eef_center").xpos + action[:3] * self._action_pos_scale
        des_pos = np.clip(des_pos, np.array([-0.3, -0.3, 0.50]), np.array([0.3, 0.3, 1.2]))
        self.data.mocap_pos[:] = des_pos
        if action.shape[-1] == 8:
            # Scale rotation to at most self.action_rot_scale * pi radians
            des_rot = R.from_quat(action[3:7])
            rot = R.from_matrix(self.eef_rot)
            delta = rot.inv() * des_rot
            delta_mag = R.magnitude(delta)
            scale = np.minimum(1, self._action_rot_scale / delta_mag)
            des_quat = (rot * delta**scale).as_quat(scalar_first=True)
        else:
            des_quat = np.array([0, 1, 0, 0])
        self.data.mocap_quat[:] = des_quat
        q_left, q_right = self.data.qpos[[self._left_finger_id, self._right_finger_id]]
        q_gripper = np.array([q_left, q_right]) + action[-1] * self._action_grip_scale
        self.data.ctrl[:] = q_gripper
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
        eef_pos = self.eef_pos
        eef_rot = self.eef_rot
        cube_pos = self.cube_pos
        cube_rot = self.cube_rot
        # End-effector velocity
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_center")
        eef_twist = np.zeros(6)  # 3 for translational, 3 for rotational
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, eef_id, eef_twist, 0
        )
        eef_vel, eef_ang_vel = eef_twist[:3], eef_twist[3:]
        # Cube velocity
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cube")
        cube_twist = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, cube_id, cube_twist, 0
        )
        cube_vel, cube_ang_vel = cube_twist[:3], cube_twist[3:]
        dt = 1 / 25
        # Finger states and velocities. Scale to [0, 1]
        finger_state = self.data.qpos[self._left_finger_id : self._right_finger_id + 1] * 25.0
        finger_vel = self.data.qvel[self._left_finger_id : self._right_finger_id + 1] * 25.0
        obs = np.concatenate(
            [
                eef_pos,  # 0:3
                eef_rot.flatten(),  # 3:12
                eef_vel,  # 12:15
                eef_ang_vel,  # 15:18
                cube_pos,  # 18:21
                cube_rot.flatten(),  # 21:30
                cube_vel * dt,  # 30:33
                cube_ang_vel * dt,  # 33:36
                cube_pos - eef_pos,  # 36:39
                (R.from_matrix(eef_rot).inv() * R.from_matrix(cube_rot))
                .as_matrix()
                .flatten(),  # 39:48
                (cube_vel - eef_vel) * dt,  # 48:51
                finger_state,  # 51:53
                finger_vel * dt,  # 53:55
            ],
            dtype=np.float32,
        )
        cube_pose = np.concat([cube_pos, cube_rot.flatten()], axis=-1).astype(np.float32)
        return {"observation": obs, "desired_goal": self._goal, "achieved_goal": cube_pose}

    def reward(self) -> float:
        """Return the reward of the robot."""
        pose = np.concat([self.cube_pos, self.cube_rot.flatten()], axis=-1).astype(np.float32)
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

    @property
    def cube_pos(self) -> NDArray[np.float64]:
        """Return the current cube position in the robot frame."""
        return self.data.body("cube").xpos - self._robot_origin

    @property
    def cube_rot(self) -> NDArray[np.float64]:
        """Return the current cube rotation in the robot frame."""
        return self.data.body("cube").xmat.reshape(3, 3)

    def _initialize_simulation(self) -> tuple:
        model, data = super()._initialize_simulation()
        data.qpos[:] = np.copy(model.key_qpos[0])  # Set start pose to the first keyframe
        mujoco.mj_forward(model, data)
        return model, data

    def _sample_goal(self) -> NDArray[np.float64]:
        """Sample a goal position in the robot frame."""
        pos = self.eef_pos + self.np_random.uniform(
            self._goal_pos_range[0], self._goal_pos_range[1]
        )
        # With 50% probability, sample a goal in the air
        if self.np_random.uniform() < 0.5:
            # Magnitude is in [0, pi], so to limit it to 0.5*pi we need to take R to the power of
            # 0.5, not 0.5*pi.
            random_rot = R.random() ** (self._goal_rot_range / np.pi)
            assert random_rot.magnitude() <= self._goal_rot_range + 1e-6  # Sanity check
            rot = (R.from_quat(self.data.qpos[-4:]) * random_rot).as_matrix().flatten()
            return np.concat([pos, rot], axis=-1).astype(np.float32)
        # Otherwise sample a goal on the table
        pos[2] = 0.025
        rot = random_rot_face_down(self.np_random).as_matrix().flatten()
        return np.concat([pos, rot], axis=-1).astype(np.float32)

    def _sample_cube(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sample a cube position and orientation in the robot frame."""
        dpos = np.zeros(3)
        while np.linalg.norm(dpos[:2]) < 0.1:
            dpos = self.np_random.uniform(self._goal_pos_range[0], self._goal_pos_range[1])
        pos = self.eef_pos + dpos
        pos[2] = 0.025  # Fixed height of the table + half cube height
        # Cube rotation randomization. First pick one of the 6 faces to align with the z axis
        cube_rot = random_rot_face_down(self.np_random)
        return pos.astype(np.float32), cube_rot.as_quat().astype(np.float32)


class PickAndPlace(PickAndPlaceOrient):
    def __init__(self, render_mode: Literal["human", "rgb_array", "depth_array"] | None = None):
        super().__init__(render_mode)
        self._goal = np.zeros(3, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, (34,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
            }
        )

    def obs(self) -> NDArray[np.float64]:
        """Return the current observation of the robot."""
        eef_pos = self.eef_pos
        cube_pos = self.cube_pos
        cube_rot = self.cube_rot
        dt = 1 / 25
        # End-effector velocity
        eef_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "eef_center")
        eef_twist = np.zeros(6)  # 3 for translational, 3 for rotational
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, eef_id, eef_twist, 0
        )
        eef_vel = eef_twist[:3]
        # Cube velocity
        cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cube")
        cube_twist = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, cube_id, cube_twist, 0
        )
        cube_vel, cube_ang_vel = cube_twist[:3], cube_twist[3:]
        # Finger states and velocities. Scale to [0, 1]
        finger_state = self.data.qpos[self._left_finger_id : self._right_finger_id + 1]
        finger_vel = self.data.qvel[self._left_finger_id : self._right_finger_id + 1]
        obs = np.concatenate(
            [
                eef_pos,  # 0:3
                eef_vel * dt,  # 3:6
                cube_pos,  # 6:9
                cube_rot.flatten(),  # 9:18
                cube_vel * dt,  # 18:21
                cube_ang_vel * dt,  # 21:24
                cube_pos - eef_pos,  # 24:27
                (cube_vel - eef_vel) * dt,  # 27:30
                finger_state,  # 30:32
                finger_vel * dt,  # 32:34
            ],
            dtype=np.float32,
        )
        return {
            "observation": obs,
            "desired_goal": self._goal,
            "achieved_goal": cube_pos.astype(np.float32),
        }

    def reward(self) -> float:
        """Return the reward of the robot."""
        return self.compute_reward(self.cube_pos, self._goal)

    def compute_reward(self, achieved_goal, desired_goal):
        xp = array_namespace(achieved_goal)
        pos_reached = xp.linalg.norm(achieved_goal - desired_goal, axis=-1) < self._goal_pos_tol
        if is_numpy_namespace(xp) and pos_reached.shape == ():
            return float(xp.where(pos_reached, 0.0, -1.0))
        return xp.where(pos_reached, 0.0, -1.0)

    def _sample_goal(self) -> NDArray[np.float64]:
        return super()._sample_goal()[:3]

    def _sample_cube(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sample a cube position and orientation in the robot frame."""
        pos, _ = super()._sample_cube()
        return pos, np.array([1.0, 0, 0, 0], dtype=np.float32)


def random_rot_face_down(rng) -> R:
    """Return a random rotation matrix with one of the cube faces pointing down."""
    # Cube rotation randomization. First pick one of the 6 faces to align with the z axis
    rx = rng.choice([0, 1, 2])
    ry = rng.choice([1, 2])
    rmat = np.zeros((3, 3))
    rmat[0, rx] = rng.choice([-1, 1])
    rmat[1, (rx + ry) % 3] = rng.choice([-1, 1])
    rmat[2, :] = np.cross(rmat[0, :], rmat[1, :])
    cube_rot = R.from_matrix(rmat)
    # Then apply a random rotation around the z axis
    return cube_rot * R.from_rotvec(rmat[2, :] * rng.uniform(-np.pi, np.pi))
