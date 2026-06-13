from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal

import gymnasium
import jax
import jax.numpy as jp
import numpy as np
from array_api_compat import array_namespace
from crazyflow.control.control import MAX_THRUST, MIN_THRUST
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax.scipy.spatial.transform import Rotation as JR
from lsy_rl.core.logger import Collector

from rotations.envs.actions import (
    euler_rotation,
    matrix_rotation,
    quat_plus_rotation,
    quat_rotation,
    r6_rotation,
    rel_euler_rotation,
    rel_matrix_rotation,
    rel_quat_plus_rotation,
    rel_quat_rotation,
    rel_r6_rotation,
    rel_tangent_rotation,
    tangent_rotation,
)
from rotations.envs.rot import rotation_dynamics
from rotations.rotations import RotType

if TYPE_CHECKING:
    from jax import Array
    from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv


class RotationDroneRacingWrapper(gymnasium.vector.VectorWrapper):
    """
    A wrapper for lsy_drone_racing environments that allows the usage of the various action types,
    control modes and rotation limits supported in the rotations package.

    Note: Unlike other similar wrappers used, this wrapper does not alter the environment's observation space.

    Args:
        env: VecDroneRacEnv to wrap.
        action_type: Rotation type to use for actions.
        obs_type: Rotation type to use for observations.
        step_len: Maximum rotation at each step is limited to step_len * pi radians.
        control_mode: Control mode for drone orientations, one of ["rel", "abs", "rel_scale"].
    """

    def __init__(
        self,
        env: VecDroneRaceEnv,
        action_type: RotType = RotType.tangent,
        obs_type: RotType = RotType.quat,
        step_len: float = 0.4,
        control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    ) -> None:
        assert env.sim.control == "attitude", (
            f"Wrapper only supports attitude control mode, got {env.sim.control}."
        )
        super().__init__(env)

        # Store rotations configs
        self.action_type = RotType(action_type)
        self.obs_type = RotType(obs_type)
        self.step_len = step_len
        self.control_mode = control_mode

        # Setup jax functions for action conversion
        self._action = self.build_action_fn()

        # Setup action space
        low = -np.ones(self.action_type.dim + 1, dtype=np.float32)
        low[-1] *= self.action_type != RotType.quat_plus  # quat_plus has a min of 0 for w
        high = np.ones(self.action_type.dim + 1, dtype=np.float32)
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def step(self, actions) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
        actions = self._action(actions, self.rotation)
        observations, rewards, terminations, truncations, infos = super().step(actions)
        return observations, rewards, terminations, truncations, infos

    def build_action_fn(self):
        step_len = self.step_len
        action_tf = self.get_action_tf(self.action_type, "rel" in self.control_mode)
        if self.control_mode == "rel_scale":
            action_tf = partial(action_tf, scale=True)

        @jax.jit
        def action(action: Array, rot: JR) -> Array:
            """Convert chosen action representation to the environment's representation
            (absolute Euler) and scale thrust. Function also enforces the set rotation limits."""
            rot_action = action[:, 1:]
            rot_desired = action_tf(rot, rot_action, step_len)
            rot_desired = rotation_dynamics(rot, rot_desired, step_len)
            rot_action_env = rot_desired.as_euler("xyz", degrees=False)

            # Scale between min and max thrust
            scale = 4 * (MAX_THRUST - MIN_THRUST) / 2
            mean = 4 * (MAX_THRUST + MIN_THRUST) / 2
            thrust = action[:, :1] * scale + mean
            return jp.concat([thrust, rot_action_env], axis=-1)

        return action

    @staticmethod
    def get_action_tf(action_type: RotType, rel: bool) -> Callable[[JR, Array, float], JR]:
        match action_type:
            case RotType.euler:
                return rel_euler_rotation if rel else euler_rotation
            case RotType.tangent:
                return rel_tangent_rotation if rel else tangent_rotation
            case RotType.quat_plus:
                return rel_quat_plus_rotation if rel else quat_plus_rotation
            case RotType.matrix:
                return rel_matrix_rotation if rel else matrix_rotation
            case RotType.r6:
                return rel_r6_rotation if rel else r6_rotation
            case RotType.quat:
                return rel_quat_rotation if rel else quat_rotation
            case _:
                raise ValueError(f"Invalid action type {action_type}")

    @property
    def rotation(self) -> JR:
        """Returns the current orientation of drones from simulation as rotation objects."""
        # Assumes only one drone is being simulated and drops n_drones dimension
        return JR.from_quat(self.env.sim.data.states.quat[:, 0, :])

    @property
    def device(self) -> Any:
        """Returns the device of the underlying DroneEnv.

        Defined to avoid attribute errors when outer wrappers try to access env.device.
        """
        return self.env.device


class ObsTF(gymnasium.vector.VectorWrapper):
    """
    A wrapper that flattens and scales the environment's dict observations and modifies
    the observations in the following way:

        1) Convert drone's orientation from quaternions to rotation matrices.
        2) Convert gates' positions from the world frame to the drone's body frame.
        3) Convert gates' orientations to the projection of their relative y-axes in the drone's xy-plane.
        4) Convert obstacles' positions from the world frame to the drone's body frame.
        5) Append the target gate's position and heading explicitly to the observations.
        6) Append the agent's previous action to the observations.

    """

    corner_offsets = jp.array(
        (  # Corner offsets in the gate frame (gate_width = 0.45)
            (0.45 / 2, 0.0, 0.45 / 2),
            (-0.45 / 2, 0.0, 0.45 / 2),
            (0.45 / 2, 0.0, -0.45 / 2),
            (-0.45 / 2, 0.0, -0.45 / 2),
        )
    )

    def __init__(self, env):
        super().__init__(env)
        assert len(self.action_space.shape) == 2
        self._last_action = jp.zeros(self.action_space.shape)
        obs_dim = 72 + self.action_space.shape[1]
        self.single_observation_space = spaces.Box(-1, 1, shape=(obs_dim,))
        self.observation_space = batch_space(self.single_observation_space, env.num_envs)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Array, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_action = jp.zeros(self.action_space.shape)
        obs, info = self.obs_and_info(obs, self._last_action)
        return obs, info

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = action
        obs, info = self.obs_and_info(obs, self._last_action)
        return obs, reward, terminated, truncated, info

    @staticmethod
    @jax.jit
    def obs_and_info(obs: dict[str, Array], action: Array) -> tuple[Array, Array]:
        # Convert target gate to one-hot encoded vector
        n_envs = obs["pos"].shape[0]
        rot = JR.from_quat(obs["quat"][..., None, :])
        target_gate = obs["target_gate"]
        target_gate_one_hot = jp.zeros((n_envs, 4))
        target_gate_one_hot = target_gate_one_hot.at[jp.arange(n_envs), target_gate].set(1)

        # Convert gate positions to relative positions in the drone frame
        gates_pos = rot.apply(obs["gates_pos"] - obs["pos"][..., None, :], inverse=True)
        assert gates_pos.shape == (n_envs, 4, 3)

        # Convert gate orientations to relative gate direction (y axis in [x, y] plane) in drone frame
        rot_gates = JR.from_quat(obs["gates_quat"])
        gates_dir = (rot.inv() * rot_gates).as_matrix()[..., :2, 1]
        assert gates_dir.shape == (n_envs, 4, 2), gates_dir.shape

        # Convert obstacle positions to relative positions in the drone frame
        obstacles_pos = rot.apply(obs["obstacles_pos"] - obs["pos"][..., None, :], inverse=True)[
            ..., :2
        ]
        assert obstacles_pos.shape == (n_envs, 4, 2)

        # Get positions of four corners of target gate in the drone frame
        target_pos = obs["gates_pos"][jp.arange(n_envs), target_gate]
        rot_target_gates = rot_gates[jp.arange(n_envs), target_gate]
        corner_offsets = jp.concat(
            (
                rot_target_gates.apply(ObsTF.corner_offsets[0]),
                rot_target_gates.apply(ObsTF.corner_offsets[1]),
                rot_target_gates.apply(ObsTF.corner_offsets[2]),
                rot_target_gates.apply(ObsTF.corner_offsets[3]),
            ),
            axis=-1,
        )
        corner_pos = jp.concat(
            (
                rot[:, 0].apply(target_pos + corner_offsets[:, 0:3] - obs["pos"], inverse=True),
                rot[:, 0].apply(target_pos + corner_offsets[:, 3:6] - obs["pos"], inverse=True),
                rot[:, 0].apply(target_pos + corner_offsets[:, 6:9] - obs["pos"], inverse=True),
                rot[:, 0].apply(target_pos + corner_offsets[:, 9:12] - obs["pos"], inverse=True),
            ),
            axis=-1,
        )
        assert corner_pos.shape == (n_envs, 12)

        # Flatten and concat all observations
        target_heading = gates_dir[jp.arange(n_envs), target_gate]
        obs = jp.concat(
            (
                obs["pos"],  # n_envs, 3
                rot.as_matrix().reshape(n_envs, 9),  # n_envs, 9
                obs["vel"],  # n_envs, 3
                obs["ang_vel"],  # n_envs, 3
                gates_pos.reshape(n_envs, 12),  # n_envs, 4 * 3
                gates_dir.reshape(n_envs, 8),  # n_envs, 4 * 2
                obstacles_pos.reshape(n_envs, 8),  # 4 * 2, only x, y
                target_gate_one_hot,  # n_envs, 4
                corner_pos,  # n_envs, 4 * 3
                target_heading,  # n_envs, 2
                obs["gates_visited"].astype(jp.float32),
                obs["obstacles_visited"].astype(jp.float32),
                action,
            ),
            axis=-1,
        )
        return obs, {"target_gate": target_gate}


class RewardTF(gymnasium.vector.VectorWrapper):
    """A wrapper that provides a dense shaped reward for the drone racing environments."""

    def __init__(
        self,
        env: gymnasium.vector.VectorEnv,
        progress_weight: float = 10.0,
        crash_weight: float = -5.0,
        vel_weight: float = -0.5,
        finished_weight: float = 10.0,
        action_weight: float = -1e-4,
    ):
        super().__init__(env)
        assert len(self.action_space.shape) == 2
        self._last_pos = None
        self._last_action = jp.zeros(self.action_space.shape)
        self._reward_weights = {
            "progress": progress_weight,
            "crash": crash_weight,
            "velocity": vel_weight,
            "finished": finished_weight,
            "action_change": action_weight,
        }

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Array], dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_pos = obs["pos"]
        return obs, info

    def step(self, actions: Array) -> tuple[dict[str, Array], Array, Array, Array, dict]:
        obs, _, terminated, truncated, info = self.env.step(actions)
        reward = self._reward(
            obs, actions, terminated, self._last_pos, self._last_action, self._reward_weights
        )
        self._last_pos = obs["pos"]
        self._last_action = actions
        return obs, reward, terminated, truncated, info

    @staticmethod
    @jax.jit
    def _reward(
        obs: dict[str, Array],
        action: Array,
        terminated: Array,
        last_pos: Array,
        last_action: Array,
        reward_weights: dict[str, float],
    ) -> Array:
        # Scaling factors
        # Get target gate position from gates_pos by selecting the target gate for each environment
        N = obs["gates_pos"].shape[0]
        gate_pos = obs["gates_pos"][jp.arange(N), obs["target_gate"], :]
        gate_quat = obs["gates_quat"][jp.arange(N), obs["target_gate"], :]
        # Progress reward
        progress = RewardTF._progress(obs["pos"], gate_pos, gate_quat)
        last_progress = RewardTF._progress(last_pos, gate_pos, gate_quat)
        r_progress = reward_weights["progress"] * (progress - last_progress)
        # Avoid excessive negative reward on gate passing
        r_progress = jp.where(obs["target_gate"] == -1, 0, r_progress)
        # Velocity penalty
        vel = jp.linalg.norm(obs["vel"], axis=-1)
        vmax = 3.0
        r_vel = (vel - vmax) ** 2
        r_vel = reward_weights["velocity"] * jp.where(vel < vmax, 0, r_vel)
        # Finished reward. If we are not finished yet, we give a constant negative reward that is
        # smaller than the crash penalty propagated by gamma (1/(1-gamma) = 100 for gamma = 0.99),
        # 200 to not destroy the "will to live"
        r_finished = jp.where(
            obs["target_gate"] == -1, reward_weights["finished"], reward_weights["crash"] / 200
        )
        # Action penalty
        action_change = jp.linalg.norm(action - last_action, axis=-1)
        r_action = reward_weights["action_change"] * action_change
        # Crash penalty
        crash = terminated & (obs["target_gate"] != -1)
        r_crash = reward_weights["crash"] * crash.astype(jp.float32)
        assert r_progress.shape == r_action.shape
        assert r_progress.shape == r_crash.shape
        assert r_progress.shape == r_finished.shape
        reward = r_progress + r_crash + r_vel + r_finished + r_action
        return reward

    @staticmethod
    @partial(jp.vectorize, signature="(n),(n),(m)->()")
    def _progress(pos: Array, gate_pos: Array, gate_quat: Array) -> Array:
        gate_rot = JR.from_quat(gate_quat).as_matrix()
        gate_nx = gate_rot[:3, 0]
        gate_ny = gate_rot[:3, 1]  # Gate direction
        gate_nz = gate_rot[:3, 2]
        gate_width = 0.45
        op = pos - gate_pos  # Vector from plane origin to point
        # Project op onto normal vector for "height" (distance from plane)
        dist_to_plane = jp.dot(op, gate_ny)
        proj_on_plane = pos - dist_to_plane * gate_ny
        # Express the projection in the (x, z) coordinate system
        local = proj_on_plane - gate_pos
        x = jp.dot(local, gate_nx)
        z = jp.dot(local, gate_nz)
        # Clamp (x, z) to the bounds of the square plane
        x_clamped = jp.clip(x, -gate_width / 2, gate_width / 2)
        z_clamped = jp.clip(z, -gate_width / 2, gate_width / 2)
        # Closest point on square
        closest_pt = gate_pos + x_clamped * gate_nx + z_clamped * gate_nz
        closest_n = (p := pos - closest_pt) / jp.linalg.norm(p + 1e-6)
        angle = jp.acos(jp.clip(jp.dot(closest_n, gate_ny), -1, 1))
        angle_progress = 2 * (angle / jp.pi - 0.5)
        # Distance
        distance = jp.linalg.norm(pos - closest_pt)
        distance_progress = jp.exp(-2 * distance)
        gamma_angle = distance_progress
        gamma_distance = 1 - gamma_angle
        progress = gamma_angle * angle_progress + gamma_distance * distance_progress
        return progress


class TrackGate(Collector):
    def __init__(self, key: str):
        self._target_gate = None
        self._key = key
        self._xp = None

    def collect(self, **kwargs):
        if (info := kwargs.get("info")) is None:
            return
        if self._xp is None:
            self._xp = array_namespace(info["target_gate"])
        self._target_gate = info["target_gate"]

    def log(self, mask):
        if self._target_gate is None:
            return {}
        self._target_gate[self._target_gate == -1] = 4
        gates = self._xp.astype(self._target_gate[mask], self._xp.float32)
        return {self._key: float(self._xp.mean(gates))}

    def clear(self, mask):
        if self._target_gate is not None:
            self._target_gate[mask] = 0
