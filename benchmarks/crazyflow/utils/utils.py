from functools import partial
from typing import Any, Callable, Literal

import gymnasium
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.control import Control
from crazyflow.envs.drone_env import DroneEnv
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from jax import Array
from jax.scipy.spatial.transform import Rotation as JR

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


class RotationCrazyflowWrapper(gymnasium.vector.VectorWrapper):
    """
    A wrapper for Crazyflow environments that allows the usage of the various observation
    types, action types, control modes and rotation limits supported in the rotations package.

    Note: the wrapper alters the observation type from Dict to Box as it flattens observations
    after the conversion of orientations from quaternions to the desired observation type.

    Args:
        env: Crazyflow DroneEnv to wrap.
        action_type: Rotation type to use for actions.
        obs_type: Rotation type to use for observations.
        step_len: Maximum rotation at each step is limited to step_len * pi radians.
        control_mode: Control mode for drone orientations, one of ["rel", "abs", "rel_scale"].
    """

    def __init__(
        self,
        env: DroneEnv,
        action_type: RotType = RotType.tangent,
        obs_type: RotType = RotType.quat,
        step_len: float = 0.1,
        control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    ) -> None:
        assert env.sim.control == Control.attitude, (
            f"Wrapper only supports attitude control mode, got {env.sim.control}."
        )
        super().__init__(env)

        # Store rotations configs
        self.action_type = RotType(action_type)
        self.obs_type = RotType(obs_type)
        self.step_len = step_len
        assert control_mode in ["rel", "abs", "rel_scale"], (
            f"Invalid control mode {control_mode}, must be one of ['rel', 'abs', 'rel_scale']"
        )
        self.control_mode = control_mode

        # Setup jax functions for observation and action conversions
        self._obs = self.build_obs_fn()
        self._action = self.build_action_fn()

        # Setup observation space
        self.single_observation_space["quat"] = spaces.Box(-np.inf, np.inf, (self.obs_type.dim,))
        self.single_observation_space = spaces.flatten_space(self.single_observation_space)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        # Setup action space
        thrust_low, thrust_high = (
            self.single_action_space.low[:1],
            self.single_action_space.high[:1],
        )
        low = np.concatenate([thrust_low, -np.ones(self.action_type.dim)], dtype=np.float32)
        low[-1] *= self.action_type != RotType.quat_plus  # quat_plus has a min of 0 for w
        high = np.concatenate([thrust_high, np.ones(self.action_type.dim)], dtype=np.float32)
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(self, *, seed=None, options=None) -> tuple[Array, dict[str, Any]]:
        observations, infos = super().reset(seed=seed, options=options)
        return self._obs(observations), infos

    def step(self, actions) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
        actions = self._action(actions, self.rotation)
        observations, rewards, terminations, truncations, infos = super().step(actions)
        return self._obs(observations), rewards, terminations, truncations, infos

    def build_obs_fn(self):
        to_array_fn = self.obs_type.as_array

        @jax.jit
        def obs(obs: dict[str, Array]) -> Array:
            obs["quat"] = to_array_fn(JR.from_quat(obs["quat"]))
            return jp.concat([v for v in obs.values()], axis=-1)

        return obs

    def build_action_fn(self):
        step_len = self.step_len
        action_tf = self.get_action_tf(self.action_type, "rel" in self.control_mode)
        if self.control_mode == "rel_scale":
            action_tf = partial(action_tf, scale=True)

        @jax.jit
        def action(action: Array, rot: JR) -> Array:
            """Convert chosen action representation to the environment's representation
            (absolute Euler). Function also enforces the set rotation limits."""
            rot_action = action[:, 1:]
            rot_desired = action_tf(rot, rot_action, step_len)
            rot_desired = rotation_dynamics(rot, rot_desired, step_len)
            rot_action_env = rot_desired.as_euler("xyz", degrees=False)
            return jp.concat([action[:, :1], rot_action_env], axis=-1)

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
