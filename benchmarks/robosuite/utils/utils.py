import os
from functools import partial
from typing import Callable, Literal, OrderedDict

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.parts.arm import OperationalSpaceController
from robosuite.environments.robot_env import RobotEnv
from robosuite.wrappers import GymWrapper
from scipy.spatial.transform import Rotation as R

from rotations.rotations import RotType, jax_rot_pow

from .actions import (
    euler_to_rel_rotation,
    matrix_to_rel_rotation,
    quat_plus_to_rel_rotation,
    quat_to_rel_rotation,
    r6_to_rel_rotation,
    rel_euler_to_rel_rotation,
    rel_matrix_to_rel_rotation,
    rel_quat_plus_to_rel_rotation,
    rel_quat_to_rel_rotation,
    rel_r6_to_rel_rotation,
    rel_tangent_to_rel_rotation,
    tangent_to_rel_rotation,
)

# Neccesary to prevent JAX from crashing the calling program due to CUDA
# running out of memory with gymnasium's async vector environments
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def load_manipulator_controller_config(keep_rot_scale: bool = False) -> dict:
    """Load the "BASIC" composite controller config for single or dual arm manipulators.

    WARNING:
        Unless keep_rot_scale is set, this function **removes** the scaling of input delta rotation vectors.
    
    Returns:
        Dictionary containing the composite controller's configuration.
    """
    # Keep the only needed config of the right arm to avoid warnings
    controller_config = load_composite_controller_config(controller="BASIC")
    right_arm_config = controller_config["body_parts"]["right"]

    # Override the rotation scaling parameters if neccesary
    if not keep_rot_scale:
        right_arm_config["input_min"] = [-1.0, -1.0, -1.0] + [-np.pi] * 3
        right_arm_config["input_max"] = [1.0, 1.0, 1.0] + [np.pi] * 3
        right_arm_config["output_min"] = right_arm_config["output_min"][:3] + [-np.pi] * 3
        right_arm_config["output_max"] = right_arm_config["output_max"][:3] + [np.pi] * 3
    controller_config["body_parts"] = {"right": right_arm_config}
    return controller_config


def split_pose_action(action: NDArray, rot_type: RotType) -> tuple[NDArray, NDArray, NDArray]:
    """Split a pose's concatenated action into its position, orientation and gripper (empty) components."""
    position = action[:3]
    orientation = action[3: 3 + rot_type.dim]
    gripper = np.zeros(0)
    return position, orientation, gripper


def split_manipulator_action(action: NDArray, rot_type: RotType) -> tuple[NDArray, NDArray, NDArray]:
    """Split a manipulator's concatenated action into its position, orientation and gripper components."""
    position = action[:3]
    orientation = action[3: 3 + rot_type.dim]
    gripper = action[-1:]
    return position, orientation, gripper


class RotationGymWrapper(GymWrapper):
    metadata = {}
    """
    A wrapper for Robosuite's environments that extends the functionality of the their 
    GymWrapper to support the various observation types, action types, control modes and 
    rotation limits supported in the rotations package.

    Args:
        env: Robosuite RobotEnv to wrap.
        action_type: Rotation type to use for actions.
        obs_type: Rotation type to use for observations.
        step_len: Maximum rotation at each step is limited to step_len * pi radians.
        control_mode: Control mode for end-effector orientations, one of ["rel", "abs", "rel_scale"].
    """
    def __init__(
        self, 
        env: RobotEnv,
        action_type: RotType = RotType.tangent,
        obs_type: RotType = RotType.quat,
        step_len: float = np.sqrt(3) * 0.5 / np.pi,
        control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    ) -> None:
        # Validate environment
        self._check_controller_configs(env)
        assert env.num_robots <= 2, f"Expected at most environment with 2 robots. Got {env.num_robots}" 

        # Filter observations to remove unwanted or invalid observations
        self.keys, keys_recover = self._filter_obs_keys(env)
        super().__init__(env, keys=self.keys, flatten_obs=False)

        # Recover needed observations that had to be discarded for parent's constructor
        obs_sample = env.reset()
        self.keys += keys_recover
        self.observation_space = spaces.Dict(
            self.observation_space.spaces | {k: spaces.Box(-np.inf, np.inf, obs_sample[k].shape) for k in keys_recover}
        )

        # Store rotations configs
        self.action_type = RotType(action_type)
        self.obs_type = RotType(obs_type)
        self.step_len = step_len
        self.control_mode = control_mode

        # Setup attributes needed for action conversion
        env_has_gripper = (env.action_dim == 7) or (env.action_dim == 14) # single or dual-arm 
        self.robot_dim = 3 + self.action_type.dim + 1 * env_has_gripper
        self.action_fn = self.build_action_fn(self.action_type, "rel" in self.control_mode)
        self.split_action_fn = split_manipulator_action if env_has_gripper else split_pose_action
        if self.control_mode == "rel_scale":
            self.action_fn = partial(self.action_fn, scale=True)

        # Pre-determine the keys of quaternion observations in the observation dict
        self.quat_keys = self._get_quat_keys(self.keys)

        # Setup observation and action spaces
        for q_key in self.quat_keys:
            assert q_key in self.observation_space.spaces
            self.observation_space.spaces[q_key] = spaces.Box(-np.inf, np.inf, (self.obs_type.dim,))
        self.observation_space = spaces.flatten_space(self.observation_space)
        self.observation_space.dtype = np.dtype(np.float32)
        self.action_space = spaces.Box(-1, 1, (self.robot_dim * env.num_robots,), dtype=np.float32) 

    def reset(self, seed: int | None = None, options=None) -> tuple[NDArray, dict]:
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        obs = self._obs(ob_dict)
        return obs, {} 

    def step(self, action: NDArray) -> tuple[NDArray, float, bool, bool, dict]:
        # IMPORTANT 
        # 
        # We reverse the logic used by the parent's step function for the terminated and truncated 
        # flags. The parent incorrectly treats truncation as termination, while Robosuite's environments 
        # never terminate as per https://robosuite.ai/docs/modules/environments.html#rewards-and-termination.
        # 
        # The returned flag is for truncation as computed here: 
        # https://github.com/ARISE-Initiative/robosuite/blob/75a4c9f4d242c1b7fe7c7fc247b564ec5d8550a2/robosuite/environments/base.py#L506
        action = self._action(action)
        ob_dict, reward, truncated, info = self.env.step(action)
        obs = self._obs(ob_dict)
        return obs, reward, False, truncated, info
    
    @property
    def osc_r1(self) -> OperationalSpaceController:
        return self.env.robots[0].composite_controller.part_controllers["right"]
    
    @property
    def osc_r2(self) -> OperationalSpaceController | None:
        return self.env.robots[1].composite_controller.part_controllers["right"] \
            if self.env.num_robots == 2 else None
    
    @staticmethod
    def _check_controller_configs(env: RobotEnv) -> None:
        """Verifies that all robots are using the composite controller configs assumed by the wrapper."""
        expected_config = load_manipulator_controller_config(keep_rot_scale=False)
        for i, robot in enumerate(env.robots):
            assert robot.composite_controller_config == expected_config, \
                f"Robot{i}'s composite controller config does not match the expected config."
    
    @staticmethod
    def _filter_obs_keys(env: RobotEnv) -> tuple[list[str], list[str]]:
        """Filter the observation keys to remove unwanted or invalid observations.
        
        Args:
            env: Robosuite RoboEnv to be wrapped.
        
        Returns:
            tuple
            - keys: Keys of observations after filtering.
            - keys_recover: Keys of observations to recover after parent's initialization.
        """
        # Get the common observation keys between _observables and _get_observations.
        # Observables can contain unused observations (e.g. PickPlaceCan). Get observations 
        # has the combined robot proprio-state and object-state, which are also not needed.
        obs = env.reset()
        keys = [k for k in env._observables.keys() if k in obs.keys()]

        # Remove joint_pos and eef_quat observations. joint_pos is removed since it is 
        # already represented by its sin and cos. For an explanation why eef_quat is 
        # removed and eef_quat_site is used instead, check the following link:
        # https://github.com/ARISE-Initiative/robosuite/blob/75a4c9f4d242c1b7fe7c7fc247b564ec5d8550a2/robosuite/robots/robot.py#L415 
        discard = [f"robot{i}_joint_pos" for i in range(env.num_robots)]
        discard += [f"robot{i}_eef_quat" for i in range(env.num_robots)]
        
        # Remove camera image observations as they're only used for rendering
        discard += [k for k in keys if "image" in k] # Filter image observations 
        
        # Remove observations that trigger exceptions in the parent's constructor due 
        # to their dtype (although they can be returned by their own environments!)
        keys_recover = [k for k in keys if not (np.issubdtype(obs[k].dtype, np.integer) or \
                                                np.issubdtype(obs[k].dtype, np.inexact))]
        discard += keys_recover
        for d_key in discard:
            keys.remove(d_key)
        return keys, keys_recover
    
    @staticmethod
    def _get_quat_keys(obs_keys: list[str]) -> list[str]:
        """Determine the keys for all stored quaternions in the observation dict."""
        quat_keys = []
        for key in obs_keys:
            if "quat" in key:
                quat_keys.append(key)
        return quat_keys
    
    @staticmethod
    def build_action_fn(action_type: RotType, rel: bool = True) -> Callable[[R, NDArray, float], R]:
        match action_type:
            case RotType.euler:
                return rel_euler_to_rel_rotation if rel else euler_to_rel_rotation
            case RotType.tangent:
                return rel_tangent_to_rel_rotation if rel else tangent_to_rel_rotation
            case RotType.quat_plus:
                return rel_quat_plus_to_rel_rotation if rel else quat_plus_to_rel_rotation
            case RotType.matrix:
                return rel_matrix_to_rel_rotation if rel else matrix_to_rel_rotation
            case RotType.r6:
                return rel_r6_to_rel_rotation if rel else r6_to_rel_rotation
            case RotType.quat:
                return rel_quat_to_rel_rotation if rel else quat_to_rel_rotation
            case _:
                raise ValueError(f"Invalid action type {action_type}")
    
    def _obs(self, obs_dict: OrderedDict[str, NDArray]) -> NDArray:
        """
        Converts orientations in observations from quaternions to observation type, 
        discards unwanted observations and flattens observations into an array.
        """
        for q_key in self.quat_keys:
            # Add small epsilon to real component as Robosuite can return zero quaternions
            q = obs_dict[q_key] + np.array([0, 0, 0, 1e-8])
            obs_dict[q_key] = self.obs_type.as_array(R.from_quat(q))
        # Robosuite environments can return 0-dimensional observations
        return np.concatenate([np.atleast_1d(obs_dict[key]) for key in self.keys], axis=0, dtype=np.float32)
            
    def _action(self, action: NDArray) -> NDArray:
        """
        Enforces action limits and converts actions from their chosen representation 
        to the relative tangent representation used by Robosuite's OSC controller. 
        The OSC controller uses relative tangent actions whose rotational axes are
        expressed in the fixed base frame.
        """
        # Convert the first robot's action
        pos, ori, grp = self.split_action_fn(action[:self.robot_dim], self.action_type)
        rot = R.from_matrix(self.osc_r1.goal_origin_to_eef_pose()[:3, :3])
        ori = self._action_robot(rot, ori)
        action_r1 = np.concatenate((pos, ori, grp))

        # Convert the second robot's action if it exists
        if self.env.num_robots == 2:
            pos, ori, grp = self.split_action_fn(action[self.robot_dim:], self.action_type)
            rot = R.from_matrix(self.osc_r2.goal_origin_to_eef_pose()[:3, :3])
            ori = self._action_robot(rot, ori)
            action_r2 = np.concatenate((pos, ori, grp))
        else:
            action_r2 = np.zeros(0)
        return np.concatenate((action_r1, action_r2))

    def _action_robot(self, rot: R, action: NDArray) -> NDArray:
        # Convert from action to delta rotation
        drot = self.action_fn(rot, action, self.step_len)
        
        # Contrain the maximum delta rotation to step_len * pi
        drot_mag = drot.magnitude()
        scale = np.minimum(1, self.step_len * np.pi / (drot_mag + 1e-8))
        drot = jax_rot_pow(drot, scale)

        # Convert from delta rotation w.r.t. body frame axes to fixed frame 
        drot = rot * drot * rot.inv()
        return drot.as_rotvec()