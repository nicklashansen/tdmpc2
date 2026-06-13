from __future__ import annotations

from functools import partial
from typing import Callable, Literal

import gymnasium
import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
from flax.struct import dataclass, field
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
from jax import Array, Device
from jax.scipy.spatial.transform import Rotation as JR
from scipy.spatial.transform import Rotation as R

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
from rotations.rotations import RotType, angle, jax_rot_pow


def action_space(action_type: RotType) -> spaces.Box:
    (amin, amax) = (-1, 1)
    if action_type == RotType.quat_plus:
        (amin, amax) = (np.array([-1, -1, -1, 0]), np.array([1, 1, 1, 1]))
    return gymnasium.spaces.Box(amin, amax, shape=(action_type.dim,))


def observation_space(obs_type: RotType) -> spaces.Dict:
    shape = (obs_type.dim,)
    return spaces.Dict(
        {
            "observation": gymnasium.spaces.Box(-6, 6, shape=shape, dtype=np.float32),
            "desired_goal": gymnasium.spaces.Box(-6, 6, shape=shape, dtype=np.float32),
            "achieved_goal": gymnasium.spaces.Box(-6, 6, shape=shape, dtype=np.float32),
        }
    )


@partial(jax.jit, static_argnames=["step_len"])
def rotation_dynamics(rot: JR, des: JR, step_len: float) -> JR:
    """Rotation dynamics.

    Rotations are limited to at most step_len * pi radians. Actions are assumed to be delta
    rotations with respect to the current rotation.

    Warning:
        DOESN'T rescale the action space to only produce rotations in the allowed range, but does
        enforce this in the dynamics.
    """
    delta = rot.inv() * des
    delta_mag = JR.magnitude(delta)
    scale = jp.minimum(1, step_len * np.pi / delta_mag)[..., None]
    return rot * jax_rot_pow(delta, scale)


@dataclass
class RotData:
    q: Array
    q_goal: Array
    tol: float = field(pytree_node=False)
    step_len: float = field(pytree_node=False)

    max_episode_steps: int = field(pytree_node=False)
    elapsed_steps: Array
    rng_key: Array

    @classmethod
    def create(
        cls,
        n_envs: int,
        tol: float,
        step_len: float,
        max_episode_steps: int,
        seed: int,
        device: Device,
    ):
        q = jp.tile(jp.array([0, 0, 0, 1.0], device=device), (n_envs, 1))
        q_goal = jp.tile(jp.array([0, 0, 0, 1.0], device=device), (n_envs, 1))
        elapsed_steps = jp.zeros(n_envs, dtype=jp.int32)
        rng_key = jax.device_put(jax.random.key(seed), device)
        return cls(q, q_goal, tol, step_len, max_episode_steps, elapsed_steps, rng_key)


class RotEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(
        self,
        num_envs: int,
        tol: float = 0.1,
        action_type: RotType = RotType.tangent,
        obs_type: RotType = RotType.quat,
        reward_type: Literal["sparse", "dense"] = "sparse",
        step_len: float = 0.1,
        max_steps: int = 50,
        seed: int = 0,
        device: str = "cpu",
        control_mode: Literal["rel", "abs", "rel_scale"] = "rel",
    ):
        super().__init__()
        self.num_envs = num_envs
        self.action_type = RotType(action_type)
        self.obs_type = RotType(obs_type)
        self.reward_type = reward_type
        self.control_mode = control_mode
        self.device = jax.devices(device)[0]
        assert reward_type in ["sparse", "dense"], "Invalid reward type"

        self.single_action_space = action_space(self.action_type)
        self.single_observation_space = observation_space(self.obs_type)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.data = RotData.create(num_envs, tol, step_len, max_steps, seed, self.device)
        self._step = self.build_step_fn()
        self._obs = self.build_obs_fn()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self.data = self.data.replace(rng_key=jax.device_put(jax.random.key(seed), self.device))
        if options is None or options.get("reset_data", True):
            self.data = self._reset(self.data)
        return self.obs(), self.info()

    def step(self, action: Array) -> tuple[dict, float, bool, bool, dict]:
        self.data = self._step(self.data, action)
        reward = self.compute_reward(JR(self.data.q), JR(self.data.q_goal))
        return self.obs(), reward, self.terminated(), self.truncated(), self.info()

    def build_step_fn(self) -> Callable[[RotData, Array], RotData]:
        action_fn = self.build_action_fn(self.action_type, "rel" in self.control_mode)
        if self.control_mode == "rel_scale":
            action_fn = partial(action_fn, scale=True)

        @jax.jit
        def step(data: RotData, action: Array) -> RotData:
            assert action.ndim == 2, f"Action must be 2D, got {action.ndim}"
            assert action.shape[0] == data.q.shape[0], (
                f"Action shape mismatch {action.shape[0]} != {data.q.shape[0]}"
            )
            elapsed_steps = data.elapsed_steps + 1
            desired = action_fn(JR(data.q), action, data.step_len)
            next_orient = rotation_dynamics(JR(data.q), desired, data.step_len)
            data = data.replace(elapsed_steps=elapsed_steps, q=next_orient.as_quat())
            data = RotEnv._reset(data, elapsed_steps > data.max_episode_steps)
            return data

        return step

    @staticmethod
    def _step(data: RotData, action: Array) -> RotData:
        raise NotImplementedError("Called step() before function was built")

    @staticmethod
    @jax.jit
    def _reset(data: RotData, mask: Array | None = None) -> RotData:
        n_envs = len(data.elapsed_steps)
        mask = jp.ones(n_envs, bool) if mask is None else mask
        key1, key2, key3 = jax.random.split(data.rng_key, 3)
        q_rnd = (_q := jax.random.normal(key1, (n_envs, 4))) / jp.linalg.norm(_q, axis=-1)[:, None]
        q = jp.where(mask[:, None], q_rnd, data.q)
        q_rnd = (_q := jax.random.normal(key2, (n_envs, 4))) / jp.linalg.norm(_q, axis=-1)[:, None]
        q_goal = jp.where(mask[:, None], q_rnd, data.q_goal)
        elapsed_steps = jp.where(mask, 0, data.elapsed_steps)
        data = data.replace(q=q, q_goal=q_goal, elapsed_steps=elapsed_steps, rng_key=key3)
        return data

    @staticmethod
    def build_action_fn(action_type: RotType, rel: bool = True) -> Callable[[JR, Array, float], JR]:
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

    def obs(self) -> dict[str, Array]:
        orient, goal = self._obs(self.data)
        return {"observation": orient, "desired_goal": goal, "achieved_goal": orient}

    def build_obs_fn(self):
        to_array_fn = self.obs_type.as_array

        @jax.jit
        def obs(data: RotData) -> tuple[Array, Array]:
            return to_array_fn(JR.from_quat(data.q)), to_array_fn(JR.from_quat(data.q_goal))

        return obs

    @staticmethod
    def _obs(data: RotData) -> tuple[Array, Array]:
        raise NotImplementedError("Called obs() before function was built")

    def truncated(self) -> Array:
        return self.data.elapsed_steps >= self.data.max_episode_steps

    def terminated(self) -> Array:
        return jp.zeros(self.num_envs, dtype=bool, device=self.device)

    def info(self) -> dict:
        return {"angle": angle(JR(self.data.q), JR(self.data.q_goal), None)}

    def compute_reward(self, achieved_goal, desired_goal):
        return self.reward_fn(angle(achieved_goal, desired_goal, self.obs_type))

    def reward_fn(self, angle: Array) -> Array:
        if self.reward_type == "sparse":
            return -((angle > self.data.tol) * 1.0)
        return -(angle / np.pi)

    def render(self):
        if self.render_mode == "ansi":
            print(f"State: {self.R.as_euler('xyz')}, goal: {self.goal.as_euler('xyz')}")
        elif self.render_mode == "human":
            if self._render_state.get("fig") is None:
                self._setup_figure()
            fig, ax = self._render_state["fig"], self._render_state["ax"]
            ax.clear()
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_zlim(-1.5, 1.5)
            ax.set_box_aspect([1, 1, 1])
            ax.grid(False)
            ax.set_axis_off()

            self._render_state["obs"].append(self.R.as_matrix())
            for mat, alpha in zip(
                self._render_state["obs"], np.linspace(0, 1, len(self._render_state["obs"])) ** 4
            ):
                for color, x in zip("rgb", mat):
                    ax.plot([0, x[0]], [0, x[1]], [0, x[2]], color=color, linewidth=4, alpha=alpha)
            for color, x in zip("rgb", self.goal.as_matrix()):
                ax.plot([0, x[0]], [0, x[1]], [0, x[2]], color=color, linestyle="--", linewidth=4)
            fig.canvas.flush_events()

    def _setup_figure(self):
        fig = plt.figure(figsize=(1920 / 300, 1080 / 300), dpi=300)
        ax = fig.add_subplot(111, projection="3d")
        self._render_state["fig"] = fig
        self._render_state["ax"] = ax
        fig.tight_layout()
        fig.subplots_adjust(top=2, bottom=-1)
        fig.text(0.07, 0.7, "State", ha="left", va="top", fontsize=12)
        fig.text(0.07, 0.4, "Goal ", ha="left", va="top", fontsize=12)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_position([0.12, 0.57, 0.2, 0.2])
        ax.view_init(elev=30, azim=45, roll=15)
        ax.grid(False)
        ax.set_axis_off()
        for color, x in zip("rgb", R.identity().as_matrix()):
            ax.plot([0, x[0]], [0, x[1]], [0, x[2]], color=color, linewidth=3, alpha=1)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_position([0.12, 0.27, 0.2, 0.2])
        ax.view_init(elev=30, azim=45, roll=15)
        ax.grid(False)
        ax.set_axis_off()
        for color, x in zip("rgb", R.identity().as_matrix()):
            ax.plot(
                [0, x[0]], [0, x[1]], [0, x[2]], color=color, linewidth=3, alpha=1, linestyle="--"
            )
        plt.ion()
        fig.show()
