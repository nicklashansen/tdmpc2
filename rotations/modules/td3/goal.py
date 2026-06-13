import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict
from lsy_rl.td3.policy import DDPGCriticNetwork
from lsy_rl.utils import polyak_update_
from tensordict import TensorDict
from torch import Tensor

from rotations.modules.ddpg import (
    actors,
    dual_manipulator_actors,
    dual_pose_actors,
    manipulator_actors,
    pose_actors,
)


class GoalCritic(nn.Module):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__()
        assert isinstance(obs_space, Dict), f"Invalid obs space type {type(obs_space)}"
        assert isinstance(action_space, Box), f"Invalid action space type {type(action_space)}"
        obs_dim = obs_space["observation"].shape[1] + obs_space["desired_goal"].shape[1]
        action_dim = action_space.shape[1]  # Remove num_envs dimension
        self.q1 = DDPGCriticNetwork(obs_dim + action_dim)
        self.q2 = DDPGCriticNetwork(obs_dim + action_dim)
        # Initialize the target network and synchronize the weights
        self.q1_target = DDPGCriticNetwork(obs_dim + action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target = DDPGCriticNetwork(obs_dim + action_dim)
        self.q2_target.load_state_dict(self.q2.state_dict())

    def values(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([obs["observation"], obs["desired_goal"], action], dim=-1)
        return self.q1(x), self.q2(x)

    def actor_value(self, obs: TensorDict, action: Tensor) -> Tensor:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        x = torch.cat([obs["observation"], obs["desired_goal"], action], dim=-1)
        return self.q1(x)

    def target(self, obs: TensorDict, action: Tensor) -> Tensor:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        assert action.dtype == torch.float32, f"Invalid dtype {action.dtype}"
        x = torch.cat([obs["observation"], obs["desired_goal"], action], dim=-1)
        return torch.minimum(self.q1_target(x), self.q2_target(x))

    def update_target(self, tau: float):
        polyak_update_(self.q1_target, self.q1, tau)
        polyak_update_(self.q2_target, self.q2, tau)


# TD3's actors are imported from DDPG since they're identical
critics = {
    "quat": GoalCritic,
    "quat_exp": GoalCritic,
    "quat_plus": GoalCritic,
    "quat_plus_exp": GoalCritic,
    "matrix": GoalCritic,
    "matrix_exp": GoalCritic,
    "r6": GoalCritic,
    "r6_exp": GoalCritic,
    "euler": GoalCritic,
    "euler_add": GoalCritic,
    "tangent": GoalCritic,
    "tangent_riemann": GoalCritic,
}

__all__ = [
    "actors",
    "manipulator_actors",
    "dual_manipulator_actors",
    "pose_actors",
    "dual_pose_actors",
    "critics",
]