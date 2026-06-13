import torch
from gymnasium.spaces import Box, Dict
from lsy_rl.sac.policy import SACActorNet, SACCriticNet
from lsy_rl.utils import polyak_update_
from tensordict import TensorDict
from torch import Tensor, nn

import rotations.modules.sac.activations as acts


class GoalActor(nn.Module):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__()
        assert isinstance(obs_space, Dict), f"Invalid obs space type {type(obs_space)}"
        assert isinstance(action_space, Box), f"Invalid action space type {type(action_space)}"
        obs_dim = obs_space["observation"].shape[1] + obs_space["desired_goal"].shape[1]
        obs_shape = (obs_dim,)
        action_shape = action_space.shape[1:]  # Remove num_envs dimension
        self.network = SACActorNet(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()

    def action(self, obs: TensorDict) -> tuple[Tensor, Tensor, Tensor]:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        # Concatenate observations into a torch tensor
        obs_tensor = torch.cat([obs["observation"], obs["desired_goal"]], dim=-1)

        # Execute regular SAC action method
        mean, logstd = self.network(obs_tensor)
        normal = torch.distributions.Normal(mean, logstd.exp())
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = self.squash_layer(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1.0 * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = self.squash_layer(mean)
        return action, log_prob, mean

    def mean_action(self, obs: TensorDict) -> Tensor:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        # Concatenate observations and execute regular mean_action method
        x = torch.cat([obs["observation"], obs["desired_goal"]], dim=-1)
        for layer in self.network.shared_layers.values():
            x = layer(x)
        for layer in self.network.mean_head.values():
            x = layer(x)
        return self.squash_layer(x)


class GoalCritic(nn.Module):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__()
        assert isinstance(obs_space, Dict), f"Invalid obs space type {type(obs_space)}"
        assert isinstance(action_space, Box), f"Invalid action space type {type(action_space)}"
        obs_dim = obs_space["observation"].shape[1] + obs_space["desired_goal"].shape[1]
        obs_shape = (obs_dim,)
        action_shape = action_space.shape[1:]  # Remove num_envs dimension
        self.q1 = SACCriticNet(obs_shape, action_shape)
        self.q2 = SACCriticNet(obs_shape, action_shape)
        self.q1_target = SACCriticNet(obs_shape, action_shape)
        for param in self.q1_target.parameters():  # Freeze target network parameters
            param.requires_grad = False
        self.q2_target = SACCriticNet(obs_shape, action_shape)
        for param in self.q2_target.parameters():  # Freeze target network parameters
            param.requires_grad = False
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def values(self, obs: TensorDict, action: Tensor) -> tuple[Tensor, Tensor]:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        x = torch.cat([obs["observation"], obs["desired_goal"], action], dim=-1)
        return self.q1(x), self.q2(x)

    def actor_value(self, obs: TensorDict, action: Tensor) -> Tensor:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        x = torch.cat([obs["observation"], obs["desired_goal"], action], dim=-1)
        return torch.minimum(self.q1(x), self.q2(x))

    def target(self, obs: TensorDict, action: Tensor) -> Tensor:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        x = torch.cat([obs["observation"], obs["desired_goal"], action], dim=-1)
        return torch.minimum(self.q1_target(x), self.q2_target(x))

    def update_target(self, tau: float) -> None:
        polyak_update_(self.q1_target, self.q1, tau)
        polyak_update_(self.q2_target, self.q2, tau)


# Actors for orientation control
class QuatActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class QuatPlusActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = acts.TanHQuatPlus()


class EulerActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class MatrixActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class R6Actor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class TangentActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


# Actors for pose control
class QuatPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class QuatPlusPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = acts.QuatPlusPose()


class EulerPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class MatrixPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class R6PoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class TangentPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


# Actors for dual pose control
class QuatDualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class QuatPlusDualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = acts.QuatPlusDualPose()


class EulerDualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class MatrixDualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class R6DualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class TangentDualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


# Actors for manipulator control (pose + gripper)
class QuatManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class EulerManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class MatrixManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class R6ManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class TangentManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


# Actors for dual manipulator control
class QuatDualManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class EulerDualManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class MatrixDualManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class R6DualManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


class TangentDualManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space, action_space)
        self.squash_layer = nn.Tanh()


actors = {
    "euler": EulerActor,
    "tangent": TangentActor,
    "quat": QuatActor,
    "quat_plus": QuatPlusActor,
    "r6": R6Actor,
    "matrix": MatrixActor,
}

critics = {
    "euler": GoalCritic,
    "tangent": GoalCritic,
    "quat": GoalCritic,
    "quat_plus": GoalCritic,
    "r6": GoalCritic,
    "matrix": GoalCritic,
}

pose_actors = {
    "quat": QuatPoseActor,
    "quat_plus": QuatPlusPoseActor,
    "matrix": MatrixPoseActor,
    "r6": R6PoseActor,
    "euler": EulerPoseActor,
    "tangent": TangentPoseActor,
}

dual_pose_actors = {
    "quat": QuatDualPoseActor,
    "quat_plus": QuatPlusDualPoseActor,
    "matrix": MatrixDualPoseActor,
    "r6": R6DualPoseActor,
    "euler": EulerDualPoseActor,
    "tangent": TangentDualPoseActor,
}

manipulator_actors = {
    "quat": QuatManipulatorActor,
    "matrix": MatrixManipulatorActor,
    "r6": R6ManipulatorActor,
    "euler": EulerManipulatorActor,
    "tangent": TangentManipulatorActor,
}

dual_manipulator_actors = {
    "quat": QuatDualManipulatorActor,
    "matrix": MatrixDualManipulatorActor,
    "r6": R6DualManipulatorActor,
    "euler": EulerDualManipulatorActor,
    "tangent": TangentDualManipulatorActor,
}
