import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict
from lsy_rl.ddpg.policy import DDPGActorNetwork, DDPGCriticNetwork
from lsy_rl.utils import polyak_update_
from tensordict import TensorDict
from torch import Tensor

import rotations.modules.ddpg.activations as acts


class GoalActor(nn.Module):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__()
        assert isinstance(obs_space, Dict), f"Invalid obs space type {type(obs_space)}"
        assert isinstance(action_space, Box), (
            f"Invalid action space type {type(action_space)}"
        )
        obs_dim = obs_space["observation"].shape[1] + obs_space["desired_goal"].shape[1]
        action_dim = action_space.shape[1]  # Remove num_envs dimension
        self.network = DDPGActorNetwork(obs_dim, action_dim)
        # Initialize the target network and synchronize the weights
        self.target_network = DDPGActorNetwork(obs_dim, action_dim)
        self.target_network.load_state_dict(self.network.state_dict())

    def forward(self, obs: TensorDict) -> Tensor:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        return self.network(
            torch.cat([obs["observation"], obs["desired_goal"]], dim=-1)
        )

    def target(self, obs: TensorDict) -> Tensor:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        return self.target_network(
            torch.cat([obs["observation"], obs["desired_goal"]], dim=-1)
        )

    def update_target(self, tau: float) -> None:
        """Update the target network with the current weights."""
        polyak_update_(self.target_network, self.network, tau)


class GoalCritic(nn.Module):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__()
        assert isinstance(obs_space, Dict), f"Invalid obs space type {type(obs_space)}"
        assert isinstance(action_space, Box), (
            f"Invalid action space type {type(action_space)}"
        )
        obs_dim = obs_space["observation"].shape[1] + obs_space["desired_goal"].shape[1]
        action_dim = action_space.shape[1]  # Remove num_envs dimension
        self.network = DDPGCriticNetwork(obs_dim + action_dim)
        # Initialize the target network and synchronize the weights
        self.target_network = DDPGCriticNetwork(obs_dim + action_dim)
        self.target_network.load_state_dict(self.network.state_dict())

    def forward(self, obs: TensorDict, action: Tensor) -> Tensor:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        return self.network(
            torch.cat([obs["observation"], obs["desired_goal"], action], dim=-1)
        )

    def target(self, obs: TensorDict, action: Tensor) -> Tensor:
        assert isinstance(obs, TensorDict), f"Invalid obs type {type(obs)}"
        return self.target_network(
            torch.cat([obs["observation"], obs["desired_goal"], action], dim=-1)
        )

    def update_target(self, tau: float) -> None:
        polyak_update_(self.target_network, self.network, tau)


# Actors for orientation control
class QuatActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.TanHQuat()})
        self.target_network.network.update({"f_output": acts.TanHQuat()})


class QuatExpActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        d = self.network.network["output"].in_features
        self.network.network.update(
            {"output": nn.Linear(d, 3), "f_output": acts.QuatExp()}
        )
        self.target_network.network.update(
            {"output": nn.Linear(d, 3), "f_output": acts.QuatExp()}
        )
        # Reload after adding new output layer
        self.target_network.load_state_dict(self.network.state_dict())


class QuatPlusActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.TanHQuatPlus()})
        self.target_network.network.update({"f_output": acts.TanHQuatPlus()})


class QuatPlusExpActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        d = self.network.network["output"].in_features
        self.network.network.update(
            {"output": nn.Linear(d, 3), "f_output": acts.QuatPlusExp()}
        )
        self.target_network.network.update(
            {"output": nn.Linear(d, 3), "f_output": acts.QuatPlusExp()}
        )
        # Reload after adding new output layer
        self.target_network.load_state_dict(self.network.state_dict())


class EulerActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class MatrixActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.TanHMatrix()})
        self.target_network.network.update({"f_output": acts.TanHMatrix()})


class MatrixExpActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        torch.autograd.set_detect_anomaly(
            True
        )  # MatrixExpActor is currently causing NaNs
        super().__init__(obs_space=obs_space, action_space=action_space)
        d = self.network.network["output"].in_features
        self.network.network.update(
            {"output": nn.Linear(d, 3), "f_output": acts.MatrixExp()}
        )
        self.target_network.network.update(
            {"output": nn.Linear(d, 3), "f_output": acts.MatrixExp()}
        )
        # Reload after adding new output layer
        self.target_network.load_state_dict(self.network.state_dict())


class R6Actor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        assert isinstance(obs_space, Dict), f"Invalid obs space type {type(obs_space)}"
        assert isinstance(action_space, Box), (
            f"Invalid action space type {type(action_space)}"
        )
        self.network.network.update({"f_output": acts.TanHR6Ortho()})
        self.target_network.network.update({"f_output": acts.TanHR6Ortho()})


class R6ExpActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        torch.autograd.set_detect_anomaly(
            True
        )  # R6 is also using matrix_exp, so better check
        super().__init__(obs_space=obs_space, action_space=action_space)
        d = self.network.network["output"].in_features
        self.network.network.update(
            {"output": nn.Linear(d, 3), "f_output": acts.R6Exp()}
        )
        self.target_network.network.update(
            {"output": nn.Linear(d, 3), "f_output": acts.R6Exp()}
        )
        # Reload after adding new output layer
        self.target_network.load_state_dict(self.network.state_dict())


class TangentActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


# Actors for pose control
class QuatPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatPose()})
        self.target_network.network.update({"f_output": acts.QuatPose()})


class QuatPlusPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatPlusPose()})
        self.target_network.network.update({"f_output": acts.QuatPlusPose()})


class EulerPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class MatrixPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.MatrixPose()})
        self.target_network.network.update({"f_output": acts.MatrixPose()})


class R6PoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.R6Pose()})
        self.target_network.network.update({"f_output": acts.R6Pose()})


class TangentPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


# Actors for dual pose control
class QuatDualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatDualPose()})
        self.target_network.network.update({"f_output": acts.QuatDualPose()})


class QuatPlusDualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatPlusDualPose()})
        self.target_network.network.update({"f_output": acts.QuatPlusDualPose()})


class EulerDualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class MatrixDualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.MatrixDualPose()})
        self.target_network.network.update({"f_output": acts.MatrixDualPose()})


class R6DualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.R6DualPose()})
        self.target_network.network.update({"f_output": acts.R6DualPose()})


class TangentDualPoseActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


# Actors for manipulator control (pose + gripper)
class QuatManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatManipulator()})
        self.target_network.network.update({"f_output": acts.QuatManipulator()})


class EulerManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class MatrixManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.MatrixManipulator()})
        self.target_network.network.update({"f_output": acts.MatrixManipulator()})


class TangentManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class R6ManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.R6Manipulator()})
        self.target_network.network.update({"f_output": acts.R6Manipulator()})


# Actors for dual manipulator control
class QuatDualManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatDualManipulator()})
        self.target_network.network.update({"f_output": acts.QuatDualManipulator()})


class EulerDualManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class MatrixDualManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.MatrixDualManipulator()})
        self.target_network.network.update({"f_output": acts.MatrixDualManipulator()})


class TangentDualManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class R6DualManipulatorActor(GoalActor):
    def __init__(self, obs_space: Dict, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.R6DualManipulator()})
        self.target_network.network.update({"f_output": acts.R6DualManipulator()})


actors = {
    "quat": QuatActor,
    "quat_exp": QuatExpActor,
    "quat_plus": QuatPlusActor,
    "quat_plus_exp": QuatPlusExpActor,
    "matrix": MatrixActor,
    "matrix_exp": MatrixExpActor,
    "r6": R6Actor,
    "r6_exp": R6ExpActor,
    "euler": EulerActor,
    "euler_add": EulerActor,
    "tangent": TangentActor,
    "tangent_riemann": TangentActor,
}

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
