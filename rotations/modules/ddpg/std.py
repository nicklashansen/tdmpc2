import torch
import torch.nn as nn
from gymnasium.spaces import Box
from lsy_rl.ddpg.policy import DDPGActor, DDPGCritic

import rotations.modules.ddpg.activations as acts


# Actors for orientation control
class QuatActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.TanHQuat()})
        self.target_network.network.update({"f_output": acts.TanHQuat()})


class QuatExpActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        d = self.network.network["output"].in_features
        self.network.network.update({"output": nn.Linear(d, 3), "f_output": acts.QuatExp()})
        self.target_network.network.update({"output": nn.Linear(d, 3), "f_output": acts.QuatExp()})
        # Reload after adding new output layer
        self.target_network.load_state_dict(self.network.state_dict())
    

class QuatPlusActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.TanHQuatPlus()})
        self.target_network.network.update({"f_output": acts.TanHQuatPlus()})


class QuatPlusExpActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        d = self.network.network["output"].in_features
        self.network.network.update({"output": nn.Linear(d, 3), "f_output": acts.QuatPlusExp()})
        self.target_network.network.update({"output": nn.Linear(d, 3), "f_output": acts.QuatPlusExp()})
        # Reload after adding new output layer
        self.target_network.load_state_dict(self.network.state_dict())


class EulerActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class MatrixActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.TanHMatrix()})
        self.target_network.network.update({"f_output": acts.TanHMatrix()})


class MatrixExpActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        torch.autograd.set_detect_anomaly(True)  # MatrixExpActor is currently causing NaNs
        super().__init__(obs_space=obs_space, action_space=action_space)
        d = self.network.network["output"].in_features
        self.network.network.update({"output": nn.Linear(d, 3), "f_output": acts.MatrixExp()})
        self.target_network.network.update({"output": nn.Linear(d, 3), "f_output": acts.MatrixExp()})
        # Reload after adding new output layer
        self.target_network.load_state_dict(self.network.state_dict())


class R6Actor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        assert isinstance(obs_space, Box), f"Invalid obs space type {type(obs_space)}"
        assert isinstance(action_space, Box), f"Invalid action space type {type(action_space)}"
        self.network.network.update({"f_output": acts.TanHR6Ortho()})
        self.target_network.network.update({"f_output": acts.TanHR6Ortho()})


class R6ExpActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        torch.autograd.set_detect_anomaly(True)  # R6 is also using matrix_exp, so better check
        super().__init__(obs_space=obs_space, action_space=action_space)
        d = self.network.network["output"].in_features
        self.network.network.update({"output": nn.Linear(d, 3), "f_output": acts.R6Exp()})
        self.target_network.network.update({"output": nn.Linear(d, 3), "f_output": acts.R6Exp()})
        # Reload after adding new output layer
        self.target_network.load_state_dict(self.network.state_dict())


class TangentActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


# Actors for pose control
class QuatPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatPose()})
        self.target_network.network.update({"f_output": acts.QuatPose()})


class QuatPlusPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatPlusPose()})
        self.target_network.network.update({"f_output": acts.QuatPlusPose()})


class EulerPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class MatrixPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.MatrixPose()})
        self.target_network.network.update({"f_output": acts.MatrixPose()})


class R6PoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.R6Pose()})
        self.target_network.network.update({"f_output": acts.R6Pose()})


class TangentPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


# Actors for dual pose control
class QuatDualPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatDualPose()})
        self.target_network.network.update({"f_output": acts.QuatDualPose()})


class QuatPlusDualPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatPlusDualPose()})
        self.target_network.network.update({"f_output": acts.QuatPlusDualPose()})


class EulerDualPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class MatrixDualPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.MatrixDualPose()})
        self.target_network.network.update({"f_output": acts.MatrixDualPose()})


class R6DualPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.R6DualPose()})
        self.target_network.network.update({"f_output": acts.R6DualPose()})


class TangentDualPoseActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


# Actors for manipulator control (pose + gripper)
class QuatManipulatorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatManipulator()})
        self.target_network.network.update({"f_output": acts.QuatManipulator()})
    

class EulerManipulatorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space, action_space)


class MatrixManipulatorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.MatrixManipulator()})
        self.target_network.network.update({"f_output": acts.MatrixManipulator()})


class R6ManipulatorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.R6Manipulator()})
        self.target_network.network.update({"f_output": acts.R6Manipulator()})


class TangentManipulatorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


# Actors for dual manipulator control
class QuatDualManipulatorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatDualManipulator()})
        self.target_network.network.update({"f_output": acts.QuatDualManipulator()})


class EulerDualManipulatorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space, action_space)
    

class MatrixDualManipulatorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.MatrixDualManipulator()})
        self.target_network.network.update({"f_output": acts.MatrixDualManipulator()})


class R6DualManipulatorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.R6DualManipulator()})
        self.target_network.network.update({"f_output": acts.R6DualManipulator()})


class TangentDualManipulatorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


# Actors for quadrotor attitude control
class QuatQuadrotorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.QuatQuadrotor()})
        self.target_network.network.update({"f_output": acts.QuatQuadrotor()})


class EulerQuadrotorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


class MatrixQuadrotorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.MatrixQuadrotor()})
        self.target_network.network.update({"f_output": acts.MatrixQuadrotor()})


class R6QuadrotorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)
        self.network.network.update({"f_output": acts.R6Quadrotor()})
        self.target_network.network.update({"f_output": acts.R6Quadrotor()})


class TangentQuadrotorActor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space=obs_space, action_space=action_space)


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

quadrotor_actors = {
    "quat": QuatQuadrotorActor,
    "matrix": MatrixQuadrotorActor,
    "r6": R6QuadrotorActor,
    "euler": EulerQuadrotorActor,
    "tangent": TangentQuadrotorActor,
}

critics = {
    "quat": DDPGCritic,
    "quat_exp": DDPGCritic,
    "quat_plus": DDPGCritic,
    "quat_plus_exp": DDPGCritic,
    "matrix": DDPGCritic,
    "matrix_exp": DDPGCritic,
    "r6": DDPGCritic,
    "r6_exp": DDPGCritic,
    "euler": DDPGCritic,
    "euler_add": DDPGCritic,
    "tangent": DDPGCritic,
    "tangent_riemann": DDPGCritic,
}