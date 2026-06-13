from lsy_rl.ppo.policy import PPOActor, PPOCritic
from torch import nn

from rotations.modules.ddpg import acts


# Actors for orientation control
class QuatActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.TanHQuat()


class QuatPlusActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.TanHQuatPlus()


class EulerActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


class MatrixActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.TanHMatrix()


class R6Actor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.TanHR6Ortho()


class TangentActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


# Actors for pose control
class QuatPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.QuatPose()


class QuatPlusPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.QuatPlusPose()


class EulerPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


class MatrixPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.MatrixPose()


class R6PoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.R6Pose()


class TangentPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


# Actors for dual pose control
class QuatDualPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.QuatDualPose()


class QuatPlusDualPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.QuatPlusDualPose()


class EulerDualPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


class MatrixDualPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.MatrixDualPose()


class R6DualPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.R6DualPose()


class TangentDualPoseActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


# Actors for manipulator control (pose + gripper)
class QuatManipulatorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.QuatManipulator()


class EulerManipulatorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


class MatrixManipulatorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.MatrixManipulator()


class R6ManipulatorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.R6Manipulator()


class TangentManipulatorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


# Actors for dual manipulator control
class QuatDualManipulatorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.QuatDualManipulator()


class EulerDualManipulatorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


class MatrixDualManipulatorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.MatrixDualManipulator()


class R6DualManipulatorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.R6DualManipulator()


class TangentDualManipulatorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


# Actors for quadrotor attitude control
class QuatQuadrotorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.QuatQuadrotor()


class QuatPlusQuadrotorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.QuatPlusQuadrotor()


class EulerQuadrotorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


class MatrixQuadrotorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.MatrixQuadrotor()


class R6QuadrotorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = acts.R6Quadrotor()


class TangentQuadrotorActor(PPOActor):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        use_logstd_net: bool = False,
    ):
        super().__init__(obs_shape, action_shape, use_logstd_net=use_logstd_net)
        self.network.network["f_out"] = nn.Tanh()


actors = {
    "euler": EulerActor,
    "tangent": TangentActor,
    "quat": QuatActor,
    "quat_plus": QuatPlusActor,
    "r6": R6Actor,
    "matrix": MatrixActor,
}

critics = {
    "euler": PPOCritic,
    "tangent": PPOCritic,
    "quat": PPOCritic,
    "quat_plus": PPOCritic,
    "r6": PPOCritic,
    "matrix": PPOCritic,
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
    "quat_plus": QuatPlusQuadrotorActor,
    "matrix": MatrixQuadrotorActor,
    "r6": R6QuadrotorActor,
    "euler": EulerQuadrotorActor,
    "tangent": TangentQuadrotorActor,
}
