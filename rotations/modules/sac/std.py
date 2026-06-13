from lsy_rl.sac.policy import SACActor, SACCritic
from torch import nn

import rotations.modules.sac.activations as acts


# Actors for orientation control
class QuatActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class QuatPlusActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = acts.TanHQuatPlus()


class EulerActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class MatrixActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class R6Actor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class TangentActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


# Actors for pose control
class QuatPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class QuatPlusPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = acts.QuatPlusPose()


class EulerPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class MatrixPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class R6PoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class TangentPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


# Actors for dual pose control
class QuatDualPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class QuatPlusDualPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = acts.QuatPlusDualPose()


class EulerDualPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class MatrixDualPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class R6DualPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class TangentDualPoseActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


# Actors for manipulator control (pose + gripper)
class QuatManipulatorActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class EulerManipulatorActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class MatrixManipulatorActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class TangentManipulatorActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


# Actors for dual manipulator control
class QuatDualManipulatorActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class EulerDualManipulatorActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class MatrixDualManipulatorActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
        self.squash_layer = nn.Tanh()


class TangentDualManipulatorActor(SACActor):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__(obs_shape, action_shape)
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
    "euler": SACCritic,
    "tangent": SACCritic,
    "quat": SACCritic,
    "quat_plus": SACCritic,
    "r6": SACCritic,
    "matrix": SACCritic,
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
    "euler": EulerManipulatorActor,
    "tangent": TangentManipulatorActor,
}

dual_manipulator_actors = {
    "quat": QuatDualManipulatorActor,
    "matrix": MatrixDualManipulatorActor,
    "euler": EulerDualManipulatorActor,
    "tangent": TangentDualManipulatorActor,
}
