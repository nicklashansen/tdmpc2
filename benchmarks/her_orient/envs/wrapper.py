import numpy as np
from gymnasium import Wrapper, spaces
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from rotations.envs.actions import euler_scale
from rotations.rotations import RotType, rot_mat_svd


class RotationWrapper(Wrapper):
    def __init__(
        self, env, rot_type: str | RotType, relative: bool = False, rot_scale: float | None = None
    ):
        super().__init__(env)
        self.rot_type = RotType(rot_type)
        assert self.rot_type in (RotType.quat, RotType.euler, RotType.tangent, RotType.matrix)

        action_shape = (self.action_space.shape[-1] - 4 + self.rot_type.dim,)
        if self.action_space.shape not in [(7,), (8,)]:
            raise ValueError("Unexpected action space. Are you wrapping an non-orientation env?")
        self._use_gripper = self.action_space.shape[-1] == 8  # x, y, z, rot(4), gripper(1)
        self.action_space = spaces.Box(-1, 1, shape=action_shape, dtype=np.float32)
        self.relative = relative
        self.rot_scale = rot_scale

    def step(self, action: NDArray) -> NDArray:
        assert action in self.action_space
        if self._use_gripper:
            pos, rot, gripper = np.split(action, [3, -1], axis=-1)
        else:
            pos, rot = np.split(action, [3], axis=-1)
        eef_rot = R.from_matrix(self.env.unwrapped.eef_rot.reshape(3, 3))
        match (self.rot_type, self.relative):
            case (RotType.quat, False):
                quat = rot
            case (RotType.quat, True):
                quat = rel_quat2quat(rot, eef_rot, self.rot_scale)
            case (RotType.euler, False):
                quat = euler2quat(rot, eef_rot, self.rot_scale)
            case (RotType.euler, True):
                quat = rel_euler2quat(rot, eef_rot, self.rot_scale)
            case (RotType.tangent, False):
                quat = tangent2quat(rot, eef_rot, self.rot_scale)
            case (RotType.tangent, True):
                quat = rel_tangent2quat(rot, eef_rot, self.rot_scale)
            case (RotType.matrix, False):
                quat = matrix2quat(rot, eef_rot, self.rot_scale)
            case (RotType.matrix, True):
                quat = rel_matrix2quat(rot, eef_rot, self.rot_scale)
            case _:
                raise NotImplementedError
        if self._use_gripper:
            action = np.concatenate([pos, quat, gripper], axis=-1).astype(np.float32)
        else:
            action = np.concatenate([pos, quat], axis=-1).astype(np.float32)
        return super().step(action)

    def _rotation_transform(self, rot: NDArray, eef_rot: R) -> NDArray:
        raise NotImplementedError


def quat2quat(quat: NDArray, eef_rot: R, rot_scale: float | None = None) -> NDArray:
    return quat


def rel_quat2quat(rel_quat: NDArray, eef_rot: R, rot_scale: float | None = None) -> NDArray:
    delta_rot = R.from_quat(rel_quat)
    if rot_scale is not None:
        delta_rot = delta_rot**rot_scale
    return (eef_rot * delta_rot).as_quat()


def euler2quat(xyz: NDArray, eef_rot: R, rot_scale: float | None = None) -> NDArray:
    return R.from_euler("xyz", xyz * np.array([np.pi, np.pi / 2, np.pi])).as_quat()


def rel_euler2quat(rel_xyz: NDArray, eef_rot: R, rot_scale: float | None = None) -> NDArray:
    rel_xyz = rel_xyz * np.array([np.pi, np.pi / 2, np.pi])
    if rot_scale is not None:
        rel_xyz = rel_xyz * euler_scale[rot_scale]
    return R.from_euler("xyz", eef_rot.as_euler("xyz") + rel_xyz).as_quat()


def tangent2quat(rotvec: NDArray, eef_rot: R, rot_scale: float | None = None) -> NDArray:
    return R.from_rotvec(rotvec * np.pi).as_quat()


def rel_tangent2quat(rel_rotvec: NDArray, eef_rot: R, rot_scale: float | None = None) -> NDArray:
    if rot_scale is not None:
        rel_rotvec = rel_rotvec * rot_scale * np.sqrt(3)  # Undo later normalization
    return (eef_rot * R.from_rotvec(rel_rotvec * np.pi / np.sqrt(3))).as_quat()


def matrix2quat(mat: NDArray, eef_rot: R, rot_scale: float | None = None) -> NDArray:
    return R.from_matrix(rot_mat_svd(mat.reshape((-1, 9))).reshape((3, 3))).as_quat()


def rel_matrix2quat(rel_mat: NDArray, eef_rot: R, rot_scale: float | None = None) -> NDArray:
    delta_rot = R.from_matrix(rot_mat_svd(rel_mat.reshape((-1, 9))).reshape((3, 3)))
    if rot_scale is not None:
        delta_rot = delta_rot**rot_scale
    return (eef_rot * delta_rot).as_quat()
