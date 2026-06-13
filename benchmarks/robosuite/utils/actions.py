import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from rotations.rotations import jax_rot_pow, r6_2mat, rot_mat_svd

euler_scale = {0.1: 0.0488, 0.2: 0.0976, 0.05: 0.0244}


def euler_to_rel_rotation(rot: R, action: NDArray, step_len: float) -> R:
    action = np.clip(action, -1, 1) * np.array([np.pi, np.pi / 2, np.pi])
    drot = rot.inv() * R.from_euler("xyz", action, degrees=False)
    return drot


def quat_to_rel_rotation(rot: R, action: NDArray, step_len: float) -> R:
    action = action.reshape(-1, 4).squeeze()
    action = np.clip(action, -1, 1)
    # from_quat normalizes automatically
    drot = rot.inv() * R.from_quat(action)
    return drot


def tangent_to_rel_rotation(rot: R, action: NDArray, step_len: float) -> R:
    action = np.clip(action, -1, 1) * np.pi
    drot = rot.inv() * R.from_rotvec(action)
    return drot


def quat_plus_to_rel_rotation(rot: R, action: NDArray, step_len: float) -> R:
    action = np.clip(action, np.array([-1, -1, -1, 0]), np.array([1, 1, 1, 1]))
    # No norm required, from_quat normalizes by default
    drot = rot.inv() * R.from_quat(action)
    return drot


def matrix_to_rel_rotation(rot: R, action: NDArray, step_len: float) -> R:
    action = np.clip(action, -1, 1)
    action = rot_mat_svd(action).reshape(-1, 3, 3).squeeze()
    drot = rot.inv() * R.from_matrix(action)
    return drot


def r6_to_rel_rotation(rot: R, action: NDArray, step_len: float) -> R:
    assert action.shape[-1] == 6
    action = np.clip(action, -1, 1)
    action = r6_2mat(action)
    drot = rot.inv() * R.from_matrix(action)
    return drot


def rel_euler_to_rel_rotation(rot: R, action: NDArray, step_len: float, scale: bool = False) -> R:
    action = np.clip(action, -1, 1) * np.array([np.pi, np.pi / 2, np.pi])
    if scale:
        action = action * euler_scale[step_len]
    desired = R.from_euler("xyz", rot.as_euler("xyz") + action, degrees=False)
    drot = rot.inv() * desired 
    return drot


def rel_quat_to_rel_rotation(_: R, action: NDArray, step_len: float, scale: bool = False) -> R:
    action = action.reshape(-1, 4).squeeze()
    action = np.clip(action, -1, 1)
    # from_quat normalizes automatically
    drot = R.from_quat(action)
    if scale:
        drot = jax_rot_pow(drot, step_len)
    return drot


def rel_tangent_to_rel_rotation(_: R, action: NDArray, step_len: float, scale: bool = False) -> R:
    action = np.clip(action, -1, 1) / np.sqrt(3) * np.pi
    # Currently outer scaling is used.
    if scale:
        action = action * np.sqrt(3) * step_len
    drot = R.from_rotvec(action)
    return drot


def rel_quat_plus_to_rel_rotation(_: R, action: NDArray, step_len: float, scale: bool = False) -> R:
    action = np.clip(action, np.array([-1, -1, -1, 0]), np.array([1, 1, 1, 1]))
    # No norm required, from_quat normalizes by default
    drot = R.from_quat(action)
    if scale:
        drot = jax_rot_pow(drot, step_len)
    return drot


def rel_matrix_to_rel_rotation(_: R, action: NDArray, step_len: float, scale: bool = False) -> R:
    action = np.clip(action, -1, 1)
    drot = R.from_matrix(rot_mat_svd(action).reshape(-1, 3, 3).squeeze())
    if scale:
        drot = jax_rot_pow(drot, step_len)
    return drot


def rel_r6_to_rel_rotation(_: R, action: NDArray, step_len: float, scale: bool = False) -> R:
    assert action.shape[-1] == 6
    action = np.clip(action, -1, 1)
    drot = R.from_matrix(r6_2mat(action))
    if scale:
        drot = jax_rot_pow(drot, step_len)
    return drot