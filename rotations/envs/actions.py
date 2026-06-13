import jax.numpy as jp
from jax import Array
from jax.scipy.spatial.transform import Rotation as JR

from rotations.rotations import jax_rot_pow, r6_2mat, rot_mat_svd

euler_scale = {0.1: 0.0488, 0.2: 0.0976, 0.05: 0.0244}


def euler_rotation(_: JR, action: Array, step_len: float) -> JR:
    action = jp.clip(action, -1, 1) * jp.array([jp.pi, jp.pi / 2, jp.pi])
    return JR.from_euler("xyz", action, degrees=False)


def quat_rotation(_: JR, action: Array, step_len: float) -> JR:
    action = action.reshape(-1, 4).squeeze()
    action = jp.clip(action, -1, 1)
    # from_quat normalizes automatically
    return JR.from_quat(action)


def tangent_rotation(_: JR, action: Array, step_len: float) -> JR:
    action = jp.clip(action, -1, 1) * jp.pi
    return JR.from_rotvec(action)


def quat_plus_rotation(_: JR, action: Array, step_len: float) -> JR:
    action = jp.clip(action, jp.array([-1, -1, -1, 0]), jp.array([1, 1, 1, 1]))
    # No norm required, from_quat normalizes by default
    return JR.from_quat(action)


def matrix_rotation(_: JR, action: Array, step_len: float) -> JR:
    action = jp.clip(action, -1, 1)
    return JR.from_matrix(rot_mat_svd(action).reshape(-1, 3, 3).squeeze())


def r6_rotation(_: JR, action: Array, step_len: float) -> JR:
    assert action.shape[-1] == 6
    action = jp.clip(action, -1, 1)
    return JR.from_matrix(r6_2mat(action))


def rel_euler_rotation(rot: JR, action: Array, step_len: float, scale: bool = False) -> JR:
    action = jp.clip(action, -1, 1) * jp.array([jp.pi, jp.pi / 2, jp.pi])
    if scale:
        action = action * euler_scale[step_len]
    return JR.from_euler("xyz", rot.as_euler("xyz") + action, degrees=False)


def rel_quat_rotation(rot: JR, action: Array, step_len: float, scale: bool = False) -> JR:
    action = action.reshape(-1, 4).squeeze()
    action = jp.clip(action, -1, 1)
    # from_quat normalizes automatically
    drot = JR.from_quat(action)
    if scale:
        drot = jax_rot_pow(drot, step_len)
    return rot * drot


def rel_tangent_rotation(rot: JR, action: Array, step_len: float, scale: bool = False) -> JR:
    action = jp.clip(action, -1, 1) / jp.sqrt(3) * jp.pi
    # Currently outer scaling is used.
    if scale:
        action = action * jp.sqrt(3) * step_len
    return rot * JR.from_rotvec(action)  # TODO: Try better normalization


def rel_quat_plus_rotation(rot: JR, action: Array, step_len: float, scale: bool = False) -> JR:
    action = jp.clip(action, jp.array([-1, -1, -1, 0]), jp.array([1, 1, 1, 1]))
    # No norm required, from_quat normalizes by default
    drot = JR.from_quat(action)
    if scale:
        drot = jax_rot_pow(drot, step_len)
    return rot * drot


def rel_matrix_rotation(rot: JR, action: Array, step_len: float, scale: bool = False) -> JR:
    action = jp.clip(action, -1, 1)
    drot = JR.from_matrix(rot_mat_svd(action).reshape(-1, 3, 3).squeeze())
    if scale:
        drot = jax_rot_pow(drot, step_len)
    return rot * drot


def rel_r6_rotation(rot: JR, action: Array, step_len: float, scale: bool = False) -> JR:
    assert action.shape[-1] == 6
    action = jp.clip(action, -1, 1)
    drot = JR.from_matrix(r6_2mat(action))
    if scale:
        drot = jax_rot_pow(drot, step_len)
    return rot * drot


def tangent_scale_inner(action: Array, step_len: float) -> Array:
    return action / jp.sqrt(3) * jp.pi * step_len


def tangent_scale_outer(action: Array, step_len: float) -> Array:
    return action * jp.pi * step_len
