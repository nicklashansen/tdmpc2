import numpy as np
from numpy.typing import NDArray

EULER_SCALES = {0.05: np.pi / 44.7, 0.1: np.pi / 22.5, 0.2: np.pi / 11.2}
"""Scaling factors have been determined empirically by running 1e8 tests with random start
rotations and random actions and measuring the maximum angle of the rotation."""

EULER_SCALES_BALANCE = {0.05: np.pi / 41, 0.1: np.pi / 20.5, 0.2: np.pi / 10.3}
"""Scaling factors have been determined empirically by running 1e8 tests with random start
rotations and random actions and measuring the maximum angle of the rotation."""


def tangent_scale_inner(tangent: NDArray[np.floating]) -> NDArray[np.floating]:
    """The cube [-1, 1]^3 lies within the tangent space sphere.

    Args:
        tangent: The tangent space action to scale.

    Returns:
        The scaled action.
    """
    return tangent / np.sqrt(3) * np.pi


def tangent_scale_outer(tangent: NDArray[np.floating]) -> NDArray[np.floating]:
    """The cube [-1, 1]^3 contains the whole tangent space of rotations and gets projected.

    Warning:
        This function drastically reduces the performance! Adaptive scaling theoretically gives
        more maneuverability, but the agent learns significantly slower.

    Args:
        tangent: The tangent space action to scale.

    Returns:
        The scaled action.
    """
    return tangent / max(1, np.linalg.norm(tangent)) * np.pi


def euler_scale(
    euler: NDArray[np.floating], step_len: float, balance: bool = True
) -> NDArray[np.floating]:
    """Scale the euler angles so that applying them incurrs a rotation of at most `step_len`.

    Args:
        euler: Array of euler angles.
        step_len: Maximum step length to scale for.
        balance: Balance out euler angle range (pi, pi/2, pi).
    """
    if balance:
        if step_len not in EULER_SCALES_BALANCE:
            raise ValueError(f"Step length {step_len} not supported for balanced euler scaling")
        return euler * np.array([1, 0.5, 1]) * EULER_SCALES_BALANCE[step_len]
    if step_len not in EULER_SCALES:
        raise ValueError(f"Step length {step_len} not supported for euler scaling")
    return euler * EULER_SCALES[step_len]
