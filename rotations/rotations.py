from enum import Enum
from functools import partial, singledispatch
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import torch
from array_api_compat import array_namespace
from jax import Array
from jax.scipy.spatial.transform import Rotation as JR
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


class RotType(str, Enum):
    """Enum of available rotation types."""

    euler = "euler"
    tangent = "tangent"
    quat = "quat"
    quat_plus = "quat_plus"
    matrix = "matrix"
    r6 = "r6"

    def from_array(self, x: NDArray | Array) -> R | JR:
        """Convert an array or batch of arrays to a rotation object.

        Array representations are always normalized to the range of [-1, 1].
        """
        match self:
            case RotType.euler:
                return _from_euler(x)
            case RotType.tangent:
                return _from_rotvec(x)
            case RotType.quat:
                return _from_quat(x)
            case RotType.quat_plus:
                return _from_quat(x)
            case RotType.matrix:
                return _from_matrix(x)
            case RotType.r6:
                return _from_r6(x)
            case _:
                raise NotImplementedError(f"Invalid array type {type(x)}")

    def as_array(self, r: R | JR) -> NDArray[np.float64] | Array:
        """Convert a rotation object to an array.

        Array representations are always normalized to the range of [-1, 1].
        """
        match self:
            case RotType.euler:
                return r.as_euler("xyz") / np.array([np.pi, np.pi / 2, np.pi])
            case RotType.tangent:
                # Normalize tangent space from [-pi, pi] to [-1, 1]
                return r.as_rotvec() / np.pi * np.sqrt(3)
            case RotType.quat:
                return r.as_quat()
            case RotType.quat_plus:
                q = r.as_quat()
                mask = (q[..., 3] < 0)[..., None]
                if isinstance(q, Array):
                    return jp.where(mask, -q, q)
                return np.where(mask, -q, q)
            case RotType.matrix:
                mat = r.as_matrix()
                return mat.reshape(-1, 9) if mat.ndim > 2 else mat.flatten()
            case RotType.r6:
                mat = r.as_matrix()
                return mat[..., :2, :].reshape(-1, 6) if mat.ndim > 2 else mat[:2, :].flatten()
            case _:
                raise TypeError(f"Invalid rotation type {self}")

    @property
    def dim(self) -> int:
        """Return the dimension of the rotation type."""
        match self:
            case RotType.euler:
                return 3
            case RotType.tangent:
                return 3
            case RotType.quat:
                return 4
            case RotType.quat_plus:
                return 4
            case RotType.matrix:
                return 9
            case RotType.r6:
                return 6
            case _:
                raise TypeError(f"Invalid rotation type: {self}")


@singledispatch
def _from_euler(x: Any) -> R:
    raise NotImplementedError(f"Invalid array type {type(x)}")


@_from_euler.register
def _(x: np.ndarray) -> R:
    return R.from_euler("xyz", x * np.array([np.pi, np.pi / 2, np.pi]))


@_from_euler.register
@jax.jit
def _(x: Array) -> JR:
    return JR.from_euler("xyz", x * jp.array([np.pi, np.pi / 2, np.pi]))


@singledispatch
def _from_rotvec(x: Any) -> R:
    raise NotImplementedError(f"Invalid array type {type(x)}")


@_from_rotvec.register
def _(x: np.ndarray) -> R:
    return R.from_rotvec(x * np.pi / np.sqrt(3))


@_from_rotvec.register
@jax.jit
def _(x: Array) -> JR:
    return JR.from_rotvec(x * np.pi / np.sqrt(3))


@singledispatch
def _from_quat(x: Any) -> R:
    raise NotImplementedError(f"Invalid array type {type(x)}")


@_from_quat.register
def _(x: np.ndarray) -> R:
    return R.from_quat(x)


@_from_quat.register
@jax.jit
def _(x: Array) -> JR:
    return JR.from_quat(x)


@singledispatch
def _from_matrix(x: Any) -> R:
    raise NotImplementedError(f"Invalid array type {type(x)}")


@_from_matrix.register
def _(x: np.ndarray) -> R:
    return R.from_matrix(x.reshape(-1, 3, 3) if x.ndim > 1 else x.reshape(3, 3))


@_from_matrix.register
@jax.jit
def _(x: Array) -> JR:
    return JR.from_matrix(x.reshape(-1, 3, 3) if x.ndim > 1 else x.reshape(3, 3))


@singledispatch
def _from_r6(x: Any) -> R:
    raise NotImplementedError(f"Invalid array type {type(x)}")


@_from_r6.register
def _(x: np.ndarray) -> R:
    r6 = r6_2mat(x)
    return R.from_matrix(r6.reshape(-1, 3, 3) if x.ndim > 1 else r6.reshape(3, 3))


@_from_r6.register
@jax.jit
def _(x: Array) -> JR:
    r6 = r6_2mat(x)
    return JR.from_matrix(r6.reshape(-1, 3, 3) if x.ndim > 1 else r6.reshape(3, 3))


def euler2quat(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler Angles to Quaternions.

    Euler angles: xyz order. Quaternions: xyzw order for scipy Rotations compatibility.

    Args:
        euler: Array of euler angles.

    Returns:
        An array of quaternions.
    """
    assert isinstance(euler, torch.Tensor), f"Invalid type euler {type(euler)}"
    assert euler.shape[-1] == 3, f"Invalid shape euler {euler.shape}"

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = torch.empty(euler.shape[:-1] + (4,), dtype=euler.dtype, device=euler.device)
    quat[..., 0] = cj * cs - sj * sc
    quat[..., 1] = -(cj * ss + sj * cc)
    quat[..., 2] = cj * sc - sj * cs
    quat[..., 3] = cj * cc + sj * ss
    return quat


@singledispatch
def rot_mat_svd(x: Any) -> Any:
    raise NotImplementedError(f"rot_mat_svd not implemented for type {type(x)}")


@rot_mat_svd.register
def _(x: np.ndarray) -> NDArray[np.floating]:
    return rot_mat_svd(torch.as_tensor(x)).numpy()


@torch.jit.script
@rot_mat_svd.register
def _(x: torch.Tensor) -> torch.Tensor:
    """Convert a matrix to a rotation matrix using SVD.

    The algorithm computes the least squares solution to the orthogonal matrix closest to the input
    matrix. See https://arxiv.org/pdf/2404.11735.

    Args:
        x: The matrix to convert.

    Returns:
        The rotation matrix.
    """
    assert isinstance(x, torch.Tensor), f"Invalid type x {type(x)}"
    assert x.shape[-1] == 9, f"Invalid shape x {x.shape}"
    u, _, vh = torch.linalg.svd(x.view(-1, 3, 3))
    unorm = torch.zeros_like(u)  # Ensure det = 1 without in-place changes to u (breaks backprop)
    unorm[..., :2] = u[..., :2]
    unorm[..., 2] = u[..., 2] * torch.det(torch.matmul(u, vh))[:, None]
    return torch.matmul(unorm, vh).view(-1, 9)


@rot_mat_svd.register
@jax.jit
def _(x: Array) -> Array:
    assert x.shape[-1] == 9, f"Invalid shape x {x.shape}"
    u, _, vh = jp.linalg.svd(x.reshape(-1, 3, 3))
    unorm = jp.zeros_like(u)
    unorm = unorm.at[..., :2].set(u[..., :2])
    unorm = unorm.at[..., 2].set(u[..., 2] * jp.linalg.det(jp.matmul(u, vh))[:, None])
    return jp.matmul(unorm, vh).reshape(-1, 9)


@singledispatch
def r6_orthonomalization(x: Any) -> Any:
    raise NotImplementedError(f"r6_orthonomalization not implemented for type {type(x)}")


@r6_orthonomalization.register
def _(x: np.ndarray) -> NDArray[np.floating]:
    return r6_orthonomalization(torch.as_tensor(x)).numpy()


@torch.jit.script
@r6_orthonomalization.register
def _(x: torch.Tensor) -> torch.Tensor:
    """Convert a 6D vector to a rotation matrix using Gram Schmidt orthonormalization.

    See .

    Args:
        x: The matrix to convert.

    Returns:
        The rotation matrix.
    """
    assert isinstance(x, torch.Tensor), f"Invalid type x {type(x)}"
    assert x.shape[-1] == 6, f"Invalid shape x {x.shape}"
    x1 = x[..., :3] / torch.linalg.norm(x[..., :3], dim=-1, keepdim=True)
    x2 = x[..., 3:] - torch.sum(x[..., 3:] * x1, dim=-1, keepdim=True) * x1
    x2 = x2 / torch.linalg.norm(x2, dim=-1, keepdim=True)
    return torch.cat((x1, x2), dim=-1)


@singledispatch
def r6_2mat(x: Any) -> Any:
    raise NotImplementedError(f"r6_2mat not implemented for type {type(x)}")


@r6_2mat.register
def _(x: np.ndarray) -> NDArray[np.floating]:
    return r6_2mat(torch.as_tensor(x)).numpy()


@torch.jit.script
@r6_2mat.register
def _(x: torch.Tensor) -> torch.Tensor:
    """Convert an unnormalized 6D vector with Gram Schmidt orthogonalization to a quaternion.

    Args:
        x: The vector to convert.

    Returns:
        The quaternion.
    """
    # We can't reuse r6_orthonomalization here because torch jit seems to interfere with dispatching
    assert isinstance(x, torch.Tensor), f"Invalid type x {type(x)}"
    assert x.shape[-1] == 6, f"Invalid shape x {x.shape}"
    x1 = x[..., :3] / torch.linalg.norm(x[..., :3], dim=-1, keepdim=True)
    x2 = x[..., 3:] - torch.sum(x[..., 3:] * x1, dim=-1, keepdim=True) * x1
    x2 = x2 / torch.linalg.norm(x2, dim=-1, keepdim=True)
    x3 = torch.cross(x1, x2, dim=-1)
    return torch.cat((x1, x2, x3), dim=-1)


@r6_2mat.register
def _(x: Array) -> Array:
    x1 = x[..., :3] / jp.linalg.norm(x[..., :3], axis=-1, keepdims=True)
    x2 = x[..., 3:] - jp.sum(x[..., 3:] * x1, axis=-1, keepdims=True) * x1
    x2 = x2 / jp.linalg.norm(x2, axis=-1, keepdims=True)
    x3 = jp.cross(x1, x2)
    return jp.concatenate((x1[..., None, :], x2[..., None, :], x3[..., None, :]), axis=-2)


@singledispatch
def quat_exp(x: Any) -> Any:
    raise NotImplementedError(f"quat_exp not implemented for type {type(x)}")


@quat_exp.register
def _(x: np.ndarray) -> NDArray[np.floating]:
    return quat_exp(torch.as_tensor(x)).numpy()


@torch.jit.script
@quat_exp.register
def _(x: torch.Tensor) -> torch.Tensor:
    """Return the quaternion exponential of a 3D vector.

    Args:
        x: The vector to convert.

    Returns:
        The quaternion.
    """
    angle = torch.linalg.norm(x, dim=-1, keepdim=True)
    axis = x / (angle + 1e-8)
    q = torch.cat((axis * torch.sin(angle), torch.cos(angle)), dim=-1)
    return q


@singledispatch
def matrix_exp(x: Any) -> Any:
    raise NotImplementedError(f"matrix_exp not implemented for type {type(x)}")


@matrix_exp.register
def _(x: np.ndarray) -> NDArray[np.floating]:
    return matrix_exp(torch.as_tensor(x)).numpy()


@torch.jit.script
@matrix_exp.register
def _(x: torch.Tensor) -> torch.Tensor:
    """Return the matrix exponential of a 3D vector.

    Args:
        x: The vector to convert.

    Returns:
        The matrix.
    """
    assert x.shape[-1] == 3, f"Invalid shape x {x.shape}"
    assert not torch.isnan(x).any(), "NaN encountered in matrix_exp"
    x_skew = torch.zeros(x.shape[:-1] + (3, 3), dtype=x.dtype, device=x.device)
    x_skew[..., 0, 1] = -x[..., 2]
    x_skew[..., 0, 2] = x[..., 1]
    x_skew[..., 1, 0] = x[..., 2]
    x_skew[..., 1, 2] = -x[..., 0]
    x_skew[..., 2, 0] = -x[..., 1]
    x_skew[..., 2, 1] = x[..., 0]
    return torch.matrix_exp(x_skew)


def quat_scale(q: R, scale: float) -> R:
    """Return the scaled quaternion.

    Args:
        q: The quaternion to scale.
        scale: The scaling factor

    Returns:
        The scaled quaternion.
    """
    q_id_neg = R.from_quat([0, 0, 0, -1])
    if q.single:
        return q**scale if q.as_quat()[3] >= 0 else q_id_neg * (q_id_neg * q) ** scale
    mask = q.as_quat()[:, 3] >= 0
    if mask.any():
        q[mask] = q[mask] ** scale
    if not mask.all():
        q[~mask] = q_id_neg * (q_id_neg * q[~mask]) ** scale
    return q


@singledispatch
def jax_rot_pow(rotation: Any, scale: float) -> Any:
    raise NotImplementedError(f"jax_rot_pow not implemented for type {type(rotation)}")


@jax_rot_pow.register
def _(rotation: JR, scale: float) -> JR:
    """Raise a jax Rotation to a power while preserving the sign of the quaternion.

    Args:
        q: The quaternion to scale.
        scale: The scaling factor

    Returns:
        The scaled quaternion.
    """
    q_id_neg = JR.from_quat([0, 0, 0, -1])
    mask = rotation.as_quat()[..., 3] >= 0
    mask = mask.reshape((-1,) + (1,) * (rotation.quat.ndim - 1))  # Broadcast mask to match q.quat
    q_scaled = _jax_rot_pow(rotation, scale).as_quat()
    q_neg_scaled = (q_id_neg * _jax_rot_pow(q_id_neg * rotation, scale)).as_quat()
    return JR.from_quat(jp.where(mask, q_scaled, q_neg_scaled))


@jax_rot_pow.register
def _(rotation: R, scale: float) -> R:
    q_id_neg = R.from_quat([0, 0, 0, -1])
    mask = rotation.as_quat()[..., 3] >= 0
    mask = mask.reshape((-1,) + (1,) * (not rotation.single))  # Broadcast mask to match q.quat
    q_scaled = _jax_rot_pow(rotation, scale).as_quat()
    q_neg_scaled = (q_id_neg * _jax_rot_pow(q_id_neg * rotation, scale)).as_quat()
    return R.from_quat(np.where(mask, q_scaled, q_neg_scaled))


@singledispatch
def _jax_rot_pow(r: Any, n: float) -> Any:
    raise NotImplementedError(f"_jax_rot_pow not implemented for type {type(r)}")


@_jax_rot_pow.register
def _(r: JR, n: float) -> JR:
    q = JR.from_rotvec(n * r.as_rotvec()).quat
    q = jp.where(n == 0, jp.array([0, 0, 0, 1]), q)
    q = jp.where(n == -1, r.inv().as_quat(), q)
    q = jp.where(n == 1, r.as_quat(), q)
    return JR.from_quat(q)


@_jax_rot_pow.register
def _(r: R, n: float) -> R:
    q = R.from_rotvec(n * r.as_rotvec()).as_quat()
    q = np.where(n == 0, jp.array([0, 0, 0, 1]), q)
    q = np.where(n == -1, r.inv().as_quat(), q)
    q = np.where(n == 1, r.as_quat(), q)
    return R.from_quat(q)


@singledispatch
def angle(r: Any, r_ref: Any, r_type: RotType) -> Any:
    raise NotImplementedError(f"Invalid types {type(r)} and {type(r_ref)}")


@angle.register
def _(r: R, r_ref: R, r_type: RotType) -> NDArray[np.float32]:
    return (r.inv() * r_ref).magnitude()


@angle.register
@partial(jax.jit, static_argnames="r_type")
def _(r: JR, r_ref: JR, r_type: RotType) -> NDArray[np.float32]:
    return (r.inv() * r_ref).magnitude()


@angle.register
def _(r: np.ndarray, r_ref: np.ndarray, r_type: RotType) -> NDArray[np.float32]:
    assert r.shape == r_ref.shape
    assert r.shape[-1] == r_type.dim, "Input must be in obs_type format"
    return angle(r_type.from_array(r), r_type.from_array(r_ref), r_type)


@angle.register
def _(r: torch.Tensor, r_ref: torch.Tensor, r_type: RotType) -> torch.Tensor:
    assert r.shape == r_ref.shape
    assert r.shape[-1] == r_type.dim, "Input must be in obs_type format"
    r = r_type.from_array(jp.from_dlpack(r))
    r_ref = r_type.from_array(jp.from_dlpack(r_ref))
    return torch.from_dlpack(angle(r, r_ref, r_type))


@singledispatch
def parallel_transport(x: Any, y: Any) -> Any:
    raise NotImplementedError(f"parallel_transport not implemented for type {type(x)}, {type(y)}")


@parallel_transport.register
def _(x: np.ndarray, y: np.ndarray) -> NDArray[np.floating]:
    return parallel_transport(torch.as_tensor(x), torch.as_tensor(y)).numpy()


@torch.jit.script
@parallel_transport.register
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Transport a tangent vector at tangent space at the identity to the the tangent space at x.

    Args:
        x: The current orientation.
        y: The tangent vector at the tangent space at the identity

    Returns:
        The local tangent vector.
    """
    assert x.shape[-1] == 4, f"Invalid shape x {x.shape}"
    assert y.shape[-1] == 3, f"Invalid shape y {y.shape}"
    u_norm = torch.acos(x[3])
    if u_norm == 0:
        return torch.cat((y, torch.zeros(1, dtype=x.dtype, device=x.device)))
    u_bar = torch.clone(x)
    u_bar[3] = 0
    u_bar = u_bar / torch.linalg.norm(u_bar)

    A = (
        -torch.sin(u_norm) * torch.outer(torch.tensor([0, 0, 0, 1]), u_bar)
        + torch.cos(u_norm) * torch.outer(u_bar, u_bar)
        + torch.eye(4)
        - torch.outer(u_bar, u_bar)
    )
    return torch.matmul(A, torch.concatenate([y, torch.zeros(1, dtype=x.dtype, device=x.device)]))


@singledispatch
def riemann_exp(x: Any, y: Any) -> Any:
    raise NotImplementedError(f"riemann_exp not implemented for type {type(x)}, {type(y)}")


@riemann_exp.register
def _(x: np.ndarray, y: np.ndarray) -> NDArray[np.floating]:
    return riemann_exp(torch.as_tensor(x), torch.as_tensor(y)).numpy()


@torch.jit.script
@riemann_exp.register
def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Riemannian exponential map of a local tangent vector at x.

    Args:
        x: The current orientation.
        y: The tangent vector at the local tangent space at x

    Returns:
        The mapped orientation.
    """
    assert x.shape[-1] == 4, f"Invalid shape x {x.shape}"
    assert y.shape[-1] == 4, f"Invalid shape y {y.shape}"
    q_norm = torch.linalg.norm(y)
    if q_norm == 0:
        return x
    q_bar = y / q_norm

    return x * torch.cos(q_norm) + q_bar * torch.sin(q_norm)


def matrix_to_quat(matrix):
    # See https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation_xp.py
    # (not released for now)
    xp = array_namespace(matrix)
    matrix_trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    decision = xp.stack(
        [matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2], matrix_trace], axis=-1
    )
    choice = xp.argmax(decision, axis=-1, keepdims=True)
    quat = xp.empty((*matrix.shape[:-2], 4), dtype=matrix.dtype, device=matrix.device)
    # Case 0
    quat_0 = xp.stack(
        [
            1 - matrix_trace[...] + 2 * matrix[..., 0, 0],
            matrix[..., 1, 0] + matrix[..., 0, 1],
            matrix[..., 2, 0] + matrix[..., 0, 2],
            matrix[..., 2, 1] - matrix[..., 1, 2],
        ],
        axis=-1,
    )
    quat = xp.where(choice == 0, quat_0, quat)

    # Case 1
    quat_1 = xp.stack(
        [
            matrix[..., 1, 0] + matrix[..., 0, 1],
            1 - matrix_trace[...] + 2 * matrix[..., 1, 1],
            matrix[..., 2, 1] + matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
        ],
        axis=-1,
    )
    quat = xp.where(choice == 1, quat_1, quat)

    # Case 2
    quat_2 = xp.stack(
        [
            matrix[..., 2, 0] + matrix[..., 0, 2],
            matrix[..., 2, 1] + matrix[..., 1, 2],
            1 - matrix_trace[...] + 2 * matrix[..., 2, 2],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        axis=-1,
    )
    quat = xp.where(choice == 2, quat_2, quat)

    # Case 3
    quat_3 = xp.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
            1 + matrix_trace[...],
        ],
        axis=-1,
    )
    quat = xp.where(choice == 3, quat_3, quat)
    return quat / xp.linalg.norm(quat, axis=-1, keepdims=True)
