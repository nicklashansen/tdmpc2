import torch

from rotations.rotations import r6_orthonomalization, rot_mat_svd


def project_to_euler(euler: torch.Tensor) -> torch.Tensor:
    assert euler.shape[-1] == 3, "Euler angles must be of shape (..., 3)"
    return torch.clamp(euler, min=-1, max=1)


def project_to_tangent(tangent: torch.Tensor) -> torch.Tensor:
    assert tangent.shape[-1] == 3, "Tangent must be of shape (..., 3)"
    return torch.clamp(tangent, min=-1, max=1)


def project_to_quat(quat: torch.Tensor) -> torch.Tensor:
    assert quat.shape[-1] == 4, "Quaternion must be of shape (..., 4)"
    qnorm = torch.norm(quat, dim=-1, keepdim=True)
    return quat / qnorm


def project_to_quat_plus(quat: torch.Tensor) -> torch.Tensor:
    assert quat.shape[-1] == 4, "Quaternion must be of shape (..., 4)"
    qnorm = torch.norm(quat, dim=-1, keepdim=True)
    quat = quat / qnorm
    # If one element is zero, check sign of the next one (w → x → y)
    mask = quat[..., 3] < 0
    zero_w = quat[..., 3] == 0
    mask = torch.logical_or(mask, zero_w & (quat[..., 0] < 0))
    zero_wx = torch.logical_or(zero_w, quat[..., 0] == 0)
    mask = torch.logical_or(mask, zero_wx & (quat[..., 1] < 0))
    zero_wxy = torch.logical_or(zero_wx, quat[..., 1] == 0)
    mask = torch.logical_or(mask, zero_wxy & (quat[..., 2] < 0))
    return torch.where(mask[..., None], -quat, quat)


action_projections = {
    "euler": project_to_euler,
    "matrix": rot_mat_svd,
    "tangent": project_to_tangent,
    "quat": project_to_quat,
    "quat_plus": project_to_quat_plus,
    "r6": r6_orthonomalization,
}
