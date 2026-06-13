import torch
import torch.nn as nn
from torch import Tensor

from rotations.rotations import matrix_exp, quat_exp, r6_orthonomalization, rot_mat_svd


# Activations for orientation control
class TanHQuat(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 4 == 0, f"Invalid shape {x.shape}"
        shape = x.shape
        n_quats = shape[-1] // 4

        x = x.view(-1, n_quats, 4)
        x = torch.tanh(x)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x.reshape(*shape)


class TanHQuatOffset(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("eye", torch.tensor([0.0, 0.0, 0.0, 1.0]))

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 4 == 0, f"Invalid shape {x.shape}"
        shape = x.shape
        n_quats = shape[-1] // 4

        x = x.view(-1, n_quats, 4)
        x = (torch.tanh(x) + self.eye).clamp_max(1.0)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x.reshape(*shape)


class TanHQuatPlus(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 4 == 0, f"Invalid shape {x.shape}"
        shape = x.shape
        n_quats = shape[-1] // 4
        x = x.view(-1, n_quats, 4)
        x[..., 3] = torch.sigmoid(x[..., 3])
        x[..., :3] = torch.tanh(x[..., :3])
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x.reshape(*shape)


class QuatExp(nn.Module):
    """Custom exp output"""

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 3, f"Invalid shape {x.shape}"
        x = quat_exp(x)
        return x / torch.norm(x, dim=-1, keepdim=True)


class QuatPlusExp(nn.Module):
    """Custom exp output"""

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 3, f"Invalid shape {x.shape}"
        x = quat_exp(x)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        x[x[..., 3] < 0] = x[x[..., 3] < 0] * -1
        return x


class MatrixExp(nn.Module):
    """Custom exp output"""

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 3, f"Invalid shape {x.shape}"
        return matrix_exp(x).view(-1, 9)


class R6Exp(nn.Module):
    """Custom exp output"""

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 3, f"Invalid shape {x.shape}"
        return torch.clamp(matrix_exp(x)[:, :, :2].reshape(-1, 6), -1, 1)


class TanHMatrix(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 9 == 0, f"Invalid shape {x.shape}"
        shape = x.shape
        n_mats = shape[-1] // 9
        x = x.view(-1, n_mats, 9)
        x = torch.tanh(x)
        x = rot_mat_svd(x)
        return x.reshape(*shape)


class TanHMatrixOffset(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("eye", torch.eye(3).flatten())

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 9 == 0, f"Invalid shape {x.shape}"
        shape = x.shape
        n_mats = shape[-1] // 9
        x = x.view(-1, n_mats, 9)
        x = (torch.tanh(x) + self.eye).clamp_max(1.0)
        x = rot_mat_svd(x)
        return x.reshape(*shape)


class TanHR6Ortho(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 6, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        return r6_orthonomalization(x)


class TanHR6OrthoOffset(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("eye", torch.eye(3).flatten()[:6])

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 6, f"Invalid shape {x.shape}"
        x = (torch.tanh(x) + self.eye).clamp_max(1.0)
        return r6_orthonomalization(x)


# Activations for pose control
class QuatPose(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 7, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        normq = x[..., 3:] / torch.norm(x[..., 3:], dim=-1, keepdim=True)
        return torch.cat((x[..., :3], normq), dim=-1)


class QuatPlusPose(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 7, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        w = x[..., [6]] / 2.0 + 0.5
        q = torch.cat((x[..., 3:6], w), dim=-1)
        q = q / torch.norm(q, dim=-1, keepdim=True)
        return torch.cat((x[..., :3], q), dim=-1)


class MatrixPose(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 12, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        return torch.cat((x[..., :3], rot_mat_svd(x[..., 3:])), dim=-1)


class R6Pose(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 9, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        return torch.cat((x[..., :3], r6_orthonomalization(x[..., 3:])), dim=-1)


# Activations for dual pose control
class QuatDualPose(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 14, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        normq1 = x[..., 3:7] / torch.norm(x[..., 3:7], dim=-1, keepdim=True)
        normq2 = x[..., 10:] / torch.norm(x[..., 10:], dim=-1, keepdim=True)
        return torch.cat((x[..., :3], normq1, x[..., 7:10], normq2), dim=-1)
    

class QuatPlusDualPose(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 14, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        w1 = x[..., [6]] / 2.0 + 0.5
        q1 = torch.cat((x[..., 3:6], w1), dim=-1)
        q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
        w2 = x[..., [13]] / 2.0 + 0.5
        q2 = torch.cat((x[..., 10:13], w2), dim=-1)
        q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
        return torch.cat((x[..., :3], q1, x[..., 7:10], q2), dim=-1)


class MatrixDualPose(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 24, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        mat1 = rot_mat_svd(x[..., 3:12])
        mat2 = rot_mat_svd(x[..., 15:])
        return torch.cat((x[..., :3], mat1, x[..., 12:15], mat2), dim=-1)


class R6DualPose(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 18, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        r6_1 = r6_orthonomalization(x[..., 3:9])
        r6_2 = r6_orthonomalization(x[..., 12:])
        return torch.cat((x[..., :3], r6_1, x[..., 9:12], r6_2), dim=-1)


# Activations for manipulator control (pose + gripper)
class QuatManipulator(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 8, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        normq = x[..., 3:7] / torch.norm(x[..., 3:7], dim=-1, keepdim=True)
        return torch.cat((x[..., :3], normq, x[..., 7:]), dim=-1)


class MatrixManipulator(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 13, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        mat1 = rot_mat_svd(x[..., 3:12])
        return torch.cat((x[..., :3], mat1, x[..., 12:]), dim=-1)


class R6Manipulator(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 10, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        return torch.cat((x[..., :4], r6_orthonomalization(x[..., 4:])), dim=-1)


# Activations for dual manipulator control
class QuatDualManipulator(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 16, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        normq1 = x[..., 3:7] / torch.norm(x[..., 3:7], dim=-1, keepdim=True)
        normq2 = x[..., 11:15] / torch.norm(x[..., 11:15], dim=-1, keepdim=True)
        return torch.cat((x[..., :3], normq1, x[..., 7:11], normq2, x[..., 15:]), dim=-1)


class MatrixDualManipulator(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 26, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        mat1 = rot_mat_svd(x[..., 3:12])
        mat2 = rot_mat_svd(x[..., 16:25])
        return torch.cat((x[..., :3], mat1, x[..., 12:16], mat2, x[..., 25:]), dim=-1)


class R6DualManipulator(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 20, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        r6_1 = r6_orthonomalization(x[..., 4:10])
        r6_2 = r6_orthonomalization(x[..., 14:])
        return torch.cat((x[..., :4], r6_1, x[..., 10:14], r6_2), dim=-1)


# Activations for quadrotor attitude control
class QuatQuadrotor(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 5, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        normq = x[..., 1:] / torch.norm(x[..., 1:], dim=-1, keepdim=True)
        return torch.cat((x[..., :1], normq), dim=-1)


class QuatQuadrotorOffset(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "eye", torch.cat([torch.zeros(1), torch.tensor([0.0, 0.0, 0.0, 1.0])])
        )
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 5, f"Invalid shape {x.shape}"
        x = (torch.tanh(x) + self.eye).clamp_max(1.0)
        normq = x[..., 1:] / torch.norm(x[..., 1:], dim=-1, keepdim=True)
        return torch.cat((x[..., :1], normq), dim=-1)


class QuatPlusQuadrotor(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 5, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        w = x[..., [4]] / 2.0 + 0.5
        q = torch.cat((x[..., 1:4], w), dim=-1)
        q = q / torch.norm(q, dim=-1, keepdim=True)
        return torch.cat((x[..., :1], q), dim=-1)


class MatrixQuadrotor(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 10, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        return torch.cat((x[..., :1], rot_mat_svd(x[..., 1:])), dim=-1)


class MatrixQuadrotorOffset(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "eye", torch.cat([torch.zeros(1), torch.eye(3).flatten()])
        )
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 10, f"Invalid shape {x.shape}"
        x = (torch.tanh(x) + self.eye).clamp_max(1.0)
        return torch.cat((x[..., :1], rot_mat_svd(x[..., 1:])), dim=-1)


class R6Quadrotor(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 7, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        return torch.cat((x[..., :1], r6_orthonomalization(x[..., 1:])), dim=-1)


class R6QuadrotorOffset(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "eye", torch.cat([torch.zeros(1), torch.eye(3).flatten()[:6]])
        )
    
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 7, f"Invalid shape {x.shape}"
        x = (torch.tanh(x) + self.eye).clamp_max(1.0)
        return torch.cat((x[..., :1], r6_orthonomalization(x[..., 1:])), dim=-1)
