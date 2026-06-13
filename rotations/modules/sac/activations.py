import torch
from torch import Tensor, nn


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
        return x.reshape(*shape)


class TanHQuatPlus(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 4 == 0, f"Invalid shape {x.shape}"
        shape = x.shape
        n_quats = shape[-1] // 4
        x = x.view(-1, n_quats, 4)
        x[..., 3] = torch.sigmoid(x[..., 3])
        x[..., :3] = torch.tanh(x[..., :3])
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
        return x.reshape(*shape)


class TanHR6OrthoOffset(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("eye", torch.eye(3).flatten()[:6])

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 6, f"Invalid shape {x.shape}"
        x = (torch.tanh(x) + self.eye).clamp_max(1.0)
        return x


class QuatPlusPose(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 7, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        w = x[..., [6]] / 2.0 + 0.5
        q = torch.cat((x[..., 3:6], w), dim=-1)
        return torch.cat((x[..., :3], q), dim=-1)


class QuatPlusDualPose(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 14, f"Invalid shape {x.shape}"
        x = torch.tanh(x)
        w1 = x[..., [6]] / 2.0 + 0.5
        q1 = torch.cat((x[..., 3:6], w1), dim=-1)
        w2 = x[..., [13]] / 2.0 + 0.5
        q2 = torch.cat((x[..., 10:13], w2), dim=-1)
        return torch.cat((x[..., :3], q1, x[..., 7:10], q2), dim=-1)
