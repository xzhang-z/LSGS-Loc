from __future__ import annotations

import math
from typing import Tuple

import torch
import kornia.geometry.liegroup as liegroup


def skew(v: torch.Tensor) -> torch.Tensor:
    v = v.float()
    s = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)
    s[..., 0, 1] = -v[..., 2]
    s[..., 0, 2] = v[..., 1]
    s[..., 1, 0] = v[..., 2]
    s[..., 1, 2] = -v[..., 0]
    s[..., 2, 0] = -v[..., 1]
    s[..., 2, 1] = v[..., 0]
    return s


def axis_angle_to_matrix_manual(v: torch.Tensor) -> torch.Tensor:
    dtype = v.dtype
    device = v.device
    theta = torch.norm(v, dim=-1, keepdim=True)
    is_zero = theta < (1e-12 if dtype == torch.float64 else 1e-6)

    i = torch.eye(3, dtype=dtype, device=device).expand(v.shape[:-1] + (3, 3))
    axis = v / (theta + torch.tensor(1e-12, dtype=dtype, device=device))
    k = skew(axis)
    k_sq = k @ k

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    one_minus_cos_t = 1 - cos_t

    r_non_zero = i + sin_t.unsqueeze(-1) * k + one_minus_cos_t.unsqueeze(-1) * k_sq
    return torch.where(is_zero.unsqueeze(-1), i, r_non_zero)


def so3_to_rotmat(log_R: torch.Tensor) -> torch.Tensor:
    return liegroup.so3.So3.exp(log_R).matrix()


def rotmat_to_so3(R: torch.Tensor) -> torch.Tensor:
    return liegroup.so3.So3.from_matrix(R).log()


def decompose_camtoworld(camtoworld: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return camtoworld[:3, :3], camtoworld[:3, 3]


def compose_camtoworld(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    c2w = torch.eye(4, device=R.device, dtype=R.dtype)
    c2w[:3, :3] = R
    c2w[:3, 3] = T
    return c2w


def compute_pose_error(
    R_est: torch.Tensor,
    T_est: torch.Tensor,
    R_gt: torch.Tensor,
    T_gt: torch.Tensor,
) -> Tuple[float, float]:
    rot_rel = R_est @ R_gt.T
    trace = torch.clamp(torch.trace(rot_rel), -1.0, 3.0)
    rot_error_rad = torch.acos((trace - 1) / 2)
    rot_error_deg = rot_error_rad * 180 / math.pi
    trans_error = torch.linalg.norm(T_est - T_gt)
    return rot_error_deg.item(), trans_error.item()
