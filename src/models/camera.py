from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class Camera:
    K: torch.Tensor
    camtoworld: torch.Tensor
    height: int
    width: int
    name: str
    real_R: Optional[torch.Tensor] = None
    real_T: Optional[torch.Tensor] = None
    transform: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.K.shape != (3, 3):
            raise ValueError(f"K matrix must be (3, 3), got {self.K.shape}")
        if self.camtoworld.shape != (4, 4):
            raise ValueError(f"camtoworld matrix must be (4, 4), got {self.camtoworld.shape}")

    def get_K(self) -> torch.Tensor:
        return self.K

    def get_camtoworld(self) -> torch.Tensor:
        return self.camtoworld

    def get_transform(self) -> Optional[torch.Tensor]:
        return self.transform

    def get_height(self) -> int:
        return self.height

    def get_width(self) -> int:
        return self.width

    def get_height14(self) -> int:
        return (self.height // 14) * 14

    def get_width14(self) -> int:
        return (self.width // 14) * 14

    def get_width1600(self) -> int:
        if self.width >= self.height:
            return 1600
        scale = 1600 / self.height
        return int(round(self.width * scale))

    def get_height1600(self) -> int:
        if self.height >= self.width:
            return 1600
        scale = 1600 / self.width
        return int(round(self.height * scale))

    def get_K1600(self) -> torch.Tensor:
        if self.width >= self.height:
            scale = 1600 / self.width
        else:
            scale = 1600 / self.height

        scale_matrix = torch.tensor(
            [[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
            dtype=self.K.dtype,
            device=self.K.device,
        )
        return scale_matrix @ self.K

    def get_name(self) -> str:
        return self.name

    def get_viewmat(self) -> torch.Tensor:
        return torch.linalg.inv(self.camtoworld)

    def set_real_pose(self, real_R: torch.Tensor, real_T: torch.Tensor) -> None:
        if real_R.shape != (3, 3) or real_T.shape != (3,):
            raise ValueError("real_R must be (3, 3) and real_T must be (3,)")
        self.real_R = real_R
        self.real_T = real_T

    def to(self, device: torch.device) -> "Camera":
        self.K = self.K.to(device)
        self.camtoworld = self.camtoworld.to(device)
        if self.real_R is not None:
            self.real_R = self.real_R.to(device)
        if self.real_T is not None:
            self.real_T = self.real_T.to(device)
        if self.transform is not None:
            self.transform = self.transform.to(device)
        return self
