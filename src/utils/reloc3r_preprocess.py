from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import transforms as tvf


def preprocess_for_reloc3r_differentiable(image_tensor: torch.Tensor, target_size: int = 224) -> torch.Tensor:
    if target_size not in [224, 512]:
        raise ValueError("target_size must be either 224 or 512")

    if image_tensor.dim() == 3 and image_tensor.shape[0] not in [1, 3]:
        image_tensor = image_tensor.permute(2, 0, 1)

    height, width = image_tensor.shape[-2:]
    short_side = min(height, width)

    if short_side == target_size:
        resized = image_tensor
    else:
        if height < width:
            new_h = target_size
            new_w = int(width * target_size / height)
        else:
            new_w = target_size
            new_h = int(height * target_size / width)

        inp = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
        resized = F.interpolate(inp, size=(new_h, new_w), mode="bilinear", align_corners=False)
        if image_tensor.dim() == 3:
            resized = resized.squeeze(0)

    return tvf.functional.center_crop(resized, target_size)
