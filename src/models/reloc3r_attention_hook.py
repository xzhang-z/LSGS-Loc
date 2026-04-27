from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CenterAttentionCapture:

    model: torch.nn.Module
    block_id: int = 5

    def __post_init__(self) -> None:
        self.center_attn_map: Optional[torch.Tensor] = None
        self._hook_handle = None

    def _validate_target_module(self):
        if not hasattr(self.model, "dec_blocks"):
            raise AttributeError("The given model does not contain `dec_blocks`; it is not the expected Reloc3rRelpose structure")

        if self.block_id < 0 or self.block_id >= len(self.model.dec_blocks):
            raise IndexError(
                f"block_id={self.block_id} is out of range, dec_blocks length is {len(self.model.dec_blocks)}"
            )

        blk = self.model.dec_blocks[self.block_id]
        if not hasattr(blk, "cross_attn"):
            raise AttributeError(f"dec_blocks[{self.block_id}] does not contain `cross_attn`")

        return blk.cross_attn

    @staticmethod
    def _center_indices(nq: int, device: torch.device) -> torch.Tensor:
        side = int(math.sqrt(nq))
        if side * side != nq:
            raise ValueError(f"Nq={nq} is not a perfect square, cannot extract center patch on a grid")

        mid = side // 2
        if side < 2:
            raise ValueError(f"Grid side length is {side}, cannot extract center 2x2 patch")

        idx = torch.tensor(
            [
                (mid - 1) * side + (mid - 1),
                (mid - 1) * side + mid,
                mid * side + (mid - 1),
                mid * side + mid,
            ],
            dtype=torch.long,
            device=device,
        )
        return idx

    def _pre_hook(self, module: torch.nn.Module, inputs):
        if len(inputs) < 3:
            return

        query = inputs[0]
        key = inputs[1]
        value = inputs[2]
        qpos = inputs[3] if len(inputs) > 3 else None
        kpos = inputs[4] if len(inputs) > 4 else None

        if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
            return

        with torch.no_grad():
            B, Nq, C = query.shape
            Nk = key.shape[1]

            q = (
                module.projq(query)
                .reshape(B, Nq, module.num_heads, C // module.num_heads)
                .permute(0, 2, 1, 3)
            )
            k = (
                module.projk(key)
                .reshape(B, Nk, module.num_heads, C // module.num_heads)
                .permute(0, 2, 1, 3)
            )
            _ = (
                module.projv(value)
                .reshape(B, Nk, module.num_heads, C // module.num_heads)
                .permute(0, 2, 1, 3)
            )

            if getattr(module, "rope", None) is not None and qpos is not None and kpos is not None:
                q = module.rope(q, qpos)
                k = module.rope(k, kpos)

            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)  # [B, heads, Nq, Nk]

            center_idx = self._center_indices(Nq, attn.device)
            center_attn = attn[:, :, center_idx, :]  # [B, heads, 4, Nk]

            self.center_attn_map = center_attn.detach().cpu()

    def attach(self) -> None:
        if self._hook_handle is not None:
            return
        target = self._validate_target_module()
        self._hook_handle = target.register_forward_pre_hook(self._pre_hook)

    def remove(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def clear(self) -> None:
        self.center_attn_map = None
