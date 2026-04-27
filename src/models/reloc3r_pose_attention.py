from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .reloc3r_attention_hook import CenterAttentionCapture
from src.utils.reloc3r_preprocess import preprocess_for_reloc3r_differentiable


try:
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model
except ImportError:
    _repo_root = Path(__file__).resolve().parents[2]
    _reloc3r_root = _repo_root / "third_party" / "ReLoc3r"
    if _reloc3r_root.is_dir():
        sys.path.insert(0, str(_reloc3r_root))
    from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model


def setup_reloc3r_estimator(
    model_args: str,
    device: torch.device,
    block_id: int = 5,
):
    reloc3r_relpose = setup_reloc3r_relpose_model(model_args=model_args, device=device)
    for param in reloc3r_relpose.parameters():
        param.requires_grad = False
    reloc3r_relpose.eval()

    attn_capture = CenterAttentionCapture(reloc3r_relpose, block_id=block_id)
    attn_capture.attach()

    return reloc3r_relpose, attn_capture


def _remove_one_point(coord_value_pairs: List[Tuple[Tuple[int, int], float]], side: int):
    corners = [(0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1)]
    first_corner = [(0, 0)]

    for i, ((x, y), _) in enumerate(coord_value_pairs):
        if (x, y) in first_corner:
            coord_value_pairs.pop(i)
            return

    for i, ((x, y), _) in enumerate(coord_value_pairs):
        if (x, y) in corners:
            coord_value_pairs.pop(i)
            return

    for i, ((x, y), _) in enumerate(coord_value_pairs):
        if x == 0 or x == (side - 1) or y == 0 or y == (side - 1):
            coord_value_pairs.pop(i)
            return

    if coord_value_pairs:
        coord_value_pairs.sort(key=lambda x: x[1])
        coord_value_pairs.pop(0)


def extract_top_coords_from_center_attention(
    center_attn_map: torch.Tensor,
) -> List[Tuple[int, int]]:
    if center_attn_map is None:
        raise ValueError("center_attn_map is empty, please run a model forward pass first")

    if center_attn_map.dim() == 4:
        attn_per_query = center_attn_map.mean(dim=1)
    elif center_attn_map.dim() == 3:
        attn_per_query = center_attn_map
    else:
        raise ValueError(f"Unsupported center_attn_map dimensions: {center_attn_map.shape}")

    B, Nq_center, Nk = attn_per_query.shape
    if B < 1:
        return []

    side = int(math.sqrt(Nk))
    if side * side != Nk:
        raise ValueError(f"Nk={Nk} is not a perfect square, cannot reshape to a 2D attention map")

    top_coords: List[Tuple[int, int]] = []

    for query_idx in range(Nq_center):
        patch_attn = attn_per_query[0, query_idx]
        attn_map_2d = patch_attn.reshape(side, side).detach().cpu().numpy()

        flat_attn = attn_map_2d.flatten()
        top5_indices = np.argsort(flat_attn)[-5:]
        top5_values = flat_attn[top5_indices]

        top5_positions = np.unravel_index(top5_indices, attn_map_2d.shape)
        top5_coords = list(zip(top5_positions[1], top5_positions[0]))

        coord_value_pairs = list(zip(top5_coords, top5_values.tolist()))

        _remove_one_point(coord_value_pairs, side=side)
        _remove_one_point(coord_value_pairs, side=side)

        if coord_value_pairs:
            coord_value_pairs.sort(key=lambda x: x[1], reverse=True)
            best_coord, _ = coord_value_pairs[0]
            top_coords.append(best_coord)

    return top_coords


def _compute_crop_region(h_orig: int, w_orig: int) -> Tuple[int, int, int, int, int]:
    L = min(h_orig, w_orig)
    if h_orig > w_orig:
        crop_top = (h_orig - L) // 2
        crop_bottom = crop_top + L
        crop_left = 0
        crop_right = w_orig
    else:
        crop_left = (w_orig - L) // 2
        crop_right = crop_left + L
        crop_top = 0
        crop_bottom = h_orig
    return crop_top, crop_bottom, crop_left, crop_right, L


def map_grid_coords_to_pixel_regions(
    top_coords: List[Tuple[int, int]],
    h_orig: int,
    w_orig: int,
    grid_side: int = 14,
) -> List[Tuple[int, int, int, int]]:
    crop_top, crop_bottom, crop_left, crop_right, L = _compute_crop_region(h_orig, w_orig)
    grid_size = L // grid_side

    regions = []
    for x_grid, y_grid in top_coords:
        x_start = crop_left + x_grid * grid_size
        y_start = crop_top + y_grid * grid_size
        x_end = min(x_start + grid_size, w_orig)
        y_end = min(y_start + grid_size, h_orig)
        regions.append((x_start, y_start, x_end, y_end))

    return regions


@torch.no_grad()
def run_reloc3r_estimation_with_attention(
    reloc3r_relpose: torch.nn.Module,
    attn_capture: CenterAttentionCapture,
    rendered_image: torch.Tensor,
    ground_truth_image: torch.Tensor,
) -> Dict:
    h_orig, w_orig = rendered_image.shape[:2]

    rendered_proc = preprocess_for_reloc3r_differentiable(
        rendered_image.permute(2, 0, 1).unsqueeze(0)
    )
    gt_proc = preprocess_for_reloc3r_differentiable(
        ground_truth_image.permute(2, 0, 1).unsqueeze(0)
    )

    _, pos2 = reloc3r_relpose({"img": rendered_proc}, {"img": gt_proc})
    pose_matrix = pos2["pose"].squeeze(0).clone()

    center_attn_map = attn_capture.center_attn_map
    top_coords = extract_top_coords_from_center_attention(center_attn_map)
    top_regions = map_grid_coords_to_pixel_regions(top_coords, h_orig=h_orig, w_orig=w_orig, grid_side=14)

    return {
        "pose_matrix": pose_matrix,
        "center_attn_map": center_attn_map,
        "top_coords": top_coords,
        "top_regions": top_regions,
    }
