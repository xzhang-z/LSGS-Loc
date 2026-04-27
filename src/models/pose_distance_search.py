from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from src.utils.pointcloud_pose_search_utils import (
    compute_focus_center_and_radius,
    depth_to_world_points,
    gather_points_from_attention_regions,
    project_points_for_color_loss,
    sample_gt_colors_bilinear,
    select_distance_by_gradient_strength,
)


def refine_pose_with_pointcloud_and_distance_search(
    initial_camtoworld: torch.Tensor,
    pose_matrix: torch.Tensor,
    top_coords: List[Tuple[int, int]],
    points_3d: torch.Tensor,
    crop_left: int,
    crop_top: int,
    grid_size: int,
    image_h: int,
    image_w: int,
    K_1600: torch.Tensor,
    width_1600: int,
    height_1600: int,
    ground_truth_image: torch.Tensor | np.ndarray,
    render_rgbd_fn: Callable[[torch.Tensor, torch.Tensor, int, int], Tuple[torch.Tensor, torch.Tensor]],
    radius_scale: float = 1.15,
    num_distance_candidates: int = 50,
    min_valid_points: int = 1000,
    n_subsample_project: int = 200000,
    window_size: int = 10,
) -> Dict:

    est_camtoworld = initial_camtoworld @ pose_matrix
    est_R = est_camtoworld[:3, :3]

    selected_points = gather_points_from_attention_regions(
        points_3d=points_3d,
        top_coords=top_coords,
        crop_left=crop_left,
        crop_top=crop_top,
        grid_size=grid_size,
        image_h=image_h,
        image_w=image_w,
    )

    points_mean_top3, radius = compute_focus_center_and_radius(
        points_3d=points_3d,
        selected_points=selected_points,
        sample_size=10000,
        radius_quantile=0.75,
    )

    forward_dir = est_R[:, 2]
    forward_dir = forward_dir / torch.linalg.norm(forward_dir)

    desired_distance = radius * radius_scale
    new_cam_center = points_mean_top3 - desired_distance * forward_dir
    est_camtoworld[:3, 3] = new_cam_center


    est_viewmat = torch.linalg.inv(est_camtoworld)[None]
    est_render_rgb, est_depth = render_rgbd_fn(est_viewmat, K_1600[None], width_1600, height_1600)

    world_points = depth_to_world_points(est_depth, K_1600, est_camtoworld)
    world_colors = est_render_rgb.reshape(-1, 3)

    R = est_camtoworld[:3, :3]
    forward_dir = R[:, 2] / torch.linalg.norm(R[:, 2])

    C0 = est_camtoworld[:3, 3]
    d0 = torch.linalg.norm(points_mean_top3 - C0).item()
    d_min, d_max = max(0.01, d0 * 0.2), d0 * 4.0
    d_candidates = torch.linspace(d_min, d_max, num_distance_candidates, device=world_points.device)

    if isinstance(ground_truth_image, np.ndarray):
        gt_img = torch.from_numpy(ground_truth_image).to(world_points.device)
    else:
        gt_img = ground_truth_image.to(world_points.device)


    if gt_img.dtype != torch.float32:
        gt_img = gt_img.float()
    if gt_img.max() > 1.0:
        gt_img = gt_img / 255.0

    if gt_img.shape[0] != height_1600 or gt_img.shape[1] != width_1600:
        gt_img = torch.nn.functional.interpolate(
            gt_img.permute(2, 0, 1).unsqueeze(0),
            size=(height_1600, width_1600),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)

    H_gt, W_gt = height_1600, width_1600

    valid_losses: List[torch.Tensor] = []
    valid_ds: List[torch.Tensor] = []

    for d in d_candidates:
        cam_center = points_mean_top3 - d * forward_dir
        u, v, pred_colors = project_points_for_color_loss(
            world_points=world_points,
            world_colors=world_colors,
            cam_center=cam_center,
            R=R,
            K=K_1600,
            H_gt=H_gt,
            W_gt=W_gt,
            n_sub=n_subsample_project,
        )

        if u.numel() < min_valid_points:
            continue

        gt_colors = sample_gt_colors_bilinear(gt_img, u, v)
        loss = torch.mean((pred_colors - gt_colors) ** 2)
        valid_losses.append(loss)
        valid_ds.append(d)

    if len(valid_losses) == 0:
        best_d = float(d0)
        best_loss = float("inf")
        select_out = "Distance search found no valid candidates, fallback to initial d0"
    else:
        losses_tensor = torch.stack(valid_losses)
        ds_tensor = torch.stack(valid_ds)
        best_d, best_loss, select_out = select_distance_by_gradient_strength(
            losses=losses_tensor,
            d_candidates=ds_tensor,
            window_size=window_size,
        )

    final_center = points_mean_top3 - best_d * forward_dir
    best_est_camtoworld = est_camtoworld.clone()
    best_est_camtoworld[:3, 3] = final_center

    return {
        "best_est_camtoworld": best_est_camtoworld,
        "est_camtoworld_after_pointcloud": est_camtoworld,
        "points_mean_top3": points_mean_top3,
        "radius": radius,
        "best_d": best_d,
        "best_loss": best_loss,
        "select_out": select_out,
    }
