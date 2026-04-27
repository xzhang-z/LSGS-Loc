from __future__ import annotations

from typing import List, Tuple

import torch


def gather_points_from_attention_regions(
    points_3d: torch.Tensor,
    top_coords: List[Tuple[int, int]],
    crop_left: int,
    crop_top: int,
    grid_size: int,
    image_h: int,
    image_w: int,
) -> torch.Tensor:

    region_points = []
    for x_grid, y_grid in top_coords:
        x_start = crop_left + x_grid * grid_size
        y_start = crop_top + y_grid * grid_size
        x_end = x_start + grid_size
        y_end = y_start + grid_size

        x_indices = torch.arange(x_start, min(x_end, image_w), device=points_3d.device, dtype=torch.long)
        y_indices = torch.arange(y_start, min(y_end, image_h), device=points_3d.device, dtype=torch.long)
        if x_indices.numel() == 0 or y_indices.numel() == 0:
            continue

        yy, xx = torch.meshgrid(y_indices, x_indices, indexing="ij")
        point_indices = yy * image_w + xx
        region_points.append(points_3d[point_indices.flatten()])

    if region_points:
        return torch.cat(region_points, dim=0)
    return torch.empty((0, 3), device=points_3d.device, dtype=points_3d.dtype)


def compute_focus_center_and_radius(
    points_3d: torch.Tensor,
    selected_points: torch.Tensor,
    sample_size: int = 10000,
    radius_quantile: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if selected_points.numel() > 0:
        center = selected_points.mean(dim=0)
    else:
        center = points_3d.mean(dim=0)

    if points_3d.shape[0] > sample_size:
        rand_idx = torch.randint(0, points_3d.shape[0], (sample_size,), device=points_3d.device, dtype=torch.long)
        sampled = points_3d[rand_idx]
    else:
        sampled = points_3d

    sampled_dist = torch.linalg.norm(sampled - center, dim=1)
    radius = torch.quantile(sampled_dist, radius_quantile)
    return center, radius


def depth_to_world_points(depth: torch.Tensor, K: torch.Tensor, camtoworld: torch.Tensor) -> torch.Tensor:

    h, w = depth.shape
    device = depth.device
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")

    z = depth
    x = (x - K[0, 2]) * z / K[0, 0]
    y = (y - K[1, 2]) * z / K[1, 1]

    cam_points = torch.stack([x, y, z, torch.ones_like(z)], dim=-1).view(-1, 4)
    world_points = (camtoworld @ cam_points.T).T[:, :3]
    return world_points


def project_points_for_color_loss(
    world_points: torch.Tensor,
    world_colors: torch.Tensor,
    cam_center: torch.Tensor,
    R: torch.Tensor,
    K: torch.Tensor,
    H_gt: int,
    W_gt: int,
    n_sub: int = 200000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    n_total = world_points.shape[0]
    if n_total > n_sub:
        rand_indices = torch.randint(0, n_total, (n_sub,), device=world_points.device)
        cur_points = world_points[rand_indices]
        cur_colors = world_colors[rand_indices]
    else:
        cur_points = world_points
        cur_colors = world_colors

    R_inv = R.T
    Xc = (R_inv @ (cur_points - cam_center).T).T
    mask_front = Xc[:, 2] > 1e-3
    Xc = Xc[mask_front]
    cur_colors = cur_colors[mask_front]

    uv = (K[:2, :2] @ (Xc[:, :2] / Xc[:, 2:3]).T).T + K[:2, 2]
    u, v = uv[:, 0], uv[:, 1]
    valid = (u >= 0) & (u < W_gt - 1) & (v >= 0) & (v < H_gt - 1)

    return u[valid], v[valid], cur_colors[valid]


def sample_gt_colors_bilinear(gt_img: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

    H_gt, W_gt = gt_img.shape[:2]
    u0, v0 = torch.floor(u).long(), torch.floor(v).long()
    u1, v1 = u0 + 1, v0 + 1
    du, dv = u - u0.float(), v - v0.float()

    u0, u1 = u0.clamp(0, W_gt - 1), u1.clamp(0, W_gt - 1)
    v0, v1 = v0.clamp(0, H_gt - 1), v1.clamp(0, H_gt - 1)

    c00 = gt_img[v0, u0]
    c10 = gt_img[v0, u1]
    c01 = gt_img[v1, u0]
    c11 = gt_img[v1, u1]

    c0 = c00 * (1 - du).unsqueeze(-1) + c10 * du.unsqueeze(-1)
    c1 = c01 * (1 - du).unsqueeze(-1) + c11 * du.unsqueeze(-1)
    return c0 * (1 - dv).unsqueeze(-1) + c1 * dv.unsqueeze(-1)


def select_distance_by_gradient_strength(
    losses: torch.Tensor,
    d_candidates: torch.Tensor,
    window_size: int = 10,
) -> Tuple[float, float, str]:

    if losses.numel() == 0:
        raise ValueError("losses is empty; cannot select the best distance")

    if losses.numel() < 2:
        best_idx = int(torch.argmin(losses).item())
        best_d = float(d_candidates[best_idx].item())
        best_loss = float(losses[best_idx].item())
        msg = f"Insufficient candidates; directly select minimum idx={best_idx}, d={best_d:.4f}, loss={best_loss:.6f}"
        return best_d, best_loss, msg

    w = int(window_size)
    num_d = losses.shape[0]

    counts = torch.zeros(num_d, dtype=torch.int32, device=losses.device)
    avg_grad_strength = torch.zeros(num_d, dtype=torch.float32, device=losses.device)

    grads = losses[1:] - losses[:-1]
    grad_mag = grads.abs()

    for i in range(0, num_d - w + 1):
        window = losses[i : i + w]
        local_min_idx = int(torch.argmin(window).item())
        if local_min_idx == 0 or local_min_idx == (w - 1):
            continue

        global_idx = i + local_min_idx
        counts[global_idx] += 1
        grad_window = grad_mag[i : i + w - 1]
        avg_grad_strength[global_idx] += grad_window.mean()

    if avg_grad_strength.max() > 0:
        best_idx = int(torch.argmax(avg_grad_strength).item())
        best_d = float(d_candidates[best_idx].item())
        best_loss = float(losses[best_idx].item())
        msg = (
            f"Selected by accumulated gradient strength idx={best_idx}, d={best_d:.4f}, loss={best_loss:.6f}, "
            f"grad_strength={avg_grad_strength[best_idx]:.6f}, votes={counts[best_idx].item()}"
        )
        return best_d, best_loss, msg

    best_idx = int(torch.argmin(losses).item())
    best_d = float(d_candidates[best_idx].item())
    best_loss = float(losses[best_idx].item())
    msg = f"No valid local valley detected; fallback to global minimum idx={best_idx}, d={best_d:.4f}, loss={best_loss:.6f}"
    return best_d, best_loss, msg
