from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
from tqdm.auto import tqdm

from src.utils.loss import compute_loss
from src.utils.pose_utils import (
    axis_angle_to_matrix_manual,
    compose_camtoworld,
    decompose_camtoworld,
    rotmat_to_so3,
)


def optimize_pose_by_render_only(
    initial_camtoworld: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    ground_truth_image: torch.Tensor,
    render_rgbd_fn: Callable[[torch.Tensor, torch.Tensor, int, int], Tuple[torch.Tensor, torch.Tensor]],
    num_optimization_steps: int = 200,
    learning_rate: float = 1e-3,
    patience: int = 50,
    scheduler_t_max: int = 300,
    cam_dir: Optional[str] = None,
    verbose: bool = False,
    show_progress: bool = True,
    progress_desc: Optional[str] = None,
) -> Dict:
    """
    Args:
    - `initial_camtoworld`: initial pose [4,4]
    - `K`: camera intrinsics [3,3]
    - `width`, `height`: rendering resolution
    - `ground_truth_image`: GT image [H,W,3], float tensor
    - `render_rgbd_fn`: callback, input `(viewmats, Ks, width, height)`, output `(rgb, depth)`

    Returns:
    - `best_est_camtoworld`: best optimized pose
    - `best_loss`, `best_iteration`
    - `trajectory`: per-step pose trajectory [N,4,4]
    - `best_rendered_image`: best rendered image (may be None)
    """
    device = initial_camtoworld.device

    initial_R, initial_T = decompose_camtoworld(initial_camtoworld)

    optimized_log_R = rotmat_to_so3(initial_R).clone().detach().to(torch.float64)
    optimized_log_R.requires_grad_(True)

    optimized_T = initial_T.clone().detach().to(torch.float64)
    optimized_T.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [
            {"params": optimized_log_R, "lr": learning_rate * 1.8},
            {"params": optimized_T, "lr": learning_rate * 1.8},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_t_max)

    best_pose = {
        "R": initial_R.clone().detach(),
        "T": initial_T.clone().detach(),
        "loss": float("inf"),
        "iteration": 0,
        "image": None,
    }

    no_improve = 0
    trajectory = []

    step_iter = range(num_optimization_steps)
    if show_progress:
        step_iter = tqdm(
            step_iter,
            desc=progress_desc or "render_opt",
            total=num_optimization_steps,
            leave=False,
            dynamic_ncols=True,
        )

    for step in step_iter:
        optimizer.zero_grad()

        current_R = axis_angle_to_matrix_manual(optimized_log_R)
        current_camtoworld = compose_camtoworld(current_R, optimized_T)
        trajectory.append(current_camtoworld.detach().clone())

        viewmat = torch.linalg.inv(current_camtoworld)[None].to(torch.float32)
        rendered_image, _ = render_rgbd_fn(viewmat, K[None], width, height)

        loss = compute_loss(
            rendered_image,
            ground_truth_image,
            cam_dir=cam_dir,
            iter=step,
            verbose=(verbose and not show_progress),
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_pose["loss"]:
            best_pose["R"] = current_camtoworld.detach().clone()[:3, :3]
            best_pose["T"] = current_camtoworld.detach().clone()[:3, 3]
            best_pose["loss"] = loss.item()
            best_pose["iteration"] = step
            best_pose["image"] = rendered_image.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping triggered at iteration {step}")
                break

        if show_progress:
            step_iter.set_postfix(
                {
                    "loss": f"{loss.item():.5f}",
                    "best": f"{best_pose['loss']:.5f}",
                }
            )

    best_est_camtoworld = compose_camtoworld(best_pose["R"], best_pose["T"]).to(device)

    return {
        "best_est_camtoworld": best_est_camtoworld,
        "best_loss": best_pose["loss"],
        "best_iteration": best_pose["iteration"],
        "trajectory": torch.stack(trajectory) if len(trajectory) > 0 else torch.empty((0, 4, 4), device=device),
        "best_rendered_image": best_pose["image"],
    }
