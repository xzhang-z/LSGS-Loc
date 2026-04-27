from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import laplacian


image_status: Dict[str, Dict] = {}


def save_annotated_render(
    render_out: torch.Tensor,
    valid_patches: List[Tuple],
    cam_dir: str,
    iter: int,
    num_blocks_h: int,
    num_blocks_w: int,
    laplacian_threshold: float,
) -> None:
    render_np = render_out[0].detach().cpu().numpy()

    if render_np.max() > 1.0:
        render_np = (render_np - render_np.min()) / (render_np.max() - render_np.min()) * 255
    else:
        render_np = render_np * 255
    render_np = render_np.astype(np.uint8)

    if len(render_np.shape) == 2:
        render_np = np.stack([render_np] * 3, axis=-1)
    elif render_np.shape[-1] == 1:
        render_np = np.concatenate([render_np] * 3, axis=-1)

    annotated_img = render_np.copy()

    h, w = annotated_img.shape[:2]
    patch_h, patch_w = h // num_blocks_h, w // num_blocks_w

    for i in range(1, num_blocks_h):
        cv2.line(annotated_img, (0, i * patch_h), (w, i * patch_h), (128, 128, 128), 1)
    for j in range(1, num_blocks_w):
        cv2.line(annotated_img, (j * patch_w, 0), (j * patch_w, h), (128, 128, 128), 1)

    for i, j, h_start, h_end, w_start, w_end, laplacian_var, loss in valid_patches:
        cv2.rectangle(annotated_img, (w_start, h_start), (w_end, w_end), (0, 255, 0), 2)

        center_x = (w_start + w_end) // 2
        center_y = (h_start + h_end) // 2

        laplacian_text = f"Laplacian: {laplacian_var:.8f}"
        loss_text = f"Loss: {loss:.4f}"

        laplacian_size = cv2.getTextSize(laplacian_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        loss_size = cv2.getTextSize(loss_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        total_height = laplacian_size[1] + loss_size[1] + 10

        text_x = center_x - max(laplacian_size[0], loss_size[0]) // 2
        laplacian_y = center_y - total_height // 2 + laplacian_size[1]
        loss_y = laplacian_y + loss_size[1] + 5

        bg_width = max(laplacian_size[0], loss_size[0]) + 10
        bg_height = total_height + 5
        cv2.rectangle(
            annotated_img,
            (text_x - 5, center_y - total_height // 2 - 5),
            (text_x + bg_width, center_y + total_height // 2),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated_img,
            laplacian_text,
            (text_x, laplacian_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated_img,
            loss_text,
            (text_x, loss_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )

    title_text = f"Iter: {iter}, Valid patches: {len(valid_patches)}, Threshold: {laplacian_threshold}"
    cv2.putText(annotated_img, title_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    output_path = os.path.join(cam_dir, f"annotated_render_{iter:06d}.png")
    cv2.imwrite(output_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    print(f"Saved annotated rendered image: {output_path}")


def compute_loss(
    render_out: torch.Tensor,
    gt: torch.Tensor,
    cam_dir: str | None = None,
    iter: int = 0,
    laplacian_threshold: float = 0.02,
    verbose: bool = True,
) -> torch.Tensor:

    def add_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor.unsqueeze(0)
        return tensor

    render_out = add_batch_dim(render_out)
    gt = add_batch_dim(gt)

    _, h, w, _ = render_out.shape

    image_id = cam_dir if cam_dir is not None else f"{h}x{w}"

    if iter == 0:
        if image_id in image_status:
            del image_status[image_id]

        num_blocks_h, num_blocks_w = 10, 10
        patch_h = h // num_blocks_h
        patch_w = w // num_blocks_w

        total_loss = 0.0
        valid_patch_count = 0

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                h_start = i * patch_h
                h_end = (i + 1) * patch_h if i < num_blocks_h - 1 else h
                w_start = j * patch_w
                w_end = (j + 1) * patch_w if j < num_blocks_w - 1 else w

                render_patch = render_out[:, h_start:h_end, w_start:w_end, :]
                gt_patch = gt[:, h_start:h_end, w_start:w_end, :]

                patch_rgb_loss = F.l1_loss(render_patch, gt_patch)

                render_patch_permuted = render_patch.permute(0, 3, 1, 2)
                laplacian_response = laplacian(render_patch_permuted, kernel_size=3)
                laplacian_var = torch.var(laplacian_response)
                laplacian_value_scale = laplacian_var.item() if torch.is_tensor(laplacian_var) else laplacian_var
                laplacian_value = laplacian_value_scale * 255.0 * 255.0

                if laplacian_value >= laplacian_threshold:
                    total_loss += patch_rgb_loss
                    valid_patch_count += 1

        total_patches = num_blocks_h * num_blocks_w
        valid_ratio = valid_patch_count / total_patches if total_patches > 0 else 0

        image_status[image_id] = {
            "use_full_image": valid_ratio >= 0.5,
            "valid_ratio": valid_ratio,
            "valid_patch_count": valid_patch_count,
        }

        if valid_patch_count > 0:
            total_loss /= valid_patch_count
        else:
            total_loss = F.l1_loss(render_out, gt)

        if verbose:
            print(
                f"Iter: {iter}, Valid patches: {valid_patch_count}/{total_patches} ({valid_ratio:.2%}), "
                f"Use full image: {image_status[image_id]['use_full_image']}, Loss: {total_loss.item():.6f}"
            )

    else:
        if image_id in image_status and image_status[image_id]["use_full_image"]:
            total_loss = F.l1_loss(render_out, gt)
            num_blocks_h, num_blocks_w = 10, 10
            valid_patch_count = num_blocks_h * num_blocks_w

            if verbose:
                print(
                    f"Iter: {iter}, Using full image L1 loss "
                    f"(valid ratio: {image_status[image_id]['valid_ratio']:.2%}), Loss: {total_loss.item():.6f}"
                )
        else:
            num_blocks_h, num_blocks_w = 10, 10
            patch_h = h // num_blocks_h
            patch_w = w // num_blocks_w

            total_loss = 0.0
            valid_patch_count = 0

            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    h_start = i * patch_h
                    h_end = (i + 1) * patch_h if i < num_blocks_h - 1 else h
                    w_start = j * patch_w
                    w_end = (j + 1) * patch_w if j < num_blocks_w - 1 else w

                    render_patch = render_out[:, h_start:h_end, w_start:w_end, :]
                    gt_patch = gt[:, h_start:h_end, w_start:w_end, :]

                    patch_rgb_loss = F.l1_loss(render_patch, gt_patch)

                    render_patch_permuted = render_patch.permute(0, 3, 1, 2)
                    laplacian_response = laplacian(render_patch_permuted, kernel_size=3)
                    laplacian_var = torch.var(laplacian_response)
                    laplacian_value_scale = laplacian_var.item() if torch.is_tensor(laplacian_var) else laplacian_var
                    laplacian_value = laplacian_value_scale * 255.0 * 255.0

                    if laplacian_value >= laplacian_threshold:
                        total_loss += patch_rgb_loss
                        valid_patch_count += 1

            if valid_patch_count > 0:
                total_loss /= valid_patch_count
            else:
                total_loss = F.l1_loss(render_out, gt)

            if image_id not in image_status:
                total_patches = num_blocks_h * num_blocks_w
                valid_ratio = valid_patch_count / total_patches if total_patches > 0 else 0
                image_status[image_id] = {
                    "use_full_image": valid_ratio >= 0.8,
                    "valid_ratio": valid_ratio,
                    "valid_patch_count": valid_patch_count,
                }

            if verbose:
                print(
                    f"Iter: {iter}, Valid patches: {valid_patch_count}/{num_blocks_h * num_blocks_w}, "
                    f"Loss: {total_loss.item():.6f}"
                )

    return total_loss
