from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def compute_topk_matches_pure_torch(
    db: torch.Tensor,
    qu: torch.Tensor,
    top_k: int,
    method: str = "cosine",
    norm_descs: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if qu.dim() == 1:
        qu = qu.unsqueeze(0)

    if norm_descs:
        db = F.normalize(db, dim=1)
        qu = F.normalize(qu, dim=1)

    if method == "cosine":
        sim = torch.matmul(qu, db.T)
        distances, indices = torch.topk(sim, k=top_k, dim=1, largest=True)
    elif method == "l2":
        dist = torch.cdist(qu, db, p=2)
        distances, indices = torch.topk(dist, k=top_k, dim=1, largest=False)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return distances, indices


def get_top_k_recall_pure_torch(
    top_k: List[int],
    db: torch.Tensor,
    qu: torch.Tensor,
    gt_pos: np.ndarray,
    method: str = "cosine",
    norm_descs: bool = True,
    use_percentage: bool = True,
    sub_sample_db: int = 1,
    sub_sample_qu: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, float]]:
    max_k = max(top_k)
    distances, indices = compute_topk_matches_pure_torch(
        db=db, qu=qu, top_k=max_k, method=method, norm_descs=norm_descs
    )

    recalls = {k: 0 for k in top_k}
    for i in range(len(indices)):
        gt = gt_pos[i * sub_sample_qu]
        for k in top_k:
            top_k_preds = (indices[i, :k] * sub_sample_db).detach().cpu().numpy()
            if np.any(np.isin(top_k_preds, gt)):
                recalls[k] += 1

    if use_percentage and len(indices) > 0:
        n_qu = len(indices)
        recalls = {k: v / n_qu for k, v in recalls.items()}

    return distances, indices, recalls


def build_ranked_results(
    query_image_full_paths: List[str],
    database_image_full_paths: List[str],
    distances: torch.Tensor,
    indices: torch.Tensor,
    query_root_dir: str,
    database_root_dir: str,
    top_k: int,
) -> List[dict]:
    results = []
    for i in range(len(query_image_full_paths)):
        query_full_path = query_image_full_paths[i]
        query_rel_path = os.path.relpath(query_full_path, query_root_dir)

        for k_idx in range(top_k):
            if k_idx >= indices.shape[1]:
                continue
            matched_db_idx = int(indices[i, k_idx].item())
            matched_distance = float(distances[i, k_idx].item())
            matched_db_full_path = database_image_full_paths[matched_db_idx]
            matched_db_rel_path = os.path.relpath(matched_db_full_path, database_root_dir)

            results.append(
                {
                    "query_image": query_rel_path,
                    "matched_database_image": matched_db_rel_path,
                    "distance": matched_distance,
                    "rank": k_idx + 1,
                }
            )
    return results


def save_match_results_txt(
    output_txt_file: str,
    results: List[dict],
    query_images_dir: str,
    database_images_dir: str,
    top_k: int,
) -> None:
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)
    with open(output_txt_file, "w", encoding="utf-8") as f:
        f.write(f"--- Image Retrieval Matches (Top {top_k}) ---\n")
        f.write(f"Query images root: {query_images_dir}\n")
        f.write(f"Database images root: {database_images_dir}\n")
        f.write("-" * 40 + "\n")

        current_query_image = None
        for res in results:
            if res["query_image"] != current_query_image:
                if current_query_image is not None:
                    f.write("-" * 30 + "\n")
                f.write(f"Query image: {res['query_image']}\n")
                current_query_image = res["query_image"]

            f.write(
                f"  Top {res['rank']} matched database image: {res['matched_database_image']} "
                f"(distance: {res['distance']:.4f})\n"
            )

        if current_query_image is not None:
            f.write("-" * 30 + "\n")
