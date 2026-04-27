from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from pycolmap import SceneManager


@dataclass
class ColmapCameraRecord:
    name: str
    K: torch.Tensor
    camtoworld: torch.Tensor
    height: int
    width: int


def _resolve_colmap_sparse_dir(data_dir: Path) -> Path:
    p0 = data_dir / "sparse" / "0"
    p1 = data_dir / "sparse"
    if p0.exists():
        return p0
    if p1.exists():
        return p1
    raise FileNotFoundError(f"COLMAP sparse directory not found: {p0} or {p1}")


def extract_matches_from_log(log_file_path: str, keep_top1_only: bool = True) -> List[Dict[str, str]]:
    """Extract query->matched relations from retrieval result text.

    Compatible formats:
    - Query image: xxx.png
    - Top 1 matched database image: yyy.png (distance: 0.1234)
    """
    query_pattern = re.compile(r"Query image:\s*(.*?\.(?:png|jpg|jpeg|JPG|JPEG))", re.IGNORECASE)
    matched_pattern = re.compile(
        r"(?:Top\s*(\d+)\s*)?(?:matched database image|matched image):?\s*"
        r"(.*?\.(?:png|jpg|jpeg|JPG|JPEG))\s*\(distance:\s*([0-9]+(?:\.[0-9]+)?)\)",
        re.IGNORECASE,
    )

    matches: List[Dict[str, str]] = []
    current_query: Optional[str] = None

    with open(log_file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            q = query_pattern.search(line)
            if q:
                current_query = q.group(1).strip()
                continue

            m = matched_pattern.search(line)
            if m and current_query:
                rank_str = m.group(1)
                rank = int(rank_str) if rank_str else 1
                matched_image = m.group(2).strip()
                distance = m.group(3).strip()

                if (not keep_top1_only) or (rank == 1):
                    matches.append(
                        {
                            "query_image": current_query,
                            "matched_image": matched_image,
                            "distance": distance,
                            "rank": str(rank),
                        }
                    )

                if keep_top1_only:
                    current_query = None

    return matches


def load_colmap_camera_records(data_dir: str, factor: int = 1) -> Dict[str, ColmapCameraRecord]:
    """Read COLMAP cameras and poses with pycolmap, without loading image pixels."""
    data_path = Path(data_dir).expanduser().resolve()
    colmap_dir = _resolve_colmap_sparse_dir(data_path)

    manager = SceneManager(str(colmap_dir))
    manager.load_cameras()
    manager.load_images()

    records: Dict[str, ColmapCameraRecord] = {}
    for image_id in manager.images:
        im = manager.images[image_id]
        cam = manager.cameras[im.camera_id]

        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        if factor > 1:
            K[:2, :] /= float(factor)

        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        rot = im.R().astype(np.float32)
        trans = im.tvec.astype(np.float32).reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], axis=1), bottom], axis=0)
        camtoworld = np.linalg.inv(w2c).astype(np.float32)

        width = int(cam.width // factor) if factor > 1 else int(cam.width)
        height = int(cam.height // factor) if factor > 1 else int(cam.height)
        name = str(im.name)

        records[name] = ColmapCameraRecord(
            name=name,
            K=torch.from_numpy(K),
            camtoworld=torch.from_numpy(camtoworld),
            height=height,
            width=width,
        )

    return records


def _find_record(records: Dict[str, ColmapCameraRecord], image_name: str) -> Optional[ColmapCameraRecord]:
    if image_name in records:
        return records[image_name]

    q_base = Path(image_name).name
    for k, v in records.items():
        if Path(k).name == q_base:
            return v
    return None


def _decompose_w2c_from_c2w(c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w2c = torch.linalg.inv(c2w)
    return w2c[:3, :3], w2c[:3, 3]


def build_query_camera_cache_entries(
    matches: List[Dict[str, str]],
    db_records: Dict[str, ColmapCameraRecord],
    query_records: Dict[str, ColmapCameraRecord],
    world_size: int = 1,
    world_rank: int = 0,
) -> List[Dict]:
    """Build query camera cache entries based on retrieval matches.

    Fields per cache entry:
    - K, camtoworld, height, width, name
    - real_R, real_T 
    - transform (identity by default)
    """
    query_cameras: List[Dict] = []

    for match_idx, m in enumerate(matches):
        if match_idx % world_size != world_rank:
            continue

        query_img_name = m["query_image"]
        matched_img_name = m["matched_image"]

        db_rec = _find_record(db_records, matched_img_name)
        if db_rec is None:
            continue

        q_rec = _find_record(query_records, query_img_name)
        real_R, real_T = (None, None)
        if q_rec is not None:
            real_R, real_T = _decompose_w2c_from_c2w(q_rec.camtoworld)
            real_R = real_R.clone()
            real_T = real_T.clone()

        query_cameras.append(
            {
                "K": db_rec.K.clone().cpu(),
                "camtoworld": db_rec.camtoworld.clone().cpu(),
                "height": int(db_rec.height),
                "width": int(db_rec.width),
                "name": str(query_img_name),
                "real_R": real_R.cpu() if real_R is not None else None,
                "real_T": real_T.cpu() if real_T is not None else None,
                "transform": torch.eye(4, dtype=torch.float32),
            }
        )

    return query_cameras


def save_query_camera_cache(query_cameras: List[Dict], cache_path: str) -> None:
    path = Path(cache_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(query_cameras, path)


def load_query_camera_cache(cache_path: str, device: Optional[torch.device] = None) -> List[Dict]:
    path = Path(cache_path).expanduser().resolve()
    data = torch.load(path, map_location="cpu")

    if device is None:
        return data

    moved: List[Dict] = []
    for item in data:
        moved.append(
            {
                "K": item["K"].to(device),
                "camtoworld": item["camtoworld"].to(device),
                "height": item["height"],
                "width": item["width"],
                "name": item["name"],
                "real_R": item["real_R"].to(device) if item.get("real_R") is not None else None,
                "real_T": item["real_T"].to(device) if item.get("real_T") is not None else None,
                "transform": item["transform"].to(device) if item.get("transform") is not None else None,
            }
        )
    return moved


def prepare_query_camera_cache(
    retrieval_file: str,
    database_dir: str,
    query_dir: str,
    cache_path: str,
    factor: int = 1,
    world_size: int = 1,
    world_rank: int = 0,
    keep_top1_only: bool = True,
    force_rebuild: bool = False,
) -> List[Dict]:
    """Read matching relations and build query camera cache.
    
    """
    cache_file = Path(cache_path).expanduser().resolve()
    if cache_file.exists() and not force_rebuild:
        return load_query_camera_cache(str(cache_file), device=None)

    matches = extract_matches_from_log(retrieval_file, keep_top1_only=keep_top1_only)
    db_records = load_colmap_camera_records(database_dir, factor=factor)
    query_records = load_colmap_camera_records(query_dir, factor=factor)

    entries = build_query_camera_cache_entries(
        matches=matches,
        db_records=db_records,
        query_records=query_records,
        world_size=world_size,
        world_rank=world_rank,
    )
    save_query_camera_cache(entries, str(cache_file))
    return entries
