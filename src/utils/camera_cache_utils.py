from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import torch

from src.models.camera import Camera


def extract_matches_from_log(log_file_path: str) -> List[Dict[str, str]]:
    matches = []
    query_pattern = re.compile(r"Query image:\s*(.*?\.(?:png|jpg|jpeg|JPG|JPEG))", re.IGNORECASE)
    matched_pattern = re.compile(
        r"(?:matched database image|matched image):?\s*(.*?\.(?:png|jpg|jpeg|JPG|JPEG))\s*\(distance:\s*(\d+\.\d+)\)",
        re.IGNORECASE,
    )

    with open(log_file_path, "r", encoding="utf-8") as f:
        current_query_image = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = query_pattern.search(line)
            if q:
                current_query_image = q.group(1).strip()
                continue
            m = matched_pattern.search(line)
            if m and current_query_image:
                matches.append(
                    {
                        "query_image": current_query_image,
                        "matched_image": m.group(1).strip(),
                        "distance": m.group(2).strip(),
                    }
                )
                current_query_image = None

    return matches


def save_query_cameras_to_cache(query_cameras: List[Camera], cache_path: Path) -> None:
    data_to_save = []
    for cam in query_cameras:
        data_to_save.append(
            {
                "K": cam.get_K().cpu(),
                "camtoworld": cam.get_camtoworld().cpu(),
                "height": cam.get_height(),
                "width": cam.get_width(),
                "name": cam.get_name(),
                "real_R": cam.real_R.cpu() if cam.real_R is not None else None,
                "real_T": cam.real_T.cpu() if cam.real_T is not None else None,
                "transform": cam.get_transform().cpu() if cam.get_transform() is not None else None,
            }
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_to_save, cache_path)


def load_query_cameras_from_cache(cache_path: Path, device: torch.device) -> List[Camera]:
    data_loaded = torch.load(cache_path, map_location="cpu")
    query_cameras = []
    for data in data_loaded:
        cam = Camera(
            K=data["K"].to(device),
            camtoworld=data["camtoworld"].to(device),
            height=data["height"],
            width=data["width"],
            name=data["name"],
            real_R=data["real_R"].to(device) if data["real_R"] is not None else None,
            real_T=data["real_T"].to(device) if data["real_T"] is not None else None,
            transform=data["transform"].to(device) if data["transform"] is not None else None,
        )
        query_cameras.append(cam)
    return query_cameras
