from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch

SCENE_NAMES = (
    "CUHK_LOWER",
    "CUHK_UPPER",
    "HAV",
    "LFLS",
    "SMBU",
    "SZIIT",
    "SZTU",
)


def _to_tensor44(values: Iterable[Iterable[float]]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float64)


BUILTIN_COLMAP_TO_WORLD: Dict[str, torch.Tensor] = {
    "CUHK_LOWER": _to_tensor44(
        [
            [-74.169609, 39.999680, -38.050034, -280.207123],
            [55.205662, 53.347549, -51.529453, -302.662537],
            [-0.338366, -64.054466, -66.676994, -4.040910],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "CUHK_UPPER": _to_tensor44(
        [
            [-68.607330, -3.013484, 16.347895, -158.637604],
            [-12.623636, 54.613461, -42.910549, -33.341969],
            [-10.815666, -44.627232, -53.616531, 2.316843],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "HAV": _to_tensor44(
        [
            [50.430958, -32.908978, 31.110167, 94.142044],
            [-45.256268, -34.929634, 36.413132, -234.332901],
            [-1.647272, -47.864861, -47.962132, -0.130013],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "LFLS": _to_tensor44(
        [
            [-73.745850, 52.777988, -48.318748, 494.701324],
            [71.515678, 52.042469, -52.304482, -319.711426],
            [-2.393047, -71.166931, -74.082451, -44.774567],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "SMBU": _to_tensor44(
        [
            [78.857307, -7.576013, -1.808225, -15.858069],
            [-7.527516, -78.854637, 2.103759, -354.770691],
            [-2.000542, -1.921799, -79.192459, 2.015016],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "SZIIT": _to_tensor44(
        [
            [-86.384277, 56.818569, -52.998547, 349.211884],
            [77.632866, 59.836540, -62.387222, -148.857742],
            [-3.214666, -81.796555, -82.452576, -11.419686],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "SZTU": _to_tensor44(
        [
            [-121.203323, 18.855204, -10.545249, 9.201624],
            [20.763456, 85.071907, -86.536774, 5.949160],
            [5.966554, 86.972504, 86.931854, -334.705963],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}


def infer_scene_from_path(path_like: str | Path) -> Optional[str]:
    text = str(path_like).upper().replace("-", "_")
    for scene in SCENE_NAMES:
        if scene in text:
            return scene
    return None


def infer_scene_from_paths(*paths: str | Path) -> Optional[str]:
    for p in paths:
        scene = infer_scene_from_path(p)
        if scene is not None:
            return scene
    return None


def parse_transfer_world_coordinates_txt(txt_path: str | Path) -> Dict[str, torch.Tensor]:
    path = Path(txt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Transform file does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    name_pat = re.compile(r"^(CUHK_LOWER|CUHK_UPPER|HAV|LFLS|SMBU|SZIIT|SZTU)\s*:?$", re.IGNORECASE)
    num_pat = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")

    out: Dict[str, torch.Tensor] = {}
    i = 0
    while i < len(lines):
        m = name_pat.match(lines[i])
        if not m:
            i += 1
            continue

        scene = m.group(1).upper()
        mat_rows = []
        j = i + 1
        while j < len(lines) and len(mat_rows) < 4:
            parts = lines[j].replace(",", " ").split()
            if len(parts) == 4 and all(num_pat.match(x) for x in parts):
                mat_rows.append([float(x) for x in parts])
            j += 1

        if len(mat_rows) == 4:
            out[scene] = _to_tensor44(mat_rows)
        i = j

    if len(out) == 0:
        raise ValueError(f"No scene transform matrix was parsed from file: {path}")
    return out


def resolve_scene_transform(
    scene_name: str,
    source: str = "auto",
    transfer_txt_path: str | Path | None = None,
) -> Tuple[torch.Tensor, str]:
    source = source.lower()
    if source not in {"auto", "builtin", "txt"}:
        raise ValueError("source only supports 'auto' / 'builtin' / 'txt'")

    if source == "builtin":
        if scene_name not in BUILTIN_COLMAP_TO_WORLD:
            raise KeyError(f"Scene not found in built-in matrices: {scene_name}")
        return BUILTIN_COLMAP_TO_WORLD[scene_name].clone(), "builtin"

    if source == "txt":
        if transfer_txt_path is None:
            raise ValueError("transfer_txt_path must be provided when source='txt'")
        txt_map = parse_transfer_world_coordinates_txt(transfer_txt_path)
        if scene_name not in txt_map:
            raise KeyError(f"Scene not found in text matrices: {scene_name}")
        return txt_map[scene_name].clone(), "txt"

    # auto: prefer txt, then fall back to builtin
    if transfer_txt_path is not None:
        txt_map = parse_transfer_world_coordinates_txt(transfer_txt_path)
        if scene_name in txt_map:
            return txt_map[scene_name].clone(), "txt"

    if scene_name not in BUILTIN_COLMAP_TO_WORLD:
        raise KeyError(f"Scene matrix not found: {scene_name}")
    return BUILTIN_COLMAP_TO_WORLD[scene_name].clone(), "builtin"


def _compose_w2c(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    w2c = torch.eye(4, dtype=R.dtype, device=R.device)
    w2c[:3, :3] = R
    w2c[:3, 3] = T
    return w2c


def _camera_center_from_c2w(c2w: torch.Tensor) -> torch.Tensor:
    return c2w[:3, 3]


def _transform_point_homo(transform44: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    p4 = torch.cat([xyz, torch.ones(1, dtype=xyz.dtype, device=xyz.device)], dim=0)
    q4 = transform44 @ p4
    if abs(float(q4[3])) > 1e-12:
        return q4[:3] / q4[3]
    return q4[:3]


def compute_real_scale_pose_error(
    est_camtoworld: torch.Tensor,
    gt_w2c_R: torch.Tensor,
    gt_w2c_T: torch.Tensor,
    colmap_to_world: torch.Tensor | None = None,
) -> tuple[float, float]:
    est_c2w = est_camtoworld.to(torch.float64)
    gt_w2c = _compose_w2c(gt_w2c_R.to(torch.float64), gt_w2c_T.to(torch.float64))

    est_w2c = torch.linalg.inv(est_c2w)
    gt_rot = gt_w2c[:3, :3]
    est_rot = est_w2c[:3, :3]

    rot_rel = est_rot @ gt_rot.T
    cos_theta = torch.clamp((torch.trace(rot_rel) - 1.0) / 2.0, -1.0, 1.0)
    sin_theta = torch.linalg.norm(
        torch.stack(
            [
                rot_rel[2, 1] - rot_rel[1, 2],
                rot_rel[0, 2] - rot_rel[2, 0],
                rot_rel[1, 0] - rot_rel[0, 1],
            ]
        )
    ) / 2.0
    rot_error_rad = torch.atan2(sin_theta, cos_theta)
    rot_error_deg = rot_error_rad * 180.0 / math.pi

    gt_c2w = torch.linalg.inv(gt_w2c)
    est_center = _camera_center_from_c2w(est_c2w)
    gt_center = _camera_center_from_c2w(gt_c2w)

    if colmap_to_world is not None:
        tf = colmap_to_world.to(dtype=torch.float64, device=est_center.device)
        est_center = _transform_point_homo(tf, est_center)
        gt_center = _transform_point_homo(tf, gt_center)

    trans_error_m = torch.linalg.norm(est_center - gt_center)
    return float(rot_error_deg.item()), float(trans_error_m.item())
