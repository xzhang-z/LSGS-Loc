from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.read_write_model import (
    Point3D,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)


_GROUP_SIZE = 3
_SELECTED_POSITIONS: Sequence[int] = (2,)
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")


def _resolve_sparse_dir(colmap_root: Path) -> Path:
    sparse0 = colmap_root / "sparse" / "0"
    sparse = colmap_root / "sparse"
    if sparse0.is_dir():
        return sparse0
    if sparse.is_dir():
        return sparse
    raise FileNotFoundError(f"Sparse directory not found: {sparse0} or {sparse}")


def _select_test_images_default(images_dir: Path) -> List[str]:
    all_images = [
        name
        for name in os.listdir(images_dir)
        if (images_dir / name).is_file() and name.lower().endswith(_IMAGE_EXTS)
    ]
    all_images.sort()

    selected: List[str] = []
    for i in range(0, len(all_images), _GROUP_SIZE):
        group = all_images[i : i + _GROUP_SIZE]
        for pos in _SELECTED_POSITIONS:
            if pos < len(group):
                selected.append(group[pos])
    return selected


def _write_list_test(output_root: Path, image_names: Iterable[str]) -> Path:
    out_path = output_root / "list_test.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for name in image_names:
            f.write(f"{name}\n")
    return out_path


def _split_images(images: Dict[int, object], test_image_names: Set[str]):
    train_images: Dict[int, object] = {}
    test_images: Dict[int, object] = {}
    train_ids: Set[int] = set()
    test_ids: Set[int] = set()

    for image_id, image_data in images.items():
        if image_data.name in test_image_names:
            test_images[image_id] = image_data
            test_ids.add(image_id)
        else:
            train_images[image_id] = image_data
            train_ids.add(image_id)

    return train_images, test_images, train_ids, test_ids


def _filter_points(points3d: Dict[int, Point3D], allowed_img_ids: Set[int]) -> Dict[int, Point3D]:
    out: Dict[int, Point3D] = {}
    allowed = list(allowed_img_ids)
    for point_id, point_data in points3d.items():
        mask = np.isin(point_data.image_ids, allowed)
        if np.any(mask):
            out[point_id] = Point3D(
                id=point_id,
                xyz=point_data.xyz,
                rgb=point_data.rgb,
                error=point_data.error,
                image_ids=point_data.image_ids[mask],
                point2D_idxs=point_data.point2D_idxs[mask],
            )
    return out


def _materialize_images(
    src_images_root: Path,
    dst_images_root: Path,
    images: Dict[int, object],
    image_mode: str,
) -> None:
    for image_data in images.values():
        src = src_images_root / image_data.name
        dst = dst_images_root / image_data.name
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"[WARN] Image does not exist, skipped: {src}")
            continue

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if image_mode == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)


def _save_subset_project(
    subset_root: Path,
    cameras: Dict[int, object],
    images: Dict[int, object],
    points3d: Dict[int, Point3D],
    src_images_root: Path,
    image_mode: str,
) -> None:
    sparse_out = subset_root / "sparse" / "0"
    images_out = subset_root / "images"
    sparse_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    write_cameras_binary(cameras, sparse_out / "cameras.bin")
    write_images_binary(images, sparse_out / "images.bin")
    write_points3D_binary(points3d, sparse_out / "points3D.bin")

    _materialize_images(src_images_root, images_out, images, image_mode)


def run_split(colmap_project: Path, output_root: Path, image_mode: str) -> None:
    images_dir = colmap_project / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Original COLMAP images directory does not exist: {images_dir}")

    sparse_dir = _resolve_sparse_dir(colmap_project)

    test_image_names = _select_test_images_default(images_dir)
    list_test_path = _write_list_test(output_root, test_image_names)

    print(f"Generated test list: {list_test_path}")
    print(f"Number of test images: {len(test_image_names)}")

    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")
    points3d = read_points3D_binary(sparse_dir / "points3D.bin")

    train_images, test_images, train_ids, test_ids = _split_images(images, set(test_image_names))
    train_points3d = _filter_points(points3d, train_ids)
    test_points3d = _filter_points(points3d, test_ids)

    train_root = output_root / "train"
    test_root = output_root / "test"

    _save_subset_project(
        subset_root=train_root,
        cameras=cameras,
        images=train_images,
        points3d=train_points3d,
        src_images_root=images_dir,
        image_mode=image_mode,
    )
    _save_subset_project(
        subset_root=test_root,
        cameras=cameras,
        images=test_images,
        points3d=test_points3d,
        src_images_root=images_dir,
        image_mode=image_mode,
    )

    print("Split completed:")
    print(f"  Train: {len(train_images)} images, {len(train_points3d)} points3D -> {train_root}")
    print(f"  Test : {len(test_images)} images, {len(test_points3d)} points3D -> {test_root}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Split COLMAP project into train/test with fixed default logic")
    parser.add_argument(
        "--colmap_project",
        type=str,
        required=True,
        help="Original COLMAP project directory (must contain images/ and sparse/0)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root directory for split projects (train/ and test/ will be generated)",
    )
    parser.add_argument(
        "--image_mode",
        type=str,
        default="copy",
        choices=["copy", "symlink"],
        help="Whether to use copy or symlink in new project images directories (copy or symlink)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    colmap_project = Path(args.colmap_project).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    run_split(colmap_project=colmap_project, output_root=output_root, image_mode=args.image_mode)


if __name__ == "__main__":
    main()
