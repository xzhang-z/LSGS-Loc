from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from src.utils.image_utils import (
    collect_image_files,
    descriptor_cache_path,
    get_default_base_transform,
    preprocess_image_tensor,
)
from src.utils.retrieval_utils import (
    build_ranked_results,
    compute_topk_matches_pure_torch,
    save_match_results_txt,
)


@dataclass
class AnyLocVLADConfig:
    database_images_dir: str
    query_images_dir: str

    img_exts: Tuple[str, ...] = ("JPG", "png")
    domain: Literal["aerial", "indoor", "urban"] = "urban"

    dinov2_model_name: str = "dinov2_vitg14"
    desc_layer: int = 31
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    num_clusters: int = 32

    max_img_size: int = 1280
    top_k: int = 3
    search_method: Literal["cosine", "l2"] = "l2"
    norm_descs: bool = True

    cache_dir: str = "third_party/AnyLoc/demo/cache"
    output_txt_file: Optional[str] = None
    descriptor_subdir_name: str = "_descriptors"
    save_descriptors: bool = True

    device: Optional[str] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_like: str, project_root: Path) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (project_root / p).resolve()


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _load_anyloc_demo_utilities(project_root: Path):
    util_path = project_root / "third_party" / "AnyLoc" / "demo" / "utilities.py"
    if not util_path.is_file():
        raise FileNotFoundError(f"AnyLoc demo utilities file not found: {util_path}")

    spec = importlib.util.spec_from_file_location("anyloc_demo_utilities", str(util_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module: {util_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "DinoV2ExtractFeatures") or not hasattr(module, "VLAD"):
        raise ImportError("AnyLoc demo utilities is missing DinoV2ExtractFeatures or VLAD")

    return module.DinoV2ExtractFeatures, module.VLAD


def _extract_vlad_descriptors(
    image_dir: Path,
    img_exts: Tuple[str, ...],
    extractor,
    vlad,
    device: torch.device,
    max_img_size: int,
    save_descriptors: bool,
    descriptor_subdir_name: str,
) -> Tuple[np.ndarray, List[str]]:
    image_dir_str = str(image_dir)
    image_paths = collect_image_files(image_dir_str, img_exts)
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32), []

    base_tf = get_default_base_transform()
    descriptor_dir = image_dir / descriptor_subdir_name if save_descriptors else None
    if descriptor_dir is not None:
        descriptor_dir.mkdir(parents=True, exist_ok=True)

    descriptors: List[np.ndarray] = []
    full_paths: List[str] = []

    for img_path in tqdm(image_paths, desc=f"Processing {image_dir.name} images"):
        try:
            descriptor_file = None
            if descriptor_dir is not None:
                descriptor_file = descriptor_cache_path(img_path, image_dir_str, str(descriptor_dir))
                if descriptor_file.exists():
                    descriptors.append(np.load(str(descriptor_file)).squeeze())
                    full_paths.append(img_path)
                    continue

            img_pt = preprocess_image_tensor(
                image_path=img_path,
                max_img_size=max_img_size,
                device=device,
                base_transform=base_tf,
                patch_multiple=14,
            )

            with torch.no_grad():
                per_patch = extractor(img_pt)
                _sync_if_cuda(device)
                flattened = per_patch.squeeze(0).detach().cpu()
                global_descriptor = vlad.generate(flattened)
                _sync_if_cuda(device)

            global_descriptor_np = global_descriptor.squeeze().detach().cpu().numpy()
            descriptors.append(global_descriptor_np)
            full_paths.append(img_path)

            if descriptor_file is not None:
                np.save(str(descriptor_file), global_descriptor_np)

        except Exception as exc:  # pragma: no cover
            print(f"Failed to process image: {img_path}, reason: {exc}")
            continue

    if not descriptors:
        return np.empty((0, 0), dtype=np.float32), []

    return np.asarray(descriptors), full_paths


def run_anyloc_vlad_retrieval(config: AnyLocVLADConfig) -> Dict:
    project_root = _project_root()

    database_images_dir = _resolve_path(config.database_images_dir, project_root)
    query_images_dir = _resolve_path(config.query_images_dir, project_root)
    cache_dir = _resolve_path(config.cache_dir, project_root)

    if not database_images_dir.is_dir():
        raise FileNotFoundError(f"Database image directory does not exist: {database_images_dir}")
    if not query_images_dir.is_dir():
        raise FileNotFoundError(f"Query image directory does not exist: {query_images_dir}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"AnyLoc cache directory does not exist: {cache_dir}")

    device = _resolve_device(config.device)
    DinoV2ExtractFeatures, VLAD = _load_anyloc_demo_utilities(project_root)

    extractor = DinoV2ExtractFeatures(
        config.dinov2_model_name,
        config.desc_layer,
        config.desc_facet,
        device=device,
    )

    ext_specifier = (
        f"{config.dinov2_model_name}/l{config.desc_layer}_{config.desc_facet}_c{config.num_clusters}"
    )
    c_centers_file = cache_dir / "vocabulary" / ext_specifier / config.domain / "c_centers.pt"
    if not c_centers_file.is_file():
        raise FileNotFoundError(f"VLAD vocabulary file not found: {c_centers_file}")

    c_centers = torch.load(str(c_centers_file), map_location="cpu")
    num_clusters = int(c_centers.shape[0])
    if num_clusters != config.num_clusters:
        print(
            f"Warning: vocabulary cluster count is {num_clusters}, inconsistent with config {config.num_clusters}."
            " The cluster count in the vocabulary will be used automatically."
        )

    vlad = VLAD(
        num_clusters,
        desc_dim=None,
        cache_dir=str(c_centers_file.parent),
        vlad_mode="hard",
    )
    vlad.fit(None)

    database_vlads_np, database_image_full_paths = _extract_vlad_descriptors(
        image_dir=database_images_dir,
        img_exts=config.img_exts,
        extractor=extractor,
        vlad=vlad,
        device=device,
        max_img_size=config.max_img_size,
        save_descriptors=config.save_descriptors,
        descriptor_subdir_name=config.descriptor_subdir_name,
    )
    if len(database_vlads_np) == 0:
        raise RuntimeError("No database descriptors were generated")

    query_vlads_np, query_image_full_paths = _extract_vlad_descriptors(
        image_dir=query_images_dir,
        img_exts=config.img_exts,
        extractor=extractor,
        vlad=vlad,
        device=device,
        max_img_size=config.max_img_size,
        save_descriptors=config.save_descriptors,
        descriptor_subdir_name=config.descriptor_subdir_name,
    )
    if len(query_vlads_np) == 0:
        raise RuntimeError("No query descriptors were generated")

    db_tensor = torch.from_numpy(database_vlads_np).to(device)
    qu_tensor = torch.from_numpy(query_vlads_np).to(device)

    top_k = min(config.top_k, db_tensor.shape[0])
    distances, indices = compute_topk_matches_pure_torch(
        db=db_tensor,
        qu=qu_tensor,
        top_k=top_k,
        method=config.search_method,
        norm_descs=config.norm_descs,
    )

    results = build_ranked_results(
        query_image_full_paths=query_image_full_paths,
        database_image_full_paths=database_image_full_paths,
        distances=distances,
        indices=indices,
        query_root_dir=str(query_images_dir),
        database_root_dir=str(database_images_dir),
        top_k=top_k,
    )

    output_txt_file = (
        _resolve_path(config.output_txt_file, project_root)
        if config.output_txt_file
        else database_images_dir.parent / f"match_results_top{top_k}.txt"
    )
    save_match_results_txt(
        output_txt_file=str(output_txt_file),
        results=results,
        query_images_dir=str(query_images_dir),
        database_images_dir=str(database_images_dir),
        top_k=top_k,
    )

    return {
        "database_vlads_shape": tuple(database_vlads_np.shape),
        "query_vlads_shape": tuple(query_vlads_np.shape),
        "top_k": top_k,
        "search_method": config.search_method,
        "results": results,
        "output_txt_file": str(output_txt_file),
    }
