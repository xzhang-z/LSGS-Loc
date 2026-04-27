from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gsplat.rendering import rasterization

from src.datasets.camera_cache_builder import load_query_camera_cache, prepare_query_camera_cache
from src.models import Camera
from src.models.anyloc_vlad_retrieval import AnyLocVLADConfig, run_anyloc_vlad_retrieval
from src.models.pose_distance_search import refine_pose_with_pointcloud_and_distance_search
from src.models.pose_render_optimizer import optimize_pose_by_render_only
from src.models.reloc3r_pose_attention import (
    run_reloc3r_estimation_with_attention,
    setup_reloc3r_estimator,
)
from src.utils.global_pose_eval import (
    compute_real_scale_pose_error,
    infer_scene_from_paths,
    resolve_scene_transform,
)
from src.utils.pointcloud_pose_search_utils import depth_to_world_points


def _resolve_existing_path(path_like: str | None) -> Path | None:
    if path_like is None:
        return None
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()

    cand_cwd = (Path.cwd() / p).resolve()
    if cand_cwd.exists():
        return cand_cwd

    cand_repo = (REPO_ROOT / p).resolve()
    if cand_repo.exists():
        return cand_repo

    return cand_repo


def _resolve_path(path_like: str | None) -> Path | None:
    if path_like is None:
        return None
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _load_yaml_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    path = _resolve_existing_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML config must be a dictionary")
    return data


def _pick(cli_val, cfg: dict, key: str, default=None):
    return cli_val if cli_val is not None else cfg.get(key, default)


def _normalize_to_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _load_image_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def _resize_hw3_image(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Resize an [H,W,3] image to the target size."""
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img
    x = img.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return x.squeeze(0).permute(1, 2, 0)


def _load_splats(ckpt_paths: List[str], device: torch.device):
    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]
        means.append(ckpt["means"])
        quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        scales.append(torch.exp(ckpt["scales"]))
        opacities.append(torch.sigmoid(ckpt["opacities"]))
        sh0.append(ckpt["sh0"])
        if "shN" in ckpt:
            shN.append(ckpt["shN"])
        else:
            shN.append(torch.empty((ckpt["sh0"].shape[0], 0, 3), device=device))

    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    shN = torch.cat(shN, dim=0)
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(np.sqrt(colors.shape[-2]) - 1)
    return means, quats, scales, opacities, colors, sh_degree


def _load_splats_from_ply(ply_paths: List[str], device: torch.device):
    means_all, quats_all, scales_all, opacities_all, sh0_all, shN_all = [], [], [], [], [], []

    for ply_path in ply_paths:
        p = Path(ply_path)
        if not p.is_file():
            raise FileNotFoundError(f"PLY file does not exist: {p}")

        with open(p, "rb") as f:
            num_vertices = None
            prop_names: List[str] = []

            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"Invalid PLY (unexpected EOF in header): {p}")
                line_s = line.decode("utf-8", errors="ignore").strip()

                if line_s.startswith("format") and "binary_little_endian" not in line_s:
                    raise ValueError(f"Only binary_little_endian PLY is supported: {p}")

                if line_s.startswith("element vertex"):
                    parts = line_s.split()
                    if len(parts) >= 3:
                        num_vertices = int(parts[2])

                if line_s.startswith("property"):
                    parts = line_s.split()
                    # Expected: property float <name>
                    if len(parts) == 3 and parts[1] == "float":
                        prop_names.append(parts[2])

                if line_s == "end_header":
                    break

            if num_vertices is None:
                raise ValueError(f"Invalid PLY (missing vertex count): {p}")

            required = [
                "x",
                "y",
                "z",
                "f_dc_0",
                "f_dc_1",
                "f_dc_2",
                "opacity",
                "scale_0",
                "scale_1",
                "scale_2",
                "rot_0",
                "rot_1",
                "rot_2",
                "rot_3",
            ]
            missing = [name for name in required if name not in prop_names]
            if missing:
                raise ValueError(f"PLY missing required fields {missing}: {p}")

            dtype = np.dtype([(name, "<f4") for name in prop_names])
            data = np.fromfile(f, dtype=dtype, count=num_vertices)
            if data.shape[0] != num_vertices:
                raise ValueError(f"Invalid PLY data length in {p}, expected {num_vertices}, got {data.shape[0]}")

            def _stack(cols: List[str]) -> np.ndarray:
                return np.stack([data[c] for c in cols], axis=1).astype(np.float32)

            means_np = _stack(["x", "y", "z"])
            sh0_np = _stack(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None, :]

            f_rest_cols = sorted(
                [name for name in prop_names if name.startswith("f_rest_")],
                key=lambda s: int(s.split("_")[-1]),
            )
            if len(f_rest_cols) % 3 != 0:
                raise ValueError(f"Invalid f_rest field count (must be multiple of 3): {p}")
            if len(f_rest_cols) > 0:
                f_rest_np = _stack(f_rest_cols)  # [N, 3*K] (channel-major)
                k = len(f_rest_cols) // 3
                shN_np = f_rest_np.reshape(num_vertices, 3, k).transpose(0, 2, 1).astype(np.float32)
            else:
                shN_np = np.empty((num_vertices, 0, 3), dtype=np.float32)

            scales_raw_np = _stack(["scale_0", "scale_1", "scale_2"])
            quats_raw_np = _stack(["rot_0", "rot_1", "rot_2", "rot_3"])
            opacities_raw_np = data["opacity"].astype(np.float32)

            means_all.append(torch.from_numpy(means_np).to(device))
            sh0_all.append(torch.from_numpy(sh0_np).to(device))
            shN_all.append(torch.from_numpy(shN_np).to(device))
            scales_all.append(torch.exp(torch.from_numpy(scales_raw_np).to(device)))
            quats_all.append(F.normalize(torch.from_numpy(quats_raw_np).to(device), p=2, dim=-1))
            opacities_all.append(torch.sigmoid(torch.from_numpy(opacities_raw_np).to(device)))

    means = torch.cat(means_all, dim=0)
    quats = torch.cat(quats_all, dim=0)
    scales = torch.cat(scales_all, dim=0)
    opacities = torch.cat(opacities_all, dim=0)
    sh0 = torch.cat(sh0_all, dim=0)
    shN = torch.cat(shN_all, dim=0)
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(np.sqrt(colors.shape[-2]) - 1)
    return means, quats, scales, opacities, colors, sh_degree


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None, help="YAML config file path")
    pre_args, _ = pre_parser.parse_known_args()
    cfg = _load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser("Retrieval + ReLoc3r + distance search", parents=[pre_parser])
    parser.add_argument("--database_root", type=str, default=None, help="Database COLMAP root (contains images/ sparse/)")
    parser.add_argument("--query_root", type=str, default=None, help="Query COLMAP root (contains images/ sparse/)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--ckpt", type=str, nargs="+", default=None, help="3DGS ckpt path")
    parser.add_argument("--ply", type=str, nargs="+", default=None, help="3DGS PLY path(s), exported by gsplat")

    parser.add_argument("--img_exts", type=str, nargs="+", default=None)
    parser.add_argument("--model_args", type=str, default=None, choices=["224", "512"])
    parser.add_argument("--camera_model", type=str, default=None, choices=["pinhole", "ortho", "fisheye"])
    parser.add_argument("--data_factor", type=int, default=None)
    parser.add_argument("--block_id", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--retrieval_top_k", type=int, default=None)
    parser.add_argument("--retrieval_search_method", type=str, default=None, choices=["l2", "cosine"])
    parser.add_argument("--keep_top1_only", action="store_true", help="Build cache using retrieval Top1 only")
    parser.add_argument("--force_rebuild_cache", action="store_true", help="Force rebuild camera cache")
    parser.add_argument(
        "--world_transform_source",
        type=str,
        default=None,
        choices=["auto", "builtin", "txt"],
        help="World transform matrix source: auto/builtin/txt",
    )
    parser.add_argument(
        "--world_transform_txt",
        type=str,
        default=None,
        help="Path to Transfer to World Coordinates.txt (available when source=txt or auto)",
    )
    parser.add_argument(
        "--scale_pose_error",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to convert translation error to real scale using colmap->world matrix (default enabled)",
    )
    args = parser.parse_args()

    database_root_v = _pick(args.database_root, cfg, "database_root")
    query_root_v = _pick(args.query_root, cfg, "query_root")
    output_dir_v = _pick(args.output_dir, cfg, "output_dir")
    ckpt_v = _normalize_to_list(_pick(args.ckpt, cfg, "ckpt"))
    ply_v = _normalize_to_list(_pick(args.ply, cfg, "ply"))

    if not database_root_v or not query_root_v or not output_dir_v:
        raise ValueError("database_root, query_root, and output_dir must be provided (via CLI or YAML)")
    if (not ckpt_v and not ply_v) or (ckpt_v and ply_v):
        raise ValueError("Exactly one of ckpt or ply must be provided (via CLI or YAML)")

    img_exts_v = _normalize_to_list(_pick(args.img_exts, cfg, "img_exts", ["jpg", "JPG", "png"]))
    model_args_v = _pick(args.model_args, cfg, "model_args", "224")
    camera_model_v = _pick(args.camera_model, cfg, "camera_model", "pinhole")
    data_factor_v = int(_pick(args.data_factor, cfg, "data_factor", 1))
    block_id_v = int(_pick(args.block_id, cfg, "block_id", 5))
    device_v = _pick(args.device, cfg, "device", "cuda")
    retrieval_top_k_v = int(_pick(args.retrieval_top_k, cfg, "retrieval_top_k", 3))
    retrieval_search_method_v = _pick(args.retrieval_search_method, cfg, "retrieval_search_method", "l2")

    keep_top1_only_v = args.keep_top1_only or bool(cfg.get("keep_top1_only", True))
    force_rebuild_cache_v = args.force_rebuild_cache or bool(cfg.get("force_rebuild_cache", False))
    render_opt_image_size_v = str(cfg.get("render_opt_image_size", "original")).lower()
    if render_opt_image_size_v not in {"original", "1600"}:
        raise ValueError("render_opt_image_size only supports 'original' or '1600'")
    world_transform_source_v = _pick(args.world_transform_source, cfg, "world_transform_source", "auto")
    world_transform_txt_v = _pick(args.world_transform_txt, cfg, "world_transform_txt", None)
    scale_pose_error_v = bool(_pick(args.scale_pose_error, cfg, "scale_pose_error", True))

    device = torch.device(device_v if (device_v != "cuda" or torch.cuda.is_available()) else "cpu")

    database_root = _resolve_path(database_root_v)
    query_root = _resolve_path(query_root_v)
    output_dir = _resolve_path(output_dir_v)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer scene from COLMAP paths and load the colmap->world transform matrix.
    scene_name = infer_scene_from_paths(str(database_root), str(query_root))
    colmap_to_world = None
    world_tf_source_used = "none"
    if scene_name is None:
        print("Warning: Failed to infer scene from database_root/query_root; only COLMAP-scale errors will be reported.")
    else:
        txt_path = _resolve_existing_path(world_transform_txt_v)
        try:
            colmap_to_world, world_tf_source_used = resolve_scene_transform(
                scene_name=scene_name,
                source=str(world_transform_source_v),
                transfer_txt_path=str(txt_path) if txt_path is not None else None,
            )
            print(
                f"Inferred scene: {scene_name}, transform source: {world_tf_source_used}"
                + (f" ({txt_path})" if world_tf_source_used == "txt" and txt_path is not None else "")
            )
        except Exception as e:
            print(f"Warning: Failed to load transform matrix for scene {scene_name}; only COLMAP-scale errors will be reported. Error: {e}")

    if colmap_to_world is not None:
        print(f"Real-scale pose evaluation config: scene={scene_name}, source={world_tf_source_used}")
        print("Using colmap->world transform matrix:")
        for row in colmap_to_world.tolist():
            print("  " + " ".join([f"{v:.6f}" for v in row]))
    else:
        print(
            f"Real-scale pose evaluation config: scene={scene_name if scene_name is not None else 'UNKNOWN'}, "
            f"source={world_tf_source_used} (real-scale matrix not enabled)"
        )

    if not scale_pose_error_v:
        print("Error scaling disabled: translation error will be computed in COLMAP units.")

    eval_colmap_to_world = colmap_to_world if scale_pose_error_v else None

    retrieval_txt = output_dir / "match_results_top3.txt"
    cache_path = output_dir / "query_cameras_cache.pt"


    if cache_path.exists() and not force_rebuild_cache_v:
        print(f"Existing camera cache detected, loading directly and skipping retrieval/build: {cache_path}")
        cache_entries = load_query_camera_cache(str(cache_path), device=None)
    else:
        print("No reusable cache found, starting retrieval and camera cache construction.")


        retrieval_cfg = AnyLocVLADConfig(
            database_images_dir=str(database_root / "images"),
            query_images_dir=str(query_root / "images"),
            img_exts=tuple(img_exts_v),
            top_k=retrieval_top_k_v,
            search_method=retrieval_search_method_v,
            output_txt_file=str(retrieval_txt),
            device=str(device),
        )
        _ = run_anyloc_vlad_retrieval(retrieval_cfg)


        cache_entries = prepare_query_camera_cache(
            retrieval_file=str(retrieval_txt),
            database_dir=str(database_root),
            query_dir=str(query_root),
            cache_path=str(cache_path),
            factor=data_factor_v,
            world_size=1,
            world_rank=0,
            keep_top1_only=keep_top1_only_v,
            force_rebuild=force_rebuild_cache_v,
        )


    reloc3r_model, attn_capture = setup_reloc3r_estimator(
        model_args=model_args_v,
        device=device,
        block_id=block_id_v,
    )


    if ckpt_v:
        ckpt_v = [str(_resolve_path(p)) for p in ckpt_v]
        means, quats, scales, opacities, colors, sh_degree = _load_splats(ckpt_v, device)
    else:
        ply_v = [str(_resolve_path(p)) for p in ply_v]
        means, quats, scales, opacities, colors, sh_degree = _load_splats_from_ply(ply_v, device)

    def render_rgbd(viewmats: torch.Tensor, Ks: torch.Tensor, width: int, height: int):
        render_colors, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            render_mode="RGB+D",
            rasterize_mode="antialiased",
            packed=False,
            distributed=False,
            camera_model=camera_model_v,
        )
        rgb = render_colors[0, ..., :3].clamp(0, 1)
        depth = render_colors[0, ..., 3]
        return rgb, depth


    pose_error_rows = []
    optimized_pose_rows = []
    for item in tqdm(cache_entries, desc="Processing query images", unit="img"):
        cam = Camera(
            K=item["K"].to(device),
            camtoworld=item["camtoworld"].to(device),
            height=item["height"],
            width=item["width"],
            name=item["name"],
            real_R=item["real_R"].to(device) if item.get("real_R") is not None else None,
            real_T=item["real_T"].to(device) if item.get("real_T") is not None else None,
            transform=item["transform"].to(device) if item.get("transform") is not None else None,
        )

        query_img_path = query_root / "images" / cam.get_name()
        if not query_img_path.is_file():
            print(f"Skipped, query image does not exist: {query_img_path}")
            continue

        subdir = output_dir / Path(cam.get_name()).stem
        subdir.mkdir(parents=True, exist_ok=True)


        init_camtoworld = cam.get_camtoworld()
        init_viewmat = torch.linalg.inv(init_camtoworld)[None]
        K1600 = cam.get_K1600().to(device)
        w1600, h1600 = cam.get_width1600(), cam.get_height1600()
        init_rgb, init_depth = render_rgbd(init_viewmat, K1600[None], w1600, h1600)


        gt_img = _load_image_tensor(query_img_path, device)
        gt_img_1600 = _resize_hw3_image(gt_img, h1600, w1600)


        reloc_out = run_reloc3r_estimation_with_attention(
            reloc3r_relpose=reloc3r_model,
            attn_capture=attn_capture,
            rendered_image=init_rgb,
            ground_truth_image=gt_img_1600,
        )
        pose_matrix = reloc_out["pose_matrix"]
        top_coords = reloc_out["top_coords"]


        points_3d = depth_to_world_points(init_depth, K1600, cam.get_camtoworld())
        H_orig, W_orig = init_rgb.shape[:2]
        L = min(H_orig, W_orig)
        if H_orig > W_orig:
            crop_top, crop_left = (H_orig - L) // 2, 0
        else:
            crop_top, crop_left = 0, (W_orig - L) // 2
        grid_size = max(1, L // 14)


        search_out = refine_pose_with_pointcloud_and_distance_search(
            initial_camtoworld=cam.get_camtoworld(),
            pose_matrix=pose_matrix,
            top_coords=top_coords,
            points_3d=points_3d,
            crop_left=crop_left,
            crop_top=crop_top,
            grid_size=grid_size,
            image_h=H_orig,
            image_w=W_orig,
            K_1600=K1600,
            width_1600=w1600,
            height_1600=h1600,
            ground_truth_image=gt_img_1600,
            render_rgbd_fn=render_rgbd,
        )

        best_est_camtoworld = search_out["best_est_camtoworld"]

        if render_opt_image_size_v == "1600":
            render_opt_K = K1600
            render_opt_w, render_opt_h = w1600, h1600
            render_opt_gt_img = gt_img_1600
            render_opt_size_desc = "1600"
        else:
            render_opt_K = cam.get_K().to(device)
            render_opt_w, render_opt_h = cam.get_width(), cam.get_height()
            render_opt_gt_img = gt_img
            render_opt_size_desc = "original"

        print(
            f"  -> render_only optimization resolution mode: {render_opt_size_desc} "
            f"({render_opt_w}x{render_opt_h})"
        )

        render_opt_out = optimize_pose_by_render_only(
            initial_camtoworld=best_est_camtoworld,
            K=render_opt_K,
            width=render_opt_w,
            height=render_opt_h,
            ground_truth_image=render_opt_gt_img,
            render_rgbd_fn=render_rgbd,
            num_optimization_steps=int(cfg.get("render_opt_steps", 200)),
            learning_rate=float(cfg.get("render_opt_lr", 1e-3)),
            patience=int(cfg.get("render_opt_patience", 50)),
            scheduler_t_max=int(cfg.get("render_opt_scheduler_tmax", 300)),
            cam_dir=str(subdir),
            verbose=bool(cfg.get("render_opt_verbose", False)),
            show_progress=bool(cfg.get("render_opt_show_progress", True)),
            progress_desc=f"Optimizing {cam.get_name()}",
        )

        optimized_camtoworld = render_opt_out["best_est_camtoworld"]
        cam.camtoworld = optimized_camtoworld
        optimized_pose_rows.append((cam.get_name(), optimized_camtoworld.detach().cpu().to(torch.float64).numpy()))

        # Compute and report optimized pose error (real scale preferred).
        if cam.real_R is not None and cam.real_T is not None:
            real_R = cam.real_R.to(torch.float64).to(device)
            real_T = cam.real_T.to(torch.float64).to(device)
            rot_error_deg, trans_error_m = compute_real_scale_pose_error(
                est_camtoworld=optimized_camtoworld,
                gt_w2c_R=real_R,
                gt_w2c_T=real_T,
                colmap_to_world=eval_colmap_to_world.to(device) if eval_colmap_to_world is not None else None,
            )
            pose_error_rows.append((cam.get_name(), rot_error_deg, trans_error_m))

            print(f"  -> {cam.get_name()} - Pose error:")
            print(f"     Rotation error: {rot_error_deg:.7f} deg")
            if eval_colmap_to_world is None:
                print(f"     Translation error: {trans_error_m:.7f} (COLMAP scale)")
            else:
                print(f"     Translation error: {trans_error_m:.7f} m")
            print(
                f"     Real-scale transform: scene={scene_name if scene_name is not None else 'UNKNOWN'}, "
                f"source={world_tf_source_used}"
            )
        else:
            print(f"  -> {cam.get_name()} - Missing real-pose reference, cannot compute error.")
            pose_error_rows.append((cam.get_name(), None, None))

        torch.save(
            {
                "pose_matrix": pose_matrix.detach().cpu(),
                "center_attn_map": reloc_out["center_attn_map"],
                "top_coords": top_coords,
                "top_regions": reloc_out["top_regions"],
                "best_est_camtoworld": best_est_camtoworld.detach().cpu(),
                "optimized_camtoworld": optimized_camtoworld.detach().cpu(),
                "distance_search": {
                    "best_d": search_out["best_d"],
                    "best_loss": search_out["best_loss"],
                    "select_out": search_out["select_out"],
                },
                "render_optimization": {
                    "best_loss": render_opt_out["best_loss"],
                    "best_iteration": render_opt_out["best_iteration"],
                },
            },
            subdir / "pose_search_result.pt",
        )

        print(f"Done: {cam.get_name()} -> {subdir}")

    attn_capture.remove()


    optimized_pose_txt = output_dir / "optimized_camtoworlds.txt"
    with open(optimized_pose_txt, "w", encoding="utf-8") as f:
        f.write("# Optimized camera poses in COLMAP coordinate system\n")
        f.write("# Format: <image_name> - camtoworld(flat16 row-major)\n")
        for name, pose in optimized_pose_rows:
            flat = pose.reshape(4, 4).reshape(-1)
            f.write(f"{name} - camtoworld(flat16): " + " ".join([f"{v:.8e}" for v in flat]) + "\n")
    print(f"Optimized poses (COLMAP coordinate system) saved to: {optimized_pose_txt}")

    stats_path = output_dir / "pose_optimization_errors.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Scene: {scene_name if scene_name is not None else 'UNKNOWN'}\n")
        f.write(f"WorldTransformSource: {world_tf_source_used}\n")
        f.write(f"ScalePoseError: {scale_pose_error_v}\n")
        if world_transform_txt_v is not None:
            f.write(f"WorldTransformTxt: {world_transform_txt_v}\n")
        if eval_colmap_to_world is not None:
            f.write("WorldTransformMatrix(colmap->world):\n")
            for row in eval_colmap_to_world.tolist():
                f.write(" ".join([f"{v:.6f}" for v in row]) + "\n")
        f.write("\n")
        valid_rot = []
        valid_trans = []
        for name, rot, trans in pose_error_rows:
            if rot is None or trans is None:
                f.write(f"{name} - Rotation Error: N/A, Translation Error: N/A\n")
            else:
                trans_unit = "m" if eval_colmap_to_world is not None else "colmap_unit"
                f.write(f"{name} - Rotation Error: {rot:.7f} deg, Translation Error: {trans:.7f} {trans_unit}\n")
                valid_rot.append(rot)
                valid_trans.append(trans)

        if len(valid_rot) > 0:
            # Calculate median errors
            sorted_rot = sorted(valid_rot)
            sorted_trans = sorted(valid_trans)
            median_rot = sorted_rot[len(sorted_rot) // 2] if len(sorted_rot) % 2 == 1 else (sorted_rot[len(sorted_rot) // 2 - 1] + sorted_rot[len(sorted_rot) // 2]) / 2
            median_trans = sorted_trans[len(sorted_trans) // 2] if len(sorted_trans) % 2 == 1 else (sorted_trans[len(sorted_trans) // 2 - 1] + sorted_trans[len(sorted_trans) // 2]) / 2
            
            trans_unit = "m" if eval_colmap_to_world is not None else "colmap_unit"
            
            f.write("\n--- Summary ---\n")
            f.write(f"Median Rotation Error: {median_rot:.7f} deg\n")
            f.write(f"Median Translation Error: {median_trans:.7f} {trans_unit}\n")
            
            # Print summary to console
            print("\n=== Pose Error Summary ===")
            print(f"Median Rotation Error: {median_rot:.7f} deg")
            print(f"Median Translation Error: {median_trans:.7f} {trans_unit}")
            
            # Save median errors to separate file
            median_path = output_dir / "median_pose_errors.txt"
            with open(median_path, "w", encoding="utf-8") as mf:
                mf.write(f"Median Rotation Error: {median_rot:.7f} deg\n")
                mf.write(f"Median Translation Error: {median_trans:.7f} {trans_unit}\n")
            print(f"Median pose errors saved to: {median_path}")

    print(f"Pose error statistics saved to: {stats_path}")
    print(f"All done, output directory: {output_dir}")


if __name__ == "__main__":
    main()
