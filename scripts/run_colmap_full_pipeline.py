from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cmd(cmd: List[str], cwd: Path, env: Optional[dict] = None) -> None:
    print("\n[RUN]", shlex.join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _resolve_sparse_dir(colmap_root: Path) -> Path:
    p0 = colmap_root / "sparse" / "0"
    p1 = colmap_root / "sparse"
    if p0.is_dir():
        return p0
    if p1.is_dir():
        return p1
    raise FileNotFoundError(f"Sparse directory not found: {p0} or {p1}")


def _is_colmap_project_ready(project_root: Path) -> bool:
    images_dir = project_root / "images"
    sparse_dir = project_root / "sparse" / "0"
    return (
        images_dir.is_dir()
        and sparse_dir.is_dir()
        and (sparse_dir / "cameras.bin").is_file()
        and (sparse_dir / "images.bin").is_file()
        and (sparse_dir / "points3D.bin").is_file()
    )


def _is_split_ready(workspace_root: Path) -> bool:
    return _is_colmap_project_ready(workspace_root / "train") and _is_colmap_project_ready(workspace_root / "test")


def _is_pose_done(pose_out_dir: Path) -> bool:
    stats = pose_out_dir / "pose_optimization_errors.txt"
    return stats.is_file() and stats.stat().st_size > 0


def _safe_remove_dir(path: Path) -> None:
    if path.exists():
        print(f"[INFO] Removing existing directory: {path}")
        shutil.rmtree(path)


def _find_best_ckpt(ckpt_dir: Path) -> Path:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"ckpt directory does not exist: {ckpt_dir}")

    rank0 = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"))
    candidates = rank0 if rank0 else sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No ckpt file found: {ckpt_dir}")

    pat = re.compile(r"ckpt_(\d+)_rank\d+\.pt$")

    def score(p: Path) -> int:
        m = pat.search(p.name)
        return int(m.group(1)) if m else -1

    best = max(candidates, key=score)
    return best


def _project_name(colmap_project: Path) -> str:
    return colmap_project.resolve().name


def _prepare_workspace(
    python_exe: str,
    colmap_project: Path,
    output_root: Path,
    do_resize: bool,
    resize_width: int,
    force_restart_all: bool,
) -> Path:
    proj_name = _project_name(colmap_project)
    workspace_name = f"{proj_name}_{resize_width}" if do_resize else proj_name
    workspace_root = (output_root / workspace_name).resolve()

    if force_restart_all:
        _safe_remove_dir(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    if not do_resize:
        print(f"[INFO] Resize disabled, workspace: {workspace_root}")
        return workspace_root

    if _is_colmap_project_ready(workspace_root):
        print(f"[INFO] Existing resized result detected, skipping resize step: {workspace_root}")
        return workspace_root

    print(f"[INFO] Running resize, target workspace: {workspace_root}")
    resize_script = REPO_ROOT / "src" / "utils" / "resize_colmap.py"
    input_sparse = _resolve_sparse_dir(colmap_project)

    cmd = [
        python_exe,
        str(resize_script),
        "--input_colmap",
        str(input_sparse),
        "--input_images",
        str(colmap_project / "images"),
        "--output_colmap",
        str(workspace_root / "sparse" / "0"),
        "--output_images",
        str(workspace_root / "images"),
        "--max_edge",
        str(resize_width),
    ]
    _run_cmd(cmd, cwd=REPO_ROOT)
    return workspace_root


def _split_dataset(
    python_exe: str,
    source_colmap_project: Path,
    workspace_root: Path,
    image_mode: str,
) -> None:
    if _is_split_ready(workspace_root):
        print(f"[INFO] Existing train/test split detected, skipping split: {workspace_root}")
        return

    cmd = [
        python_exe,
        "-m",
        "src.utils.split_colmap",
        "--colmap_project",
        str(source_colmap_project),
        "--output_root",
        str(workspace_root),
        "--image_mode",
        image_mode,
    ]
    _run_cmd(cmd, cwd=REPO_ROOT)


def _train_3dgs(
    python_exe: str,
    train_data_path: Path,
    gsplat_result_dir: Path,
    steps_scaler: float,
    eval_steps: int,
    data_factor: int,
    render_traj_path: str,
    disable_viewer: bool,
    cuda_visible_devices: Optional[str],
    save_ply: bool,
    ply_steps: Optional[List[int]],
    normalize_world_space: bool = False,
) -> Path:
    trainer = REPO_ROOT / "third_party" / "gsplat" / "examples" / "simple_trainer.py"
    gsplat_result_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = gsplat_result_dir / "ckpts"
    if ckpt_dir.is_dir():
        try:
            ckpt = _find_best_ckpt(ckpt_dir)
            print(f"[INFO] Existing trained ckpt detected, skipping training: {ckpt}")
            return ckpt
        except FileNotFoundError:
            pass

    cmd = [
        python_exe,
        str(trainer),
        "default",
        "--steps_scaler",
        str(steps_scaler),
        "--eval_steps",
        str(eval_steps),
        "--data_factor",
        str(data_factor),
        "--render_traj_path",
        render_traj_path,
        "--data_dir",
        str(train_data_path),
        "--result_dir",
        str(gsplat_result_dir),
    ]
    # Tyro boolean flags cannot be passed as "--flag False"; use --flag / --no-flag.
    if normalize_world_space:
        cmd.append("--normalize_world_space")
    else:
        cmd.append("--no-normalize_world_space")

    if disable_viewer:
        cmd.append("--disable_viewer")
    if save_ply:
        cmd.append("--save_ply")
    if ply_steps:
        cmd.append("--ply_steps")
        cmd.extend([str(x) for x in ply_steps])

    env = os.environ.copy()
    if cuda_visible_devices is not None and cuda_visible_devices != "":
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    _run_cmd(cmd, cwd=REPO_ROOT, env=env)

    ckpt = _find_best_ckpt(gsplat_result_dir / "ckpts")
    print(f"[INFO] Training finished, selected ckpt: {ckpt}")
    return ckpt


def _run_pose_pipeline(
    python_exe: str,
    train_project: Path,
    test_project: Path,
    retrieval_output_dir: Path,
    ckpt: Path,
    device: str,
    model_args: str,
    camera_model: str,
    data_factor: int,
    block_id: int,
    retrieval_top_k: int,
    retrieval_search_method: str,
    keep_top1_only: bool,
    force_rebuild_cache: bool,
    render_opt_image_size: str,
    world_transform_source: str,
    world_transform_txt: Optional[str],
    cuda_visible_devices: Optional[str],
) -> None:
    pose_script = REPO_ROOT / "scripts" / "lsgs_loc_demo.py"
    retrieval_output_dir.mkdir(parents=True, exist_ok=True)

    pose_cfg_path = retrieval_output_dir / "pipeline_pose_config.yaml"
    with open(pose_cfg_path, "w", encoding="utf-8") as f:
        f.write(f"render_opt_image_size: {render_opt_image_size}\n")

    cmd = [
        python_exe,
        "-m",
        "scripts.lsgs_loc_demo",
        "--config",
        str(pose_cfg_path),
        "--database_root",
        str(train_project),
        "--query_root",
        str(test_project),
        "--output_dir",
        str(retrieval_output_dir),
        "--ckpt",
        str(ckpt),
        "--device",
        device,
        "--model_args",
        model_args,
        "--camera_model",
        camera_model,
        "--data_factor",
        str(data_factor),
        "--block_id",
        str(block_id),
        "--retrieval_top_k",
        str(retrieval_top_k),
        "--retrieval_search_method",
        retrieval_search_method,
        "--world_transform_source",
        world_transform_source,
    ]

    if keep_top1_only:
        cmd.append("--keep_top1_only")
    if force_rebuild_cache:
        cmd.append("--force_rebuild_cache")

    if world_transform_txt:
        cmd.extend(["--world_transform_txt", world_transform_txt])

    env = os.environ.copy()
    if cuda_visible_devices is not None and cuda_visible_devices != "":
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    _run_cmd(cmd, cwd=REPO_ROOT, env=env)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("COLMAP -> (optional resize) -> split -> 3DGS train -> retrieval+pose pipeline")

    parser.add_argument("--colmap_project", type=str, required=True, help="Original COLMAP project directory (contains images/ and sparse/)")
    parser.add_argument("--output_root", type=str, required=True, help="Root output directory")
    parser.add_argument(
        "--force_restart_all",
        action="store_true",
        help="Force restart the full pipeline from scratch (delete existing intermediates and rerun)",
    )

    parser.add_argument("--do_resize", action="store_true", help="Whether to resize to a fixed max edge first")
    parser.add_argument("--resize_width", type=int, default=1600, help="Resize max edge length (default: 1600)")

    parser.add_argument("--image_mode", type=str, default="copy", choices=["copy", "symlink"], help="Image storage mode after split")

    parser.add_argument("--trainer_cuda_visible_devices", type=str, default=None, help="CUDA_VISIBLE_DEVICES used for training")
    parser.add_argument("--trainer_steps_scaler", type=float, default=1.0)
    parser.add_argument("--trainer_eval_steps", type=int, default=-1)
    parser.add_argument("--trainer_data_factor", type=int, default=1)
    parser.add_argument(
        "--trainer_normalize_world_space",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to normalize world coordinates in 3DGS training (default off, keep COLMAP coordinates)",
    )
    parser.add_argument("--trainer_render_traj_path", type=str, default="ellipse")
    parser.add_argument(
        "--trainer_save_ply",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export ply during training (default on)",
    )
    parser.add_argument(
        "--trainer_ply_steps",
        type=int,
        nargs="*",
        default=None,
        help="Optional: override ply export steps, e.g. --trainer_ply_steps 7000 30000",
    )
    parser.add_argument(
        "--trainer_disable_viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to disable gsplat viewer (default disabled)",
    )

    parser.add_argument("--retrieval_device", type=str, default="cuda")
    parser.add_argument("--retrieval_model_args", type=str, default="224", choices=["224", "512"])
    parser.add_argument("--retrieval_camera_model", type=str, default="pinhole", choices=["pinhole", "ortho", "fisheye"])
    parser.add_argument("--retrieval_data_factor", type=int, default=1)
    parser.add_argument("--retrieval_block_id", type=int, default=5)
    parser.add_argument("--retrieval_top_k", type=int, default=3)
    parser.add_argument("--retrieval_search_method", type=str, default="l2", choices=["l2", "cosine"])
    parser.add_argument(
        "--keep_top1_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether retrieval cache keeps only Top1 per query (default: true)",
    )
    parser.add_argument("--force_rebuild_cache", action="store_true")

    parser.add_argument("--render_opt_image_size", type=str, default="original", choices=["original", "1600"])

    parser.add_argument("--world_transform_source", type=str, default="auto", choices=["auto", "builtin", "txt"])
    parser.add_argument("--world_transform_txt", type=str, default=None)

    parser.add_argument("--gsplat_result_subdir", type=str, default="gsplat_scene")
    parser.add_argument("--pose_result_subdir", type=str, default="retrieval_pose_search")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    python_exe = sys.executable
    colmap_project = Path(args.colmap_project).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not (colmap_project / "images").is_dir():
        raise FileNotFoundError(f"images directory not found: {colmap_project / 'images'}")
    _ = _resolve_sparse_dir(colmap_project)

    # Optional resize
    workspace_root = _prepare_workspace(
        python_exe=python_exe,
        colmap_project=colmap_project,
        output_root=output_root,
        do_resize=bool(args.do_resize),
        resize_width=int(args.resize_width),
        force_restart_all=bool(args.force_restart_all),
    )

    # Split train/test
    split_source = workspace_root if args.do_resize else colmap_project
    _split_dataset(
        python_exe=python_exe,
        source_colmap_project=split_source,
        workspace_root=workspace_root,
        image_mode=args.image_mode,
    )

    train_project = workspace_root / "train"
    test_project = workspace_root / "test"


    gsplat_result_dir = (output_root / args.gsplat_result_subdir / workspace_root.name).resolve()
    if args.force_restart_all:
        _safe_remove_dir(gsplat_result_dir)
    ckpt = _train_3dgs(
        python_exe=python_exe,
        train_data_path=train_project,
        gsplat_result_dir=gsplat_result_dir,
        steps_scaler=float(args.trainer_steps_scaler),
        eval_steps=int(args.trainer_eval_steps),
        data_factor=int(args.trainer_data_factor),
        render_traj_path=args.trainer_render_traj_path,
        disable_viewer=bool(args.trainer_disable_viewer),
        cuda_visible_devices=args.trainer_cuda_visible_devices,
        save_ply=bool(args.trainer_save_ply),
        ply_steps=args.trainer_ply_steps,
        normalize_world_space=bool(args.trainer_normalize_world_space),
    )


    pose_out_dir = (output_root / args.pose_result_subdir / workspace_root.name).resolve()
    if args.force_restart_all:
        _safe_remove_dir(pose_out_dir)

    if _is_pose_done(pose_out_dir) and not args.force_restart_all:
        print(f"[INFO] Pose pipeline result already completed, skipping pose pipeline: {pose_out_dir}")
    else:
        _run_pose_pipeline(
            python_exe=python_exe,
            train_project=train_project,
            test_project=test_project,
            retrieval_output_dir=pose_out_dir,
            ckpt=ckpt,
            device=args.retrieval_device,
            model_args=args.retrieval_model_args,
            camera_model=args.retrieval_camera_model,
            data_factor=int(args.retrieval_data_factor),
            block_id=int(args.retrieval_block_id),
            retrieval_top_k=int(args.retrieval_top_k),
            retrieval_search_method=args.retrieval_search_method,
            keep_top1_only=bool(args.keep_top1_only),
            force_rebuild_cache=bool(args.force_rebuild_cache),
            render_opt_image_size=args.render_opt_image_size,
            world_transform_source=args.world_transform_source,
            world_transform_txt=args.world_transform_txt,
            cuda_visible_devices=args.trainer_cuda_visible_devices,
        )

    print("\n[OK] Full pipeline completed")
    print(f"Workspace: {workspace_root}")
    print(f"3DGS results: {gsplat_result_dir}")
    print(f"Retrieval + pose results: {pose_out_dir}")


if __name__ == "__main__":
    main()
