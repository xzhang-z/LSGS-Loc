from __future__ import annotations

import argparse

from src.datasets.camera_cache_builder import (
    prepare_query_camera_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser("Build query camera cache from retrieval matches + COLMAP")
    parser.add_argument("--retrieval_file", type=str, required=True, help="Retrieval result txt file")
    parser.add_argument("--database_dir", type=str, required=True, help="Database scene root directory (contains sparse)")
    parser.add_argument("--query_dir", type=str, required=True, help="Query scene root directory (contains sparse)")
    parser.add_argument("--cache_path", type=str, required=True, help="Output cache pt file path")
    parser.add_argument("--factor", type=int, default=1, help="Scale factor consistent with COLMAP parsing")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--world_rank", type=int, default=0)
    parser.add_argument("--keep_topk", action="store_true", help="By default only keep Top1; enable this to keep all ranks in the log")
    args = parser.parse_args()

    query_cameras = prepare_query_camera_cache(
        retrieval_file=args.retrieval_file,
        database_dir=args.database_dir,
        query_dir=args.query_dir,
        cache_path=args.cache_path,
        factor=args.factor,
        world_size=args.world_size,
        world_rank=args.world_rank,
        keep_top1_only=not args.keep_topk,
        force_rebuild=True,
    )

    print(f"Number of cached cameras: {len(query_cameras)}")
    print(f"Cache written to: {args.cache_path}")


if __name__ == "__main__":
    main()
