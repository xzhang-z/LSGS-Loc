from __future__ import annotations

import argparse

from src.models.anyloc_vlad_retrieval import AnyLocVLADConfig, run_anyloc_vlad_retrieval


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("AnyLoc VLAD retrieval")
    parser.add_argument("--database_images_dir", type=str, required=True, help="Path to database/training images directory")
    parser.add_argument("--query_images_dir", type=str, required=True, help="Path to query/test images directory")
    parser.add_argument("--top_k", type=int, default=3, help="Retrieval top-k (default: 3)")
    parser.add_argument(
        "--output_txt_file",
        type=str,
        default=None,
        help="Optional output txt path for retrieval results (default: auto-generated)",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = AnyLocVLADConfig(
        database_images_dir=args.database_images_dir,
        query_images_dir=args.query_images_dir,
        img_exts=("jpg", "png"),
        domain="urban",
        top_k=args.top_k,
        search_method="l2",
        output_txt_file=args.output_txt_file,
    )

    out = run_anyloc_vlad_retrieval(cfg)
    print("Retrieval completed")
    print(f"Database VLAD shape: {out['database_vlads_shape']}")
    print(f"Query VLAD shape: {out['query_vlads_shape']}")
    print(f"Result file: {out['output_txt_file']}")
