"""Compatibility shim.

Please use `src.models.reloc3r_pose_attention` instead.
"""

from src.models.reloc3r_pose_attention import (  # noqa: F401
    setup_reloc3r_estimator,
    run_reloc3r_estimation_with_attention,
    extract_top_coords_from_center_attention,
    map_grid_coords_to_pixel_regions,
)
