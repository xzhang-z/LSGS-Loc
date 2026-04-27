from __future__ import annotations

"""LSGS-Loc end-to-end pipeline from raw COLMAP project.

This script is an alias entry for the full workflow including preprocessing,
split, 3DGS training, and LSGS-Loc inference.
"""

from scripts.run_colmap_full_pipeline import main


if __name__ == "__main__":
    main()
