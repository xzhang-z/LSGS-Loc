from __future__ import annotations

"""LSGS-Loc method demo pipeline entry.

This script is an alias entry for the full
pipeline used by the paper method.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_workflow import main


if __name__ == "__main__":
    main()
