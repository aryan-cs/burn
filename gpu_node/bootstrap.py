from __future__ import annotations

import sys
from pathlib import Path


def ensure_backend_path() -> None:
    """Allow gpu_node to import shared backend modules."""
    backend_dir = Path(__file__).resolve().parents[1] / "backend"
    backend_str = str(backend_dir)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)
