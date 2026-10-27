"""Root conftest: puts the repo root on sys.path so tests can import
`mining.*`, `training.*`, `rag.*`, `backend.*` as plain packages."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
