"""Vector index over repo chunks.

Default backend is a local NumPy store (no infra needed); Qdrant stays the
documented production path (see the project plan §6) and can be added behind
the same interface.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.chunker import chunk_repo
from rag.embedder import get_embedder

INDEX_DIR = Path("data/index")


def _store_paths(index_path: str | Path) -> tuple[Path, Path]:
    index_path = Path(index_path)
    return index_path.with_suffix(".npz"), index_path.with_suffix(".jsonl")


def build_index(repo_path: Path, index_path: str | Path, max_files: int | None = None) -> dict:
    """Chunk a repo, embed the chunks, save vectors + metadata."""
    chunks = chunk_repo(repo_path, max_files=max_files)
    if not chunks:
        return {"chunks": 0}
    embedder = get_embedder()
    texts = [c["code"] for c in chunks]
    vecs = embedder.embed(texts)
    if isinstance(vecs, np.ndarray):
        vecs = vecs.astype(np.float32)
    vpath, mpath = _store_paths(index_path)
    vpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(vpath, vectors=vecs)
    with mpath.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return {"chunks": len(chunks), "vectors": vpath, "meta": mpath, "dim": int(vecs.shape[1])}


class LocalIndex:
    def __init__(self, index_path: str | Path):
        vpath, mpath = _store_paths(index_path)
        if not vpath.exists():
            raise FileNotFoundError(f"index not found: {vpath} (run `python -m rag.indexer index ...`)")
        data = np.load(vpath)
        self.vectors: np.ndarray = data["vectors"]
        self.meta = [json.loads(ln) for ln in mpath.read_text().splitlines() if ln.strip()]
        self.embedder = get_embedder()

    def search(self, query_text: str, k: int = 5, exclude_path: str | None = None) -> list[tuple[dict, float]]:
        q = self.embedder.embed([query_text])
        if isinstance(q, np.ndarray):
            q = q[0]
        sims = self.vectors @ q / (np.linalg.norm(self.vectors, axis=1) * (np.linalg.norm(q) + 1e-9))
        order = np.argsort(-sims)
        out = []
        for i in order:
            m = self.meta[int(i)]
            if exclude_path and m["path"] == exclude_path:
                continue
            out.append((m, float(sims[int(i)])))
            if len(out) >= k:
                break
        return out
