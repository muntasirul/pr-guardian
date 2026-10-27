"""Embedders: a deterministic dummy (tests/mock mode, no torch) and UniXcoder
(the real one used when the ML stack is installed).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np


class DummyEmbedder:
    """Deterministic hash-projection embeddings — no model needed.

    Not semantically meaningful; good for plumbing tests and offline demos.
    """

    dim = 128

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8", "ignore")).digest()  # 32 bytes
            reps = max(2, self.dim * 8 // 32 + 1)  # enough bytes for dim float64 values
            raw = np.frombuffer(h * reps, dtype=np.float64)[: self.dim]
            # sign * log1p keeps huge/overflowing values in a stable range
            v = np.sign(raw) * np.log1p(np.abs(raw))
            v = (v - v.mean()) / (v.std() + 1e-9)
            out[i] = v
        return out


class UniXcoderEmbedder:
    """microsoft/unixcoder-base mean-pooled, L2-normalized embeddings."""

    def __init__(self, model_name: str = "microsoft/unixcoder-base"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def embed(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        import torch

        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            outs.append(pooled.cpu().numpy())
        v = np.concatenate(outs, axis=0)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def get_embedder(kind: str | None = None) -> DummyEmbedder | UniXcoderEmbedder:
    kind = kind or os.getenv("PRGUARDIAN_EMBEDDER", "dummy")
    if kind == "unixcoder":
        return UniXcoderEmbedder()
    return DummyEmbedder()
