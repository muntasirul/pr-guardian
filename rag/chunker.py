"""AST-based chunking of Python code (functions/classes with line ranges).

tree-sitter remains the stretch goal for multi-language support; for Python the
built-in `ast` module is dependency-free and just as precise.
"""

from __future__ import annotations

import ast
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}


def chunk_python_source(source: str, path: str = "") -> list[dict]:
    chunks: list[dict] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seg = ast.get_source_segment(source, node)
            if not seg or seg.count("\n") < 1:  # need at least 2 lines
                continue
            chunks.append(
                {
                    "path": path,
                    "name": node.name,
                    "kind": type(node).__name__,
                    "code": seg,
                    "start_line": node.lineno,
                }
            )
    return chunks


def chunk_repo(repo_path: Path, max_files: int | None = None) -> list[dict]:
    repo_path = Path(repo_path)
    chunks: list[dict] = []
    n_files = 0
    for py in sorted(repo_path.rglob("*.py")):
        if any(part in SKIP_DIRS for part in py.relative_to(repo_path).parts):
            continue
        try:
            source = py.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = py.relative_to(repo_path).as_posix()
        chunks.extend(chunk_python_source(source, path=rel))
        n_files += 1
        if max_files and n_files >= max_files:
            break
    return chunks
