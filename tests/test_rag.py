from pathlib import Path

import pytest

from rag.chunker import chunk_python_source
from rag.embedder import DummyEmbedder
from rag.indexer import LocalIndex, build_index
from rag.retrieve import query_text_from_diff, retrieve_context

SRC = '''
import os

def alpha(x):
    return x + 1

class Widget:
    def spin(self):
        return "spin"

def beta(y):
    return y * 2
'''


def test_chunker_extracts_functions_and_classes():
    chunks = chunk_python_source(SRC, path="src/thing.py")
    names = {(c["kind"], c["name"]) for c in chunks}
    assert ("FunctionDef", "alpha") in names
    assert ("FunctionDef", "beta") in names
    assert ("ClassDef", "Widget") in names
    assert ("FunctionDef", "spin") in names
    assert all(c["path"] == "src/thing.py" for c in chunks)


def test_chunker_syntax_error_returns_empty():
    assert chunk_python_source("def broken(:\n") == []


@pytest.fixture()
def indexed_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "core.py").write_text("def process(items):\n    return [i for i in items]\n")
    (repo / "pkg" / "util.py").write_text("def double(n):\n    return 2 * n\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_core.py").write_text("def test_process():\n    assert process([]) == []\n")
    index = tmp_path / "index" / "repo"
    stats = build_index(repo, index)
    assert stats["chunks"] >= 3
    return repo, index


def test_build_and_search_index(indexed_repo):
    _, index = indexed_repo
    local = LocalIndex(index)
    # the dummy embedder has no semantics, but querying with a chunk's exact
    # stored text produces an identical vector -> cosine ~1.0 -> that chunk top
    target = next(m for m in local.meta if m["name"] == "process")
    hits = local.search(target["code"], k=3)
    assert hits
    assert hits[0][1] > 0.99
    assert hits[0][0]["name"] == "process"


def test_search_excludes_path(indexed_repo):
    _, index = indexed_repo
    local = LocalIndex(index)
    hits = local.search("process items", k=10, exclude_path="pkg/core.py")
    assert all(h["path"] != "pkg/core.py" for h, _ in hits)


def test_query_text_from_diff():
    diff = "@@ -1,2 +1,3 @@\n-old\n+new stuff\n context"
    assert "new stuff" in query_text_from_diff(diff)


def test_retrieve_context_assembles_blocks(indexed_repo):
    _, index = indexed_repo
    ctx = retrieve_context(index, "pkg/core.py", "@@ -1 +1 @@\n+x = process([])\n")
    assert ctx is not None
    assert "---" in ctx
    assert "# pkg/core.py (" not in ctx  # changed file is excluded from context


def test_retrieve_context_missing_index(tmp_path):
    assert retrieve_context(tmp_path / "nope", "a.py", "+ x") is None


def test_dummy_embedder_deterministic():
    d = DummyEmbedder()
    a = d.embed(["hello world"])
    b = d.embed(["hello world"])
    assert (a == b).all()
    assert a.shape == (1, DummyEmbedder.dim)
