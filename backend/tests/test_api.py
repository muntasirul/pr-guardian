import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("PRGUARDIAN_MOCK", "1")

from backend.app.main import app


@pytest.fixture()
def client():
    return TestClient(app)


DIFF_BUG = """@@ -1,3 +1,4 @@
 def run(user_input):
+    result = eval(user_input)
     return result
"""

DIFF_CLEAN = """@@ -1,2 +1,2 @@
-value = 0
+value = 1
"""

DIFF_TWO_FILES = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1 +1 @@
-x = open("f")
+x = 1
diff --git a/src/b.py b/src/b.py
--- a/src/b.py
+++ b/src/b.py
@@ -1 +1 @@
-y = 2
+f = open("data")
"""


def test_review_detects_bug(client):
    r = client.post("/review", json={"diff": DIFF_BUG})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "BUG"
    assert len(body["comments"]) == 1
    assert body["comments"][0]["type"] == "security"
    assert body["model_mode"] == "mock"

    r2 = client.get(f"/reviews/{body['id']}")
    assert r2.status_code == 200
    assert r2.json()["verdict"] == "BUG"


def test_review_clean_diff_no_comments(client):
    r = client.post("/review", json={"diff": DIFF_CLEAN})
    assert r.status_code == 200
    assert r.json()["verdict"] == "CLEAN"
    assert r.json()["comments"] == []


def test_review_splits_multi_file_diff(client):
    r = client.post("/review", json={"diff": DIFF_TWO_FILES})
    body = r.json()
    assert body["n_files"] == 2
    assert body["verdict"] == "BUG"  # open() without `with` in the first file


def test_review_requires_input(client):
    r = client.post("/review", json={})
    assert r.status_code == 400


def test_review_pr_url_without_token(client, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    r = client.post("/review", json={"pr_url": "https://github.com/o/r/pull/1"})
    assert r.status_code == 400
    r2 = client.post("/review", json={"pr_url": "not-a-url"})
    assert r2.status_code == 400


def test_stats_and_feedback(client):
    client.post("/review", json={"diff": DIFF_BUG})
    client.post("/review", json={"diff": DIFF_CLEAN})
    stats = client.get("/stats").json()
    assert stats["total_reviews"] >= 2
    assert stats["bug_reviews"] >= 1

    from backend.app.main import _REVIEWS

    rid = next(iter(_REVIEWS))
    fb = client.post("/feedback", json={"review_id": rid, "helpful": True})
    assert fb.status_code == 200
    assert client.post("/feedback", json={"review_id": "nope", "helpful": True}).status_code == 404


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
