from backend.app import github_client as gc


class FakeResp:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_parse_pr_url():
    assert gc.parse_pr_url("https://github.com/psf/requests/pull/6712") == ("psf", "requests", 6712)
    assert gc.parse_pr_url("https://github.com/o/r/pull/12") == ("o", "r", 12)
    assert gc.parse_pr_url("https://gitlab.com/o/r/-/merge_requests/1") is None
    assert gc.parse_pr_url("github.com/o/r/pull/1/files") == ("o", "r", 1)


def test_fetch_pr_files_maps_and_skips_binary(monkeypatch):
    api_payload = [
        {"filename": "a.py", "patch": "@@ -1 +1 @@\n-x\n+y"},
        {"filename": "logo.png", "patch": None},
        {"filename": "b.py", "patch": "@@ -2 +2 @@\n-z\n+w"},
    ]
    captured = {}

    def fake_request(method, url, token=None, json_body=None):
        captured["url"] = url
        captured["token"] = token
        return FakeResp(api_payload)

    monkeypatch.setattr(gc, "_request", fake_request)
    monkeypatch.setenv("GITHUB_TOKEN", "tok123")
    files = gc.fetch_pr_files("psf", "requests", 12)
    assert files == [
        {"path": "a.py", "patch": "@@ -1 +1 @@\n-x\n+y"},
        {"path": "b.py", "patch": "@@ -2 +2 @@\n-z\n+w"},
    ]
    assert "pulls/12/files" in captured["url"]
    assert captured["token"] == "tok123"


def test_post_inline_comment_payload(monkeypatch):
    captured = {}

    def fake_request(method, url, token=None, json_body=None):
        captured.update(method=method, url=url, body=json_body)
        return FakeResp({"id": 1, "html_url": "https://github.com/o/r/pull/1#discussion_r1"})

    monkeypatch.setattr(gc, "_request", fake_request)
    out = gc.post_inline_comment("o", "r", 5, "abc123", "src/a.py", 3, "watch out", token="t")
    assert captured["method"] == "POST"
    assert "pulls/5/comments" in captured["url"]
    assert captured["body"] == {
        "commit_id": "abc123",
        "path": "src/a.py",
        "line": 3,
        "body": "watch out",
        "subject_type": "line",
    }
    assert out["html_url"].startswith("https://")


def test_get_pr_head(monkeypatch):
    monkeypatch.setattr(gc, "_request", lambda *a, **k: FakeResp({"head": {"sha": "dead beef".replace(" ", "")}}))
    assert gc.get_pr_head("o", "r", 9) == "deadbeef"
