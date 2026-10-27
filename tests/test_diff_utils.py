from mining.diff_utils import diff_stats, parse_unified_diff, split_multi_file_diff

DIFF = """diff --git a/f.py b/f.py
index 1234567..89abcde 100644
--- a/f.py
+++ b/f.py
@@ -1,4 +1,4 @@
 old1
-old2
+new2
 old3
-- a removed line that looks like a header
+normal added
"""

EMPTY_HEADER_ONLY = "diff --git a/x.py b/x.py\nindex 111..222 100644\n--- a/x.py\n+++ b/x.py\n"


def test_parse_hunks_line_numbers():
    hunks = parse_unified_diff(DIFF)
    assert len(hunks) == 1
    h = hunks[0]
    assert h.old_start == 1
    assert h.new_start == 1
    assert h.removed == {
        2: "old2",
        4: "- a removed line that looks like a header",
    }
    assert h.added == {2: "new2", 4: "normal added"}


def test_parse_header_only_diff():
    assert parse_unified_diff(EMPTY_HEADER_ONLY) == []
    assert parse_unified_diff("") == []


def test_diff_stats():
    s = diff_stats(DIFF)
    assert s == {"added": 2, "removed": 2, "changed": 4}


MULTI = """diff --git a/src/a.py b/src/a.py
index 111..222 100644
--- a/src/a.py
+++ b/src/a.py
@@ -1,2 +1,2 @@
 x = 1
-x = 2
+x = 3
diff --git a/src/b.py b/src/b.py
index 333..444 100644
--- a/src/b.py
+++ b/src/b.py
@@ -1 +1 @@
-y = 1
+y = 2
"""


def test_split_multi_file_diff():
    parts = split_multi_file_diff(MULTI)
    assert [p for p, _ in parts] == ["src/a.py", "src/b.py"]
    assert "x = 3" in parts[0][1]
    assert "y = 2" in parts[1][1]
    assert split_multi_file_diff("") == []
    assert split_multi_file_diff("no git headers here") == [("unknown", "no git headers here")]
