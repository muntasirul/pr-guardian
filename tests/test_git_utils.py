from mining.git_utils import parse_blame_porcelain

A_SHA = "a" * 40
B_SHA = "b" * 40

PORCELAIN = f"""{A_SHA} 1 1 2
author Tester
author-mail <t@example.com>
summary add app module
\tline one content
\tline two content
{B_SHA} 3 3 1
author Tester
\tthird line content
\tfourth line blamed to same commit as third (no new header)
{A_SHA} 5 5 1
author Tester
\tfifth line
"""


def test_parse_blame_porcelain_groups():
    mapping = parse_blame_porcelain(PORCELAIN)
    assert mapping[1] == A_SHA
    assert mapping[2] == A_SHA
    assert mapping[3] == B_SHA
    assert mapping[4] == B_SHA  # no header emitted for same-commit groups
    assert mapping[5] == A_SHA
    assert set(mapping) == {1, 2, 3, 4, 5}


def test_parse_blame_empty():
    assert parse_blame_porcelain("") == {}
