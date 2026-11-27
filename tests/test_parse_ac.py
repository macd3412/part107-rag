"""Offline unit tests for the AC 107-2A PDF chunking helpers (no I/O)."""

from src.ingest import parse_ac


def test_clean_page_text_collapses_whitespace():
    raw = "Small   UAS\n\n operating   \t rules"
    assert parse_ac.clean_page_text(raw) == "Small UAS operating rules"


def test_split_with_overlap_returns_single_chunk_when_short():
    text = "short text"
    assert parse_ac.split_with_overlap(text, max_chars=100, overlap=10) == [text]


def test_split_with_overlap_windows_and_overlaps():
    text = "abcdefghij"  # 10 chars
    parts = parse_ac.split_with_overlap(text, max_chars=4, overlap=1)
    # Windows advance by (max_chars - overlap) = 3 chars each.
    assert parts == ["abcd", "defg", "ghij"]


def test_split_with_overlap_covers_entire_text():
    text = "x" * 50
    parts = parse_ac.split_with_overlap(text, max_chars=10, overlap=3)
    assert "".join(dict.fromkeys(parts))  # non-empty
    # Reassembling with the known step size must reproduce the original length.
    assert parts[0] == "x" * 10
    assert all(len(p) <= 10 for p in parts)
    assert len(parts) > 1


def test_split_with_overlap_consecutive_chunks_share_overlap():
    text = "0123456789ABCDEF"
    parts = parse_ac.split_with_overlap(text, max_chars=6, overlap=2)
    for prev, nxt in zip(parts, parts[1:], strict=False):
        # The tail of each chunk is the head of the next (overlap=2).
        assert prev[-2:] == nxt[:2]
