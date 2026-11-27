"""Offline unit tests for the eCFR XML parser (no network/API calls)."""

import xml.etree.ElementTree as ET

from src.ingest import parse_cfr

# A minimal eCFR-shaped document: one subpart (DIV6) with one section (DIV8)
# containing two paragraphs, one of which carries cross-references.
SAMPLE_XML = """
<DIV5>
  <DIV6>
    <HEAD>Subpart B—Operating Rules</HEAD>
    <DIV8 N="107.51">
      <HEAD>§ 107.51 Operating limitations for small unmanned aircraft.</HEAD>
      <P>A remote pilot must comply with the following operating limitations.</P>
      <P>The groundspeed may not exceed 87 knots, see § 107.3 and part 89.</P>
    </DIV8>
  </DIV6>
</DIV5>
"""


def _write_sample(tmp_path):
    path = tmp_path / "part107.xml"
    path.write_text(SAMPLE_XML, encoding="utf-8")
    return path


def test_clean_text_collapses_whitespace():
    elem = ET.fromstring("<P>Hello   world\n\tagain</P>")
    assert parse_cfr.clean_text(elem) == "Hello world again"


def test_parse_part_builds_expected_chunk(tmp_path):
    chunks = parse_cfr.parse_part(_write_sample(tmp_path), "107")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["chunk_id"] == "14cfr_107_51"
    assert chunk["citation"] == "14 CFR § 107.51"
    assert chunk["part"] == "107"
    assert chunk["section_number"] == "107.51"
    assert chunk["source_type"] == "cfr_regulation"
    assert chunk["subpart"] == "Subpart B—Operating Rules"
    # The breadcrumb precedes the paragraph body.
    assert chunk["text"].startswith("[14 CFR § 107.51")
    assert "groundspeed may not exceed 87 knots" in chunk["text"]


def test_parse_part_extracts_cross_references(tmp_path):
    chunk = parse_cfr.parse_part(_write_sample(tmp_path), "107")[0]
    refs = chunk["cross_references"]
    assert any("107.3" in r for r in refs)
    assert any(r.lower().startswith("part 89") for r in refs)


def test_parse_part_splits_long_sections(tmp_path):
    # Two paragraphs each longer than MAX_CHARS must land in separate chunks.
    big = "word " * (parse_cfr.MAX_CHARS // 2)
    xml = f"""
    <DIV5><DIV6><HEAD>Subpart X</HEAD>
      <DIV8 N="107.99">
        <HEAD>§ 107.99 Big section.</HEAD>
        <P>{big}</P>
        <P>{big}</P>
      </DIV8>
    </DIV6></DIV5>
    """
    path = tmp_path / "big.xml"
    path.write_text(xml, encoding="utf-8")

    chunks = parse_cfr.parse_part(path, "107")
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "14cfr_107_99_p1"
    assert chunks[1]["chunk_id"] == "14cfr_107_99_p2"


def test_parse_part_skips_sections_without_paragraphs(tmp_path):
    xml = """
    <DIV5><DIV6><HEAD>Subpart Y</HEAD>
      <DIV8 N="107.1"><HEAD>§ 107.1 Empty.</HEAD></DIV8>
    </DIV6></DIV5>
    """
    path = tmp_path / "empty.xml"
    path.write_text(xml, encoding="utf-8")
    assert parse_cfr.parse_part(path, "107") == []
