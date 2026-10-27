from pathlib import Path

import pytest

from ingestion.chunker import _pack, chunk_pdf
from tests.helpers import build_pdf, word_count

FILES = Path(__file__).parent.parent / "files"
HEADER = "Acme General Insurance - Policy Wording"
BODY, SUB, SEC = 9.3, 10.5, 12.5


def page(n, *lines):
    return [(HEADER, BODY, False), *lines, (f"Page {n}", BODY, False)]


@pytest.fixture(scope="module")
def chunks():
    body = [(f"Body line {i} with ordinary policy wording text.", BODY, False) for i in range(6)]
    pdf = build_pdf([
        page(1,
             ("1. Policy Details", SEC, True), *body,
             # A numbered table row in body font must not become a section.
             ("6. Personal accident cover", BODY, False)),
        page(2,
             ("2. Claims", SEC, True), *body[:2],
             ("A. Cashless", SUB, True), ("Inform the insurer 48 hours before admission.", BODY, False),
             ("B. Reimbursement", SUB, True), ("Submit documents within 30 days.", BODY, False)),
        page(3, ("3. Exclusions", SEC, True), *body),
    ])
    return chunk_pdf(pdf, "policies/Acme_Health_Policy.pdf", word_count, max_tokens=200)


def test_splits_on_section_headings(chunks):
    assert list(dict.fromkeys(c["metadata"]["section"] for c in chunks)) == [
        "1. Policy Details", "2. Claims", "3. Exclusions"]


def test_subsections_keep_their_parent_section(chunks):
    subs = [(c["metadata"]["section"], c["metadata"]["subsection"]) for c in chunks if c["metadata"]["subsection"]]
    assert subs == [("2. Claims", "A. Cashless"), ("2. Claims", "B. Reimbursement")]
    cashless = next(c for c in chunks if c["metadata"]["subsection"] == "A. Cashless")
    assert "Section: 2. Claims > A. Cashless" in cashless["text"]
    assert "48 hours" in cashless["text"]


def test_numbered_body_row_is_not_a_section(chunks):
    first = next(c for c in chunks if c["metadata"]["section"] == "1. Policy Details")
    assert "6. Personal accident cover" in first["text"]


def test_running_headers_and_page_numbers_are_removed(chunks):
    for c in chunks:
        assert HEADER not in c["text"]
        assert "Page 2" not in c["text"]


def test_metadata(chunks):
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert {c["metadata"]["title"] for c in chunks} == {"Acme Health Policy"}
    assert {c["metadata"]["source"] for c in chunks} == {"policies/Acme_Health_Policy.pdf"}
    assert next(c for c in chunks if c["metadata"]["section"] == "2. Claims")["metadata"]["page"] == 2
    assert all(c["text"].startswith("Acme Health Policy\nSection: ") for c in chunks)


def test_long_sections_are_split_within_the_token_limit():
    lines = [(f"Exclusion clause number {i} applies to this policy.", BODY, False) for i in range(40)]
    pdf = build_pdf([[("1. Exclusions", SEC, True), *lines]])
    chunks = chunk_pdf(pdf, "doc.pdf", word_count, max_tokens=60)
    assert len(chunks) > 1
    assert all(word_count(c["text"]) <= 60 for c in chunks)
    # One line of overlap between consecutive chunks.
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a["text"].splitlines()[-1] == b["text"].splitlines()[2]


@pytest.mark.parametrize(("overlap", "expected"), [
    (0, [["a b", "c d"], ["e f"]]),
    (1, [["a b", "c d"], ["c d", "e f"]]),
])
def test_pack(overlap, expected):
    assert _pack("", ["a b", "c d", "e f"], max_tokens=4, count_tokens=word_count, overlap=overlap) == expected


def test_pack_keeps_an_oversized_line_as_its_own_chunk():
    assert _pack("", ["one two three four five"], max_tokens=2, count_tokens=word_count, overlap=1) == [
        ["one two three four five"]]


@pytest.mark.parametrize(("pdf", "expected"), [
    ("Health_Insurance_Policy_Suraksha_Family_Health_Plus.pdf",
     {("6. Waiting Periods", ""), ("7. Sub-limits", ""), ("9. Permanent Exclusions", ""),
      ("11. Claims Procedure", "A. Cashless claims (network hospitals)"),
      ("11. Claims Procedure", "B. Reimbursement claims")}),
    ("Home_Insurance_Policy_Griha_Kavach.pdf",
     {("3. Schedule of Cover & Sums Insured", ""), ("7. Deductibles (Excess)", ""),
      ("4. Valuation Report Summary", "C. Jewellery & valuables (Section 3)"), ("13. Claims Procedure", "")}),
])
def test_real_policy_documents(pdf, expected):
    chunks = chunk_pdf((FILES / pdf).read_bytes(), pdf, word_count, max_tokens=200)
    found = {(c["metadata"]["section"], c["metadata"]["subsection"]) for c in chunks}
    assert expected <= found
    sections = {c["metadata"]["section"] for c in chunks}
    assert len([s for s in sections if s != "Overview"]) == 15
