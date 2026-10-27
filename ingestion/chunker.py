import re
from collections import Counter
from pathlib import PurePosixPath

import pymupdf

# Headings are detected by font (bold + larger than body text) AND numbering, so that
# numbered table rows such as "6. Personal accident" are not mistaken for sections.
SECTION_RE = re.compile(r"^\d{1,2}\.\s+\S")
SUBSECTION_RE = re.compile(r"^[A-Z]\.\s+\S")
PAGE_NO_RE = re.compile(r"^Page \d+$")
SECTION_SIZE_RATIO = 1.25
SUBSECTION_SIZE_RATIO = 1.08


def _read_lines(doc):
    """Return (page_no, text, size, bold) for every visual line in the PDF."""
    lines = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = " ".join(s["text"].strip() for s in spans)
                size = max(s["size"] for s in spans)
                bold = all(s["flags"] & 16 or "Bold" in s["font"] for s in spans)
                lines.append((page.number + 1, text, size, bold))
    return lines


def _repeated_lines(lines, page_count):
    """Running headers/footers: lines that appear on (nearly) every page."""
    if page_count < 2:
        return set()
    pages_per_text = Counter(text for _, text in {(page, text) for page, text, _, _ in lines})
    return {text for text, n in pages_per_text.items() if n >= 0.8 * page_count}


def _body_size(lines):
    sizes = Counter()
    for _, text, size, _ in lines:
        sizes[round(size, 1)] += len(text)
    return sizes.most_common(1)[0][0]


def _split_sections(lines, page_count):
    """Group lines into sections / subsections based on heading fonts."""
    skip = _repeated_lines(lines, page_count)
    body = _body_size(lines)
    sections = []
    current = {"section": "Overview", "subsection": "", "page": 1, "lines": []}

    for page, text, size, bold in lines:
        if text in skip or PAGE_NO_RE.match(text):
            continue
        if bold and size >= body * SECTION_SIZE_RATIO and SECTION_RE.match(text):
            sections.append(current)
            current = {"section": text, "subsection": "", "page": page, "lines": []}
        elif bold and size >= body * SUBSECTION_SIZE_RATIO and SUBSECTION_RE.match(text):
            sections.append(current)
            current = {"section": current["section"], "subsection": text, "page": page, "lines": []}
        else:
            current["lines"].append(text)
    sections.append(current)
    return [s for s in sections if s["lines"]]


def _pack(header, lines, max_tokens, count_tokens, overlap):
    """Greedily pack lines into chunks that fit the embedding model's token window."""
    chunks, current = [], []
    for line in lines:
        candidate = current + [line]
        if current and count_tokens(header + "\n".join(candidate)) > max_tokens:
            chunks.append(current)
            current = current[-overlap:] + [line] if overlap else [line]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_pdf(data: bytes, source: str, count_tokens, max_tokens: int, overlap: int = 1):
    """Section-aware chunking of a PDF. Returns a list of {"text", "metadata"} dicts.

    Each section (and subsection) becomes one chunk; sections longer than the model's
    token window are split on line boundaries with `overlap` lines carried over.
    Every chunk is prefixed with the document title and section path for context.
    """
    title = PurePosixPath(source).stem.replace("_", " ")
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        sections = _split_sections(_read_lines(doc), doc.page_count)

    chunks = []
    for sec in sections:
        path = f"{sec['section']} > {sec['subsection']}" if sec["subsection"] else sec["section"]
        header = f"{title}\nSection: {path}\n"
        for part in _pack(header, sec["lines"], max_tokens, count_tokens, overlap):
            chunks.append({
                "text": header + "\n".join(part),
                "metadata": {
                    "source": source,
                    "title": title,
                    "section": sec["section"],
                    "subsection": sec["subsection"],
                    "page": sec["page"],
                    "chunk_index": len(chunks),
                },
            })
    return chunks
