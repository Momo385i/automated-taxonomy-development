from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# PyMuPDF4LLM: Markdown, kein OCR, keine Bilder/Vektorgrafiken (schnell, für durchsuchbare PDFs).
_PYMUPDF4LLM_KWARGS: dict = {
    "use_ocr": False,
    "force_ocr": False,
    "ignore_images": True,
    "ignore_graphics": True,
    "write_images": False,
    "embed_images": False,
    "header": False,
    "footer": False
}

# Whole-word only (avoids "preferences", "bibliographical references" partials on wrong anchor, etc.)
_REF_HEADING = re.compile(r"\breferences\b", re.IGNORECASE)
_ACK_HEADING = re.compile(
    r"\b(?:acknowledgements|acknowledgments|acknowledgment|acknowledgement)\b",
    re.IGNORECASE,
)


def _last_regex_match_start(pattern: re.Pattern[str], s: str) -> int:
    last = -1
    for m in pattern.finditer(s):
        last = m.start()
    return last


def list_pdfs(input_dir: Path) -> list[Path]:  # returns a list of all pdf files in the input directory.
    if not input_dir.exists():
        return []
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def sha256_file(path: Path) -> str:  # returns the sha256 hash of the file. Useful for uniquely identifying a file or detecting changes.
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def truncate_trailing_acknowledgements_and_references(text: str) -> str:
    """
    Drop acknowledgements / bibliography tails to save tokens.
    Matches only **whole words** (word boundaries), case-insensitive:
    ``references``; ``acknowledgement(s)`` / ``acknowledgment(s)`` (UK/US singular & plural).
    Each hit counts only if it lies after the first 10% of the doc string.
    Cut at the *earlier* valid index so everything from the first of those sections onward is removed.
    """
    if not text:
        return text
    n = len(text)
    thresh = n * 0.10

    last_ref = _last_regex_match_start(_REF_HEADING, text)
    ref_ok = last_ref != -1 and last_ref > thresh

    last_ack = _last_regex_match_start(_ACK_HEADING, text)
    ack_ok = last_ack != -1 and last_ack > thresh

    candidates: list[int] = []
    if ref_ok:
        candidates.append(last_ref)
    if ack_ok:
        candidates.append(last_ack)
    if not candidates:
        return text

    cut = min(candidates)
    return text[:cut].rstrip()


def _extract_plain_pypdf(path: Path) -> tuple[str, int]: #fallback function if pymudpdf4llm fails
    reader = PdfReader(str(path))
    pages = reader.pages
    texts: list[str] = []
    for page in pages:
        t = page.extract_text() or ""
        texts.append(t)
    text = "\n\n".join(texts).strip()
    text = truncate_trailing_acknowledgements_and_references(text)
    return text, len(pages)


def extract_text_from_pdf(path: Path) -> tuple[str, int]:
    """
    PDF → Markdown via PyMuPDF4LLM ``to_markdown``, then remove trailing References/Acknowledgements.

    Ingest continues to store ``.txt`` files; the content is Markdown and is loaded unchanged as ``text`` by the graph — so LLMs can use it directly.

    On extraction errors, fallback to plain text via pypdf (without Markdown/layout).
    """
    try:
        with fitz.open(str(path)) as doc:
            page_count = len(doc)
            md = pymupdf4llm.to_markdown(doc, **_PYMUPDF4LLM_KWARGS)
    except Exception as e:
        logger.warning(
            "pymupdf4llm extraction failed for %s, falling back to pypdf: %s",
            path.name,
            e,
        )
        plain, page_count = _extract_plain_pypdf(path)
        plain = truncate_trailing_acknowledgements_and_references(plain)
        return plain.strip(), page_count

    if not isinstance(md, str):
        raise TypeError(f"expected str from to_markdown, got {type(md)}")
    md = truncate_trailing_acknowledgements_and_references(md)
    return md.strip(), page_count
