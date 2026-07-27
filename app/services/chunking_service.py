from dataclasses import dataclass
from typing import Optional

from app.services.extraction_service import ExtractedPage
from app.utils.text_utils import estimate_tokens, is_heading, split_sentences

MIN_CHUNK_CHARS = 200


@dataclass
class ChunkDraft:
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    section_label: Optional[str]
    token_count: int
    char_count: int


def _paragraphs_with_pages(pages: list[ExtractedPage]) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    for page in pages:
        for para in page.text.split("\n\n"):
            para = para.strip()
            if para:
                items.append((page.page_number, para))
    return items


def chunk_document(pages: list[ExtractedPage], target_chars: int, overlap_chars: int) -> list[ChunkDraft]:
    paragraphs = _paragraphs_with_pages(pages)

    drafts: list[ChunkDraft] = []
    buffer_parts: list[str] = []
    buffer_pages: list[int] = []
    buffer_len = 0
    current_section: Optional[str] = None
    chunk_index = 0

    def flush(carry_overlap: bool) -> None:
        nonlocal buffer_parts, buffer_pages, buffer_len, chunk_index
        if not buffer_parts:
            return
        text = "\n\n".join(buffer_parts).strip()
        if not text:
            buffer_parts, buffer_pages, buffer_len = [], [], 0
            return

        if drafts and len(text) < MIN_CHUNK_CHARS:
            # Too small to stand alone (e.g. a trailing fragment) — merge into the previous chunk.
            prev = drafts[-1]
            merged_text = f"{prev.text}\n\n{text}"
            drafts[-1] = ChunkDraft(
                chunk_index=prev.chunk_index,
                text=merged_text,
                page_start=prev.page_start,
                page_end=max(prev.page_end, max(buffer_pages)),
                section_label=prev.section_label,
                token_count=estimate_tokens(merged_text),
                char_count=len(merged_text),
            )
        else:
            drafts.append(
                ChunkDraft(
                    chunk_index=chunk_index,
                    text=text,
                    page_start=min(buffer_pages),
                    page_end=max(buffer_pages),
                    section_label=current_section,
                    token_count=estimate_tokens(text),
                    char_count=len(text),
                )
            )
            chunk_index += 1

        if carry_overlap and overlap_chars > 0:
            overlap_text = text[-overlap_chars:]
            # Snap forward to the start of a sentence so we never resume mid-sentence.
            sentences = split_sentences(overlap_text)
            carried = " ".join(sentences[-2:]) if len(sentences) > 1 else overlap_text
            buffer_parts = [carried] if carried.strip() else []
            buffer_pages = [buffer_pages[-1]] if buffer_parts else []
            buffer_len = len(carried) if buffer_parts else 0
        else:
            buffer_parts, buffer_pages, buffer_len = [], [], 0

    for page_number, paragraph in paragraphs:
        if is_heading(paragraph):
            flush(carry_overlap=False)
            current_section = paragraph
            continue

        # Split oversized paragraphs into sentences so we never break mid-sentence.
        pieces = [paragraph] if len(paragraph) <= target_chars else split_sentences(paragraph)

        for piece in pieces:
            if buffer_len + len(piece) > target_chars and buffer_parts:
                flush(carry_overlap=True)
            buffer_parts.append(piece)
            buffer_pages.append(page_number)
            buffer_len += len(piece)

    flush(carry_overlap=False)
    return drafts
