from dataclasses import dataclass

import fitz  # PyMuPDF

from app.core.exceptions import ProcessingError

BOLD_FLAG = 1 << 4


@dataclass
class ExtractedBlock:
    text: str
    y0: float
    x0: float
    x1: float


@dataclass
class ExtractedPage:
    page_number: int  # 1-indexed
    text: str  # blocks joined in reading order, "\n\n" separated (paragraphs)


def _block_reading_order(blocks: list[ExtractedBlock], page_width: float) -> list[ExtractedBlock]:
    """Column-major reading order: left column top-to-bottom, then right column.
    Works well for standard single/two-column academic paper layouts; documents with
    unusual multi-column-spanning figures may occasionally misorder (documented limitation).
    """
    mid = page_width / 2
    left = sorted((b for b in blocks if (b.x0 + b.x1) / 2 < mid), key=lambda b: b.y0)
    right = sorted((b for b in blocks if (b.x0 + b.x1) / 2 >= mid), key=lambda b: b.y0)
    return left + right


def extract_pdf(file_path: str) -> list[ExtractedPage]:
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise ProcessingError(f"Failed to open PDF: {exc}") from exc

    pages: list[ExtractedPage] = []
    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            raw_blocks = page.get_text("dict")["blocks"]
            blocks: list[ExtractedBlock] = []
            for raw_block in raw_blocks:
                if "lines" not in raw_block:
                    continue
                line_texts = []
                for line in raw_block["lines"]:
                    spans_text = "".join(span["text"] for span in line["spans"])
                    if spans_text.strip():
                        line_texts.append(spans_text)
                block_text = "\n".join(line_texts).strip()
                if not block_text:
                    continue
                x0, y0, x1, _y1 = raw_block["bbox"]
                blocks.append(ExtractedBlock(text=block_text, y0=y0, x0=x0, x1=x1))

            ordered = _block_reading_order(blocks, page.rect.width)
            page_text = "\n\n".join(b.text for b in ordered)
            pages.append(ExtractedPage(page_number=page_index + 1, text=page_text))
    finally:
        doc.close()

    return pages
