import re
import unicodedata
from collections import Counter

from app.services.extraction_service import ExtractedPage
from app.utils.text_utils import PAGE_NUMBER_LINE_RE, dehyphenate

WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")

REPEATED_LINE_MIN_PAGES = 4
REPEATED_LINE_FREQUENCY = 0.3


def _find_repeated_lines(pages: list[ExtractedPage]) -> set[str]:
    """Lines (e.g. running headers/footers) appearing on >=30% of pages, only checked
    when there are enough pages for the signal to be meaningful."""
    if len(pages) < REPEATED_LINE_MIN_PAGES:
        return set()

    counter: Counter = Counter()
    for page in pages:
        lines = {line.strip() for line in page.text.split("\n") if line.strip()}
        counter.update(lines)

    threshold = max(2, int(len(pages) * REPEATED_LINE_FREQUENCY))
    return {line for line, count in counter.items() if count >= threshold}


def clean_pages(pages: list[ExtractedPage]) -> list[ExtractedPage]:
    repeated_lines = _find_repeated_lines(pages)
    cleaned: list[ExtractedPage] = []

    for page in pages:
        text = unicodedata.normalize("NFKC", page.text)
        text = dehyphenate(text)

        kept_lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                kept_lines.append("")
                continue
            if stripped in repeated_lines:
                continue
            if PAGE_NUMBER_LINE_RE.match(stripped):
                continue
            kept_lines.append(WHITESPACE_RE.sub(" ", stripped))

        cleaned_text = "\n".join(kept_lines)
        cleaned_text = BLANK_LINES_RE.sub("\n\n", cleaned_text).strip()
        cleaned.append(ExtractedPage(page_number=page.page_number, text=cleaned_text))

    return cleaned
