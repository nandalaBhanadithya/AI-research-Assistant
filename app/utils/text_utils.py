import re

# Matches a hyphen at the end of a visual line, wrapping into a lowercase continuation
# on the next line — the standard PDF line-wrap artifact, e.g. "infor-\nmation".
DEHYPHEN_RE = re.compile(r"-\n(?=[a-z])")

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

PAGE_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")

HEADING_RE = re.compile(
    r"^\s*(?:[0-9]+\.?|[IVXLC]+\.)?\s*"
    r"(abstract|introduction|related works?|background|methodology|methods?|approach|"
    r"proposed method|experiments?(?:\s+and\s+results?)?|evaluation|results?(?:\s+and\s+discussion)?|"
    r"discussion(?:\s+and\s+conclusions?)?|conclusions?|limitations?|future work|"
    r"acknowledge?ments?|references|appendix[a-z]*)\s*$",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    """Cheap chars/4 heuristic — avoids pulling in a tokenizer that doesn't match the
    actual Ollama/Groq model's vocabulary anyway."""
    return max(1, len(text) // 4)


def dehyphenate(text: str) -> str:
    return DEHYPHEN_RE.sub("", text)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def is_heading(paragraph: str) -> bool:
    if len(paragraph) > 80:
        return False
    return bool(HEADING_RE.match(paragraph.strip()))
