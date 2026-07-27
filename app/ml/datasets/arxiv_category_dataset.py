import json
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
PAGE_SIZE = 100
REQUEST_DELAY_SECONDS = 3.0  # arXiv API etiquette: avoid hammering the endpoint
MAX_RETRIES = 5
MIN_VIABLE_SAMPLES = 20  # below this, don't cache — treat as a failed fetch worth retrying


def fetch_abstracts(arxiv_category: str, total: int) -> list[dict]:
    """Fetches title+abstract pairs for one arXiv category via the public arXiv API
    (no key required). Used as free, real-world labeled text for the 7 document
    categories — arXiv categories are a close but imperfect proxy (see README
    limitations, especially for "Cloud Computing" which has no exact arXiv tag)."""
    samples: list[dict] = []
    start = 0
    while len(samples) < total:
        page_size = min(PAGE_SIZE, total - len(samples))
        params = urllib.parse.urlencode(
            {
                "search_query": f"cat:{arxiv_category}",
                "start": start,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        url = f"{ARXIV_API_URL}?{params}"

        data = None
        for attempt in range(MAX_RETRIES):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = resp.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < MAX_RETRIES - 1:
                    backoff = REQUEST_DELAY_SECONDS * (2 ** (attempt + 1))
                    logger.warning("arXiv rate-limited (429) for %s; backing off %.0fs", arxiv_category, backoff)
                    time.sleep(backoff)
                    continue
                logger.warning("arXiv fetch failed for %s at start=%s: %s", arxiv_category, start, exc)
                break
            except urllib.error.URLError as exc:
                logger.warning("arXiv fetch failed for %s at start=%s: %s", arxiv_category, start, exc)
                break

        if data is None:
            break

        root = ET.fromstring(data)
        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            break

        for entry in entries:
            title_el = entry.find("atom:title", ATOM_NS)
            summary_el = entry.find("atom:summary", ATOM_NS)
            if title_el is None or summary_el is None:
                continue
            title = " ".join((title_el.text or "").split())
            abstract = " ".join((summary_el.text or "").split())
            if title and abstract:
                samples.append({"title": title, "abstract": abstract})

        start += page_size
        time.sleep(REQUEST_DELAY_SECONDS)

    return samples[:total]


def build_dataset(category_map: dict[str, str], samples_per_category: int, cache_dir: Path) -> dict[str, list[dict]]:
    """Returns {category_label: [{"title", "abstract"}, ...]}, caching each category's
    raw samples to disk so re-running training doesn't re-hit the arXiv API."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset: dict[str, list[dict]] = {}

    for label, arxiv_category in category_map.items():
        cache_file = cache_dir / f"{arxiv_category.replace('.', '_')}.json"
        if cache_file.exists():
            dataset[label] = json.loads(cache_file.read_text())
            logger.info("Loaded %d cached samples for %s (%s)", len(dataset[label]), label, arxiv_category)
            continue

        logger.info("Fetching %d samples for %s (arXiv category %s)...", samples_per_category, label, arxiv_category)
        samples = fetch_abstracts(arxiv_category, total=samples_per_category)
        if len(samples) < MIN_VIABLE_SAMPLES:
            # Don't cache a failed/rate-limited fetch — an empty cache file would
            # otherwise permanently block retrying this category on the next run.
            logger.warning(
                "Only fetched %d samples for %s (below viability threshold %d) — not caching, will retry next run",
                len(samples), label, MIN_VIABLE_SAMPLES,
            )
        else:
            cache_file.write_text(json.dumps(samples, indent=2))
        dataset[label] = samples
        logger.info("Fetched %d samples for %s", len(samples), label)

    return dataset
