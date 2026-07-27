from app.core.constants import ConfidenceLevel


def compute_confidence(
    top_similarity: float,
    total_sentences: int,
    stripped_sentences: int,
    num_citations: int,
    multihop_used: bool,
) -> tuple[float, str]:
    """Code-derived confidence — never the LLM's self-reported confidence, which is
    known to be unreliable/overconfident. Weighted heuristic, documented as a starting
    point rather than an empirically tuned score:
        0.4 * top retrieval similarity
      + 0.4 * citation coverage (has >=1 verified citation)
      + 0.2 * (1 - fraction of answer sentences stripped by the grounding check)
      - 0.1 penalty if bounded multi-hop expansion was used (longer inference chain)
    """
    citation_coverage = 1.0 if num_citations > 0 else 0.0
    stripped_ratio = (stripped_sentences / total_sentences) if total_sentences else 1.0

    score = 0.4 * top_similarity + 0.4 * citation_coverage + 0.2 * (1 - stripped_ratio)
    if multihop_used:
        score -= 0.1
    score = max(0.0, min(1.0, score))

    if score > 0.75:
        level = ConfidenceLevel.HIGH.value
    elif score >= 0.45:
        level = ConfidenceLevel.MEDIUM.value
    else:
        level = ConfidenceLevel.LOW.value

    return round(score, 3), level
