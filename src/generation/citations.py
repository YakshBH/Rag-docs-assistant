import re


_CITATION_PATTERN = re.compile(
    r"\[(?:(?:Source\s+\d+):\s*)?([^,\[\]]+),\s*page\s+(\d+)\]",
    re.IGNORECASE,
)
_INSUFFICIENT_CONTEXT = "I could not find this answer in the indexed documents."


def normalize_citations(answer: str) -> str:
    """Normalize valid citations without changing their source-page references."""
    if not answer:
        return answer

    def replace(match: re.Match[str]) -> str:
        source = match.group(1).strip()
        page = match.group(2)
        if not source:
            return match.group(0)
        return f"[{source}, page {page}]"

    return _CITATION_PATTERN.sub(replace, answer)


def ground_citations(answer: str, passages: list[dict]) -> str:
    """Keep only citations that refer to the retrieved source pages."""
    normalized = normalize_citations(answer)
    allowed = {
        (str(passage["source"]), int(passage["page"]))
        for passage in passages
    }
    citation_found = False

    def replace(match: re.Match[str]) -> str:
        nonlocal citation_found
        source = match.group(1).strip()
        page = int(match.group(2))
        if (source, page) not in allowed:
            return ""
        citation_found = True
        return f"[{source}, page {page}]"

    grounded = _CITATION_PATTERN.sub(replace, normalized).strip()
    if citation_found or _INSUFFICIENT_CONTEXT.lower() in grounded.lower():
        return grounded
    return _INSUFFICIENT_CONTEXT
