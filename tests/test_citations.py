from src.generation.citations import ground_citations, normalize_citations


def test_normalizes_existing_citation():
    assert normalize_citations("See [act.pdf, page 7].") == "See [act.pdf, page 7]."


def test_removes_source_label_from_citation():
    answer = "See [Source 2: act.pdf, page 7]."
    assert normalize_citations(answer) == "See [act.pdf, page 7]."


def test_normalizes_multiple_sources():
    answer = "[Source 1: act.pdf, page 7] and [policy.pdf, page 12]."
    expected = "[act.pdf, page 7] and [policy.pdf, page 12]."
    assert normalize_citations(answer) == expected


def test_preserves_citation_without_complete_metadata():
    answer = "See [Source 2: page 7] and [act.pdf]."
    assert normalize_citations(answer) == answer


def test_grounding_keeps_only_retrieved_citations():
    passages = [{"source": "act.pdf", "page": 7}]
    answer = "The rule applies. [act.pdf, page 7] Another claim. [act.pdf, page 9]"
    assert ground_citations(answer, passages) == "The rule applies. [act.pdf, page 7] Another claim."


def test_grounding_returns_insufficient_context_without_valid_citation():
    passages = [{"source": "act.pdf", "page": 7}]
    assert ground_citations("The answer is three.", passages) == (
        "I could not find this answer in the indexed documents."
    )


def test_grounding_preserves_explicit_insufficient_context():
    passages = [{"source": "act.pdf", "page": 7}]
    answer = "I could not find this answer in the indexed documents."
    assert ground_citations(answer, passages) == answer
