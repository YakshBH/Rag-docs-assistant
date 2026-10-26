from src.index.vector_store import _phrase_matches


def test_matches_requested_exact_phrases_case_insensitively():
    assert _phrase_matches(
        "Who is a Data Fiduciary outside India?",
        "A Data Fiduciary processes data outside India.",
    ) == ["data fiduciary", "outside india"]


def test_does_not_match_unrelated_query_or_document():
    assert _phrase_matches("What is prompt injection?", "Data Fiduciary") == []
    assert _phrase_matches("What is Confabulation?", "The answer discusses hallucinations.") == []


def test_existing_non_phrase_query_has_no_phrase_bonus():
    assert _phrase_matches(
        "What is prompt injection in an LLM application?",
        "Prompt injection changes model behavior.",
    ) == []
