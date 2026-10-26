from src.retrieval import retriever


def test_phrase_evidence_requires_meaningful_keyword_overlap(monkeypatch):
    monkeypatch.setattr(
        retriever,
        "search_chunks",
        lambda question, top_k: [
            {"source": "doc.pdf", "page": 3, "text": "semantic", "distance": 0.2}
        ],
    )
    monkeypatch.setattr(
        retriever,
        "keyword_search_chunks",
        lambda question, top_k: [
            {
                "source": "doc.pdf",
                "page": 2,
                "text": "Data Fiduciary means a person who determines purposes.",
                "distance": 0.1,
                "keyword_score": 6.0,
                "phrase_matches": ["data fiduciary"],
            },
            {
                "source": "doc.pdf",
                "page": 1,
                "text": "Data Fiduciary",
                "distance": 0.1,
                "keyword_score": 2.0,
                "phrase_matches": ["data fiduciary"],
            },
        ],
    )

    results = retriever.retrieve("Who is a Data Fiduciary?", top_k=2)

    assert results[0]["page"] == 2
    assert results[1]["page"] == 3
