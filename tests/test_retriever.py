from src.index import vector_store
from src.retrieval import retriever


def test_index_bootstrap_is_exported_from_vector_store_module():
    assert callable(vector_store.ensure_index_built)


def test_retrieve_builds_index_when_missing(monkeypatch):
    calls = []

    def fake_ensure_index_built():
        calls.append("built")
        return True

    monkeypatch.setattr(retriever, "ensure_index_built", fake_ensure_index_built)
    monkeypatch.setattr(retriever, "search_chunks", lambda question, top_k: [])
    monkeypatch.setattr(retriever, "keyword_search_chunks", lambda question, top_k: [])

    assert retriever.retrieve("What is in this document?", top_k=5) == []
    assert calls == ["built"]


def test_ensure_index_built_skips_if_collection_already_has_documents(monkeypatch):
    collection = type("Collection", (), {"count": lambda self: 7})()
    monkeypatch.setattr(vector_store, "get_collection", lambda: collection)

    assert vector_store.ensure_index_built() is True


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
