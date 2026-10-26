import pytest
from src.ingest.chunker import chunk_pages


def test_chunker_preserves_metadata():
    pages = [{"text": "one two three four five six", "source": "sample.pdf", "page": 2}]
    chunks = chunk_pages(pages, chunk_size=4, overlap=1)
    assert len(chunks) == 2
    assert chunks[0]["source"] == "sample.pdf"
    assert chunks[0]["page"] == 2
    assert chunks[0]["text"] == "one two three four"


def test_chunker_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=4, overlap=4)
