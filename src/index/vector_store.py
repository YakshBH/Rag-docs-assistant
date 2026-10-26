import math
import re

import chromadb
from src.config import CHROMA_DIR, COLLECTION_NAME
from src.index.embedder import embed_texts, embed_query

_client = None
_EXACT_PHRASES = (
    "data fiduciary",
    "outside india",
    "confabulation",
    "environmental impacts",
)


def get_collection():
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    global _client
    collection = get_collection()
    client = _client
    client.delete_collection(COLLECTION_NAME)
    return client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict], batch_size: int = 64) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset:offset + batch_size]
        embeddings = embed_texts([item["text"] for item in batch])
        collection.add(
            ids=[f"chunk-{item['chunk_id']}" for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[{"source": item["source"], "page": int(item["page"])} for item in batch],
            embeddings=embeddings,
        )
    return len(chunks)


def search_chunks(query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    matches = []
    for i, document in enumerate(result["documents"][0]):
        metadata = result["metadatas"][0][i]
        matches.append({
            "text": document,
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", "?"),
            "distance": result["distances"][0][i],
        })
    return matches
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", text.lower())


def _phrase_matches(query: str, document: str) -> list[str]:
    query_text = query.lower()
    document_text = document.lower()
    return [
        phrase
        for phrase in _EXACT_PHRASES
        if re.search(rf"\b{re.escape(phrase)}\b", query_text)
        and re.search(rf"\b{re.escape(phrase)}\b", document_text)
    ]


def keyword_search_chunks(query: str, top_k: int = 50) -> list[dict]:
    """Search the indexed chunks with BM25 rather than substring counting."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    result = collection.get(
        include=["documents", "metadatas"]
    )

    query_terms = _tokens(query)
    documents = [_tokens(document) for document in result["documents"]]
    document_frequency = {}
    for terms in documents:
        for term in set(terms):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    average_length = sum(map(len, documents)) / max(len(documents), 1)
    matches = []
    k1 = 1.5
    b = 0.75
    for document, metadata, terms in zip(
        result["documents"], result["metadatas"], documents
    ):
        term_counts = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            frequency = term_counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            normalization = 1 - b + b * len(terms) / max(average_length, 1)
            score += idf * frequency * (k1 + 1) / (frequency + k1 * normalization)
        phrase_matches = _phrase_matches(query, document)
        # Keep this additive and capped: phrase evidence refines BM25 rather
        # than allowing a heading-only match to replace explanatory overlap.
        score += min(1.5, 0.75 * len(phrase_matches))
        if score > 0:
            matches.append({
                "text": document,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", "?"),
                "distance": 1.0 / (1.0 + score),
                "keyword_score": score,
                "phrase_matches": phrase_matches,
            })

    matches.sort(key=lambda item: item["keyword_score"], reverse=True)
    return matches[:top_k]