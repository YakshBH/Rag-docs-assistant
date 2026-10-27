from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL
from src.performance import timed_stage

_model = None


def get_model():
    global _model
    if _model is None:
        with timed_stage("embedding model load", model=EMBEDDING_MODEL):
            _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    with timed_stage("embedding generation", count=len(texts)):
        vectors = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
