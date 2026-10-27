from src.performance import timed_stage
from src.config import TOP_K
from src.index.vector_store import (
    ensure_index_built,
    search_chunks,
    keyword_search_chunks,
)


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    question = question.strip()

    if not question:
        return []

    with timed_stage("index bootstrap"):
        ensure_index_built()
    candidate_k = max(top_k * 10, 50)

    with timed_stage("semantic retrieval", top_k=candidate_k):
        semantic_results = search_chunks(question, top_k=candidate_k)

    with timed_stage("BM25 retrieval", top_k=candidate_k):
        keyword_results = keyword_search_chunks(question, top_k=candidate_k)

    with timed_stage("hybrid retrieval merge", semantic=len(semantic_results), bm25=len(keyword_results)):
        fused = {}
        phrase_evidence = {
            (item["source"], int(item["page"])): item
            for item in keyword_results
            if item.get("phrase_matches") and item.get("keyword_score", 0.0) >= 5.0
        }

        def add_results(results):
            seen_pages = set()
            page_rank = 0
            for item in results:
                # Evaluation and citations are page-based. Fuse at that same unit
                # so multiple chunks from one page cannot crowd out other pages.
                key = (item["source"], int(item["page"]))
                if key in seen_pages:
                    continue
                seen_pages.add(key)
                page_rank += 1

                if key not in fused:
                    fused[key] = item.copy()
                    fused[key]["rrf_score"] = 0.0

                fused[key]["rrf_score"] += 1.0 / (60 + page_rank)

        add_results(semantic_results)
        add_results(keyword_results)

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                item["rrf_score"]
                + (0.0015 if (item["source"], int(item["page"])) in phrase_evidence else 0.0),
                -float(item.get("distance", 1.0)),
            ),
            reverse=True,
        )

    return ranked[:top_k]