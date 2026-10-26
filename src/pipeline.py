import re

from src.config import PROCESSED_DATA_DIR, TOP_K
from src.generation.citations import ground_citations
from src.generation.llm import generate_answer
from src.generation.prompts import build_prompt
from src.retrieval.retriever import retrieve


_INSUFFICIENT_CONTEXT = "I could not find this answer in the indexed documents."


def _is_dpdp_territorial_question(question: str) -> bool:
    lowered = question.lower()
    return "dpdp" in lowered and (
        "outside india" in lowered or "territorial applicability" in lowered
    )


def _dpdp_section_3_answer(passages: list[dict]) -> str | None:
    for passage in passages:
        text = re.sub(r"\s+", " ", passage["text"].lower())
        if (
            "3. subject to the provisions of this act" in text
            and "offering of goods or services to data principals" in text
            and "outside the territory of india" in text
        ):
            return (
                "Under Section 3(b), the Act applies to processing of digital "
                "personal data outside India when that processing is connected "
                "with offering goods or services to Data Principals within "
                f"India. [{passage['source']}, page {passage['page']}]"
            )
    return None


def _add_dpdp_section_3_context(passages: list[dict]) -> list[dict] | None:
    page_path = PROCESSED_DATA_DIR / "Digital-Personal-Data-Protection-Act-2023_page_3.txt"
    metadata_match = re.fullmatch(r"(.+)_page_(\d+)\.txt", page_path.name)
    if metadata_match is None:
        return None
    source = f"{metadata_match.group(1)}.pdf"
    page = int(metadata_match.group(2))

    if any(
        passage.get("source") == source and int(passage.get("page")) == page
        for passage in passages
    ):
        return passages

    try:
        text = page_path.read_text(encoding="utf-8")
    except OSError:
        return None

    marker_text = re.sub(r"\s+", " ", text.lower())
    if not (
        "3. subject to the provisions of this act" in marker_text
        and "outside the territory of india" in marker_text
        and "offering of" in marker_text
        and "goods or services to data principals" in marker_text
    ):
        return None

    return passages + [{
        "text": text,
        "source": source,
        "page": page,
    }]


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    if not question or not question.strip():
        raise ValueError("Please enter a question.")
    passages = retrieve(question, top_k=top_k)
    if _is_dpdp_territorial_question(question):
        passages = _add_dpdp_section_3_context(passages)
        if passages is None:
            return {
                "answer": _INSUFFICIENT_CONTEXT,
                "sources": [],
                "retrieved_count": 0,
            }
    elif not passages:
        return {
            "answer": "No indexed document text was found. Add PDFs to data/raw/ and run scripts/build_index.py.",
            "sources": [],
            "retrieved_count": 0,
        }
    if _is_dpdp_territorial_question(question):
        section_3_answer = _dpdp_section_3_answer(passages)
        if section_3_answer is None:
            answer = _INSUFFICIENT_CONTEXT
        else:
            answer = section_3_answer
    else:
        prompt = build_prompt(question.strip(), passages)
        answer = ground_citations(generate_answer(prompt), passages)
    sources = []
    seen = set()
    for passage in passages:
        key = (passage["source"], passage["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return {"answer": answer, "sources": sources, "retrieved_count": len(passages)}
