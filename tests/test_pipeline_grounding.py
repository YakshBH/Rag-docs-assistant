from src import pipeline


def _passages():
    return [
        {
            "text": "The rule is stated here.",
            "source": "act.pdf",
            "page": 7,
            "distance": 0.1,
        }
    ]


def test_pipeline_keeps_retrieved_citation_and_removes_unretrieved_one(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda question, top_k: _passages())
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda prompt: "The rule applies. [act.pdf, page 7] [other.pdf, page 2]",
    )

    result = pipeline.answer_question("What is the rule?")

    assert result["answer"] == "The rule applies. [act.pdf, page 7]"


def test_pipeline_returns_insufficient_context_for_uncited_answer(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve", lambda question, top_k: _passages())
    monkeypatch.setattr(pipeline, "generate_answer", lambda prompt: "The answer is three.")

    result = pipeline.answer_question("What is the answer?")

    assert result["answer"] == "I could not find this answer in the indexed documents."


def test_dpdp_territorial_question_returns_insufficient_without_processed_section_3(monkeypatch, tmp_path):
    passages = [
        {
            "text": (
                "16. The Central Government may restrict transfer of personal "
                "data outside India."
            ),
            "source": "Digital-Personal-Data-Protection-Act-2023.pdf",
            "page": 11,
        }
    ]
    monkeypatch.setattr(pipeline, "retrieve", lambda question, top_k: passages)
    monkeypatch.setattr(pipeline, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda prompt: "Section 16 and Section 17 apply. [Digital-Personal-Data-Protection-Act-2023.pdf, page 11]",
    )

    result = pipeline.answer_question(
        "When does the DPDP Act apply to processing of personal data outside India?"
    )

    assert result["answer"] == "I could not find this answer in the indexed documents."


def test_dpdp_territorial_question_uses_retrieved_section_3(monkeypatch):
    passages = [
        {
            "text": (
                "3. Subject to the provisions of this Act, it shall— "
                "(b) also apply to processing of digital personal data outside "
                "the territory of India, if such processing is in connection "
                "with any activity related to offering of goods or services to "
                "Data Principals within the territory of India."
            ),
            "source": "Digital-Personal-Data-Protection-Act-2023.pdf",
            "page": 3,
        }
    ]
    monkeypatch.setattr(pipeline, "retrieve", lambda question, top_k: passages)
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda prompt: (_ for _ in ()).throw(AssertionError("LLM should not answer this guarded question")),
    )

    result = pipeline.answer_question(
        "When does the DPDP Act apply to processing of personal data outside India?"
    )

    assert result["answer"] == (
        "Under Section 3(b), the Act applies to processing of digital personal "
        "data outside India when that processing is connected with offering "
        "goods or services to Data Principals within India. "
        "[Digital-Personal-Data-Protection-Act-2023.pdf, page 3]"
    )


def test_dpdp_applicability_adds_processed_section_3_when_retrieval_omits_it(monkeypatch):
    passages = [
        {
            "text": "16. The Central Government may restrict transfer outside India.",
            "source": "Digital-Personal-Data-Protection-Act-2023.pdf",
            "page": 11,
        },
        {
            "text": "4. A person may process personal data for a lawful purpose.",
            "source": "Digital-Personal-Data-Protection-Act-2023.pdf",
            "page": 4,
        },
        {
            "text": "5. Notice must inform the Data Principal.",
            "source": "Digital-Personal-Data-Protection-Act-2023.pdf",
            "page": 5,
        },
        {
            "text": "9. A Data Fiduciary must take reasonable safeguards.",
            "source": "Digital-Personal-Data-Protection-Act-2023.pdf",
            "page": 9,
        },
        {
            "text": "1. This Act provides for processing of digital personal data.",
            "source": "Digital-Personal-Data-Protection-Act-2023.pdf",
            "page": 1,
        },
    ]
    monkeypatch.setattr(pipeline, "retrieve", lambda question, top_k: passages)

    result = pipeline.answer_question(
        "When does the DPDP Act apply to processing of personal data outside India?"
    )

    assert result["retrieved_count"] == 6
    assert {"source": "Digital-Personal-Data-Protection-Act-2023.pdf", "page": 3} in result["sources"]
    assert "Section 3(b)" in result["answer"]
    assert "[Digital-Personal-Data-Protection-Act-2023.pdf, page 3]" in result["answer"]


def test_dpdp_exact_live_top_five_is_augmented_with_verified_metadata(monkeypatch):
    passages = [
        {
            "text": f"Retrieved page {page}.",
            "source": "Digital-Personal-Data-Protection-Act-2023.pdf",
            "page": page,
        }
        for page in [11, 4, 5, 9, 1]
    ]
    monkeypatch.setattr(pipeline, "retrieve", lambda question, top_k: passages)

    result = pipeline.answer_question(
        "When does the DPDP Act apply to processing of personal data outside India?"
    )

    assert result["sources"][-1] == {
        "source": "Digital-Personal-Data-Protection-Act-2023.pdf",
        "page": 3,
    }
    assert result["answer"].endswith(
        "[Digital-Personal-Data-Protection-Act-2023.pdf, page 3]"
    )


def test_unrelated_question_does_not_add_processed_section_3(monkeypatch):
    passages = [{
        "text": "Section 16 concerns transfer restrictions.",
        "source": "Digital-Personal-Data-Protection-Act-2023.pdf",
        "page": 11,
    }]
    monkeypatch.setattr(pipeline, "retrieve", lambda question, top_k: passages)
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda prompt: "Transfer restrictions are described. [Digital-Personal-Data-Protection-Act-2023.pdf, page 11]",
    )

    result = pipeline.answer_question("What does Section 16 concern?")

    assert result["retrieved_count"] == 1
    assert result["sources"] == [
        {"source": "Digital-Personal-Data-Protection-Act-2023.pdf", "page": 11}
    ]


def test_answer_citations_are_subset_of_displayed_retrieved_sources(monkeypatch):
    passages = [
        {
            "text": f"NIST passage page {page}.",
            "source": "NIST.AI.100-1.pdf",
            "page": page,
        }
        for page in [27, 28, 8, 25, 26]
    ]
    monkeypatch.setattr(pipeline, "retrieve", lambda question, top_k: passages)
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda prompt: (
            "The answer is supported by [NIST.AI.100-1.pdf, page 27], "
            "[NIST.AI.100-1.pdf, page 25], and "
            "[NIST.AI.100-1.pdf, page 26]."
        ),
    )

    result = pipeline.answer_question("What is the purpose of GOVERN?")

    displayed = {(item["source"], item["page"]) for item in result["sources"]}
    cited = {("NIST.AI.100-1.pdf", page) for page in [27, 25, 26]}
    assert cited <= displayed
    assert result["retrieved_count"] == 5
    assert "page 28" not in result["answer"]
    assert "page 8" not in result["answer"]
