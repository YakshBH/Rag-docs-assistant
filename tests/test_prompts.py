from src.generation.prompts import build_prompt


def test_prompt_includes_question_and_source():
    prompt = build_prompt(
        "What is the rule?",
        [{"text": "The rule is described here.", "source": "act.pdf", "page": 7}],
    )
    assert "What is the rule?" in prompt
    assert "act.pdf, page 7" in prompt
    assert "not contain enough evidence" in prompt
    assert "avoid repeating the same information" in prompt


def test_prompt_restricts_citations_to_retrieved_pages():
    prompt = build_prompt(
        "What is the rule?",
        [{"text": "The rule is described here.", "source": "act.pdf", "page": 7}],
    )
    assert "Use only these retrieved source-page citations" in prompt
    assert "[act.pdf, page 7]" in prompt
    assert "Never invent filenames" in prompt


def test_dpdp_outside_india_prompt_distinguishes_territorial_scope():
    prompt = build_prompt(
        "When does the DPDP Act apply to processing of personal data outside India?",
        [{"text": "Section 3 applies to processing outside India.", "source": "act.pdf", "page": 4}],
    )
    assert "territorial-applicability provision" in prompt
    assert "Do not substitute" in prompt
    assert "Section 16" in prompt
