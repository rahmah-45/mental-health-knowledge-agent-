"""Regression tests for complete, source-grounded generated sentences."""

import answer_generator


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self, token_count=5):
        self.token_count = token_count

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return list(range(self.token_count)) if tokenize else "prompt"

    def encode(self, prompt, add_special_tokens=False):
        return list(range(self.token_count))


class FakeChain:
    def __init__(self, answer):
        self.answer = answer

    def invoke(self, values):
        return self.answer


def test_incomplete_generated_sentence_uses_complete_supported_source_text(monkeypatch):
    question = "What are some evidence-based ways to manage stress?"
    evidence = [{
        "source_id": "CDC_009",
        "title": "Managing Stress",
        "page": 1,
        "text": (
            "Taking small steps in daily life to manage stress can have a big impact. "
            "You can find and manage what triggers your stress and the right "
            "combination of healthy techniques that work for you."
        ),
    }]
    monkeypatch.setattr(
        answer_generator,
        "get_answer_chain",
        lambda tokenizer, model: FakeChain(
            "Taking small steps in daily life to manage stress can have a big impact.\n"
            "You can find and manage what triggers your stress and the right "
            "combination of healthy techniques that."
        ),
    )

    answer, gaps = answer_generator.generate_answer_with_gaps(
        question, evidence, FakeTokenizer(), object()
    )

    assert "Taking small steps in daily life to manage stress" in answer
    assert "healthy techniques that." not in answer
    assert "healthy techniques that work for you." in answer
    assert "[CDC_009, Page 1]" in answer
    assert gaps == []


def test_extractive_fallback_does_not_finish_a_truncated_source_line():
    answer = answer_generator.build_extractive_fallback(
        "What are some evidence-based ways to manage stress?",
        [{
            "source_id": "CDC_009",
            "page": 1,
            "text": (
                "Learning to cope in a healthy way can help reduce stress.\n"
                "You can find and manage what triggers your stress and the "
                "right combination of healthy techniques that"
            ),
        }],
    )
    assert "Learning to cope in a healthy way can help reduce stress." in answer
    assert "healthy techniques that." not in answer


def test_incomplete_gerund_lines_are_not_sentenced_as_answers():
    from answer_validator import is_complete_sentence

    assert not is_complete_sentence("Taking breaks from news and social media")
    assert not is_complete_sentence("Making time to unwind")
    assert not is_complete_sentence("Getting help now")
    assert is_complete_sentence(
        "Taking breaks from news and social media can reduce stress."
    )


def test_incomplete_draft_filtered_as_off_topic_still_creates_gap(monkeypatch):
    question = "What are some evidence-based ways to manage stress?"
    evidence = [{
        "source_id": "CDC_009",
        "title": "Managing Stress",
        "page": 1,
        "text": "Taking small steps in daily life can help manage stress.",
    }]
    monkeypatch.setattr(
        answer_generator,
        "get_answer_chain",
        lambda tokenizer, model: FakeChain(
            "Taking breaks from news and social media\nMaking time to unwind"
        ),
    )

    answer, gaps = answer_generator.generate_answer_with_gaps(
        question, evidence, FakeTokenizer(), object()
    )

    assert "Taking breaks from news and social media" not in answer
    assert gaps
    assert any("incomplete generated statement was omitted" in gap for gap in gaps)


def test_token_budget_uses_tokenizer_and_keeps_all_evidence_for_fallback(monkeypatch):
    question = (
        "How can adequate sleep support emotional well-being, and what effects "
        "can long-term stress have on mental health?"
    )
    evidence = [
        {
            "source_id": "CDC_008",
            "title": "About Sleep",
            "page": 1,
            "question_part_indices": [0],
            "text": "Adequate sleep supports emotional well-being and mental health.",
        },
        {
            "source_id": "CDC_009",
            "title": "Managing Stress",
            "page": 2,
            "question_part_indices": [1],
            "text": "Long-term stress can affect mental health.",
        },
    ]
    tokenizer = FakeTokenizer(
        token_count=(
            answer_generator.MAX_CONTEXT_LENGTH
            - answer_generator.MAX_NEW_TOKENS
            - answer_generator.CONTEXT_SPECIAL_TOKEN_RESERVE
            + 1
        )
    )
    monkeypatch.setattr(
        answer_generator,
        "get_answer_chain",
        lambda *_: (_ for _ in ()).throw(AssertionError("LLM should not receive an over-budget prompt")),
    )

    answer, gaps = answer_generator.generate_answer_with_gaps(
        question, evidence, tokenizer, object()
    )

    assert "[CDC_008, Page 1]" in answer
    assert "[CDC_009, Page 2]" in answer
    assert gaps == []


def test_duplicate_evidence_passages_are_removed_without_losing_distinct_pages():
    documents = [
        {"source_id": "CDC_008", "page": 1, "chunk_index": 1, "text": "Sleep supports health."},
        {"source_id": "CDC_008", "page": 1, "chunk_index": 4, "text": "Sleep supports health."},
        {"source_id": "CDC_008", "page": 2, "chunk_index": 5, "text": "Sleep supports health."},
    ]

    unique = answer_generator.deduplicate_evidence_documents(documents)

    assert len(unique) == 2
    assert [item["page"] for item in unique] == [1, 2]
