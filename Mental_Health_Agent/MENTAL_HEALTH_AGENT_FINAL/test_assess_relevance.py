"""Regression tests for cautious stress-management relevance fallback."""

import assess_relevance
import torch
from evidence_retriever import filter_relevant_documents


QUESTION = "What are some evidence-based ways to manage stress?"


class BatchTokenizer:
    def __init__(self, responses):
        self.responses = responses
        self.padding_side = "right"
        self.pad_token = None
        self.eos_token = "<eos>"
        self.eos_token_id = 0
        self.prompt_batch = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[-1]["content"]

    def __call__(self, prompts, **kwargs):
        self.prompt_batch = prompts
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == assess_relevance.MAX_CONTEXT_LENGTH
        width = 3
        return {
            "input_ids": torch.ones((len(prompts), width), dtype=torch.long),
            "attention_mask": torch.ones((len(prompts), width), dtype=torch.long),
        }

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is True
        token_id = int(tokens[-1])
        return self.responses.get(token_id, "invalid output")


class BatchModel:
    def __init__(self, response_ids):
        self.response_ids = response_ids
        self.generate_calls = 0

    def generate(self, input_ids, attention_mask, **kwargs):
        self.generate_calls += 1
        assert kwargs["max_new_tokens"] == 12
        generated = torch.tensor(self.response_ids, dtype=torch.long).reshape(-1, 1)
        return torch.cat((input_ids, generated), dim=1)


class ShortBatchModel(BatchModel):
    def generate(self, input_ids, attention_mask, **kwargs):
        self.generate_calls += 1
        generated = torch.tensor([[self.response_ids[0]]], dtype=torch.long)
        return torch.cat((input_ids[:1], generated), dim=1)


def test_actionable_stress_coping_passage_is_not_discarded(monkeypatch):
    monkeypatch.setattr(
        assess_relevance,
        "assess_with_qwen",
        lambda question, passage, tokenizer, model: "IRRELEVANT",
    )
    passage = (
        "Learning to cope in a healthy way can help reduce your stress. "
        "Taking small steps in daily life to manage stress can have a big impact."
    )

    result = assess_relevance.assess_relevance(
        QUESTION, passage, tokenizer=object(), model=object()
    )

    assert result == "PARTIALLY_RELEVANT"


def test_stress_background_without_a_coping_method_stays_irrelevant(monkeypatch):
    monkeypatch.setattr(
        assess_relevance,
        "assess_with_qwen",
        lambda question, passage, tokenizer, model: "IRRELEVANT",
    )
    passage = "Long-term stress can worsen health problems and sleep quality."

    result = assess_relevance.assess_relevance(
        QUESTION, passage, tokenizer=object(), model=object()
    )

    assert result == "IRRELEVANT"


def test_stress_title_without_substantive_content_stays_irrelevant(monkeypatch):
    monkeypatch.setattr(
        assess_relevance,
        "assess_with_qwen",
        lambda question, passage, tokenizer, model: "IRRELEVANT",
    )
    passage = "Doing What Matters in Times of Stress: An Illustrated Guide"

    result = assess_relevance.assess_relevance(
        QUESTION, passage, tokenizer=object(), model=object()
    )

    assert result == "IRRELEVANT"


def test_job_stress_description_without_coping_guidance_stays_irrelevant(monkeypatch):
    monkeypatch.setattr(
        assess_relevance,
        "assess_with_qwen",
        lambda question, passage, tokenizer, model: "IRRELEVANT",
    )
    passage = (
        "4.2 Managing Stress: healthy work and life habits. A main source of "
        "stress for helpers is day-to-day job stress, particularly during a "
        "crisis. Long working hours and overwhelming responsibilities can "
        "affect helpers."
    )

    result = assess_relevance.assess_relevance(
        QUESTION, passage, tokenizer=object(), model=object()
    )

    assert result == "IRRELEVANT"


def test_batch_relevance_calls_qwen_once_and_keeps_one_result_per_passage(capsys):
    tokenizer = BatchTokenizer({
        100: "RELEVANT",
        101: "PARTIALLY_RELEVANT",
        102: "IRRELEVANT",
    })
    model = BatchModel([100, 101, 102])
    passages = ["direct evidence", "partial evidence", "unrelated content"]

    labels = assess_relevance.assess_relevance_batch(
        "question", passages, tokenizer=tokenizer, model=model
    )

    single_model = BatchModel([100])
    for passage in passages:
        assess_relevance.assess_with_qwen(
            "question", passage, BatchTokenizer({100: "RELEVANT"}), single_model
        )

    assert labels == ["RELEVANT", "PARTIALLY_RELEVANT", "IRRELEVANT"]
    assert model.generate_calls == 1
    assert single_model.generate_calls == len(passages)
    assert len(tokenizer.prompt_batch) == len(passages)
    assert tokenizer.padding_side == "right"
    assert tokenizer.pad_token is None
    output = capsys.readouterr().out
    assert "passages=3" in output
    assert "model.generate calls=1" in output


def test_invalid_batch_output_is_uncertain_and_not_accepted(capsys):
    tokenizer = BatchTokenizer({100: "RELEVANT"})
    model = BatchModel([100, 999])
    documents = [
        {"source_id": "CDC_1", "page": 1, "text": "first passage"},
        {"source_id": "CDC_2", "page": 2, "text": "second passage"},
    ]

    accepted, assessments = filter_relevant_documents(
        "question", documents, tokenizer, model
    )

    assert model.generate_calls == 1
    assert len(assessments) == 2
    assert [item["relevance"] for item in assessments] == ["RELEVANT", "UNCERTAIN"]
    assert accepted == [assessments[0]]
    output = capsys.readouterr().out
    assert "passages=2" in output
    assert "model.generate calls=1" in output


def test_missing_batch_rows_are_uncertain_not_accepted(capsys):
    tokenizer = BatchTokenizer({100: "RELEVANT"})
    model = ShortBatchModel([100])

    labels = assess_relevance.assess_relevance_batch(
        "question", ["one", "two", "three"], tokenizer, model
    )

    assert labels == ["RELEVANT", "UNCERTAIN", "UNCERTAIN"]
    assert model.generate_calls == 1
    assert "passages=3" in capsys.readouterr().out
