"""Read-only tests for the standalone evaluation result collector."""

import json
from pathlib import Path

from evaluation_collector import collect_agent_result


PROJECT_ROOT = Path(__file__).resolve().parent
SAVED_EVALUATION = (
    PROJECT_ROOT
    / "execution_logs"
    / "examination3_final_e2e_evaluation_2026-11-13.json"
)


def test_collect_saved_stress_scenario_without_running_agent():
    saved = json.loads(SAVED_EVALUATION.read_text(encoding="utf-8"))
    first_scenario = saved["scenarios"][0]

    parsed = collect_agent_result(first_scenario)

    assert parsed["question"] == "What are some evidence-based ways to manage stress?"
    assert parsed["question_classification"] == "IN_SCOPE"
    assert parsed["validation_status"] == "SUPPORTED"
    assert parsed["retry_count"] == 0
    assert parsed["human_review_status"] == "NOT_REQUIRED"
    assert parsed["retrieved_sources"][1]["source_id"] == "CDC_009"
    assert parsed["citations"] == [{"source_id": "CDC_009", "page": 1}]
    assert "healthy techniques that work for you" in parsed["final_answer"]


def test_collect_nested_interrupted_graph_result_with_optional_fields_missing():
    parsed = collect_agent_result(
        {
            "result": {
                "values": {
                    "question": "Sensitive case",
                    "requires_human_review": True,
                    "__interrupt__": [{"value": {"reason": "Review required"}}],
                }
            }
        }
    )

    assert parsed["question"] == "Sensitive case"
    assert parsed["execution_interrupted"] is True
    assert parsed["human_review_required"] is True
    assert parsed["human_review_status"] == "PENDING"
    assert parsed["retrieved_sources"] == []
    assert parsed["citations"] == []
    assert parsed["retry_count"] is None


def test_collect_empty_result_safely():
    parsed = collect_agent_result(None)

    assert parsed["final_answer"] == ""
    assert parsed["retrieved_sources"] == []
    assert parsed["citations"] == []
    assert parsed["human_review_status"] == "NOT_REQUIRED"


def test_validation_citations_are_not_reported_as_final_answer_citations():
    parsed = collect_agent_result(
        {
            "question": "Pakistan-specific prevalence?",
            "question_classification": "IN_SCOPE",
            "final_response": "I could not find sufficient evidence.",
            "answer_supported": False,
            "validation_notes": "Geographic scope mismatch: Pakistan-specific data was not found.",
            "citation_results": [
                {
                    "valid_citations": [
                        {"source_id": "WHO_002", "page": 1},
                    ]
                }
            ],
        }
    )

    assert parsed["citations"] == []
    assert parsed["validation_citations"] == [{"source_id": "WHO_002", "page": 1}]
    assert parsed["validation_status"] == "NOT_SUPPORTED"
