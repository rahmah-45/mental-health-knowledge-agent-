"""Headless tests for evidence grouping in the Streamlit UI."""

from streamlit.testing.v1 import AppTest


def test_app_separates_supporting_and_rejected_passages():
    at = AppTest.from_file("app.py")
    at.session_state["conversation_history"] = [{
        "question": "What are some evidence-based ways to manage stress?",
        "result": {
            "final_response": "Take breaks to manage stress. [CDC_009, Page 1]",
            "answer_supported": True,
            "validation_notes": "All claims passed evidence checks.",
            "human_review_status": "NOT_REQUIRED",
            "retry_count": 0,
            "assessment_results": [
                {
                    "source_id": "CDC_009", "title": "Managing Stress", "page": 1,
                    "relevance": "RELEVANT", "text": "Take breaks to manage stress."
                },
                {
                    "source_id": "WHO_002", "title": "Anxiety", "page": 1,
                    "relevance": "IRRELEVANT", "text": "Unrelated anxiety information."
                },
            ],
            "relevant_documents": [
                {"source_id": "CDC_009", "title": "Managing Stress", "page": 1}
            ],
            "citation_results": [{
                "valid_citations": [{"source_id": "CDC_009", "page": "1"}]
            }],
            "question_part_coverage": [],
            "evidence_gaps": [],
            "requires_human_review": False,
        },
        "interrupt": None,
        "error": "",
    }]

    at.run(timeout=30)

    assert not at.exception
    assert at.success
    assert any("Supporting evidence" in item.value for item in at.markdown)
    rejected_sections = [
        item.label for item in at.get("expander")
        if "rejected or uncertain" in item.label.casefold()
    ]
    assert rejected_sections == ["Retrieved but rejected or uncertain passages (1)"]
