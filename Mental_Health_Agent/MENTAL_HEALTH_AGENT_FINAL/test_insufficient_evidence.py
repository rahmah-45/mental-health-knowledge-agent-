"""Repeatable test that an in-scope, unsupported question is declined."""

from unittest.mock import patch

from mental_health_agent import run_agent


QUESTION = "What percentage of people in Pakistan have an anxiety disorder?"


class EmptyRetriever:
    """Deterministic retriever stand-in that has no supporting passages."""

    def invoke(self, query):
        return []


def test_insufficient_evidence_abstains_without_inventing():
    with (
        patch(
            "agent_nodes.get_components",
            return_value=(None, EmptyRetriever(), None, None),
        ),
        patch(
            "agent_nodes.filter_relevant_documents",
            return_value=([], []),
        ),
    ):
        result = run_agent(QUESTION)

    response = result["final_response"]
    assert result["question_classification"] == "IN_SCOPE"
    assert result["retry_count"] == 1
    assert result["retrieved_documents"] == []
    assert result["relevant_documents"] == []
    assert result["answer_supported"] is False
    assert "could not find sufficient evidence" in response.casefold()
    assert "[" not in response

    print("Question:", QUESTION)
    print("Classification:", result["question_classification"])
    print("Corrective retrieval attempts:", result["retry_count"])
    print("Final response:", response)
    print("Test result: PASS")


def main():
    test_insufficient_evidence_abstains_without_inventing()


if __name__ == "__main__":
    main()
