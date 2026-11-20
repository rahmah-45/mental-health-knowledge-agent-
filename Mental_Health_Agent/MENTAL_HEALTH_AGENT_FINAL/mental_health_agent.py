
from typing import Literal
import argparse
from uuid import uuid4

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent_state import AgentState

from agent_nodes import (
    triage_node,
    human_review_node,
    out_of_scope_node,
    retrieve_node,
    assess_evidence_node,
    corrective_retrieval_node,
    generate_answer_node,
    validate_answer_node,
    abstain_node,
    finalize_answer_node
)


def check_evidence(state: AgentState) -> Literal["generate", "correct", "abstain"]:
    """Decide whether to generate, retry retrieval, or abstain."""

    if state.get("evidence_sufficient", False):
        return "generate"

    if state.get("retry_count", 0) < 1:
        return "correct"

    return "abstain"


def route_question(state: AgentState) -> Literal["retrieve", "human_review", "out_of_scope"]:
    classification = state.get("question_classification", "OUT_OF_SCOPE")
    if classification in {"CRISIS", "HUMAN_REVIEW"}:
        return "human_review"
    if classification == "OUT_OF_SCOPE":
        return "out_of_scope"
    return "retrieve"



def check_validation(state) -> Literal["finalize", "correct", "reject", "abstain"]:
    scope_gaps = state.get("evidence_gaps", [])
    if any(
        gap.startswith((
            "Geographic scope mismatch:",
            "Population scope mismatch:",
        ))
        for gap in scope_gaps
    ):
        print("\n[Decision] Retrieved evidence does not match the requested scope. Abstaining...")
        return "abstain"

    if state.get("answer", "").strip().lower().startswith(
        "i could not find sufficient supporting evidence"
    ):
        return "abstain"

    if state.get("answer_supported", False):
        return "finalize"

    if state.get("retry_count", 0) < 1:
        print("\n[Decision] Answer validation failed. Retrying retrieval...")
        return "correct"

    return "reject"

def reject_answer_node(state):
    """Reject answers that fail evidence validation."""

    print("\n[Node] Unsupported claims detected.")

    evidence_gaps = state.get("evidence_gaps", [])
    if not evidence_gaps:
        question = (state.get("question") or "").strip()
        gap = (
            "The retrieved evidence did not establish a fully supported "
            f"answer to: {question}"
            if question
            else "The retrieved evidence did not establish a fully supported answer."
        )
        evidence_gaps = [gap]

    return {
        "evidence_gaps": evidence_gaps,
        "final_response": (
            "I could not verify that all claims in the generated "
            "answer are supported by the available evidence. "
            "Therefore, I cannot provide that answer confidently."
        )
    }


def build_agent(checkpointer=None):
    workflow = StateGraph(AgentState)

    workflow.add_node("triage", triage_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("out_of_scope", out_of_scope_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("assess", assess_evidence_node)
    workflow.add_node("correct", corrective_retrieval_node)
    workflow.add_node("generate", generate_answer_node)
    workflow.add_node("validate", validate_answer_node)
    workflow.add_node("abstain", abstain_node)
    workflow.add_node("finalize", finalize_answer_node)
    workflow.add_node("reject", reject_answer_node)

    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges(
        "triage",
        route_question,
        {
            "retrieve": "retrieve",
            "human_review": "human_review",
            "out_of_scope": "out_of_scope",
        },
    )
    workflow.add_edge("human_review", END)
    workflow.add_edge("out_of_scope", END)
    workflow.add_edge("retrieve", "assess")

    workflow.add_conditional_edges(
        "assess",
        check_evidence,
        {
            "generate": "generate",
            "correct": "correct",
            "abstain": "abstain"
        }
    )

    workflow.add_edge("correct", "assess")

    workflow.add_edge("generate", "validate")


    workflow.add_conditional_edges(
    "validate",
    check_validation,
    {
        "finalize": "finalize",
        "correct": "correct",
        "reject": "reject",
        "abstain": "abstain",
    }
)


    workflow.add_edge("finalize", END)
    workflow.add_edge("abstain", END)
    workflow.add_edge("reject", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())


def run_agent(question, review_callback=None):
    agent = build_agent()
    config = {"configurable": {"thread_id": str(uuid4())}}

    initial_state = {
        "question": question,
        "question_parts": [],
        "question_part_coverage": [],
        "question_classification": "",
        "human_review_reason": "",
        "human_review_status": "NOT_REQUIRED",
        "search_query": question,
        "corrected_query": "",
        "retry_count": 0,
        "corrective_question_part_indices": [],
        "corrective_retrieved_documents": [],
        "retrieved_documents": [],
        "relevant_documents": [],
        "assessment_results": [],
        "evidence_gaps": [],
        "evidence_sufficient": False,
        "answer": "",
        "answer_supported": False,
        "validation_notes": "",
        "claim_results": [],
        "citation_results": [],
        "unsupported_claims": [],
        "requires_human_review": False,
        "final_response": ""
    }

    result = agent.invoke(initial_state, config=config)

    interruptions = result.get("__interrupt__", ())
    if interruptions:
        review_payload = interruptions[0].value
        result["requires_human_review"] = True
        result["human_review_reason"] = review_payload["reason"]
        result["human_review_status"] = "PENDING"
        if review_callback is not None:
            decision = review_callback(review_payload)
            result = agent.invoke(
                Command(resume={"reviewed": bool(decision)}),
                config=config,
            )

    return result


def print_retrieved_evidence(passages):
    """Print unique assessed passages with their source provenance."""
    print("\nRetrieved evidence passages:")

    unique_passages = []
    seen = set()
    for passage in passages:
        text = passage.get("text", "") or ""
        dedupe_key = (
            str(passage.get("source_id", "")).casefold(),
            str(passage.get("page", "")),
            " ".join(text.split()).casefold(),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        unique_passages.append((passage, text))

    if not unique_passages:
        print("No passages were retrieved for assessment.")
        return

    for index, (passage, text) in enumerate(unique_passages, start=1):
        print(f"\nPassage {index}")
        print("Source title:", passage.get("title", "Unknown"))
        print("Source ID:", passage.get("source_id", "Unknown"))
        print("Page:", passage.get("page", "Unknown"))
        print("Relevance assessment:", passage.get("relevance", "NOT ASSESSED"))
        print("Passage text:")
        print(text if text else "(No passage text available.)")


def print_evidence_gaps(result):
    print("\nEvidence gaps:")
    evidence_gaps = result.get("evidence_gaps", [])
    if evidence_gaps:
        for gap in evidence_gaps:
            print("-", gap)
    elif result.get("requires_human_review") or result.get("question_classification") == "OUT_OF_SCOPE":
        print("Evidence gaps were not assessed for this route.")
    elif result.get("answer_supported"):
        print("No unsupported claims were identified by validation.")
    else:
        print(
            "The available evidence was insufficient; the specific "
            "missing information could not be determined reliably."
        )


def request_human_review(review_payload):
    print("\n[HUMAN REVIEW REQUIRED]")
    print("Reason:", review_payload["reason"])
    print("Question:", review_payload["question"])
    print("Suggested safe response:", review_payload["suggested_action"])
    decision = input("Has a qualified reviewer reviewed this case? (yes/no): ")
    return decision.strip().casefold() in {"yes", "y"}


def main():
    print("=" * 60)
    print("MENTAL HEALTH AGENT - CORRECTIVE RAG TEST")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Mental health RAG education agent")
    parser.add_argument("--question", help="Question to ask the knowledge agent")
    args = parser.parse_args()
    question = args.question or input("Ask a mental health education question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    print("\nUser question:", question)

    result = run_agent(question, review_callback=request_human_review)

    print("\n" + "=" * 60)
    print("FINAL AGENT RESPONSE")
    print("=" * 60)

    print(result.get("final_response", "No response generated."))

    print_retrieved_evidence(result.get("assessment_results", []))
    print_evidence_gaps(result)

    if result.get("requires_human_review"):
        print("\nHuman review required:", result.get("human_review_reason"))
        print("Human review status:", result.get("human_review_status"))
    elif result.get("relevant_documents"):
        print("\nRetrieved references:")
        seen = set()
        for document in result["relevant_documents"]:
            reference = (document["source_id"], document["title"], document["page"])
            if reference not in seen:
                seen.add(reference)
                print(f"- {reference[0]} — {reference[1]}, page {reference[2]}")

    print("\nValidation supported:", result.get("answer_supported"))
    print("Validation notes:", result.get("validation_notes"))
    print("Retrieval assessment:", len(result.get("assessment_results", [])), "passages assessed")
    print("Corrective retrieval attempts:", result.get("retry_count"))

    print("\n" + "=" * 60)
    print("AGENT WORKFLOW COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
