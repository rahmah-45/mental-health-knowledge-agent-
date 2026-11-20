
from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    question: str
    question_parts: List[str]
    question_part_coverage: List[Dict[str, Any]]
    question_classification: str
    search_query: str
    corrected_query: str
    retry_count: int
    corrective_question_part_indices: List[int]
    corrective_retrieved_documents: List[Dict[str, Any]]

    retrieved_documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    assessment_results: List[Dict[str, Any]]
    evidence_gaps: List[str]

    evidence_sufficient: bool

    answer: str
    answer_supported: bool
    validation_notes: str

    # Validation details
    claim_results: List[Dict[str, Any]]
    citation_results: List[Dict[str, Any]]
    unsupported_claims: List[str]

    requires_human_review: bool
    human_review_reason: str
    human_review_status: str
    final_response: str
