
import re
from functools import lru_cache

from evidence_retriever import (
    load_retrieval_components,
    retrieve_documents,
    filter_relevant_documents
)

from answer_generator import (
    load_answer_model,
    generate_answer_with_gaps,
    evidence_document_key,
)

from answer_validator import (
    validate_answer,
    split_question_parts,
    extract_keywords,
)
from scope_router import classify_question
from langgraph.types import interrupt


CORRECTIVE_QUERY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "of",
    "on", "or", "the", "their", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with",
}


def build_corrective_query(question_part, evidence_gaps=None, unsupported_claims=None):
    """Make a distinct retrieval query from the unsupported part and its gap."""
    part_terms = re.findall(r"[a-zA-Z0-9]+", question_part.casefold())
    part_terms = [
        term for term in part_terms
        if term not in CORRECTIVE_QUERY_STOP_WORDS and len(term) > 1
    ]

    focus_terms = []
    for gap in evidence_gaps or []:
        gap_text = str(gap)
        marker = "requested part:"
        if marker in gap_text.casefold():
            gap_text = gap_text[gap_text.casefold().index(marker) + len(marker):]
        gap_terms = re.findall(r"[a-zA-Z0-9]+", gap_text.casefold())
        gap_terms = [
            term for term in gap_terms
            if term not in CORRECTIVE_QUERY_STOP_WORDS and len(term) > 1
        ]
        if set(gap_terms).intersection(part_terms):
            focus_terms.extend(gap_terms)

    claim_terms = []
    for claim in unsupported_claims or []:
        clean_claim = re.sub(r"\[[^\]]*\]", "", str(claim))
        terms = re.findall(r"[a-zA-Z0-9]+", clean_claim.casefold())
        terms = [
            term for term in terms
            if term not in CORRECTIVE_QUERY_STOP_WORDS and len(term) > 1
        ]
        if set(terms).intersection(part_terms):
            claim_terms.extend(terms)

    ordered_terms = []
    for term in [*focus_terms, *claim_terms, *part_terms]:
        if term not in ordered_terms:
            ordered_terms.append(term)

    # Domain and evidence terms make the corrective query distinct while
    # leaving all factual selection to retrieval and the existing validator.
    query_terms = ["mental", "health", "evidence", *ordered_terms]
    corrected = " ".join(dict.fromkeys(query_terms)).strip()
    if corrected.casefold() == question_part.casefold().strip():
        corrected = f"{corrected} information"
    return corrected


@lru_cache(maxsize=1)
def get_components():

    return load_retrieval_components()


@lru_cache(maxsize=1)
def get_answer_components():

    return load_answer_model()


def retrieve_node(state):

    print("\n[Node 1] Retrieving evidence...")

    embedding_model, collection, _, _ = get_components()

    search_query = (
        state.get("search_query")
        or state["question"]
    )

    question_parts = split_question_parts(state["question"])
    documents = []
    seen = set()
    for part_index, part in enumerate(question_parts):
        query = search_query if len(question_parts) == 1 else part
        for document in retrieve_documents(query, embedding_model, collection):
            key = (str(document.get("source_id", "")).casefold(),
                   str(document.get("page", "")),
                   str(document.get("chunk_index", "")),
                   " ".join(document.get("text", "").split()).casefold())
            if key not in seen:
                seen.add(key)
                document["question_part_indices"] = [part_index]
                documents.append(document)
            else:
                for existing in documents:
                    if (str(existing.get("source_id", "")).casefold(),
                        str(existing.get("page", "")),
                        str(existing.get("chunk_index", "")),
                        " ".join(existing.get("text", "").split()).casefold()) == key:
                        existing["question_part_indices"].append(part_index)
                        break

    print("Search query:", search_query)
    print("Retrieved documents:", len(documents))

    return {
        "search_query": search_query,
        "question_parts": question_parts,
        "retrieved_documents": documents
    }


def order_evidence_for_generation(question_parts, documents):
    """Prioritize the strongest unique evidence for each question part."""
    ordered = []
    seen = set()
    relevance_order = {
        "RELEVANT": 0,
        "PARTIALLY_RELEVANT": 1,
        "UNCERTAIN": 2,
        "IRRELEVANT": 3,
    }

    for part_index in range(len(question_parts)):
        group = []
        for document in documents:
            indices = document.get("question_part_indices")
            if indices is None and len(question_parts) == 1:
                indices = [0]
            if part_index in (indices or []):
                labels = document.get("relevance_by_part") or {}
                label = labels.get(part_index, labels.get(str(part_index), document.get("relevance", "UNCERTAIN")))
                try:
                    rank = float(document.get("rank", float("inf")))
                except (TypeError, ValueError):
                    rank = float("inf")
                group.append((relevance_order.get(label, 2), rank, document))

        for _, _, document in sorted(group, key=lambda item: (item[0], item[1])):
            key = evidence_document_key(document)
            if key not in seen:
                seen.add(key)
                ordered.append(document)

    remaining = sorted(
        documents,
        key=lambda document: (
            min(
                (relevance_order.get(label, 2)
                 for label in (document.get("relevance_by_part") or {}).values()),
                default=relevance_order.get(document.get("relevance", "UNCERTAIN"), 2),
            ),
            document.get("rank", float("inf")),
        ),
    )
    for document in remaining:
        key = evidence_document_key(document)
        if key not in seen:
            seen.add(key)
            ordered.append(document)
    return ordered


def triage_node(state):
    """Route high-risk requests before they reach the generation model."""
    classification = classify_question(state.get("question", ""))
    print("\n[Node 0] Question classification:", classification)
    return {"question_classification": classification}


def human_review_node(state):
    """Pause the graph until a human reviewer submits a decision."""
    crisis = state.get("question_classification") == "CRISIS"
    if crisis:
        message = (
            "This request needs prompt review by a qualified mental health "
            "professional. If you may be in immediate danger, contact your "
            "local emergency service now or reach out to someone you trust."
        )
        reason = "Potential crisis or a request for an individualized clinical decision."
    else:
        message = (
            "This assistant provides general education only. Please discuss "
            "personal diagnosis or medication decisions with a qualified "
            "mental health professional."
        )
        reason = "Individualized diagnosis, medication, or treatment request."

    review_decision = interrupt({
        "type": "mental_health_human_review",
        "question": state.get("question", ""),
        "reason": reason,
        "suggested_action": message,
    })

    reviewed = bool(review_decision.get("reviewed", False))
    return {
        "requires_human_review": True,
        "human_review_reason": reason,
        "human_review_status": "REVIEWED" if reviewed else "PENDING",
        "final_response": message,
    }


def out_of_scope_node(state):
    return {
        "requires_human_review": False,
        "human_review_reason": "",
        "final_response": (
            "This question is outside the mental health education topics "
            "covered by the available knowledge base."
        ),
    }


def assess_evidence_node(state):

    print("\n[Node 2] Assessing evidence relevance...")

    _, _, tokenizer, qwen_model = get_components()

    search_query = (
        state.get("search_query")
        or state["question"]
    )

    parts = state.get("question_parts") or split_question_parts(state["question"])
    retrieved = state.get("retrieved_documents", [])
    previously_relevant = state.get("relevant_documents", [])
    corrective_indices = set(state.get("corrective_question_part_indices") or [])
    corrective_documents = state.get("corrective_retrieved_documents", [])
    accepted_documents = []
    assessment_results = []
    missing_parts = []

    def add_accepted(document):
        key = evidence_document_key(document)
        if all(evidence_document_key(existing) != key for existing in accepted_documents):
            accepted_documents.append(document)

    for part_index, part in enumerate(parts):
        prior_for_part = [
            doc for doc in previously_relevant
            if part_index in doc.get("question_part_indices", [part_index])
        ]
        for document in prior_for_part:
            add_accepted(document)

        if corrective_indices and part_index not in corrective_indices:
            if not prior_for_part:
                missing_parts.append(part)
            continue

        assessment_pool = corrective_documents if corrective_indices else retrieved
        group = [
            doc for doc in assessment_pool
            if part_index in doc.get("question_part_indices", [part_index])
        ]
        accepted, assessments = filter_relevant_documents(part, group, tokenizer, qwen_model)
        for doc in assessments:
            doc.setdefault("relevance_by_part", {})[part_index] = doc.get("relevance")
            doc["question_part_indices"] = list(set(doc.get("question_part_indices", [])) | {part_index})
            assessment_results.append(doc)
        for doc in accepted:
            add_accepted(doc)
        if not prior_for_part and not accepted:
            missing_parts.append(part)

    fully_relevant_count = sum(
        1 for doc in accepted_documents
        if doc["relevance"] == "RELEVANT"
    )

    partially_relevant_count = sum(
        1 for doc in accepted_documents
        if doc["relevance"] == "PARTIALLY_RELEVANT"
    )

    directly_relevant = [
        doc for doc in accepted_documents
        if doc.get("relevance") == "RELEVANT"
    ]
    # Keep partial matches in the trace, but prefer direct evidence for
    # generation when the assessor found any.
    answer_documents = (
        accepted_documents if len(parts) > 1 else (directly_relevant or accepted_documents)
    )
    evidence_sufficient = bool(answer_documents) and not missing_parts

    print("Fully relevant:", fully_relevant_count)
    print("Partially relevant:", partially_relevant_count)
    print("Evidence available:", evidence_sufficient)

    return {
        "relevant_documents": answer_documents,
        "assessment_results": [
            *state.get("assessment_results", []),
            *assessment_results,
        ],
        "evidence_gaps": [
            "No relevant retrieved evidence was found for the requested part: " + part
            for part in missing_parts
        ],
        "evidence_sufficient": evidence_sufficient,
        "corrective_question_part_indices": [],
        "corrective_retrieved_documents": [],
    }


def identify_corrective_question_parts(parts, state):
    """Return confidently failed parts, or None when evidence is ambiguous."""
    coverage = state.get("question_part_coverage", [])
    passed_indices = set()
    failed_indices = set()

    def normalize(value):
        return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))

    normalized_parts = [normalize(part) for part in parts]
    for row in coverage:
        if not isinstance(row, dict) or not row.get("part"):
            continue
        part_key = normalize(row["part"])
        for index, normalized_part in enumerate(normalized_parts):
            if part_key == normalized_part:
                (passed_indices if row.get("supported") is True else failed_indices).add(index)
                break

    explicit_gap_indices = set()
    for gap in state.get("evidence_gaps", []):
        gap_key = normalize(gap)
        for index, normalized_part in enumerate(normalized_parts):
            if normalized_part and normalized_part in gap_key:
                explicit_gap_indices.add(index)

    claims = list(state.get("unsupported_claims", []))
    claims.extend(
        row.get("claim", "")
        for row in state.get("claim_results", [])
        if isinstance(row, dict) and row.get("supported") is False
    )
    claims.extend(
        row.get("claim", "")
        for row in state.get("citation_results", [])
        if isinstance(row, dict)
        and row.get("claim")
        and (row.get("claim_supported") is False or row.get("valid_citation") is False)
    )

    claim_indices = set()
    part_terms = [extract_keywords(part) for part in parts]
    for claim in claims:
        clean_claim = re.sub(r"\[[^\]]*\]", "", str(claim))
        claim_terms = extract_keywords(clean_claim)
        scores = [len(claim_terms.intersection(terms)) for terms in part_terms]
        if not scores:
            continue
        best = max(scores)
        best_indices = [index for index, score in enumerate(scores) if score == best]
        # Require multiple shared terms and a unique best part before narrowing.
        if best >= 2 and len(best_indices) == 1:
            index = best_indices[0]
            if index not in passed_indices:
                claim_indices.add(index)

    targets = (failed_indices | explicit_gap_indices | claim_indices) - passed_indices
    return sorted(targets) if targets else None


def corrective_retrieval_node(state):

    print("\n[Corrective Retrieval] Starting...")

    original_question = state["question"]
    parts = state.get("question_parts") or split_question_parts(original_question)
    unsupported_claims = state.get("unsupported_claims", [])
    evidence_gaps = state.get("evidence_gaps", [])
    previously_retrieved = state.get("retrieved_documents", [])
    relevant_documents = state.get("relevant_documents", [])

    supported_part_indices = {
        part_index
        for document in relevant_documents
        for part_index in document.get("question_part_indices", [])
    }
    targeted_indices = identify_corrective_question_parts(parts, state)
    if targeted_indices is not None:
        parts_to_search = [(index, parts[index]) for index in targeted_indices]
        selection_reason = "validation gaps or unsupported claims"
    else:
        missing_parts = [
            (index, part) for index, part in enumerate(parts)
            if index not in supported_part_indices
        ]
        # Keep the existing conservative fallback when the failed part cannot
        # be located reliably from validation state.
        parts_to_search = missing_parts or list(enumerate(parts))
        selection_reason = "safe fallback; failed part was not identified confidently"

    print("Original question:", original_question)
    print("Evidence gaps:", evidence_gaps)
    print("Unsupported claims:", unsupported_claims)
    print("Corrective selection:", selection_reason)
    print("Question-part indices to search:", [index for index, _ in parts_to_search])

    embedding_model, collection, _, _ = get_components()
    documents = []
    seen = set()
    corrected_queries = []
    for part_index, part in parts_to_search:
        query = build_corrective_query(
            part,
            evidence_gaps=evidence_gaps,
            unsupported_claims=unsupported_claims,
        )
        corrected_queries.append(query)
        previous_for_part = [
            document for document in previously_retrieved
            if part_index in document.get("question_part_indices", [part_index])
        ]
        for document in retrieve_documents(
            query,
            embedding_model,
            collection,
            exclude_documents=previous_for_part,
            corrective=True,
        ):
            key = (str(document.get("source_id", "")).casefold(),
                   str(document.get("page", "")),
                   " ".join(document.get("text", "").split()).casefold())
            if key not in seen:
                seen.add(key)
                document["question_part_indices"] = [part_index]
                documents.append(document)
            else:
                for existing in documents:
                    existing_key = (str(existing.get("source_id", "")).casefold(),
                                    str(existing.get("page", "")),
                                    " ".join(existing.get("text", "").split()).casefold())
                    if existing_key == key:
                        existing["question_part_indices"].append(part_index)
                        break

    corrected_query = " | ".join(corrected_queries)
    print("Corrected query:", corrected_query)

    print("Corrective retrieval results:", len(documents))

    merged_retrieved = []
    merged_keys = set()
    for document in [*previously_retrieved, *documents]:
        key = evidence_document_key(document)
        if key not in merged_keys:
            merged_keys.add(key)
            merged_retrieved.append(document)

    return {
        "corrected_query": corrected_query,
        "search_query": corrected_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "retrieved_documents": merged_retrieved,
        "relevant_documents": relevant_documents,
        "assessment_results": state.get("assessment_results", []),
        "corrective_question_part_indices": [index for index, _ in parts_to_search],
        "corrective_retrieved_documents": documents,
        "evidence_gaps": [],
        "evidence_sufficient": False,
        "answer": "",
        "answer_supported": False,
        "validation_notes": "",
        "claim_results": [],
        "citation_results": [],
        "unsupported_claims": [],
        "question_part_coverage": [],
    }


def generate_answer_node(state):

    print("\n[Node 3] Generating evidence-grounded answer...")

    tokenizer, model = get_answer_components()

    parts = state.get("question_parts") or [state["question"]]
    documents = order_evidence_for_generation(
        parts,
        state.get("relevant_documents", []),
    )
    answer, generation_gaps = generate_answer_with_gaps(
        state["question"], documents, tokenizer, model
    )

    print("Answer generated.")

    return {
        "answer": answer,
        "evidence_gaps": [
            *state.get("evidence_gaps", []),
            *generation_gaps,
        ],
    }

def validate_answer_node(state):

    print("\n[Node 4] Validating answer...")

    result = validate_answer(
        state["question"],
        state["answer"],
        state["relevant_documents"]
    )

    print("Validation status:", result["status"])
    print("Validation notes:", result["notes"])

    print("\nClaim-by-claim validation:")

    claims_needing_evidence = []

    citation_results = result["citation_results"]

    for index, claim in enumerate(result["claim_results"]):

        citation = (
            citation_results[index]
            if index < len(citation_results)
            else {}
        )

        print(f"\nClaim {index + 1}:")
        print("Text:", claim["claim"])
        print("Supported:", claim["supported"])
        print("Keyword overlap:", claim["keyword_overlap"])
        print("Best evidence source:", claim["best_evidence_source"])
        print("Best evidence page:", claim["best_evidence_page"])
        print("Has citation:", citation.get("has_citation", False))
        print("Valid citation:", citation.get("valid_citation", False))

        if (
            not claim["supported"]
            or not citation.get("valid_citation", False)
        ):
            claims_needing_evidence.append(
                claim["claim"]
            )

    print("\nUnsupported claims:", claims_needing_evidence)

    evidence_gaps = list(result.get("scope_gaps", []))
    evidence_gaps.extend(result.get("coverage_gaps", []))
    evidence_gaps.extend(state.get("evidence_gaps", []))
    evidence_gaps.extend(
        f"Retrieved evidence did not support: {claim}"
        for claim in claims_needing_evidence
    )

    return {
        "answer_supported": result["supported"],
        "validation_notes": result["notes"],
        "claim_results": result["claim_results"],
        "citation_results": result["citation_results"],
        "unsupported_claims": claims_needing_evidence,
        "evidence_gaps": evidence_gaps,
        "question_part_coverage": result.get("question_part_coverage", []),
    }




def abstain_node(state):

    print("\n[Node] Insufficient evidence. Abstaining.")

    message = (
        "I could not find sufficient evidence in the "
        "available knowledge base to answer this question. "
        "Please consult a qualified professional or a "
        "reliable source for further information."
    )

    evidence_gaps = state.get("evidence_gaps", [])
    if not evidence_gaps:
        question = (state.get("question") or "").strip()
        gap = (
            "The retrieved evidence did not establish an answer to: "
            f"{question}"
            if question
            else "The available evidence did not establish an answer."
        )
        evidence_gaps = [gap]

    has_scope_mismatch = any(
        gap.startswith((
            "Geographic scope mismatch:",
            "Population scope mismatch:",
        ))
        for gap in evidence_gaps
    )
    validation_notes = (
        "The retrieved evidence did not match the requested geographic or "
        "population scope."
        if has_scope_mismatch
        else (
            "The available knowledge base did not provide sufficient "
            "evidence after corrective retrieval."
        )
    )

    return {
        "answer": message,
        "answer_supported": False,
        "validation_notes": validation_notes,
        "evidence_gaps": evidence_gaps,
        "final_response": message
    }


def finalize_answer_node(state):

    print("\n[Node 5] Finalizing response...")

    if state.get("answer_supported", False):

        final_response = state["answer"]

    else:

        final_response = (
            "I could not verify that all claims in the "
            "generated answer are supported by the available "
            "evidence. Therefore, I cannot provide that answer "
            "confidently."
        )

    return {
        "final_response": final_response
    }
