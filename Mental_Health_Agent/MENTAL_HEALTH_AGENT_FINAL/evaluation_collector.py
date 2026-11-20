"""Normalize complete, nested, or interrupted LangGraph results for evaluation logs.

This module only reads an agent result. It does not import or invoke the agent.
"""

from collections.abc import Mapping


_NESTED_RESULT_KEYS = ("result", "state", "values", "output", "final_state", "graph_state")


def _result_layers(result):
    """Yield result mappings from outer wrappers toward nested graph state."""
    queue = [result]
    seen = set()
    while queue:
        current = queue.pop(0)
        if not isinstance(current, Mapping) or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for key in _NESTED_RESULT_KEYS:
            nested = current.get(key)
            if isinstance(nested, Mapping):
                queue.append(nested)


def _field(layers, *names, default=None):
    for layer in layers:
        for name in names:
            value = layer.get(name)
            if value is not None:
                return value
    return default


def _items(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _read(obj, key, default=None):
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _document_record(document):
    metadata = _read(document, "metadata", {}) or {}
    text = _read(document, "text")
    if text is None:
        text = _read(document, "page_content", "")
    record = {
        "source_id": _read(document, "source_id", _read(metadata, "source_id")),
        "title": _read(document, "title", _read(metadata, "title")),
        "page": _read(document, "page", _read(metadata, "page")),
        "relevance": _read(document, "relevance", _read(metadata, "relevance")),
    }
    if text:
        record["text"] = text
    return {key: value for key, value in record.items() if value is not None}


def _citation_from_text(text):
    """Read [SOURCE_ID, Page N] citations without a regular-expression parser."""
    citations = []
    if not isinstance(text, str):
        return citations
    for fragment in text.split("[")[1:]:
        if ", Page " not in fragment:
            continue
        source_id, page_part = fragment.split(", Page ", 1)
        page_text = page_part.split("]", 1)[0].strip()
        if source_id.strip() and page_text.isdigit():
            citation = {"source_id": source_id.strip(), "page": int(page_text)}
            if citation not in citations:
                citations.append(citation)
    return citations


def _citation_records(layers):
    answer_citations = []
    validation_citations = []
    answer = _field(layers, "final_response", "final_answer", "answer", default="")
    if isinstance(answer, Mapping):
        answer = answer.get("text", answer.get("content", ""))
    answer_citations.extend(_citation_from_text(answer))
    for layer in layers:
        for row in _items(layer.get("citation_results")):
            for citation in _items(_read(row, "valid_citations")):
                source_id = _read(citation, "source_id")
                page = _read(citation, "page")
                if source_id is not None and page is not None:
                    validation_citations.append({"source_id": source_id, "page": page})

    def unique(items):
        result = []
        for item in items:
            if item not in result:
                result.append(item)
        return result

    return unique(answer_citations), unique(validation_citations)


def collect_agent_result(result):
    """Return a stable, JSON-safe report from an optional or wrapped graph state."""
    layers = list(_result_layers(result))
    answer = _field(layers, "final_response", "final_answer", "answer", default="")
    if isinstance(answer, Mapping):
        answer = answer.get("text", answer.get("content", ""))

    retrieved_value = _field(layers, "retrieved_documents", "retrieved_sources", default=[])
    relevant_value = _field(layers, "relevant_documents", "relevant_sources", default=[])
    retrieved = [_document_record(item) for item in _items(retrieved_value)]
    relevant = [_document_record(item) for item in _items(relevant_value)]

    assessments = _items(_field(layers, "assessment_results", default=[]))
    for document in retrieved:
        if document.get("relevance") is not None:
            continue
        for assessment in assessments:
            if not isinstance(assessment, Mapping):
                continue
            same_source = assessment.get("source_id") == document.get("source_id")
            same_page = assessment.get("page") == document.get("page")
            if same_source and same_page:
                relevance = next(
                    (assessment.get(key) for key in ("assessment", "relevance", "label") if assessment.get(key) is not None),
                    None,
                )
                if relevance is not None:
                    document["relevance"] = relevance
                    break

    interrupts = _items(_field(layers, "__interrupt__", "interrupts", default=[]))
    requires_review = bool(_field(layers, "requires_human_review", default=False)) or bool(interrupts)
    review_status = _field(layers, "human_review_status", default=None)
    if review_status is None:
        review_status = "PENDING" if requires_review else "NOT_REQUIRED"

    validation_notes = _field(layers, "validation_notes", "validation_result")
    validation_status = _field(layers, "validation_status")
    question_classification = _field(layers, "question_classification")
    answer_supported = _field(layers, "answer_supported")
    if validation_status is None and isinstance(validation_notes, str):
        if validation_notes.casefold().startswith("geographic scope mismatch"):
            validation_status = "NOT_SUPPORTED"
        elif question_classification == "OUT_OF_SCOPE":
            validation_status = "NOT_RUN"
        elif any(
            phrase in validation_notes.casefold()
            for phrase in ("did not provide sufficient evidence", "no relevant retrieved evidence")
        ):
            validation_status = "NOT_RUN"
    answer_citations, validation_citations = _citation_records(layers)

    return {
        "question": _field(layers, "question", default=""),
        "question_classification": question_classification,
        "retrieved_sources": retrieved,
        "relevant_sources": relevant,
        "final_answer": answer or "",
        "citations": answer_citations,
        "validation_citations": validation_citations,
        "validation_status": validation_status,
        "validation_notes": validation_notes,
        "answer_supported": answer_supported,
        "evidence_gaps": _items(_field(layers, "evidence_gaps", default=[])),
        "question_parts": _items(_field(layers, "question_parts", default=[])),
        "question_part_coverage": _items(_field(layers, "question_part_coverage", default=[])),
        "retry_count": _field(layers, "retry_count", "corrective_retrieval_attempts"),
        "human_review_required": requires_review,
        "human_review_status": review_status,
        "human_review_reason": _field(layers, "human_review_reason", default=""),
        "execution_interrupted": bool(interrupts),
    }
