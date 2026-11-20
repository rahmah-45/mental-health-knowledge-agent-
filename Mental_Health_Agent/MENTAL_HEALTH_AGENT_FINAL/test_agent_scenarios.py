"""Requirement scenarios for the Exam 3 knowledge agent, without model downloads."""

from langchain_core.documents import Document

import agent_nodes
from agent_nodes import assess_evidence_node, corrective_retrieval_node
from evidence_retriever import ChromaLangChainRetriever, TOP_K, retrieve_documents
from mental_health_agent import run_agent
from answer_validator import validate_answer as validate_answer_with_evidence
from answer_generator import build_extractive_fallback, split_answer_sentences


def evidence(source_id, text, page=1):
    return Document(
        page_content=text,
        metadata={
            "source_id": source_id,
            "title": f"Synthetic {source_id} evidence",
            "page": page,
            "chunk_index": 1,
        },
    )


class SequenceRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = 0
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


def configure_agent(monkeypatch, retrieved):
    retriever = SequenceRetriever(retrieved)
    monkeypatch.setattr(
        agent_nodes,
        "get_components",
        lambda: (None, retriever, None, None),
    )
    monkeypatch.setattr(
        agent_nodes,
        "get_answer_components",
        lambda: (object(), object()),
    )

    def assess(question, documents, tokenizer, model):
        accepted = []
        results = []
        for document in documents:
            item = dict(document)
            item["relevance"] = "RELEVANT" if "UNRELATED" not in item["text"] else "IRRELEVANT"
            results.append(item)
            if item["relevance"] == "RELEVANT":
                accepted.append(item)
        return accepted, results

    def generate(question, documents, tokenizer, model):
        if len(documents) > 1:
            return (
                "Sleep supports health and emotional well-being. "
                "[CDC_008, Page 1]\n"
                "Long-term stress can affect mental health. "
                "[CDC_009, Page 2]",
                [],
            )
        return "Sleep supports health and emotional well-being. [CDC_008, Page 1]", []

    monkeypatch.setattr(agent_nodes, "filter_relevant_documents", assess)
    monkeypatch.setattr(agent_nodes, "generate_answer_with_gaps", generate)
    monkeypatch.setattr(
        agent_nodes,
        "validate_answer",
        lambda question, answer, documents: {
            "supported": True,
            "status": "SUPPORTED",
            "notes": "Synthetic validator accepted the sample.",
            "claim_results": [],
            "citation_results": [],
        },
    )
    return retriever


def test_relevant_question_returns_cited_education(monkeypatch):
    configure_agent(monkeypatch, [[evidence("CDC_008", "sleep health")]])
    result = run_agent("How does sleep support mental health?")
    assert result["question_classification"] == "IN_SCOPE"
    assert result["answer_supported"] is True
    assert "[CDC_008, Page 1]" in result["final_response"]


def test_multi_document_question_includes_both_references(monkeypatch):
    configure_agent(monkeypatch, [[
        evidence("CDC_008", "sleep health"),
        evidence("CDC_009", "stress mental health", 2),
    ]])
    result = run_agent("How do sleep and stress affect mental health?")
    assert "[CDC_008, Page 1]" in result["final_response"]
    assert "[CDC_009, Page 2]" in result["final_response"]


def test_exact_sleep_and_stress_question_covers_both_parts_with_citations(monkeypatch):
    sleep = evidence(
        "CDC_008",
        "Adequate sleep supports emotional well-being and mental health.",
        1,
    )
    stress = evidence(
        "CDC_009",
        "Long-term stress can affect mental health.",
        2,
    )
    configure_agent(monkeypatch, [[sleep, stress]])
    monkeypatch.setattr(agent_nodes, "validate_answer", validate_answer_with_evidence)
    generation_calls = []

    def generate_once(question, documents, tokenizer, model):
        generation_calls.append((question, documents))
        return (
            "Adequate sleep supports emotional well-being. [CDC_008, Page 1]\n"
            "Long-term stress can affect mental health. [CDC_009, Page 2]",
            [],
        )

    monkeypatch.setattr(agent_nodes, "generate_answer_with_gaps", generate_once)
    question = (
        "How can adequate sleep support emotional well-being, and what effects "
        "can long-term stress have on mental health?"
    )

    result = run_agent(question)

    assert len(generation_calls) == 1
    assert generation_calls[0][0] == question
    assert result["answer_supported"] is True
    assert len(result["question_part_coverage"]) == 2
    assert all(part["supported"] for part in result["question_part_coverage"])
    assert "[CDC_008, Page 1]" in result["final_response"]
    assert "[CDC_009, Page 2]" in result["final_response"]


def test_irrelevant_first_retrieval_uses_corrective_retrieval(monkeypatch):
    retriever = configure_agent(monkeypatch, [
        [evidence("WHO_002", "UNRELATED anxiety passage")],
        [evidence("CDC_008", "sleep health")],
    ])
    result = run_agent("How does sleep support mental health?")
    assert retriever.calls == 2
    assert retriever.queries[1] != retriever.queries[0]
    assert result["retry_count"] == 1
    assert result["answer_supported"] is True


def test_repeated_irrelevant_passages_are_excluded_and_agent_abstains(monkeypatch):
    irrelevant = evidence("WHO_002", "UNRELATED anxiety passage")
    retriever = configure_agent(monkeypatch, [[irrelevant], [irrelevant]])

    result = run_agent("What is the latitude of anxiety?")

    assert retriever.calls == 2
    assert retriever.queries[1] != retriever.queries[0]
    assert result["retry_count"] == 1
    assert len(result["retrieved_documents"]) == 1
    assert result["relevant_documents"] == []
    assert result["answer_supported"] is False
    assert "could not find sufficient evidence" in result["final_response"].casefold()


def test_omitted_incomplete_statement_is_reported_as_evidence_gap(monkeypatch):
    configure_agent(monkeypatch, [[
        evidence("CDC_009", "Taking small steps can help manage stress.")
    ]])
    gap = "An incomplete generated statement was omitted; no complete cited version was produced."
    monkeypatch.setattr(
        agent_nodes,
        "generate_answer_with_gaps",
        lambda question, documents, tokenizer, model: (
            "Taking small steps can help manage stress. [CDC_009, Page 1]",
            [gap],
        ),
    )

    result = run_agent("What are ways to manage stress?")

    assert result["answer_supported"] is True
    assert gap in result["evidence_gaps"]


def test_insufficient_evidence_abstains(monkeypatch):
    configure_agent(monkeypatch, [[], []])
    result = run_agent("What is the prevalence of anxiety in Pakistan?")
    assert result["question_classification"] == "IN_SCOPE"
    assert result["final_response"]
    assert result["requires_human_review"] is False


def test_pakistan_prevalence_scope_mismatch_abstains(monkeypatch):
    global_prevalence = (
        "An estimated 5.8% of the global population currently experience "
        "an anxiety disorder. In 2023, 470 million people in the world had "
        "an anxiety disorder."
    )
    retriever = SequenceRetriever([[
        evidence("WHO_002", global_prevalence)
    ]])
    monkeypatch.setattr(
        agent_nodes,
        "get_components",
        lambda: (None, retriever, None, None),
    )
    monkeypatch.setattr(
        agent_nodes,
        "get_answer_components",
        lambda: (object(), object()),
    )

    def assess(question, documents, tokenizer, model):
        accepted = []
        results = []
        for document in documents:
            item = dict(document)
            item["relevance"] = "RELEVANT"
            accepted.append(item)
            results.append(item)
        return accepted, results

    monkeypatch.setattr(agent_nodes, "filter_relevant_documents", assess)
    monkeypatch.setattr(
        agent_nodes,
        "generate_answer_with_gaps",
        lambda question, documents, tokenizer, model: ((
            "An estimated 5.8% of the global population currently "
            "experience an anxiety disorder. [WHO_002, Page 1]"
        ), []),
    )

    question = "What percentage of people in Pakistan have an anxiety disorder?"
    result = run_agent(question)

    assert result["question_classification"] == "IN_SCOPE"
    assert result["answer_supported"] is False
    assert result["retry_count"] == 0
    assert "could not find sufficient evidence" in result["final_response"].casefold()
    assert "5.8%" not in result["final_response"]
    assert "470 million" not in result["final_response"]
    assert any(
        "Pakistan-specific prevalence data was not found" in gap
        for gap in result["evidence_gaps"]
    )


def test_out_of_scope_question_is_declined_without_retrieval(monkeypatch):
    retriever = configure_agent(monkeypatch, [[]])
    result = run_agent("What is the population of Islamabad?")
    assert result["question_classification"] == "OUT_OF_SCOPE"
    assert result["requires_human_review"] is False
    assert retriever.calls == 0


def test_clinical_decision_routes_to_human_review(monkeypatch):
    retriever = configure_agent(monkeypatch, [[]])
    result = run_agent(
        "Which medication should I take for anxiety?",
        review_callback=lambda payload: True,
    )
    assert result["question_classification"] == "HUMAN_REVIEW"
    assert result["requires_human_review"] is True
    assert result["human_review_status"] == "REVIEWED"
    assert result["human_review_reason"]
    assert retriever.calls == 0


def test_crisis_language_routes_to_human_review(monkeypatch):
    retriever = configure_agent(monkeypatch, [[]])
    result = run_agent(
        "I am thinking of suicide and cannot stay safe",
        review_callback=lambda payload: True,
    )
    assert result["question_classification"] == "CRISIS"
    assert result["requires_human_review"] is True
    assert result["human_review_status"] == "REVIEWED"
    assert "emergency service" in result["final_response"]
    assert retriever.calls == 0


def test_sensitive_request_pauses_until_a_reviewer_responds(monkeypatch):
    configure_agent(monkeypatch, [[]])
    result = run_agent("Could I have depression?")
    assert result["requires_human_review"] is True
    assert result["human_review_status"] == "PENDING"
    assert result["__interrupt__"]


def test_chroma_adapter_exposes_page_metadata_as_langchain_documents():
    class FakeEmbedding:
        def encode(self, query, normalize_embeddings):
            assert normalize_embeddings is True
            class EncodedVector(list):
                def tolist(self):
                    return list(self)
            return EncodedVector([0.25, 0.75])

    class FakeCollection:
        def count(self):
            return 1

        def query(self, **kwargs):
            assert kwargs["n_results"] == 1
            return {
                "documents": [["Sleep education passage."]],
                "metadatas": [[{
                    "source_id": "CDC_008",
                    "title": "About Sleep",
                    "page": 1,
                    "chunk_index": 2,
                }]],
                "distances": [[0.12]],
            }

    retriever = ChromaLangChainRetriever(
        embedding_model=FakeEmbedding(),
        collection=FakeCollection(),
        top_k=1,
    )
    documents = retriever.invoke("sleep")
    assert documents[0].metadata["source_id"] == "CDC_008"
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["distance"] == 0.12


def test_corrective_chroma_retrieval_overfetches_only_to_return_new_passages():
    class FakeEmbedding:
        def encode(self, query, normalize_embeddings):
            class Vector(list):
                def tolist(self):
                    return list(self)
            return Vector([0.25, 0.75])

    rows = [
        {
            "text": f"Passage {index} about mental health.",
            "metadata": {
                "source_id": f"WHO_{index:03d}",
                "title": f"Source {index}",
                "page": 1,
                "chunk_index": index,
            },
            "distance": index / 100,
        }
        for index in range(1, 9)
    ]

    class FakeCollection:
        def count(self):
            return len(rows)

        def query(self, n_results, **kwargs):
            selected = rows[:n_results]
            return {
                "documents": [[row["text"] for row in selected]],
                "metadatas": [[row["metadata"] for row in selected]],
                "distances": [[row["distance"] for row in selected]],
            }

    retriever = ChromaLangChainRetriever(
        embedding_model=FakeEmbedding(),
        collection=FakeCollection(),
        top_k=TOP_K,
    )
    initial = retrieve_documents("original phrasing", None, retriever)
    corrected = retrieve_documents(
        "corrected evidence phrasing",
        None,
        retriever,
        exclude_documents=initial,
        corrective=True,
    )

    assert TOP_K == 5
    assert len(initial) == TOP_K
    assert len(corrected) == 3
    assert {item["source_id"] for item in corrected}.isdisjoint(
        item["source_id"] for item in initial
    )


def test_corrective_retrieval_targets_failed_part_and_keeps_passed_evidence(monkeypatch):
    question = (
        "How does good sleep support emotional well-being, and what are the "
        "effects of long-term stress on health?"
    )
    parts = [
        "How does good sleep support emotional well-being",
        "what are the effects of long-term stress on health",
    ]
    sleep = {
        "source_id": "CDC_008",
        "title": "Sleep",
        "page": 1,
        "text": "Good sleep supports emotional well-being.",
        "question_part_indices": [0],
        "relevance": "RELEVANT",
        "relevance_by_part": {0: "RELEVANT"},
    }
    stress = {
        "source_id": "CDC_009",
        "title": "Stress",
        "page": 2,
        "text": "Long-term stress can affect health.",
    }
    retrieval_calls = []
    monkeypatch.setattr(agent_nodes, "get_components", lambda: (object(), object(), None, None))

    def retrieve(query, embedding_model, collection, exclude_documents=None, corrective=False):
        retrieval_calls.append((query, exclude_documents, corrective))
        return [stress]

    monkeypatch.setattr(agent_nodes, "retrieve_documents", retrieve)
    state = {
        "question": question,
        "question_parts": parts,
        "retry_count": 0,
        "search_query": question,
        "corrected_query": "",
        "retrieved_documents": [sleep],
        "relevant_documents": [sleep],
        "assessment_results": [sleep],
        "evidence_gaps": [
            "The available evidence did not support a complete cited answer for this requested part: "
            + parts[1]
        ],
        "unsupported_claims": ["Long-term stress can affect health."],
        "claim_results": [{"claim": "Long-term stress can affect health.", "supported": False}],
        "citation_results": [{
            "claim": "Long-term stress can affect health.",
            "claim_supported": False,
            "valid_citation": False,
        }],
        "question_part_coverage": [
            {"part": parts[0], "supported": True},
            {"part": parts[1], "supported": False},
        ],
        "corrective_question_part_indices": [],
        "corrective_retrieved_documents": [],
    }

    correction = corrective_retrieval_node(state)

    assert len(retrieval_calls) == 1
    assert retrieval_calls[0][2] is True
    assert correction["corrective_question_part_indices"] == [1]
    assert "sleep" not in retrieval_calls[0][0].casefold()
    assert correction["relevant_documents"] == [sleep]
    assert {item["source_id"] for item in correction["retrieved_documents"]} == {
        "CDC_008", "CDC_009"
    }

    assessed_parts = []

    def assess_only_targeted_part(part, documents, tokenizer, qwen_model):
        assessed_parts.append(part)
        assert [item["source_id"] for item in documents] == ["CDC_009"]
        item = dict(documents[0])
        item["relevance"] = "RELEVANT"
        return [item], [item]

    monkeypatch.setattr(agent_nodes, "get_components", lambda: (None, None, object(), object()))
    monkeypatch.setattr(agent_nodes, "filter_relevant_documents", assess_only_targeted_part)
    assessed = assess_evidence_node({**state, **correction})

    assert assessed_parts == [parts[1]]
    assert {item["source_id"] for item in assessed["relevant_documents"]} == {
        "CDC_008", "CDC_009"
    }
    assert assessed["evidence_sufficient"] is True


def test_unclear_corrective_target_uses_safe_all_parts_fallback(monkeypatch):
    parts = ["How does good sleep support emotional well-being", "What does stress affect"]
    relevant = [
        {"source_id": "CDC_008", "page": 1, "text": "sleep", "question_part_indices": [0]},
        {"source_id": "CDC_009", "page": 2, "text": "stress", "question_part_indices": [1]},
    ]
    calls = []
    monkeypatch.setattr(agent_nodes, "get_components", lambda: (None, None, None, None))
    monkeypatch.setattr(
        agent_nodes,
        "retrieve_documents",
        lambda query, *args, **kwargs: calls.append(query) or [],
    )
    state = {
        "question": "How does sleep relate to stress?",
        "question_parts": parts,
        "retry_count": 0,
        "retrieved_documents": relevant,
        "relevant_documents": relevant,
        "assessment_results": relevant,
        "evidence_gaps": [],
        "unsupported_claims": ["It causes an effect."],
        "claim_results": [],
        "citation_results": [],
        "question_part_coverage": [],
    }

    result = corrective_retrieval_node(state)

    assert len(calls) == 2
    assert result["corrective_question_part_indices"] == [0, 1]


def test_unsupported_claim_identifies_its_unique_question_part():
    parts = [
        "How does good sleep support emotional well-being",
        "What are the effects of long-term stress on health",
    ]
    state = {
        "unsupported_claims": ["Long-term stress can affect physical health."],
        "claim_results": [],
        "citation_results": [],
        "question_part_coverage": [],
        "evidence_gaps": [],
    }

    assert agent_nodes.identify_corrective_question_parts(parts, state) == [1]


def test_no_new_relevant_corrective_evidence_still_abstains(monkeypatch):
    sleep = evidence("CDC_008", "Good sleep supports emotional well-being.")
    unrelated = evidence("WHO_002", "UNRELATED astronomy passage.", 2)
    configure_agent(monkeypatch, [[sleep], [unrelated], []])

    result = run_agent(
        "How does good sleep support emotional well-being, and what are the effects of long-term stress on health?"
    )

    assert result["retry_count"] == 1
    assert result["answer_supported"] is False
    assert "could not find sufficient evidence" in result["final_response"].casefold()


def test_answer_splitter_joins_lowercase_line_continuations():
    sentences = split_answer_sentences(
        "Adults aged 18–60 need 7\nor more hours of sleep."
    )
    assert sentences == ["Adults aged 18–60 need 7 or more hours of sleep."]


def test_extractive_fallback_quotes_age_specific_sleep_rows():
    passage = (
        "Adult 18–60 years 7 or more hours\n"
        "Adult 61–64 years 7–9 hours\n"
        "Adult 65 years and older 7–8 hours"
    )
    result = build_extractive_fallback(
        "How many hours are recommended for adults?",
        [{"source_id": "CDC_008", "page": 1, "text": passage}],
    )
    assert "Adult 18–60 years 7 or more hours. [CDC_008, Page 1]" in result
    assert "Adult 61–64 years 7–9 hours. [CDC_008, Page 1]" in result
