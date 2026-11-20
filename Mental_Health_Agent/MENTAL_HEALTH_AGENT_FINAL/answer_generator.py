
import re
from answer_validator import (
    is_complete_sentence,
    question_part_anchors,
    question_part_target_terms,
    split_question_parts,
)
import torch
from pathlib import Path
from transformers import pipeline as transformers_pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


MODEL_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen2.5-1.5B-Instruct"
    / "snapshots"
    / "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

MAX_CONTEXT_LENGTH = 4096
MAX_NEW_TOKENS = 64
CONTEXT_SPECIAL_TOKEN_RESERVE = 32

_LANGCHAIN_LLM = None
_LANGCHAIN_CHAIN = None

CITATION_PATTERN = re.compile(
    r"\s*\[[A-Za-z0-9_-]+\s*,\s*Page\s+\d+\]",
    re.IGNORECASE
)

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "of", "in", "on", "for", "to", "and", "or",
    "with", "by", "from", "that", "this", "as",
    "at", "it", "their", "they", "be", "has",
    "have", "had", "can", "how", "what", "when",
    "who", "which", "also", "more", "than", "into",
    "such", "your", "you", "our", "both", "may",
    "might", "should", "could", "would", "aged", "aim", "per"
}

STRESS_MANAGEMENT_QUESTION = re.compile(
    r"\b(stress|stressed)\b.*\b(manag\w*|cop\w*|reliev\w*|"
    r"reduc\w*|strateg\w*|techniqu\w*|ways|steps|tips)\b|"
    r"\b(manag\w*|cop\w*|reliev\w*|reduc\w*|strateg\w*|"
    r"techniqu\w*|ways|steps|tips)\b.*\b(stress|stressed)\b",
    re.IGNORECASE,
)
STRESS_STRATEGY_LINE = re.compile(
    r"\b(?:take breaks|mak(?:e|ing) time to unwind|take care of your mind|"
    r"deep breaths|practice gratitude|keep a journal|spend time outdoors|"
    r"connect with others|meditat\w*|mindfulness)\b",
    re.IGNORECASE,
)


def load_answer_model():
    print("Loading answer generation model...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        torch_dtype=torch.float32
    )

    model.eval()

    print("Answer model loaded successfully.")

    return tokenizer, model


def get_answer_chain(tokenizer, model):
    """Create a LangChain generation chain around the local HF model."""
    global _LANGCHAIN_LLM, _LANGCHAIN_CHAIN

    if _LANGCHAIN_CHAIN is None:
        text_pipeline = transformers_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        _LANGCHAIN_LLM = HuggingFacePipeline(pipeline=text_pipeline)
        _LANGCHAIN_CHAIN = (
            PromptTemplate.from_template("{formatted_prompt}")
            | _LANGCHAIN_LLM
            | StrOutputParser()
        )

    return _LANGCHAIN_CHAIN


def evidence_document_key(document):
    """Stable key for dropping exact duplicate chunks without losing provenance."""
    return (
        str(document.get("source_id", "")).casefold(),
        str(document.get("page", "")),
        " ".join((document.get("text", "") or "").split()).casefold(),
    )


def deduplicate_evidence_documents(evidence_documents):
    unique = []
    seen = set()
    for document in evidence_documents or []:
        key = evidence_document_key(document)
        if key in seen:
            continue
        seen.add(key)
        unique.append(document)
    return unique


def count_prompt_tokens(tokenizer, formatted_prompt):
    """Count the actual prompt with the loaded model tokenizer."""
    try:
        token_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(formatted_prompt)
    if hasattr(token_ids, "numel"):
        return int(token_ids.numel())
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        return len(token_ids[0])
    return len(token_ids)


def normalize_word(word):
    word = word.lower()

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("ing") and len(word) > 5:
        return word[:-3]

    if word.endswith("ed") and len(word) > 4:
        return word[:-2]

    if word.endswith("s") and len(word) > 3:
        return word[:-1]

    return word


def get_keywords(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    return {
        normalize_word(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    }


def extract_numbers(text):
    return set(
        re.findall(r"\b\d+(?:\.\d+)?\b", text)
    )


def split_answer_sentences(answer):
    """Split generated answers into separate factual statements."""

    if not answer:
        return []

    answer = CITATION_PATTERN.sub("", answer).strip()

    lines = re.split(r"\r?\n+", answer)

    sentences = []

    for line in lines:

        line = re.sub(
            r"^\s*[-*•\d.)]+\s*",
            "",
            line
        ).strip()

        if not line:
            continue

        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z])|\s{2,}(?=[A-Z])",
            line
        )

        for part in parts:

            part = part.strip()

            if len(part) > 10:
                is_continuation = (
                    bool(sentences)
                    and part[:1].islower()
                )
                if is_continuation:
                    sentences[-1] = f"{sentences[-1]} {part}"
                else:
                    sentences.append(part)

    return sentences


def find_supporting_document(sentence, evidence_documents):
    """
    Find one evidence passage with sufficient lexical support.

    The threshold is aligned with answer_validator.py:
    at least two shared keywords and 50% keyword overlap.
    """

    sentence_keywords = get_keywords(sentence)
    sentence_numbers = extract_numbers(sentence)

    if not sentence_keywords:
        return None

    best_document = None
    best_score = 0.0

    for document in evidence_documents:

        evidence_text = document.get("text", "")

        evidence_keywords = get_keywords(evidence_text)

        common_keywords = sentence_keywords.intersection(
            evidence_keywords
        )

        overlap = (
            len(common_keywords) / len(sentence_keywords)
        )

        evidence_numbers = extract_numbers(evidence_text)

        numbers_supported = sentence_numbers.issubset(
            evidence_numbers
        )

        if (
            len(common_keywords) >= 2
            and overlap >= 0.50
            and numbers_supported
            and overlap > best_score
        ):
            best_score = overlap
            best_document = document

    return best_document


def add_source_citations(answer, evidence_documents):
    """
    Keep only supported sentences.
    Attach an exact source citation to each sentence.
    """

    sentences = split_answer_sentences(answer)

    supported_sentences = []
    unsupported_sentences = []

    for sentence in sentences:

        if not is_complete_sentence(sentence):
            unsupported_sentences.append(sentence)
            continue

        document = find_supporting_document(
            sentence,
            evidence_documents
        )

        if document is None:

            unsupported_sentences.append(sentence)

            continue

        citation = (
            f"[{document['source_id']}, "
            f"Page {document['page']}]"
        )

        supported_sentences.append(
            f"{sentence.rstrip('.!?')}. {citation}"
        )

    return "\n".join(supported_sentences), unsupported_sentences


def join_wrapped_evidence_lines(text):
    """Join a source line only when it visibly ends mid-clause."""
    continuation = re.compile(
        r"\b(?:and|or|but|that|which|who|with|for|to|of|in|on|at|by|from|about|as)$",
        re.IGNORECASE,
    )
    lines = []
    for raw_line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if lines and continuation.search(lines[-1]) and line[:1].islower():
            lines[-1] = f"{lines[-1]} {line}"
        else:
            lines.append(line)
    return lines


def build_extractive_fallback(question, evidence_documents, limit=5):
    """Return matching evidence lines verbatim when generation is unusable."""
    question_keywords = get_keywords(question)
    topic_anchors = question_part_anchors(question)
    stress_management_question = bool(STRESS_MANAGEMENT_QUESTION.search(question or ""))
    excerpts = []
    seen = set()

    for document in evidence_documents:
        passage_text = document.get("text", "") or ""
        candidates = []
        for source_line in join_wrapped_evidence_lines(passage_text):
            candidates.extend(split_answer_sentences(source_line) or [source_line])
        for raw_line in candidates:
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line).strip()
            line = re.sub(r"\s+", " ", line)
            if len(line) < 15 or line in seen:
                continue

            if not is_complete_sentence(line):
                continue

            # Exclude headings and label fragments while retaining structured
            # evidence rows such as age-specific sleep recommendations.
            has_action_or_predicate = bool(re.search(
                r"\b(?:is|are|was|were|has|have|can|may|helps?|improves?|"
                r"reduces?|supports?|leads?|causes?|take|avoid|make|practice|"
                r"talk|keep|spend|connect|try|need|should|do|includes?)\b",
                line,
                re.I,
            ))
            structured_numeric_row = bool(
                extract_numbers(line)
                and re.search(r"\b(?:adult|child|teen|year|hour|month)\w*\b", line, re.I)
            )
            if not has_action_or_predicate and not structured_numeric_row:
                continue

            # Do not turn reference-list fragments into answer sentences.
            if (re.search(r"\b(?:doi|issn|vol\.?|issue|et al\.)\b", line, re.I)
                    or re.search(r"\b(?:19|20)\d{2}\s*;\s*\d", line)):
                continue

            shared = question_keywords.intersection(get_keywords(line))
            contextual_strategy = bool(
                stress_management_question
                and re.search(r"\bstress\b", passage_text, re.I)
                and STRESS_STRATEGY_LINE.search(line)
            )
            if len(shared) < 2 and not contextual_strategy:
                continue
            if (topic_anchors and not topic_anchors.intersection(get_keywords(line))
                    and not contextual_strategy):
                continue

            if not extract_numbers(line).issubset(
                extract_numbers(document.get("text", ""))
            ):
                continue

            seen.add(line)
            excerpts.append(line.rstrip(" .!?") + ".")
            if len(excerpts) >= limit:
                break
        if len(excerpts) >= limit:
            break

    if not excerpts:
        return ""

    answer, unsupported = add_source_citations(
        "\n".join(excerpts), evidence_documents
    )
    return answer if not unsupported else ""


def incomplete_claim_replaced(fragment, fallback_answer, evidence_documents):
    fragment_keywords = get_keywords(fragment)
    for candidate in split_answer_sentences(fallback_answer):
        shared = fragment_keywords.intersection(get_keywords(candidate))
        if len(shared) >= 2 and find_supporting_document(candidate, evidence_documents):
            return True
    return False


def incomplete_statement_gaps(incomplete_sentences, fallback_answer, evidence_documents):
    unresolved = [
        sentence for sentence in incomplete_sentences
        if not incomplete_claim_replaced(sentence, fallback_answer, evidence_documents)
    ]
    return (
        [
            "An incomplete generated statement was omitted because no complete, "
            "cited version could be established from the retrieved evidence."
        ]
        if unresolved else []
    )


def generate_answer_with_gaps(
    question,
    evidence_documents,
    tokenizer,
    model
):
    """Generate an evidence-grounded answer with claim-level citations."""

    evidence_documents = deduplicate_evidence_documents(evidence_documents)

    if not evidence_documents:

        return (
            "I could not find sufficient evidence in the "
            "available knowledge base to answer this question.",
            [],
        )

    question_parts = split_question_parts(question)
    evidence_parts = []
    for index, document in enumerate(evidence_documents, start=1):
        part_indices = document.get("question_part_indices") or []
        part_labels = ", ".join(str(int(i) + 1) for i in sorted(set(part_indices)))
        relevance_by_part = document.get("relevance_by_part") or {}
        relevance = document.get("relevance", "NOT ASSESSED")
        if relevance_by_part:
            relevance = "; ".join(
                f"part {int(part_index) + 1}: {label}"
                for part_index, label in sorted(
                    relevance_by_part.items(), key=lambda item: int(item[0])
                )
            )

        evidence_parts.append(
            f"Evidence {index}\n"
            f"Applicable question parts: {part_labels or 'all'}\n"
            f"Relevance: {relevance}\n"
            f"Source ID: {document['source_id']}\n"
            f"Title: {document['title']}\n"
            f"Page: {document['page']}\n"
            f"Text:\n{document['text']}"
        )

    evidence_text = "\n".join(evidence_parts)
    parts_text = "\n".join(
        f"{index}. {part}" for index, part in enumerate(question_parts, start=1)
    )

    prompt = f"""
You are an evidence-grounded mental health psychoeducation assistant.

Answer the user's question using ONLY the supplied evidence.

Rules:
1. Use only information explicitly supported by the evidence.
2. Do not use outside knowledge.
3. Do not diagnose or prescribe treatment.
4. Answer only the parts of the question supported by evidence.
5. Do not invent recommendations.
6. Write short, simple English sentences.
7. Put each factual idea on a separate line.
8. Never combine different recommendations in one sentence.
9. Do not write citations.
10. Do not repeat the same idea.
11. If evidence is missing for a part, do not guess.
12. Do not add a general recommendation unless the evidence supports it.
13. Keep the evidence's wording for ages, quantities, and time periods. Do not add qualifiers.
14. If quantities differ between age groups or categories, state each group and quantity separately. Never combine them into one range.
15. If the evidence does not clearly answer the question, say that the knowledge base is insufficient.
16. Every line must be a complete, self-contained sentence. Do not output sentence fragments or bullets.
17. Address every independent part of the user's question in a separate sentence when evidence supports it.
18. Do not imply that one supported part answers another part of the question.

USER QUESTION:
{question}

INDEPENDENT QUESTION PARTS:
{parts_text}

AVAILABLE EVIDENCE:
{evidence_text}

Write a concise answer using the supported information only.
Return each factual statement on a separate line.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful evidence-only assistant. "
                "Never invent facts, advice, or references. "
                "Use short factual statements."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    input_budget = MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS - CONTEXT_SPECIAL_TOKEN_RESERVE
    prompt_tokens = count_prompt_tokens(tokenizer, formatted_prompt)
    if prompt_tokens > input_budget:
        print(
            f"Prompt uses {prompt_tokens} tokens; safe input budget is "
            f"{input_budget}. Using the full evidence for extractive fallback."
        )
        fallback = build_extractive_fallback(
            question,
            evidence_documents,
            limit=max(5, 3 * len(question_parts)),
        )
        if fallback:
            return fallback, []
        return (
            "I could not fit all relevant evidence into the model context "
            "and cannot provide a complete evidence-supported answer.",
            [
                "All relevant passages were retained; the model context budget "
                "was exceeded, so a complete answer could not be generated."
            ],
        )

    answer = get_answer_chain(tokenizer, model).invoke(
        {"formatted_prompt": formatted_prompt}
    ).strip()

    print("Raw model draft:", answer)

    if not answer:

        return (
            "I could not generate a reliable answer "
            "from the available evidence.",
            [],
        )

    answer = CITATION_PATTERN.sub(
        "",
        answer
    ).strip()
    draft_sentences = split_answer_sentences(answer)
    relevant_incomplete = []
    for sentence in draft_sentences:
        if is_complete_sentence(sentence):
            continue
        keywords = get_keywords(sentence)
        if (keywords.intersection(question_part_anchors(question))
                or STRESS_STRATEGY_LINE.search(sentence)):
            relevant_incomplete.append(sentence)
    incomplete_draft_sentences = relevant_incomplete

    # Keep generated sentences tied to the requested topic for this clause.
    anchors = question_part_anchors(question)
    if anchors:
        filtered_sentences = []
        topic_established = False
        targets = question_part_target_terms(question)
        for sentence in draft_sentences:
            sentence_keywords = get_keywords(sentence)
            if sentence_keywords.intersection(anchors):
                topic_established = True
                filtered_sentences.append(sentence)
            elif (topic_established and targets.intersection(sentence_keywords)
                    and re.match(r"^(?:it|this|they|these)\b", sentence, re.I)):
                filtered_sentences.append(sentence)
        answer = "\n".join(filtered_sentences)

    cited_answer, unsupported_sentences = add_source_citations(
        answer,
        evidence_documents
    )
    incomplete_sentences = list(dict.fromkeys([
        *incomplete_draft_sentences,
        *(
            sentence for sentence in unsupported_sentences
            if not is_complete_sentence(sentence)
        ),
    ]))
    generation_gaps = []

    if not cited_answer:

        print("\nNo supported sentences were generated.")

        if unsupported_sentences:

            print("Unsupported statements:")

            for sentence in unsupported_sentences:
                print("-", sentence)

        extractive_answer = build_extractive_fallback(
            question, evidence_documents, limit=10 if incomplete_sentences else 5
        )
        if extractive_answer:
            print("Returning exact evidence excerpts with source citations.")
            generation_gaps = incomplete_statement_gaps(
                incomplete_sentences, extractive_answer, evidence_documents
            )
            return extractive_answer, generation_gaps

        if incomplete_sentences:
            generation_gaps.append(
                "A requested point could not be stated "
                "as a complete evidence-supported sentence."
            )
        return (
            "I could not find sufficient supporting evidence "
            "in the available knowledge base to answer this question.",
            generation_gaps,
        )

    if incomplete_sentences:
        extractive_answer = build_extractive_fallback(
            question, evidence_documents, limit=10
        )
        if extractive_answer:
            existing = {
                " ".join(sentence.casefold().split()).rstrip(".!?")
                for sentence in split_answer_sentences(cited_answer)
            }
            additions = []
            for line in extractive_answer.splitlines():
                clean_line = CITATION_PATTERN.sub("", line).strip()
                key = " ".join(clean_line.casefold().split()).rstrip(".!?")
                if key not in existing:
                    existing.add(key)
                    additions.append(line)
            if additions:
                cited_answer = "\n".join([cited_answer, *additions])
            generation_gaps = incomplete_statement_gaps(
                incomplete_sentences, extractive_answer, evidence_documents
            )
        else:
            generation_gaps = incomplete_statement_gaps(
                incomplete_sentences, "", evidence_documents
            )

    if unsupported_sentences:

        print(
            "\nWarning: Unsupported generated statements were removed:"
        )

        for sentence in unsupported_sentences:
            print("-", sentence)

    return cited_answer, generation_gaps


def generate_answer(question, evidence_documents, tokenizer, model):
    """Backward-compatible string API used by the CLI."""
    answer, _ = generate_answer_with_gaps(
        question, evidence_documents, tokenizer, model
    )
    return answer


def main():

    print("=" * 60)
    print("PARTIAL EVIDENCE-GROUNDED ANSWER TEST")
    print("=" * 60)

    question = (
        "What are the recommended sleep hours for adults?"
    )

    evidence_documents = [
        {
            "source_id": "CDC_008",
            "title": "About Sleep",
            "page": 1,
            "relevance": "RELEVANT",
            "text": (
                "Adults aged 18–60 years need 7 or more hours "
                "of sleep. Adults aged 61–64 years need 7–9 hours. "
                "Adults aged 65 years and older need 7–8 hours."
            )
        }
    ]

    tokenizer, model = load_answer_model()

    answer = generate_answer(
        question,
        evidence_documents,
        tokenizer,
        model
    )

    print("\nQuestion:", question)

    print("\nGenerated answer:")
    print(answer)

    print("\n" + "=" * 60)
    print("ANSWER GENERATION TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
