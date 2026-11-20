
import re


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "of", "in", "on", "for", "to", "and", "or",
    "with", "by", "from", "that", "this", "as",
    "at", "it", "their", "they", "be", "has",
    "have", "had", "need", "needs", "according",
    "following", "years", "year", "can", "should",
    "also", "through", "these", "those", "such",
    "both", "more", "most", "may", "might",
    "your", "you", "our", "them", "include",
    "including", "help", "helps", "aged", "aim", "per"
}

CITATION_PATTERN = re.compile(
    r"\[([A-Za-z0-9_-]+)\s*,\s*Page\s+(\d+)\]",
    re.IGNORECASE
)
INCOMPLETE_ENDING_PATTERN = re.compile(
    r"\b(?:and|or|but|that|which|who|whose|because|although|while|"
    r"if|when|where|with|for|to|of|in|on|at|by|from|about|as|"
    r"the|a|an|their|its|can|could|will|would|should|may|might|"
    r"than|including|such as)$",
    re.IGNORECASE,
)
GERUND_OPENING_PATTERN = re.compile(
    r"^(?:taking|making|getting|finding|using|avoiding|spending|"
    r"practicing|maintaining|eating|seeking|connecting|doing|"
    r"learning|keeping|moving|trying|building|developing)\b",
    re.IGNORECASE,
)
FINITE_PREDICATE_PATTERN = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being|has|have|had|can|could|"
    r"will|would|should|may|might|must|help(?:s|ed)|reduces?|relieves?|"
    r"improves?|supports?|causes?|leads?|works?|provides?|prevents?|"
    r"affects?|contributes?|keeps?|makes?|means?|promotes?|enhances?|"
    r"allows?|increases?|decreases?|needs?|requires?|offers?|includes?)\b",
    re.IGNORECASE,
)

GEOGRAPHIC_SCOPE_ALIASES = {
    "Pakistan": ("pakistan", "pakistani"),
    "United States": ("united states", "u.s.", "usa", "american"),
    "United Kingdom": ("united kingdom", "uk", "british"),
    "India": ("india", "indian"),
    "Bangladesh": ("bangladesh", "bangladeshi"),
    "Canada": ("canada", "canadian"),
    "Australia": ("australia", "australian"),
    "global": (
        "global", "globally", "worldwide", "world", "world population"
    ),
}

POPULATION_SCOPE_ALIASES = {
    "adults": ("adult", "adults"),
    "children": ("child", "children", "pediatric", "paediatric"),
    "adolescents": ("adolescent", "adolescents", "teen", "teenagers"),
    "women": ("woman", "women", "female"),
    "men": ("man", "men", "male"),
    "older adults": ("older adult", "older adults", "elderly"),
}

STATISTICAL_QUESTION_PATTERN = re.compile(
    r"\b(prevalence|percentage|percent|rate|proportion|incidence|"
    r"how many|what number|number of people|statistics?)\b",
    re.IGNORECASE,
)
NAMED_LOCATION_PATTERN = re.compile(
    r"\b(?:in|within|across|throughout|among|for)\s+(?:the\s+)?"
    r"([A-Z][A-Za-z]+(?:[ -]+[A-Z][A-Za-z]+)*)"
)
SPECIFIC_LOCATION_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z]+(?:[ -]+[A-Z][A-Za-z]+)*)-specific\b"
)


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


def extract_keywords(text):
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


def is_complete_sentence(sentence):
    """Reject clear sentence truncations ending in an unfinished connector."""
    clean = CITATION_PATTERN.sub("", sentence or "").strip()
    clean = re.sub(r"[.!?…]+[\"'’”)]*$", "", clean).strip()
    if not clean or re.search(r"(?:\.\.\.|…|[-—])\s*$", sentence or ""):
        return False
    if GERUND_OPENING_PATTERN.search(clean) and not FINITE_PREDICATE_PATTERN.search(clean):
        return False
    if re.match(r"^(?:healthy ways? to|ways? to|tips? for|strategies for)\b", clean, re.I) and not FINITE_PREDICATE_PATTERN.search(clean):
        return False
    final_word = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", clean)
    return bool(final_word) and not INCOMPLETE_ENDING_PATTERN.search(final_word[-1])


QUESTION_PART_PATTERN = re.compile(
    r"\s*(?:,\s*(?:and\s+)?|;\s*|\?\s*)(?="
    r"(?:what|how|why|when|where|which|who|does|do|did|can|could|"
    r"would|should|is|are|was|were|has|have|will|may)\b)",
    re.IGNORECASE,
)
NON_TOPICAL_COVERAGE_TERMS = {
    "adequate", "enough", "support", "effect", "effects", "impact",
    "long", "term", "emotional", "emotion", "well", "being", "mental",
    "health", "wellbeing", "can", "could", "would", "should", "what",
    "how", "why", "when", "where", "which", "who", "have", "does",
    "do", "did", "is", "are", "was", "were", "their", "they",
}


def split_question_parts(question):
    """Split clearly independent question clauses while preserving wording."""
    if not question or not question.strip():
        return []
    parts = [part.strip(" ,;?") for part in QUESTION_PART_PATTERN.split(question)]
    parts = [part for part in parts if len(extract_keywords(part)) >= 2]
    return parts or [question.strip()]


def question_part_anchors(part):
    """Return topical terms that distinguish the requested part."""
    return extract_keywords(part) - NON_TOPICAL_COVERAGE_TERMS


def question_part_target_terms(part):
    """Return explicit wellbeing domains that an answer must mention."""
    keywords = extract_keywords(part)
    return keywords.intersection({"mental", "emotional"})


def assess_question_part_coverage(question, answer, evidence_documents, citation_results):
    """Require each independent question part to have a cited supported claim."""
    parts = split_question_parts(question)
    claims = split_claims(answer)
    gaps = []
    coverage = []
    for part in parts:
        part_keywords = extract_keywords(part)
        anchors = question_part_anchors(part)
        target_terms = question_part_target_terms(part)
        matching_claims = []
        for index, claim in enumerate(claims):
            overlap = part_keywords.intersection(extract_keywords(CITATION_PATTERN.sub("", claim)))
            if len(overlap) >= 2 and (
                not anchors
                or overlap.intersection(anchors)
                or overlap.intersection(target_terms)
            ):
                matching_claims.append(index)
        supported = False
        covered_terms = set()
        for index in matching_claims:
            claim_result = citation_results[index] if index < len(citation_results) else {}
            claim = claims[index]
            cited = CITATION_PATTERN.findall(claim)
            cited_docs = [
                doc for source_id, page in cited for doc in evidence_documents
                if str(doc.get("source_id", "")).casefold() == source_id.casefold()
                and str(doc.get("page", "")) == page
            ]
            if (claim_result.get("claim_supported") and claim_result.get("valid_citation")
                    and any(check_evidence_support(claim, [doc])[0] for doc in cited_docs)):
                covered_terms.update(
                    part_keywords.intersection(
                        extract_keywords(CITATION_PATTERN.sub("", claim))
                    )
                )
        supported = (
            bool(covered_terms.intersection(anchors))
            and target_terms.issubset(covered_terms)
        )
        coverage.append({"part": part, "supported": supported})
        if not supported:
            gaps.append(
                "The available evidence did not support a complete cited answer for this requested part: "
                + part
            )
    return coverage, gaps


def _contains_scope_term(text, terms):
    text = text.casefold()
    return any(
        re.search(rf"\b{re.escape(term.casefold())}\b", text)
        for term in terms
    )


def _requested_geographic_scopes(question):
    requested = []
    question_text = question or ""
    folded_question = question_text.casefold()

    for scope, aliases in GEOGRAPHIC_SCOPE_ALIASES.items():
        if scope == "global":
            continue
        if _contains_scope_term(folded_question, aliases):
            requested.append((scope, aliases))

    for pattern in (NAMED_LOCATION_PATTERN, SPECIFIC_LOCATION_PATTERN):
        for match in pattern.finditer(question_text):
            name = match.group(1).strip()
            if name.casefold() in {scope.casefold() for scope, _ in requested}:
                continue
            requested.append((name, (name,)))

    return requested


def _scope_evidence_for_claim(claim, evidence_documents):
    cited_documents = []
    for source_id, page in CITATION_PATTERN.findall(claim):
        cited_documents.extend(
            document
            for document in evidence_documents
            if str(document.get("source_id", "")).casefold()
            == source_id.casefold()
            and str(document.get("page", "")) == page
        )

    if not cited_documents:
        return evidence_documents

    supporting_documents = [
        document
        for document in cited_documents
        if check_evidence_support(claim, [document])[0]
    ]
    return supporting_documents or cited_documents


def check_requested_scope(question, answer, evidence_documents):
    """Return scope gaps when evidence does not match the requested population."""
    requested_geographies = _requested_geographic_scopes(question)
    statistical_question = bool(
        STATISTICAL_QUESTION_PATTERN.search(question or "")
    )
    requested_global_scope = (
        statistical_question
        and _contains_scope_term(
            question or "",
            GEOGRAPHIC_SCOPE_ALIASES["global"],
        )
    )
    requested_populations = []
    if statistical_question:
        for population, aliases in POPULATION_SCOPE_ALIASES.items():
            if _contains_scope_term(question or "", aliases):
                requested_populations.append((population, aliases))

    numeric_claims = [
        claim
        for claim in split_claims(answer)
        if extract_numbers(CITATION_PATTERN.sub("", claim))
    ]
    scope_document_groups = [
        _scope_evidence_for_claim(claim, evidence_documents)
        for claim in numeric_claims
    ] or [evidence_documents]

    def scope_is_present(aliases):
        return bool(scope_document_groups) and all(
            any(
                _contains_scope_term(
                    "\n".join(document.get("text", "") for document in group),
                    aliases,
                )
                for document in group
            )
            for group in scope_document_groups
        )

    has_global_evidence = scope_is_present(
        GEOGRAPHIC_SCOPE_ALIASES["global"]
    )

    gaps = []
    if requested_global_scope and not has_global_evidence:
        gaps.append(
            "Geographic scope mismatch: global statistics were requested, "
            "but the retrieved evidence does not establish global scope."
        )

    for geography, aliases in requested_geographies:
        if scope_is_present(aliases):
            continue

        if statistical_question and has_global_evidence:
            gaps.append(
                "Geographic scope mismatch: "
                f"{geography}-specific prevalence data was not found; "
                "the retrieved evidence reports global statistics."
            )
        else:
            gaps.append(
                "Geographic scope mismatch: retrieved evidence does not "
                f"establish information specific to {geography}."
            )

    for population, aliases in requested_populations:
        if not scope_is_present(aliases):
            gaps.append(
                "Population scope mismatch: retrieved evidence does not "
                f"establish statistics specific to {population}."
            )

    return gaps


def split_claims(answer):
    """Split sentences while keeping trailing citations attached."""

    if not answer or not answer.strip():
        return []

    text = re.sub(
        r"(?m)^\s*[-*•]\s*",
        "",
        answer.strip()
    )

    text = re.sub(
        r"(?m)^\s*\d+[.)]\s*",
        "",
        text
    )

    # Split after sentence punctuation, not inside citations.
    parts = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    claims = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # If this part contains only citations, attach them
        # to the preceding sentence.
        if claims and CITATION_PATTERN.fullmatch(part):
            claims[-1] += " " + part
            continue

        # Remove citations from the beginning of a sentence.
        # They belong to the preceding sentence.
        leading = []

        while True:
            match = CITATION_PATTERN.match(part)

            if not match:
                break

            leading.append(match.group(0))
            part = part[match.end():].strip()

        if leading and claims:
            claims[-1] += " " + " ".join(leading)

        if len(part) > 10:
            claims.append(part)

        elif part and claims:
            claims[-1] += " " + part

    # Remove duplicate citations from each claim.
    cleaned_claims = []

    for claim in claims:
        seen = set()
        unique_citations = []

        for match in CITATION_PATTERN.finditer(claim):
            citation = match.group(0).strip()
            key = citation.lower()

            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)

        clean_text = CITATION_PATTERN.sub("", claim).strip()

        if unique_citations:
            clean_text += " " + " ".join(unique_citations)

        if len(clean_text) > 10:
            cleaned_claims.append(clean_text)

    return cleaned_claims



def get_document_keywords(documents):
    keywords = set()

    for document in documents:
        keywords.update(
            extract_keywords(document.get("text", ""))
        )

    return keywords


def check_evidence_support(claim, documents):
    """
    Check lexical support across the supplied evidence passages.

    This is a heuristic check, not semantic entailment.
    """

    clean_claim = CITATION_PATTERN.sub("", claim).strip()

    if not is_complete_sentence(clean_claim):
        return False, 0.0

    claim_keywords = extract_keywords(clean_claim)

    if not claim_keywords or not documents:
        return False, 0.0

    evidence_keywords = get_document_keywords(documents)

    common = claim_keywords.intersection(evidence_keywords)

    overlap = len(common) / len(claim_keywords)

    claim_numbers = extract_numbers(clean_claim)

    evidence_numbers = set()

    for document in documents:
        evidence_numbers.update(
            extract_numbers(document.get("text", ""))
        )

    numbers_supported = claim_numbers.issubset(
        evidence_numbers
    )

    supported = (
        len(common) >= 2
        and overlap >= 0.50
        and numbers_supported
    )

    return supported, overlap


def check_claim_support(claim, evidence_documents):
    """Check the claim against all available evidence passages."""

    supported, score = check_evidence_support(
        claim,
        evidence_documents
    )

    best_document = None
    best_score = 0.0

    claim_keywords = extract_keywords(
        CITATION_PATTERN.sub("", claim)
    )

    for document in evidence_documents:
        doc_keywords = extract_keywords(
            document.get("text", "")
        )

        if not claim_keywords:
            continue

        overlap = (
            len(claim_keywords.intersection(doc_keywords))
            / len(claim_keywords)
        )

        if overlap > best_score:
            best_score = overlap
            best_document = document

    return supported, score, best_document


def check_citations(answer, evidence_documents):
    """Check whether each claim has an exact, supporting citation."""

    claims = split_claims(answer)
    results = []

    for claim in claims:
        citations = CITATION_PATTERN.findall(claim)

        claim_supported, score, _ = check_claim_support(
            claim,
            evidence_documents
        )

        valid_citations = []

        for source_id, page in citations:
            matching_documents = [
                document
                for document in evidence_documents
                if str(document.get("source_id", "")).lower()
                == source_id.lower()
                and str(document.get("page", "")) == page
            ]

            if not matching_documents:
                continue

            citation_supported, _ = check_evidence_support(
                claim,
                matching_documents
            )

            if citation_supported:
                valid_citations.append({
                    "source_id": source_id,
                    "page": page
                })

        results.append({
            "claim": claim,
            "has_citation": bool(citations),
            "valid_citation": bool(valid_citations),
            "valid_citations": valid_citations,
            "claim_supported": claim_supported,
            "evidence_overlap": round(score, 2)
        })

    all_valid = (
        len(results) > 0
        and all(
            item["has_citation"]
            and item["valid_citation"]
            and item["claim_supported"]
            for item in results
        )
    )

    return all_valid, results



def validate_answer(question, answer, evidence_documents):
    """Validate answer claims and their source citations."""

    claims = split_claims(answer)
    claim_results = []

    for claim in claims:
        supported, overlap, best_document = check_claim_support(
            claim,
            evidence_documents
        )

        claim_results.append({
            "claim": claim,
            "supported": supported,
            "keyword_overlap": round(overlap, 2),
            "best_evidence_source": (
                best_document.get("source_id")
                if best_document else None
            ),
            "best_evidence_page": (
                best_document.get("page")
                if best_document else None
            )
        })

    all_claims_supported = (
        len(claim_results) > 0
        and all(item["supported"] for item in claim_results)
    )

    citations_valid, citation_results = check_citations(
        answer,
        evidence_documents
    )

    question_part_coverage, coverage_gaps = assess_question_part_coverage(
        question, answer, evidence_documents, citation_results
    )

    scope_gaps = check_requested_scope(
        question,
        answer,
        evidence_documents
    )

    all_supported = (
        all_claims_supported
        and citations_valid
        and not scope_gaps
        and not coverage_gaps
    )

    print("\n[Validation Debug]")
    print("Number of claims:", len(claim_results))
    print("All claims supported:", all_claims_supported)
    print("Citations valid:", citations_valid)

    for item in citation_results:
        print(
            "Citation:",
            item["has_citation"],
            "| Valid:",
            item["valid_citation"],
            "| Claim supported:",
            item["claim_supported"]
        )

    status = "SUPPORTED" if all_supported else "NOT_SUPPORTED"

    if scope_gaps or coverage_gaps:
        notes = " ".join([*scope_gaps, *coverage_gaps])
    elif all_supported:
        notes = "All claims passed the lexical evidence and citation checks."
    else:
        notes = (
            "One or more claims lack sufficient lexical evidence "
            "or a valid citation to the cited source and page."
        )

    return {
        "supported": all_supported,
        "status": status,
        "notes": notes,
        "claim_results": claim_results,
        "citation_results": citation_results,
        "citations_valid": citations_valid,
        "scope_gaps": scope_gaps,
        "question_part_coverage": question_part_coverage,
        "coverage_gaps": coverage_gaps,
    }



if __name__ == "__main__":
    import chromadb
    from pathlib import Path

    client = chromadb.PersistentClient(
        path=str(Path(__file__).parent / "chroma_db")
    )

    collection = client.get_collection(
        name="mental_health_knowledge"
    )

    results = collection.get(
        where={
            "$and": [
                {"source_id": "CDC_009"},
                {"page": 1}
            ]
        },
        include=["documents", "metadatas"]
    )

    claim = (
        "Long-term stress can worsen health problems "
        "and mental health conditions"
    )

    documents = [
        {
            "source_id": metadata["source_id"],
            "page": metadata["page"],
            "text": document
        }
        for document, metadata in zip(
            results["documents"],
            results["metadatas"]
        )
    ]

    print("Evidence support:")
    print(check_evidence_support(claim, documents))

    print("\nCitation validation:")
    answer = claim + " [CDC_009, Page 1]"
    print(check_citations(answer, documents))
