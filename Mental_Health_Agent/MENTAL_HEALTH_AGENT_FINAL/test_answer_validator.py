
from answer_validator import validate_answer
from answer_validator import split_question_parts
from answer_generator import add_source_citations


def test_supported_claim_requires_exact_source_and_page():
    evidence = [{
        "source_id": "CDC_008",
        "page": 1,
        "text": "Adults aged 18 to 60 years need 7 or more hours of sleep.",
    }]
    result = validate_answer(
        "How much sleep do adults need?",
        "Adults aged 18 to 60 years need 7 or more hours of sleep. "
        "[CDC_008, Page 1]",
        evidence,
    )
    assert result["supported"] is True


def test_wrong_page_citation_is_rejected():
    evidence = [{
        "source_id": "CDC_008",
        "page": 1,
        "text": "Adults aged 18 to 60 years need 7 or more hours of sleep.",
    }]
    result = validate_answer(
        "How much sleep do adults need?",
        "Adults aged 18 to 60 years need 7 or more hours of sleep. "
        "[CDC_008, Page 2]",
        evidence,
    )
    assert result["supported"] is False


def test_unsupported_numeric_claim_is_rejected():
    evidence = [{
        "source_id": "CDC_008",
        "page": 1,
        "text": "Adults aged 18 to 60 years need 7 or more hours of sleep.",
    }]
    result = validate_answer(
        "How much sleep do adults need?",
        "Adults need 12 hours of sleep. [CDC_008, Page 1]",
        evidence,
    )
    assert result["supported"] is False


def test_incomplete_answer_claim_is_rejected_even_with_matching_evidence():
    evidence = [{
        "source_id": "CDC_009",
        "page": 1,
        "text": (
            "Everyone manages stress differently. You can find and manage "
            "what triggers your stress and the right combination of healthy "
            "techniques that work for you."
        ),
    }]
    result = validate_answer(
        "What are some evidence-based ways to manage stress?",
        "You can find and manage what triggers your stress and the right "
        "combination of healthy techniques that. [CDC_009, Page 1]",
        evidence,
    )
    assert result["supported"] is False
    assert result["claim_results"][0]["supported"] is False


def test_pakistan_prevalence_rejects_global_statistics():
    evidence = [{
        "source_id": "WHO_002",
        "page": 1,
        "text": (
            "An estimated 5.8% of the global population currently "
            "experience an anxiety disorder. In 2023, 470 million people "
            "in the world had an anxiety disorder."
        ),
    }]
    result = validate_answer(
        "What percentage of people in Pakistan have an anxiety disorder?",
        "An estimated 5.8% of the global population currently experience "
        "an anxiety disorder. [WHO_002, Page 1]",
        evidence,
    )

    assert result["supported"] is False
    assert result["status"] == "NOT_SUPPORTED"
    assert any(
        "Pakistan-specific prevalence data was not found" in gap
        and "global statistics" in gap
        for gap in result["scope_gaps"]
    )


def test_global_prevalence_question_accepts_global_statistics():
    evidence = [{
        "source_id": "WHO_002",
        "page": 1,
        "text": (
            "An estimated 5.8% of the global population currently "
            "experience an anxiety disorder."
        ),
    }]
    result = validate_answer(
        "What is the global prevalence of anxiety disorders?",
        "An estimated 5.8% of the global population currently experience "
        "an anxiety disorder. [WHO_002, Page 1]",
        evidence,
    )

    assert result["supported"] is True
    assert result["status"] == "SUPPORTED"
    assert result["scope_gaps"] == []


def test_population_specific_prevalence_rejects_unscoped_global_data():
    evidence = [{
        "source_id": "WHO_002",
        "page": 1,
        "text": (
            "An estimated 5.8% of the global population currently "
            "experience an anxiety disorder."
        ),
    }]
    result = validate_answer(
        "What percentage of adults worldwide experience anxiety disorders?",
        "An estimated 5.8% of the global population currently experience "
        "an anxiety disorder. [WHO_002, Page 1]",
        evidence,
    )

    assert result["supported"] is False
    assert any(
        gap.startswith("Population scope mismatch:")
        for gap in result["scope_gaps"]
    )


SLEEP_STRESS_QUESTION = (
    "How can adequate sleep support emotional well-being, and what effects "
    "can long-term stress have on mental health?"
)


def multipart_evidence():
    return [
        {
            "source_id": "CDC_008", "title": "About Sleep", "page": 1,
            "text": "Adequate sleep supports emotional well-being and mental health."
        },
        {
            "source_id": "CDC_009", "title": "Stress", "page": 2,
            "text": "Long-term stress can worsen mental health conditions and affect emotional well-being."
        },
    ]


def test_exact_sleep_and_stress_question_has_two_identified_parts():
    assert len(split_question_parts(SLEEP_STRESS_QUESTION)) == 2


def test_multipart_answer_is_rejected_when_only_one_part_is_supported():
    evidence = [multipart_evidence()[1]]
    answer = "Long-term stress can worsen mental health conditions. [CDC_009, Page 2]"
    result = validate_answer(SLEEP_STRESS_QUESTION, answer, evidence)
    assert result["supported"] is False
    assert len(result["coverage_gaps"]) == 1
    assert any("adequate sleep" in gap for gap in result["coverage_gaps"])


def test_multipart_answer_passes_when_every_part_has_cited_evidence():
    evidence = multipart_evidence()
    answer = (
        "Adequate sleep supports emotional well-being. [CDC_008, Page 1]\n"
        "Long-term stress can worsen mental health conditions. [CDC_009, Page 2]"
    )
    result = validate_answer(SLEEP_STRESS_QUESTION, answer, evidence)
    assert result["supported"] is True
    assert result["coverage_gaps"] == []
    assert [part["supported"] for part in result["question_part_coverage"]] == [True, True]


def test_multipart_citations_are_attached_to_supported_sentences():
    evidence = multipart_evidence()
    answer, unsupported = add_source_citations(
        "Adequate sleep supports emotional well-being.\n"
        "Long-term stress can worsen mental health conditions.", evidence
    )
    assert unsupported == []
    assert answer.count("[CDC_008, Page 1]") == 1
    assert answer.count("[CDC_009, Page 2]") == 1
    assert "[CDC_008, Page 1] [CDC_009" not in answer
    assert validate_answer(SLEEP_STRESS_QUESTION, answer, evidence)["supported"] is True


def main():
    question = "What are the recommended sleep hours for adults?"

    evidence_documents = [
        {
            "source_id": "CDC_008",
            "page": 1,
            "text": (
                "Adults aged 18–60 years need 7 or more hours "
                "of sleep. Adults aged 61–64 years need 7–9 hours. "
                "Adults aged 65 years and older need 7–8 hours."
            )
        }
    ]

    test_cases = [
        {
            "name": "Supported answer with valid citation",
            "answer": (
                "Adults aged 18–60 years need 7 or more hours "
                "of sleep. [CDC_008, Page 1]"
            ),
            "expected": "SUPPORTED"
        },
        {
            "name": "Supported information without citation",
            "answer": (
                "Adults aged 18–60 years need 7 or more hours "
                "of sleep."
            ),
            "expected": "NOT_SUPPORTED"
        },
        {
            "name": "Unsupported claim with citation",
            "answer": (
                "Maintaining a regular sleep schedule improves "
                "memory. [CDC_008, Page 1]"
            ),
            "expected": "NOT_SUPPORTED"
        }
    ]

    passed = 0

    for test in test_cases:
        print("\n" + "=" * 60)
        print("TEST:", test["name"])
        print("=" * 60)

        result = validate_answer(
            question,
            test["answer"],
            evidence_documents
        )

        print("Expected:", test["expected"])
        print("Actual:", result["status"])
        print("Notes:", result["notes"])
        print("Claim results:", result["claim_results"])
        print("Citation results:", result["citation_results"])
        print("Citations valid:", result["citations_valid"])

        if result["status"] == test["expected"]:
            print("Test result: PASS")
            passed += 1
        else:
            print("Test result: FAIL")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print("Total tests:", len(test_cases))
    print("Passed:", passed)
    print("Failed:", len(test_cases) - passed)


if __name__ == "__main__":
    main()

from answer_validator import split_claims

print("\n" + "=" * 60)
print("TEST: Splitting multiple claims")
print("=" * 60)

sample_answer = """
Teenagers can protect their emotional well-being.
- They should maintain healthy sleep patterns.
- They should exercise regularly.
- They should develop coping and interpersonal skills.
"""

claims = split_claims(sample_answer)

print("Number of claims:", len(claims))

for index, claim in enumerate(claims, start=1):
    print(f"Claim {index}: {claim}")

if len(claims) == 4:
    print("Test result: PASS")
else:
    print("Test result: FAIL")

from answer_validator import split_claims

print("\n" + "=" * 60)
print("TEST: Splitting a paragraph with citations")
print("=" * 60)

sample = (
    "Teenagers can protect their emotional well-being "
    "through healthy habits. [WHO_010, Page 1] "
    "They should maintain healthy sleep patterns and "
    "exercise regularly. [WHO_010, Page 2] "
    "Supportive environments are also important. "
    "[WHO_010, Page 2] "
    "Early treatment is always necessary."
)

claims = split_claims(sample)

for i, claim in enumerate(claims, start=1):
    print(f"Claim {i}: {claim}")

print("Number of claims:", len(claims))
print("Test result:", "PASS" if len(claims) == 4 else "FAIL")
