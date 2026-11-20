"""Run real local Qwen generation on five synthetic, evidence-backed questions."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import answer_generator


CASES = [
    {
        "id": "sleep_hours",
        "question": "How many hours of sleep are recommended for adults?",
        "evidence": [{
            "source_id": "CDC_008", "title": "About Sleep", "page": 1,
            "text": (
                "Adult 18–60 years: 7 or more hours. Adult 61–64 years: "
                "7–9 hours. Adult 65 years and older: 7–8 hours."
            ),
        }],
    },
    {
        "id": "stress_effects",
        "question": "What are some effects of long-term stress?",
        "evidence": [{
            "source_id": "CDC_009", "title": "Managing Stress", "page": 1,
            "text": (
                "Long-term stress can lead to worsening health problems. "
                "Stress can cause trouble concentrating and making decisions, "
                "problems sleeping, and worsening of chronic health problems "
                "and mental health conditions."
            ),
        }],
    },
    {
        "id": "anxiety_information",
        "question": "What experiences may occur with anxiety disorders?",
        "evidence": [{
            "source_id": "WHO_002", "title": "Anxiety Disorders", "page": 2,
            "text": (
                "Other symptoms of anxiety disorders may include trouble "
                "concentrating or making decisions, feeling irritable, tense "
                "or restless, trouble sleeping, nausea, heart palpitations, "
                "sweating, trembling, or shaking."
            ),
        }],
    },
    {
        "id": "wellbeing_support",
        "question": "How can relationships support mental wellbeing?",
        "evidence": [{
            "source_id": "NHS_004", "title": "5 Steps to Mental Wellbeing", "page": 1,
            "text": (
                "Good relationships are important for mental wellbeing. "
                "They can help build a sense of belonging and self-worth, "
                "provide emotional support, and allow people to share "
                "positive experiences."
            ),
        }],
    },
    {
        "id": "depression_education",
        "question": "How can depression affect everyday activities?",
        "evidence": [{
            "source_id": "NIMH_005", "title": "Depression", "page": 1,
            "text": (
                "Depression can cause severe symptoms that affect how a person "
                "feels, thinks, and handles daily activities, such as sleeping, "
                "eating, or working."
            ),
        }],
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=len(CASES))
    args = parser.parse_args()
    selected_cases = CASES[:max(1, min(args.limit, len(CASES)))]

    # Keep this integration check short while exercising the real checkpoint.
    answer_generator.MAX_NEW_TOKENS = 96
    tokenizer, model = answer_generator.load_answer_model()

    log_path = Path(__file__).resolve().parent / "execution_logs" / "model_smoke_test.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model": "Qwen2.5-1.5B-Instruct via LangChain HuggingFacePipeline",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "real_model": True,
        "cases": [],
    }

    for case in selected_cases:
        print(f"\n[{case['id']}] {case['question']}")
        try:
            answer = answer_generator.generate_answer(
                case["question"], case["evidence"], tokenizer, model
            )
            outcome = {"id": case["id"], "question": case["question"], "answer": answer}
            print(answer)
        except Exception as error:
            outcome = {
                "id": case["id"],
                "question": case["question"],
                "error": f"{type(error).__name__}: {error}",
            }
            print(outcome["error"])

        results["cases"].append(outcome)
        log_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    completed = sum("answer" in item for item in results["cases"])
    supported = sum(
        "answer" in item
        and not item["answer"].startswith("I could not find sufficient")
        for item in results["cases"]
    )
    results["completed_model_calls"] = completed
    results["answers_passing_evidence_filter"] = supported
    results["total_cases"] = len(selected_cases)
    log_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nModel calls completed: {completed}/{len(selected_cases)}")
    print(f"Answers passing evidence filter: {supported}/{len(selected_cases)}")
    print("Execution log:", log_path)


if __name__ == "__main__":
    main()
