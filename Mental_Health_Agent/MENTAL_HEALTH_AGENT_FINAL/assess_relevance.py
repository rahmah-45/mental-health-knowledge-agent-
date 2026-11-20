
import re
import time
import torch
from pathlib import Path
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen2.5-1.5B-Instruct"
    / "snapshots"
    / "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
)

MAX_CONTEXT_LENGTH = 2048

STOP_WORDS = {
    "what", "are", "the", "is", "a", "an", "of", "for",
    "to", "do", "does", "how", "many", "much", "can",
    "could", "should", "would", "in", "on", "and", "or",
    "with", "about", "from", "need", "recommended",
    "their", "they", "this", "that", "these", "those",
    "your", "you", "it", "be", "has", "have", "had",
    "will", "may", "might", "also", "more", "most",
    "using", "into", "than", "both", "each"
}

LABELS = {
    "RELEVANT",
    "PARTIALLY_RELEVANT",
    "IRRELEVANT"
}

STRESS_MANAGEMENT_INTENT = re.compile(
    r"\b(stress|stressed)\b.*\b(manag\w*|cop\w*|reliev\w*|"
    r"reduc\w*|strateg\w*|techniqu\w*|ways|steps|tips)\b|"
    r"\b(manag\w*|cop\w*|reliev\w*|reduc\w*|strateg\w*|"
    r"techniqu\w*|ways|steps|tips)\b.*\b(stress|stressed)\b",
    re.IGNORECASE,
)
STRESS_MANAGEMENT_VERB = re.compile(
    r"\b(?:cop\w*|reduc\w*|reliev\w*|manag\w*)\b",
    re.IGNORECASE,
)
STRESS_MANAGEMENT_CUE = re.compile(
    r"\b(?:can|could|help|helps|by|through|daily|steps?|ways?|"
    r"strategies|techniques|skills|practice|using|include|including)\b",
    re.IGNORECASE,
)
CONCRETE_COPING_METHOD = re.compile(
    r"\b(?:taking breaks|take breaks|make time to unwind|deep breaths|"
    r"practice gratitude|keep a journal|keeping a journal|spend time outdoors|"
    r"connect with others|meditat\w*|mindfulness|take care of your mind)\b",
    re.IGNORECASE,
)


def has_stress_management_evidence(question, passage):
    """Require a stress-management request and an actionable coping passage."""
    text = passage or ""
    if not STRESS_MANAGEMENT_INTENT.search(question or ""):
        return False
    if not re.search(r"\b(?:stress|stressed)\b", text, re.I):
        return False

    clauses = re.split(r"[.!?;\n]+", text)
    has_actionable_clause = any(
        STRESS_MANAGEMENT_VERB.search(clause)
        and re.search(r"\b(?:stress|stressed)\b", clause, re.I)
        and STRESS_MANAGEMENT_CUE.search(clause)
        for clause in clauses
    )
    has_specific_method = bool(CONCRETE_COPING_METHOD.search(text))
    return has_actionable_clause or has_specific_method


@lru_cache(maxsize=1)
def load_model():

    print("Loading Qwen relevance model...")

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

    print("Relevance model loaded successfully.")

    return tokenizer, model


def normalize_word(word):

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

    words = re.findall(r"[a-zA-Z]+", text.lower())

    return {
        normalize_word(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    }


def lexical_relevance_check(question, retrieved_text):

    question_keywords = extract_keywords(question)
    text_keywords = extract_keywords(retrieved_text)

    matching_keywords = question_keywords.intersection(
        text_keywords
    )

    return len(matching_keywords), sorted(matching_keywords)


def _assessment_prompt(question, retrieved_text):
    prompt = f"""
You are an evidence relevance evaluator.

Evaluate whether the retrieved passage helps answer
the user's question.

Use these definitions:

RELEVANT:
The passage directly provides evidence answering
the question.

PARTIALLY_RELEVANT:
The passage provides useful evidence for at least
one meaningful part of the question.

IRRELEVANT:
The passage does not provide useful evidence
for the question.

Do not judge by topic similarity alone.
Judge the actual information in the passage.
Use only the supplied text.

Return exactly one label:
RELEVANT
PARTIALLY_RELEVANT
IRRELEVANT

Question:
{question}

Retrieved passage:
{retrieved_text}

Label:
"""

    return prompt


def _parse_assessment_label(generated_text):
    response = str(generated_text or "").strip().upper()
    for line in response.splitlines():
        label = line.strip().strip(" .,:;*`_")
        if label in LABELS:
            return label
    return "UNCERTAIN"


def assess_with_qwen_batch(question, retrieved_texts, tokenizer, model):
    """Classify every passage in one padded generation batch.

    Invalid, missing, or failed outputs remain UNCERTAIN and are rejected by
    the evidence filter. Each returned label keeps the input passage order.
    """
    texts = list(retrieved_texts or [])
    if not texts:
        return []

    started = time.perf_counter()
    model_generate_calls = 0
    labels = ["UNCERTAIN"] * len(texts)
    original_padding_side = getattr(tokenizer, "padding_side", None)
    original_pad_token = getattr(tokenizer, "pad_token", None)

    try:
        prompts = []
        for text in texts:
            messages = [
                {"role": "system", "content": "You are a careful evidence evaluator."},
                {"role": "user", "content": _assessment_prompt(question, text or "")},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ))
        if original_padding_side is not None:
            tokenizer.padding_side = "left"
        if original_pad_token is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_CONTEXT_LENGTH,
        )
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
        if original_pad_token is None and getattr(tokenizer, "pad_token", None):
            tokenizer.pad_token = original_pad_token

        model_generate_calls = 1
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )

        sequences = getattr(outputs, "sequences", outputs)
        input_ids = inputs["input_ids"]
        input_width = input_ids.shape[1]
        parsed = []
        for row in sequences:
            generated_tokens = row[input_width:]
            decoded = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            parsed.append(_parse_assessment_label(decoded))
        labels[:min(len(labels), len(parsed))] = parsed[:len(labels)]
    except Exception as error:
        # A failed batch must never turn into accepted evidence.
        print("Qwen relevance batch failed; marking outputs UNCERTAIN:", repr(error))
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
        if original_pad_token is None and getattr(tokenizer, "pad_token", None):
            tokenizer.pad_token = original_pad_token
        elapsed = time.perf_counter() - started
        print(
            "Relevance batch:",
            f"passages={len(texts)}",
            f"model.generate calls={model_generate_calls}",
            f"elapsed_seconds={elapsed:.3f}",
        )
    return labels


def assess_with_qwen(question, retrieved_text, tokenizer, model):
    """Compatibility wrapper for assessing one passage."""
    labels = assess_with_qwen_batch(
        question,
        [retrieved_text],
        tokenizer,
        model,
    )
    return labels[0] if labels else "UNCERTAIN"


def _apply_lexical_fallback(question, retrieved_text, qwen_result):
    match_count, matching_keywords = lexical_relevance_check(
        question,
        retrieved_text,
    )
    print("Matching keywords:", matching_keywords)

    if qwen_result == "IRRELEVANT" and (
        match_count >= 2
        or has_stress_management_evidence(question, retrieved_text)
    ):
        print("Qwen rejected the passage, but direct evidence supports a partial match.")
        return "PARTIALLY_RELEVANT"
    return qwen_result if qwen_result in LABELS else "UNCERTAIN"


def assess_relevance_batch(question, retrieved_texts, tokenizer=None, model=None):
    """Assess each passage independently while sharing one Qwen batch call."""
    texts = list(retrieved_texts or [])
    if not texts:
        print("Relevance batch: passages=0, model.generate calls=0, elapsed_seconds=0.000")
        return []
    if tokenizer is None or model is None:
        tokenizer, model = load_model()
    qwen_results = assess_with_qwen_batch(question, texts, tokenizer, model)
    if len(qwen_results) < len(texts):
        qwen_results.extend(["UNCERTAIN"] * (len(texts) - len(qwen_results)))
    return [
        _apply_lexical_fallback(question, text, qwen_results[index])
        for index, text in enumerate(texts)
    ]


def assess_relevance(
    question,
    retrieved_text,
    tokenizer=None,
    model=None
):

    if tokenizer is None or model is None:
        tokenizer, model = load_model()

    qwen_result = assess_with_qwen(
        question,
        retrieved_text,
        tokenizer,
        model
    )

    return _apply_lexical_fallback(question, retrieved_text, qwen_result)


def main():

    print("=" * 60)
    print("EVIDENCE RELEVANCE ASSESSMENT TEST")
    print("=" * 60)

    tokenizer, model = load_model()

    test_cases = [
        {
            "question": "How do sleep and stress affect mental health?",
            "source": "CDC_008",
            "text": (
                "Good sleep is essential for health and emotional "
                "well-being. Getting enough sleep can reduce stress "
                "and improve mood."
            ),
            "expected": "RELEVANT"
        },
        {
            "question": "How do sleep and stress affect mental health?",
            "source": "CDC_009",
            "text": (
                "Long-term stress can lead to worsening health "
                "problems, including worsening mental health "
                "conditions and problems sleeping."
            ),
            "expected": "RELEVANT"
        },
        {
            "question": "How do sleep and stress affect mental health?",
            "source": "WHO_010",
            "text": (
                "Healthy sleep patterns, regular exercise, "
                "coping skills and supportive environments "
                "are important for adolescent mental well-being."
            ),
            "expected": "PARTIALLY_RELEVANT"
        },
        {
            "question": "How do sleep and stress affect mental health?",
            "source": "WHO_002",
            "text": (
                "Anxiety disorders may include excessive worry, "
                "difficulty concentrating and physical tension."
            ),
            "expected": "IRRELEVANT"
        }
    ]

    correct = 0

    for index, item in enumerate(test_cases, start=1):

        print("\n" + "-" * 60)
        print("Test:", index)
        print("Source:", item["source"])
        print("Expected:", item["expected"])

        result = assess_relevance(
            item["question"],
            item["text"],
            tokenizer,
            model
        )

        print("Actual:", result)

        if result == item["expected"]:
            print("Test result: PASS")
            correct += 1
        else:
            print("Test result: REVIEW")

    print("\n" + "=" * 60)
    print("ASSESSMENT SUMMARY")
    print("=" * 60)
    print("Total examples:", len(test_cases))
    print("Correct classifications:", correct)
    print("Incorrect classifications:", len(test_cases) - correct)


if __name__ == "__main__":
    main()
