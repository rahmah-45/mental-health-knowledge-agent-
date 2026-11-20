"""Conservative routing for sensitive and out-of-scope questions."""

import re


CRISIS_PATTERNS = (
    r"\b(suicid(?:e|al)|kill myself|end my life|self[- ]?harm|"
    r"hurt myself|overdose|can't stay safe|cannot stay safe)\b",
    r"\b(harming myself|thinking of suicide|want to die)\b",
)

CLINICAL_DECISION_PATTERNS = (
    r"\b(diagnos(?:e|is|ing)|prescri(?:be|ption)|medication dosage|"
    r"change my medication|should i take|which medication|"
    r"treatment plan|stop taking my medication|do i have|could i have|"
    r"am i suffering from)\b",
)

IN_SCOPE_TERMS = {
    "mental health", "wellbeing", "well-being", "anxiety", "depression",
    "stress", "sleep", "mood", "emotional", "coping", "panic",
    "psychological", "psychoeducation", "support", "therapy",
    "mental illness", "mental disorder", "burnout", "loneliness",
}


def classify_question(question: str) -> str:
    """Return IN_SCOPE, HUMAN_REVIEW, or OUT_OF_SCOPE."""
    text = (question or "").casefold()

    if any(re.search(pattern, text) for pattern in CRISIS_PATTERNS):
        return "CRISIS"

    if any(re.search(pattern, text) for pattern in CLINICAL_DECISION_PATTERNS):
        return "HUMAN_REVIEW"

    if any(term in text for term in IN_SCOPE_TERMS):
        return "IN_SCOPE"

    return "OUT_OF_SCOPE"
