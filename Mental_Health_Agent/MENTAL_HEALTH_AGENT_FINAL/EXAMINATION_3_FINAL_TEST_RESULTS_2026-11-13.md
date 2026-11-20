# Examination 3 — Final End-to-End Test Results

**Evaluation date:** 2026-11-13  
**Workflow:** Real LangGraph `run_agent` using local Qwen and the existing Chroma database. No mocks, ingestion, or database rebuild were used. The saved stress-management result was reused; that scenario was not rerun. The remaining cases had a 90-second per-case limit.

| Scenario and question | Result | Actual result |
|---|---|---|
| Relevant information — “What are some evidence-based ways to manage stress?” | **PASS** | Saved run classified `IN_SCOPE`. Retrieved CDC **Managing Stress** (CDC_009, p. 1) was relevant/partially relevant. The final answer gave six evidence-based suggestions with CDC_009 p. 1 citations; all six claims passed validation. Retry: 0. Human review: not required. |
| Multiple documents — “How can adequate sleep support emotional well-being, and what effects can long-term stress have on mental health?” | **PARTIAL** | Timed out at 90 seconds during answer generation. Retrieval completed with 10 passages, including CDC **About Sleep** (CDC_008, pp. 1–2), CDC **Managing Stress** (CDC_009, pp. 1–3), CDC **About Mental Health** (CDC_006, pp. 1, 2, 4), and WHO **Mental Health** (WHO_001, p. 2). Three passages were labeled relevant and six partially relevant. No final answer or citations were produced; validation did not run. Retry: 0. Human review: not required. |
| Initially irrelevant retrieval — “What is the latitude of anxiety?” | **PARTIAL** | Classified `IN_SCOPE`. The initial five passages and the five after corrective retrieval were all labeled irrelevant; the retry did not improve retrieval. The agent abstained: “I could not find sufficient evidence in the available knowledge base to answer this question.” No citations; validation did not run. Retry: 1. Human review: not required. |
| Insufficient evidence — “What percentage of people in Pakistan have an anxiety disorder?” | **PASS** | Classified `IN_SCOPE`. WHO **Anxiety Disorders** (WHO_002, pp. 1–2) and WHO **Mental Health of Adolescents** (WHO_010, p. 2) were retrieved, including global statistics. Geographic-scope validation rejected them as support for Pakistan-specific prevalence. The agent abstained and recorded that Pakistan-specific data was not found; it did not present the global figures as the answer. Final-answer citations: none. Internal validation reference: WHO_002 p. 1. Status: `NOT_SUPPORTED`. Retry: 0. Human review: not required. |
| Out of scope — “What is the current population of Islamabad?” | **PASS** | Classified `OUT_OF_SCOPE`; no sources retrieved. The agent declined: “This question is outside the mental health education topics covered by the available knowledge base.” Validation did not run. Retry: 0. Human review: not required. |

## Summary

**3 PASS, 2 PARTIAL, 0 FAIL.** The multi-document case did not finish before its timeout. Corrective retrieval ran in the initially irrelevant case but returned the same irrelevant evidence, so that case remains PARTIAL. These results do not establish that the unfinished multi-part answer works end to end.

Detailed current-run results for the four newly executed cases: `execution_logs/examination3_remaining_e2e_evaluation_2026-11-13.json`. The first scenario’s saved result remains in `execution_logs/examination3_final_e2e_evaluation_2026-11-13.json`.
