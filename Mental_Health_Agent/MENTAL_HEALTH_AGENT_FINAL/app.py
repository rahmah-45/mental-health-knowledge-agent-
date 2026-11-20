"""Streamlit interface for the existing Mental Health Knowledge Agent."""

import re

import streamlit as st

from mental_health_agent import run_agent


st.set_page_config(
    page_title="Mental Health Knowledge Agent",
    page_icon="💙",
    layout="wide",
)

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp { background: #f4f8fc; color: #17324d; }
      [data-testid="stHeader"] { background: rgba(244,248,252,.92); }
      .block-container { max-width: 1120px; padding-top: 2rem; }
      .mh-hero {
        padding: 1.5rem 1.7rem; border-radius: 18px;
        background: linear-gradient(120deg, #e7f2ff, #ffffff);
        border: 1px solid #d6e6f5; margin-bottom: 1.25rem;
      }
      .mh-hero h1 { color: #174a78; margin: 0 0 .35rem 0; }
      .mh-hero p { color: #48657f; margin: 0; }
      [data-testid="stForm"] {
        background: #fff; border: 1px solid #dbe7f1;
        border-radius: 16px; padding: 1.2rem;
      }
      [data-testid="stExpander"] {
        background: #fff; border: 1px solid #dbe7f1;
        border-radius: 12px; margin-bottom: .5rem;
      }
      div.stButton > button, div.stFormSubmitButton > button {
        background: #1769aa; color: #fff; border: 0; border-radius: 9px;
      }
      div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: #12568e; color: #fff;
      }
      .mh-muted { color: #536d83; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _interrupt_payload(result):
    interruptions = result.get("__interrupt__", ())
    if not interruptions:
        return None
    first = interruptions[0]
    return getattr(first, "value", None)


def _deduplicated_passages(result):
    passages = result.get("assessment_results") or result.get("retrieved_documents") or []
    unique = {}
    for passage in passages:
        text = passage.get("text", "") or ""
        key = (
            str(passage.get("source_id", "")).casefold(),
            str(passage.get("page", "")),
            " ".join(text.split()).casefold(),
        )
        if key not in unique:
            unique[key] = dict(passage)
            continue

        existing = unique[key]
        relevance_by_part = dict(existing.get("relevance_by_part", {}))
        existing.update(passage)
        relevance_by_part.update(passage.get("relevance_by_part", {}))
        if relevance_by_part:
            existing["relevance_by_part"] = relevance_by_part
    return list(unique.values())


def _citations(result, answer):
    citations = {}
    for item in result.get("citation_results", []):
        for citation in item.get("valid_citations", []):
            key = (str(citation.get("source_id", "")), str(citation.get("page", "")))
            citations[key] = True

    for source_id, page in re.findall(
        r"\[([A-Za-z0-9_-]+)\s*,\s*Page\s+(\d+)\]", answer or "", re.I
    ):
        citations.setdefault((source_id, page), False)
    return citations


def _references(result):
    seen = set()
    references = []
    for document in result.get("relevant_documents", []):
        key = (
            str(document.get("source_id", "")),
            str(document.get("title", "")),
            str(document.get("page", "")),
        )
        if key not in seen:
            seen.add(key)
            references.append(key)
    return references


def _render_passage(passage, index):
    st.markdown(f"**Passage {index}: {passage.get('title', 'Unknown source')}**")
    st.write("Source ID:", passage.get("source_id", "Unknown"))
    st.write("Page:", passage.get("page", "Unknown"))
    st.write("Relevance:", passage.get("relevance", "NOT ASSESSED"))
    if passage.get("relevance_by_part"):
        st.write("Relevance by question part:", passage["relevance_by_part"])
    st.text(passage.get("text", "") or "(No passage text available.)")


def render_result(record):
    question = record["question"]
    result = record.get("result") or {}
    interrupt = record.get("interrupt")

    st.markdown(f"#### You asked\n{question}")

    if record.get("error"):
        st.error(record["error"])
        return

    pending_review = bool(
        result.get("requires_human_review") or interrupt
    )
    answer = result.get("final_response", "") or ""
    if pending_review and interrupt:
        answer = interrupt.get("suggested_action", "") or answer
    elif not answer:
        answer = result.get("answer", "") or "No response was returned."

    st.markdown("#### Educational response")
    st.markdown(answer)

    validation_ran = bool(
        result.get("validation_notes")
        or result.get("claim_results")
        or result.get("citation_results")
    )
    if pending_review:
        st.warning("PENDING HUMAN REVIEW")
        st.caption(
            "The app has not notified a clinician and cannot confirm that a clinician has reviewed this case."
        )
        if result.get("human_review_reason"):
            st.write("Reason:", result["human_review_reason"])
        st.caption("Human-review status: PENDING")
    elif validation_ran:
        if result.get("answer_supported"):
            st.success("Validation: SUPPORTED")
        else:
            st.error("Validation: NOT SUPPORTED")
    else:
        st.info("Validation: NOT ASSESSED for this route.")

    if not pending_review:
        st.caption(
            "Human-review status: "
            + str(result.get("human_review_status", "NOT_REQUIRED"))
        )

    if result.get("validation_notes"):
        st.caption("Validation notes: " + result["validation_notes"])

    passages = _deduplicated_passages(result)
    supporting_passages = [
        passage for passage in passages
        if str(passage.get("relevance", "")).upper()
        in {"RELEVANT", "PARTIALLY_RELEVANT"}
    ]
    rejected_passages = [
        passage for passage in passages if passage not in supporting_passages
    ]

    st.markdown("### Supporting evidence")
    if supporting_passages:
        for index, passage in enumerate(supporting_passages, start=1):
            with st.container(border=True):
                _render_passage(passage, index)
    else:
        st.info("No passages were accepted as supporting evidence.")

    with st.expander(
        f"Retrieved but rejected or uncertain passages ({len(rejected_passages)})",
        expanded=False,
    ):
        if rejected_passages:
            st.caption(
                "These passages were retrieved but were not used as supporting evidence."
            )
            for index, passage in enumerate(rejected_passages, start=1):
                with st.container(border=True):
                    _render_passage(passage, index)
        else:
            st.write("No rejected or uncertain passages were retrieved.")

    with st.expander("Citations and references", expanded=False):
        citations = _citations(result, answer)
        if citations:
            st.markdown("**Citations in the answer**")
            for (source_id, page), valid in citations.items():
                status = "validated" if valid else "not validated"
                st.write(f"[{source_id}, Page {page}] — {status}")
        else:
            st.write("No citations were returned.")

        references = _references(result)
        st.markdown("**Deduplicated references**")
        if references:
            for source_id, title, page in references:
                st.write(f"{source_id} — {title}, page {page}")
        else:
            st.write("No retrieved references are available for this response.")

    with st.expander("Question-part coverage", expanded=False):
        coverage = result.get("question_part_coverage", [])
        if coverage:
            for item in coverage:
                label = "Covered" if item.get("supported") else "Not supported"
                st.write(f"**{label}:** {item.get('part', '')}")
        else:
            st.write("Question-part coverage was not assessed.")

    with st.expander("Evidence gaps", expanded=False):
        gaps = result.get("evidence_gaps", [])
        if gaps:
            for gap in gaps:
                st.write("•", gap)
        else:
            st.write("No evidence gaps were reported." if validation_ran else "Evidence gaps were not assessed for this route.")

    st.caption(f"Corrective retrieval attempts: {result.get('retry_count', 0)}")


st.markdown(
    """
    <div class="mh-hero">
      <h1>Mental Health Knowledge &amp; Psychoeducation</h1>
      <p>Evidence-based mental health education grounded in the project knowledge base.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="mh-muted">This tool provides general education, not diagnosis or individualized treatment.</p>',
    unsafe_allow_html=True,
)

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

with st.form("question_form", clear_on_submit=True):
    question = st.text_area(
        "What would you like to learn about?",
        placeholder="For example: How can sleep support emotional well-being?",
        height=100,
        max_chars=2000,
        key="question_input",
    )
    submitted = st.form_submit_button("Ask the agent", use_container_width=True)

if submitted:
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        st.warning("Enter a question before submitting.")
    else:
        try:
            with st.spinner("Searching the knowledge base and validating evidence…"):
                result = run_agent(cleaned_question)
            interrupt = _interrupt_payload(result)
            # Store plain result data and the interruption payload, rather than
            # retaining the LangGraph Interrupt wrapper in browser session state.
            stored_result = {
                key: value for key, value in result.items() if key != "__interrupt__"
            }
            st.session_state.conversation_history.append({
                "question": cleaned_question,
                "result": stored_result,
                "interrupt": interrupt,
                "error": "",
            })
        except Exception as exc:
            st.session_state.conversation_history.append({
                "question": cleaned_question,
                "result": {},
                "interrupt": None,
                "error": (
                    "The agent could not complete this request. Check that the local "
                    "model files and Chroma database are available, then review the "
                    f"server log. Error: {type(exc).__name__}: {exc}"
                ),
            })

if st.session_state.conversation_history:
    st.markdown("### Conversation")
    for record in st.session_state.conversation_history:
        with st.container(border=True):
            render_result(record)
