# Mental Health Knowledge and Psychoeducation Agent

This repository implements **Examination 3** from the Agentic AI midterm: an evidence-grounded mental health education assistant using RAG and LangGraph. It retrieves the five highest-ranked passages from a local knowledge base, assesses relevance, generates concise educational answers with page citations, validates those claims, and retries retrieval once when needed.

The assistant is educational only. It does not diagnose, prescribe, or make personal treatment decisions. Requests about medication, diagnosis, or treatment pause the LangGraph workflow until a reviewer responds. Crisis language is routed away from the language model and shown to the reviewer with a brief emergency-support message. Other topics outside the knowledge-base scope are declined.

## Requirements

- Python 3.10 or later
- CPU or compatible GPU; the checked-in settings use CPU
- The local Qwen2.5-1.5B-Instruct model for generation and relevance assessment
- The `sentence-transformers/all-MiniLM-L6-v2` embedding model
- The embedding model and Qwen checkpoint must be present in the local Hugging Face cache; the agent runs models in local-only mode

## Setup

From the repository root, create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `knowledge_base/` folder contains 13 supplied PDF files. **Ten documents are registered in `source_metadata.json` and form the active knowledge base; three additional PDFs are present in the folder but are not registered or processed by the current pipeline.** The metadata file lists the registered sources' IDs, titles, organizations, dates, and source URLs.

Before running the ingestion pipeline or asking questions, make sure both model checkpoints are already available in the local Hugging Face cache:

- `Qwen/Qwen2.5-1.5B-Instruct` for answer generation and relevance assessment.
- `sentence-transformers/all-MiniLM-L6-v2` for embeddings.

The project uses local-only model loading and does not download model files automatically at runtime. A first-time setup therefore requires internet access (or separately supplied model files) to populate the Hugging Face cache. The GitHub-ready package intentionally excludes the generated `chroma_db/` folder. After downloading the required model checkpoints, build the vector database by running the ingestion commands below.

## Build the knowledge base

If you are reviewing the supplied project, first check whether `chroma_db/` is already present. If it is present and corresponds to the supplied registered sources and embedding model, you can try running the agent without rebuilding it. Rebuild only when needed.

Run these steps in order after adding or changing source PDFs:

```powershell
python clean_text.py
python chunk_documents.py
python build_vector_db.py
```

This extracts page text using LangChain's `PyPDFLoader`, cleans it, splits it with LangChain's recursive text splitter, generates normalized embeddings, and writes page-aware chunks to ChromaDB. The agent uses a LangChain `BaseRetriever` adapter over the persisted Chroma collection. The generated collection is stored locally in `chroma_db/`.

## Ask a question

```powershell
python mental_health_agent.py
```

Or provide a question directly:

```powershell
python mental_health_agent.py --question "How can stress affect sleep?"
```

The output includes the response, references, retrieval/validation details, and whether human review is required. The command-line trace is the local execution log. To enable LangSmith traces, set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in the environment before starting the program; do not commit credentials.

For a human-review case, the command line displays the question and reason, then pauses for an operator response. Answer `yes` or `no` to resume the graph and record the review status. This is a prototype operator acknowledgment only: it does not verify the reviewer's qualifications, contact a clinician, or provide clinical or emergency-service review.

To exercise actual local Qwen generation with five synthetic evidence-backed questions and save the outputs:

```powershell
python run_model_smoke_test.py
```

The run writes a sanitized synthetic execution log to `execution_logs/model_smoke_test.json`. A sanitized trace summary from a full local RAG run is in `execution_logs/full_rag_smoke_test.json`. Use `--limit 1` for a quick model integration check.

## Run the requirement scenarios

```powershell
python -m pytest -q
```

`test_agent_scenarios.py` uses synthetic evidence and injected model results, so workflow tests do not load multi-gigabyte models. It covers relevant evidence, multiple sources, corrective retrieval, insufficient evidence, out-of-scope questions, clinical decisions, crisis routing, human-review interruption, and Chroma metadata preservation. `test_answer_validator.py` covers citations and unsupported claims. `pytest.ini` limits automatic collection to these model-free suites. To test actual local Qwen generation on five synthetic questions, run `python run_model_smoke_test.py`.

### Final end-to-end evaluation (2026-11-13)

The final evaluation used the real local Qwen model, the existing Chroma database, and the LangGraph workflow. It recorded **3 PASS and 2 PARTIAL**:

- **PASS:** The stress-management question returned six supported claims cited to CDC Managing Stress, page 1. The Pakistan-specific anxiety prevalence question was safely answered with an abstention because the retrieved figures were global, and the out-of-scope Islamabad population question was declined without retrieval.
- **PARTIAL:** The multi-source sleep-and-stress question retrieved evidence but timed out after 90 seconds during answer generation; it produced no final answer and was not validated.
- **PARTIAL:** In the initially irrelevant retrieval case, the corrective retrieval returned the same five irrelevant passages. The agent abstained, so retrieval was not improved.

The separate full-pipeline question about the recommended number of adult sleep hours completed RAG with a supported answer. It is distinct from the multi-source sleep-and-stress question that timed out. Detailed results are in `EXAMINATION_3_FINAL_TEST_RESULTS_2026-11-13.md` and `execution_logs/examination3_remaining_e2e_evaluation_2026-11-13.json`.

## Project files

- `knowledge_base/`: 13 source PDFs; 10 are registered and active in the current knowledge base, while 3 are not registered or processed.
- `source_metadata.json`: source register and URLs.
- `clean_text.py`, `chunk_documents.py`, `build_vector_db.py`: ingestion pipeline.
- `evidence_retriever.py`, `assess_relevance.py`: LangChain retrieval and Corrective RAG assessment.
- `answer_generator.py`, `answer_validator.py`: LangChain model call, source citations, and reflection check.
- `mental_health_agent.py`, `agent_nodes.py`, `agent_state.py`, `scope_router.py`: LangGraph workflow, interrupt-based human review, routing, and typed state.
- `run_model_smoke_test.py`: manual five-question test through the real local Qwen generation chain.
- `create_submission_zip.py`: recreates and integrity-checks the submission archive.
- `architecture.md`: workflow and component diagram.
- `REPORT/`: contains the final PDF technical report for submission.
- `test_cases.json`: synthetic test inputs and expected outcomes.

## Safety and limitations

Source-grounding is checked with lexical overlap and citation matching; this is a transparent heuristic, not a clinical or semantic entailment guarantee. Human review is represented by a review-required state and response; the command-line prototype does not connect to a clinician or emergency service. Do not enter identifiable patient information. Retrieved sources are educational references and may not cover every question.


## Portability note for reviewers

The repository is provided as source code and local project data, not as a self-contained installer. Create a new virtual environment on the reviewer's computer and install `requirements.txt`; do not copy the developer's `.venv/`, `__pycache__/`, or `.pytest_cache/` folders. The reviewer must also have the required Hugging Face model checkpoints available locally. CPU inference may be slow, and the recorded multi-source real-model scenario did not complete within its time limit; see the evaluation results above.
