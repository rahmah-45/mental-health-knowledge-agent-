
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from assess_relevance import (
    load_model,
    assess_relevance_batch,
)


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "mental_health_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


def _document_identity(document):
    metadata = getattr(document, "metadata", None) or {}
    if isinstance(document, dict):
        metadata = document.get("metadata") or document
        text = document.get("text", document.get("page_content", ""))
    else:
        text = getattr(document, "page_content", "")
    return (
        str(metadata.get("source_id", "")).casefold(),
        str(metadata.get("page", "")),
        " ".join((text or "").split()).casefold(),
    )


class ChromaLangChainRetriever(BaseRetriever):
    """LangChain retriever adapter for the existing Chroma collection."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    embedding_model: object
    collection: object
    top_k: int = TOP_K

    def _query_documents(self, query, candidate_count, exclude_documents=None, result_limit=None):
        count = min(candidate_count, self.collection.count())
        if count == 0:
            return []

        excluded = {
            _document_identity(document)
            for document in (exclude_documents or [])
        }

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=count,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        for rank, (text, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), start=1):
            document = Document(
                page_content=text,
                metadata={
                    **metadata,
                    "rank": rank,
                    "distance": float(distance),
                },
            )
            if _document_identity(document) in excluded:
                continue
            documents.append(document)
            if result_limit is not None and len(documents) >= result_limit:
                break
        return documents

    def _get_relevant_documents(self, query, *, run_manager):
        return self._query_documents(query, self.top_k)

    def invoke_candidates(self, query, exclude_documents, result_limit=None):
        """Overfetch only on correction, then return new unique passages."""
        excluded_count = len(exclude_documents or [])
        candidate_count = max(self.top_k * 3, self.top_k + excluded_count)
        return self._query_documents(
            query,
            candidate_count,
            exclude_documents=exclude_documents,
            result_limit=result_limit or self.top_k,
        )


def load_retrieval_components():

    print("Loading embedding model and LangChain Chroma retriever...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        local_files_only=True,
    )

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    retriever = ChromaLangChainRetriever(
        embedding_model=embedding_model,
        collection=collection,
        top_k=TOP_K,
    )

    tokenizer, qwen_model = load_model()

    return (
        embedding_model,
        retriever,
        tokenizer,
        qwen_model
    )


def retrieve_documents(
    question,
    embedding_model,
    collection,
    exclude_documents=None,
    corrective=False,
):
    """Retrieve evidence through a LangChain VectorStoreRetriever."""
    retrieved_documents = []
    excluded_keys = {
        _document_identity(document)
        for document in (exclude_documents or [])
    }
    if corrective and hasattr(collection, "invoke_candidates"):
        source_documents = collection.invoke_candidates(
            question,
            exclude_documents=exclude_documents or [],
            result_limit=TOP_K,
        )
    else:
        source_documents = collection.invoke(question)

    for index, document in enumerate(source_documents, start=1):
        metadata = document.metadata
        if _document_identity(document) in excluded_keys:
            continue

        retrieved_documents.append({
            "text": document.page_content,
            "source_id": metadata["source_id"],
            "title": metadata["title"],
            "page": metadata["page"],
            "chunk_index": metadata["chunk_index"],
            "rank": metadata.get("rank", index),
            "distance": metadata.get("distance"),
        })

    return retrieved_documents


def filter_relevant_documents(
    question,
    documents,
    tokenizer,
    qwen_model
):

    accepted_documents = []
    assessment_results = []
    documents = list(documents or [])
    labels = assess_relevance_batch(
        question,
        [document.get("text", "") for document in documents],
        tokenizer,
        qwen_model,
    )
    if len(labels) < len(documents):
        labels.extend(["UNCERTAIN"] * (len(documents) - len(labels)))

    for document, label in zip(documents, labels):
        document["relevance"] = label
        print(
            "Assessment:",
            document.get("source_id", "Unknown"),
            "| Page:",
            document.get("page", "Unknown"),
            "| Label:",
            label,
        )
        assessment_results.append(document)
        if label in {"RELEVANT", "PARTIALLY_RELEVANT"}:
            accepted_documents.append(document)

    print(
        "Relevance filter:",
        f"passages={len(documents)}",
        f"accepted={len(accepted_documents)}",
    )

    return accepted_documents, assessment_results


def run_retrieval_pipeline(question):

    (
        embedding_model,
        collection,
        tokenizer,
        qwen_model
    ) = load_retrieval_components()

    print("\nSearching the knowledge base...")

    documents = retrieve_documents(
        question,
        embedding_model,
        collection
    )

    accepted_documents, all_assessments = (
        filter_relevant_documents(
            question,
            documents,
            tokenizer,
            qwen_model
        )
    )

    fully_relevant = [
        doc for doc in accepted_documents
        if doc["relevance"] == "RELEVANT"
    ]

    partially_relevant = [
        doc for doc in accepted_documents
        if doc["relevance"] == "PARTIALLY_RELEVANT"
    ]

    print("\n" + "=" * 60)
    print("RETRIEVAL SUMMARY")
    print("=" * 60)

    print("Retrieved:", len(documents))
    print("Relevant:", len(fully_relevant))
    print("Partially relevant:", len(partially_relevant))
    print("Accepted evidence:", len(accepted_documents))

    return {
        "question": question,
        "retrieved_count": len(documents),
        "relevant_count": len(fully_relevant),
        "partially_relevant_count": len(partially_relevant),
        "accepted_count": len(accepted_documents),
        "relevant_documents": accepted_documents,
        "all_assessments": all_assessments
    }


def main():

    print("=" * 60)
    print("EVIDENCE-GROUNDED RETRIEVAL PIPELINE")
    print("=" * 60)

    question = "What are the recommended sleep hours for adults?"

    result = run_retrieval_pipeline(question)

    print("\nFINAL RESULTS")
    print("Question:", result["question"])
    print("Retrieved:", result["retrieved_count"])
    print("Relevant:", result["relevant_count"])
    print("Partially relevant:", result["partially_relevant_count"])
    print("Accepted:", result["accepted_count"])

    for index, document in enumerate(
        result["relevant_documents"],
        start=1
    ):

        print(f"\nEvidence {index}")
        print("Source:", document["source_id"])
        print("Title:", document["title"])
        print("Page:", document["page"])
        print("Relevance:", document["relevance"])
        print("Rank:", document.get("rank", "unknown"))
        if document.get("distance") is not None:
            print("Distance:", round(document["distance"], 4))
        print("Text:", document["text"][:500])

    print("\nRETRIEVAL PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
