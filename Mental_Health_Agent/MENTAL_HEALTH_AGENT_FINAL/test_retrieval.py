
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "mental_health_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


def search_documents(model, collection, question):

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    return results


def display_results(question, results):

    print("\n" + "=" * 65)
    print("QUESTION:", question)
    print("=" * 65)

    found_sources = set()

    for index, (document, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):

        source = metadata.get("source_id", "Unknown")
        page = metadata.get("page", "Unknown")

        found_sources.add(source)

        print(f"\nResult {index}")
        print("Source:", source)
        print("Page:", page)
        print("Distance:", round(distance, 4))
        print("Text:", document[:700].replace("\n", " "))

    print("\nSources found:", sorted(found_sources))


def main():

    print("=" * 65)
    print("SLEEP AND STRESS RETRIEVAL INVESTIGATION")
    print("=" * 65)

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    questions = [
        "What are the health benefits of getting enough sleep?",
        "What habits can improve sleep quality?",
        "How can people manage stress?",
        "What are the effects of long-term stress on health?",
        "How does insufficient sleep affect mental health?",
        "How do sleep and stress affect mental health?"
    ]

    for question in questions:

        results = search_documents(
            model,
            collection,
            question
        )

        display_results(question, results)

    print("\n" + "=" * 65)
    print("RETRIEVAL INVESTIGATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
