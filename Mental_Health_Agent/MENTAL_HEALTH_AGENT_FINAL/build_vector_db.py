
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "mental_health_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("No chunks were found.")

    return chunks


def main():
    print("=" * 55)
    print("MENTAL HEALTH VECTOR DATABASE BUILDER")
    print("=" * 55)

    print("\nLoading chunks...")
    chunks = load_chunks()
    print("Loaded chunks:", len(chunks))

    print("\nLoading embedding model...")
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        local_files_only=True,
    )
    print("Embedding model loaded successfully.")

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    # Rebuild the collection to avoid duplicate records.
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Previous collection removed.")
    except Exception:
        print("No previous collection found.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("\nGenerating embeddings...")

    total = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [
            item["text"]
            for item in batch
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        collection.add(
            ids=[
                item["chunk_id"]
                for item in batch
            ],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                item["metadata"]
                for item in batch
            ]
        )

        print(
            f"Processed {min(start + len(batch), total)} / {total}"
        )

    print("\nVerifying vector database...")

    stored_count = collection.count()

    test_result = collection.query(
        query_texts=["What is mental health?"],
        n_results=3
    )

    print("Stored chunks:", stored_count)
    print("Test retrieval results:", len(test_result["ids"][0]))

    print("\n" + "=" * 55)
    print("VECTOR DATABASE BUILD COMPLETE")
    print("=" * 55)
    print("Database directory:", DB_DIR)
    print("Collection:", COLLECTION_NAME)
    print("Total stored chunks:", stored_count)


if __name__ == "__main__":
    main()
