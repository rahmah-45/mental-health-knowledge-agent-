
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "mental_health_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5

questions = [
    "What are the symptoms of anxiety?",
    "What are the recommended sleep hours for adults?",
    "What is the exact medication dosage for treating depression?",
    "What is the population of Islamabad?",
]


def main():
    print("=" * 60)
    print("EVIDENCE GAP RETRIEVAL TEST")
    print("=" * 60)

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    for question in questions:
        print("\n" + "=" * 60)
        print("QUESTION:", question)
        print("=" * 60)

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

        for index, (document, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ),
            start=1
        ):
            print(f"\nResult {index}")
            print("Source:", metadata["source_id"])
            print("Page:", metadata["page"])
            print("Distance:", round(distance, 4))
            print("Text:", document[:350].replace("\n", " "))


if __name__ == "__main__":
    main()
