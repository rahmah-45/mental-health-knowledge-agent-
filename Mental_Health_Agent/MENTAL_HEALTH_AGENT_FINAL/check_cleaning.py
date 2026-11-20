
import json
from pathlib import Path

chunks_file = Path("processed/chunks.jsonl")

chunks = []

with chunks_file.open("r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            chunks.append(json.loads(line))

print("=" * 50)
print("CHUNK QUALITY CHECK")
print("=" * 50)

print("Total chunks:", len(chunks))

missing_text = sum(
    1 for chunk in chunks
    if not chunk.get("text", "").strip()
)

missing_metadata = sum(
    1 for chunk in chunks
    if not all(
        key in chunk.get("metadata", {})
        for key in ["source_id", "title", "page"]
    )
)

unique_ids = len({
    chunk.get("chunk_id")
    for chunk in chunks
})

print("Empty chunks:", missing_text)
print("Missing metadata:", missing_metadata)
print("Unique chunk IDs:", unique_ids)
print("Duplicate IDs:", len(chunks) - unique_ids)

print("\nFirst chunk:")
if chunks:
    print(json.dumps(
        chunks[0],
        indent=4,
        ensure_ascii=False
    ))
