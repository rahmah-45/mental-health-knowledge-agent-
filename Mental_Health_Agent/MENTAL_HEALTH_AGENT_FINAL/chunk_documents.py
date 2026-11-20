
import json
import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"
METADATA_FILE = BASE_DIR / "source_metadata.json"

OUTPUT_FILE = PROCESSED_DIR / "chunks.jsonl"
REPORT_FILE = PROCESSED_DIR / "chunking_report.json"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)


def split_into_chunks(text):
    """Split text into overlapping chunks using word boundaries."""

    words = text.split()
    chunks = []

    current_words = []
    current_length = 0

    for word in words:
        additional_length = len(word) + (1 if current_words else 0)

        if current_length + additional_length > CHUNK_SIZE and current_words:
            chunk_text = " ".join(current_words)
            chunks.append(chunk_text)

            overlap_words = []
            overlap_length = 0

            for previous_word in reversed(current_words):
                word_length = len(previous_word) + 1

                if overlap_length + word_length > CHUNK_OVERLAP:
                    break

                overlap_words.insert(0, previous_word)
                overlap_length += word_length

            current_words = overlap_words
            current_length = len(" ".join(current_words))

        current_words.append(word)
        current_length = len(" ".join(current_words))

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def extract_page_metadata(page_text):
    """Read source ID, title and page number from page header."""

    match = re.match(
        r"\[SOURCE:\s*(.*?)\s*\|\s*TITLE:\s*(.*?)\s*\|\s*PAGE:\s*(\d+)\]\s*",
        page_text,
        flags=re.DOTALL
    )

    if not match:
        return None, None, None, page_text.strip()

    source_id = match.group(1).strip()
    title = match.group(2).strip()
    page_number = int(match.group(3))

    content = page_text[match.end():].strip()

    return source_id, title, page_number, content


def main():
    if not METADATA_FILE.exists():
        print("ERROR: source_metadata.json was not found.")
        return

    metadata = json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )

    all_chunks = []
    report = []

    print("=" * 55)
    print("MENTAL HEALTH DOCUMENT CHUNKING")
    print("=" * 55)

    for source in metadata["sources"]:
        filename = Path(source["filename"]).stem + ".txt"
        input_path = PROCESSED_DIR / filename

        print("\nProcessing:", filename)

        if not input_path.exists():
            report.append({
                "source_id": source["source_id"],
                "status": "missing_file",
                "chunks": 0
            })
            print("Status: missing_file")
            continue

        text = input_path.read_text(encoding="utf-8")

        pages = re.split(
            r"(?=\[SOURCE:)",
            text
        )

        source_chunks = 0

        for page_text in pages:
            if not page_text.strip():
                continue

            source_id, title, page_number, content = (
                extract_page_metadata(page_text)
            )

            if source_id is None or not content:
                continue

            page_chunks = TEXT_SPLITTER.split_documents([
                Document(
                    page_content=content,
                    metadata={
                        "source_id": source_id,
                        "title": title,
                        "page": page_number,
                    },
                )
            ])

            for chunk_index, chunk_document in enumerate(
                page_chunks,
                start=1
            ):
                chunk_text = chunk_document.page_content
                chunk_id = (
                    f"{source_id}_P{page_number:03d}_C{chunk_index:03d}"
                )

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source_id": source_id,
                        "title": title,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "filename": filename
                    }
                })

                source_chunks += 1

        report.append({
            "source_id": source["source_id"],
            "filename": filename,
            "status": "processed" if source_chunks else "empty",
            "chunks": source_chunks
        })

        print("Status: processed")
        print("Chunks:", source_chunks)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in all_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    successful = sum(
        1 for item in report
        if item["status"] == "processed"
    )

    report_data = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_sources": len(report),
        "successful_sources": successful,
        "total_chunks": len(all_chunks),
        "sources": report
    }

    REPORT_FILE.write_text(
        json.dumps(report_data, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "=" * 55)
    print("CHUNKING SUMMARY")
    print("=" * 55)
    print("Total sources:", len(report))
    print("Successfully processed:", successful)
    print("Total chunks:", len(all_chunks))
    print("Output:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
