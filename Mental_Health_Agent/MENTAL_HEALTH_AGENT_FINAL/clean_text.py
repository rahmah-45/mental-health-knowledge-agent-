
import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
OUTPUT_DIR = BASE_DIR / "processed"
METADATA_FILE = BASE_DIR / "source_metadata.json"

OUTPUT_DIR.mkdir(exist_ok=True)


def fix_encoding(text):
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "â€‹": "",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    return text


def is_spaced_design_text(line):
    compact = re.sub(r"\s+", "", line)

    if len(compact) < 12:
        return False

    letters = re.findall(r"[A-Za-z]", compact)

    if len(letters) < 8:
        return False

    spaces = line.count(" ")

    return spaces >= len(letters) * 0.45


def clean_page_text(text):
    if not text:
        return ""

    text = fix_encoding(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove URLs and obvious page timestamps.
        # Remove NIH website transition announcement.
    text = re.sub(
        r"NIH Websites Are Changing.*?"
        r"An official website of the United States government",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(
        r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]M.*",
        "",
        text,
        flags=re.IGNORECASE
    )

    noise_patterns = [
        r"^Donate$",
        r"^Menu$",
        r"^Search$",
        r"^Share$",
        r"^On this page$",
        r"^For Everyone$",
        r"^For Health Care Providers$",
        r"^MORE INFORMATION$",
        r"^SOURCES AND PAGE INFO$",
        r"^CONTENT SOURCE$",
        r"^ABOUT THIS PAGE$",
        r"^Disclaimer$",
        r"^NIMH Resources$",
        r"^Health Topics$",
        r"^Brochures and Fact Sheets$",
        r"^Contact Us$",
        r"^Policies and Notices$",
        r"^Privacy Policy$",
        r"^Website Policies$",
        r"^Accessibility$",
        r"^Federal Resources$",
        r"^NIH Virtual Tour$",
        r"^USA\.gov$",
        r"^Here.s how you know$",
    ]

    cleaned_lines = []

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        # Remove Arabic OCR fragments from English-only sources.
        if re.search(r"[\u0600-\u06FF]", line):
            continue

        if any(
            re.match(pattern, line, re.IGNORECASE)
            for pattern in noise_patterns
        ):
            continue

        # Remove page counters such as 1/3 or 2/10.
        if re.fullmatch(r"\d+\s*/\s*\d+", line):
            continue

        # Remove decorative spaced-out design text.
        if is_spaced_design_text(line):
            continue

        # Remove isolated decorative symbols.
        if re.fullmatch(r"[\W_]+", line):
            continue

        line = re.sub(r"\s+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    # Remove repeated consecutive lines only.
    final_lines = []
    previous = None

    for line in cleaned_lines:
        if line != previous:
            final_lines.append(line)

        previous = line

    return "\n".join(final_lines)


def process_source(source):
    filename = source["filename"]
    pdf_path = KB_DIR / filename

    if not pdf_path.exists():
        return {
            "source_id": source["source_id"],
            "filename": filename,
            "status": "missing_pdf",
            "pages": 0,
            "characters": 0
        }

    # Use LangChain's PDF loader so ingestion follows the same
    # document abstraction used by the rest of the RAG pipeline.
    pages = PyPDFLoader(str(pdf_path)).load()
    page_outputs = []
    total_characters = 0

    for page in pages:
        # PyPDFLoader stores pages as zero-based indices.
        page_number = int(page.metadata.get("page", 0)) + 1
        raw_text = page.page_content or ""
        cleaned_text = clean_page_text(raw_text)

        if cleaned_text:
            page_outputs.append(
                f"[SOURCE: {source['source_id']} | "
                f"TITLE: {source['title']} | "
                f"PAGE: {page_number}]\n\n"
                f"{cleaned_text}"
            )

            total_characters += len(cleaned_text)

    output_name = Path(filename).stem + ".txt"
    output_path = OUTPUT_DIR / output_name

    output_path.write_text(
        "\n\n".join(page_outputs),
        encoding="utf-8"
    )

    return {
        "source_id": source["source_id"],
        "filename": filename,
        "output_file": output_name,
        "status": "processed" if total_characters > 0 else "empty_text",
        "pages": len(pages),
        "characters": total_characters
    }


def main():
    if not METADATA_FILE.exists():
        print("ERROR: source_metadata.json was not found.")
        return

    metadata = json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )

    results = []

    print("=" * 55)
    print("MENTAL HEALTH KNOWLEDGE BASE CLEANING")
    print("=" * 55)

    for source in metadata["sources"]:
        print(f"\nProcessing: {source['filename']}")

        try:
            result = process_source(source)
        except Exception as error:
            result = {
                "source_id": source["source_id"],
                "filename": source["filename"],
                "status": "error",
                "error": str(error),
                "pages": 0,
                "characters": 0
            }

        results.append(result)

        print("Status:", result["status"])
        print("Pages:", result["pages"])
        print("Characters:", result["characters"])

    report_path = OUTPUT_DIR / "cleaning_report.json"

    report_path.write_text(
        json.dumps(results, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    successful = sum(
        1 for item in results
        if item["status"] == "processed"
    )

    print("\n" + "=" * 55)
    print("CLEANING SUMMARY")
    print("=" * 55)
    print("Total sources:", len(results))
    print("Successfully processed:", successful)
    print("Needs attention:", len(results) - successful)
    print("Output folder:", OUTPUT_DIR)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
