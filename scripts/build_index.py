import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.ingest.parse_pdf import parse_directory
from src.ingest.chunker import chunk_pages
from src.index.vector_store import reset_collection, add_chunks


def main():
    pdfs = list(RAW_DATA_DIR.glob("*.pdf")) if RAW_DATA_DIR.exists() else []
    if not pdfs:
        print(f"No PDF files found in: {RAW_DATA_DIR}")
        print("Copy at least one text-based PDF into data/raw/ and run this script again.")
        raise SystemExit(1)

    print(f"Found {len(pdfs)} PDF(s). Extracting text...")
    pages = parse_directory(RAW_DATA_DIR)
    if not pages:
        print("No selectable text was extracted. Your PDFs may be scanned; OCR is not included yet.")
        raise SystemExit(1)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for page in pages:
        output = PROCESSED_DATA_DIR / f"{Path(page['source']).stem}_page_{page['page']}.txt"
        output.write_text(page["text"], encoding="utf-8")

    chunks = chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"Extracted {len(pages)} non-empty page(s), created {len(chunks)} chunk(s).")
    print("Resetting the old index and embedding chunks. The first run downloads the model...")
    reset_collection()
    count = add_chunks(chunks)
    print(f"Done. Indexed {count} chunks.")


if __name__ == "__main__":
    main()
