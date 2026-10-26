from pathlib import Path
import unicodedata
import pymupdf


def _repair_mojibake(text: str) -> str:
    """Repair common UTF-8 bytes decoded as Windows-1252 during extraction."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ï¬‚", "fl").replace("ï¬", "fi")
    markers = ("Ã", "Â", "â", "ð", "ï")
    if not any(marker in text for marker in markers):
        return text
    try:
        repaired = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            repaired = text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return repaired if sum(text.count(marker) for marker in markers) > sum(
        repaired.count(marker) for marker in markers
    ) else text


def parse_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract one record per PDF page, preserving page number and file name."""
    pdf_path = Path(pdf_path)
    pages = []
    with pymupdf.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            text = _repair_mojibake(page.get_text("text")).strip()
            if text:
                pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_index + 1,
                })
    return pages


def parse_directory(directory: str | Path) -> list[dict]:
    directory = Path(directory)
    if not directory.exists():
        return []
    records = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        records.extend(parse_pdf(pdf_path))
    return records
