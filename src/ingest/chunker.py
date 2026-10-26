import re


def chunk_pages(pages: list[dict], chunk_size: int = 450, overlap: int = 80) -> list[dict]:
    """Split page text into overlapping word chunks and preserve source metadata."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for page_record in pages:
        words = re.findall(r"\S+", page_record["text"])
        for start in range(0, len(words), step):
            part = words[start:start + chunk_size]
            if not part:
                continue
            chunks.append({
                "text": " ".join(part),
                "source": page_record["source"],
                "page": int(page_record["page"]),
                "chunk_id": len(chunks),
            })
            if start + chunk_size >= len(words):
                break
    return chunks
