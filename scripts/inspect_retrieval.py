"""Print the top retrieval candidates for one question."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.retriever import retrieve


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit("Usage: python scripts/inspect_retrieval.py <question>")

    for rank, item in enumerate(retrieve(question, top_k=20), start=1):
        preview = " ".join(item["text"].split())[:180]
        print(
            f"#{rank:02d} {item['source']} p.{item['page']} "
            f"rrf={item.get('rrf_score', 0.0):.6f} "
            f"distance={float(item.get('distance', 0.0)):.4f} | {preview}"
        )


if __name__ == "__main__":
    main()
