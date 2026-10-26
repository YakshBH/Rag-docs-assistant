
"""Evaluate retrieval and show the top results for every failed question."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.retriever import retrieve


def main():
    questions_path = ROOT / "eval" / "questions.json"
    cases = json.loads(questions_path.read_text(encoding="utf-8"))
    valid_cases = [
        case for case in cases
        if case.get("source") != "your_document.pdf"
    ]

    if not valid_cases:
        print("Add verified questions to eval/questions.json first.")
        return

    hits = 0
    reciprocal_ranks = []
    misses = []

    for case in valid_cases:
        results = retrieve(case["question"], top_k=5)
        expected_source = case["source"]
        expected_page = int(case["page"])
        rank = None

        for i, item in enumerate(results, start=1):
            if (
                item["source"] == expected_source
                and int(item["page"]) == expected_page
            ):
                rank = i
                break

        if rank:
            hits += 1
            reciprocal_ranks.append(1 / rank)
            print(
                f"HIT  | {case['question']} | "
                f"expected {expected_source} p.{expected_page} | rank={rank}"
            )
        else:
            reciprocal_ranks.append(0)
            misses.append((case, results))
            print(
                f"MISS | {case['question']} | "
                f"expected {expected_source} p.{expected_page}"
            )

            for i, item in enumerate(results, start=1):
                print(
                    f"       #{i} {item['source']} "
                    f"p.{item['page']} | distance={item['distance']:.4f}"
                )

    n = len(valid_cases)

    print("\n========== SUMMARY ==========")
    print(f"Evaluated: {n}")
    print(f"Hits: {hits}")
    print(f"Misses: {len(misses)}")
    print(f"Recall@5 (source-page hit): {hits / n:.3f}")
    print(f"MRR: {sum(reciprocal_ranks) / n:.3f}")


if __name__ == "__main__":
    main()
