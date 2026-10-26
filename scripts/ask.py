import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import answer_question


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Ask a question about your PDFs: ").strip()
    try:
        result = answer_question(question)
        print("\nANSWER\n------")
        print(result["answer"])
        print("\nRETRIEVED SOURCES\n-----------------")
        if result["sources"]:
            for source in result["sources"]:
                print(f"- {source['source']}, page {source['page']}")
        else:
            print("- No sources found")
    except Exception as exc:
        print(f"Error: {exc}")
        print("Check your .env settings and make sure your selected LLM is running/configured.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
