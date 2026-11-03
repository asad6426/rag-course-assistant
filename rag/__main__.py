"""Usage: python -m rag "How much is the tuition?" [--claude]"""
import sys

from .answerers import ClaudeAnswerer
from .pipeline import RagPipeline


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--claude"]
    if len(args) != 1:
        sys.exit(__doc__)
    answerer = ClaudeAnswerer() if "--claude" in sys.argv else None
    answer = RagPipeline("docs", answerer).ask(args[0])
    print(answer.text)
    for citation in answer.citations:
        print(f"  source: {citation}")


if __name__ == "__main__":
    main()
