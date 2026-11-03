"""Score retrieval, abstention and offline answers on the labelled questions."""
import json
from pathlib import Path

from rag.pipeline import RagPipeline

ROOT = Path(__file__).resolve().parent.parent


def evaluate(pipeline: RagPipeline, rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["source"]]
    unanswerable = [r for r in rows if not r["source"]]
    hit_at_1 = hit_at_3 = reciprocal = answered_right = 0.0
    failures = []
    for row in answerable:
        sources = [h.chunk.source for h in pipeline.retrieve(row["q"])]
        rank = sources.index(row["source"]) + 1 if row["source"] in sources else None
        hit_at_1 += rank == 1
        hit_at_3 += rank is not None
        reciprocal += 1 / rank if rank else 0
        answer = pipeline.ask(row["q"])
        ok = not answer.abstained and row["answer"].lower() in answer.text.lower()
        answered_right += ok
        if not ok:
            failures.append({"q": row["q"], "got": answer.text})
    abstained = 0
    for row in unanswerable:
        answer = pipeline.ask(row["q"])
        abstained += answer.abstained
        if not answer.abstained:
            failures.append({"q": row["q"], "got": answer.text})
    n = len(answerable)
    return {
        "answerable_questions": n,
        "unanswerable_questions": len(unanswerable),
        "hit_rate_at_1": round(hit_at_1 / n, 3),
        "hit_rate_at_3": round(hit_at_3 / n, 3),
        "mrr": round(reciprocal / n, 3),
        "offline_answer_accuracy": round(answered_right / n, 3),
        "abstention_accuracy": round(abstained / len(unanswerable), 3),
        "failures": failures,
    }


def main() -> None:
    lines = (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    report = evaluate(RagPipeline(ROOT / "docs"), [json.loads(l) for l in lines])
    print(json.dumps(report, indent=2))
    (ROOT / "eval" / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
