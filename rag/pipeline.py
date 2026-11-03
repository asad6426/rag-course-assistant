"""Retrieve, decide whether the evidence is good enough, then answer."""
from pathlib import Path

from .answerers import REFUSAL_TEXT, Answer, Answerer, ExtractiveAnswerer
from .chunking import load_directory
from .retrieval import BM25Index, Hit


class RagPipeline:
    def __init__(self, docs_dir: str | Path, answerer: Answerer | None = None,
                 k: int = 3, min_coverage: float = 0.5):
        self.index = BM25Index(load_directory(docs_dir))
        self.answerer = answerer or ExtractiveAnswerer()
        self.k, self.min_coverage = k, min_coverage

    def retrieve(self, question: str) -> list[Hit]:
        return self.index.search(question, self.k)

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        if not hits or hits[0].coverage < self.min_coverage:
            return Answer(REFUSAL_TEXT, abstained=True)
        return self.answerer.answer(question, hits)
