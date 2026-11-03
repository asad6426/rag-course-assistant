"""BM25 retrieval written from the formula, with no search library."""
import math
import re
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk

STOPWORDS = frozenset(
    "a an and are at be by can comes do does for from get how i if in is it long many much need "
    "my of on or the there this to what when where which who will with".split()
)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_stem(t) for t in tokens if t not in STOPWORDS]


def _stem(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    coverage: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(f"{c.heading} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        self.doc_freq = Counter(term for doc in self.docs for term in doc)

    def idf(self, term: str) -> float:
        n, df = len(self.docs), self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 3) -> list[Hit]:
        terms = set(tokenize(query))
        total_idf = sum(self.idf(t) for t in terms)
        hits = []
        for i, doc in enumerate(self.docs):
            score = matched = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf(term) * tf * (self.k1 + 1) / (tf + norm)
                    matched += self.idf(term)
            if score > 0:
                # IDF-weighted share of the question this chunk actually covers.
                hits.append(Hit(self.chunks[i], score, matched / total_idf))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
