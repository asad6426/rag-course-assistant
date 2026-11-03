import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from rag.answerers import ClaudeAnswerer, ExtractiveAnswerer
from rag.chunking import Chunk, chunk_markdown
from rag.pipeline import RagPipeline
from rag.retrieval import BM25Index, tokenize

DOCS = Path(__file__).resolve().parent.parent / "docs"


class FakeClient:
    def __init__(self, response):
        self.response, self.kwargs = response, None
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class ChunkingTest(unittest.TestCase):
    def test_one_chunk_per_section(self):
        chunks = chunk_markdown("a.md", "# T\n## One\nFirst.\n\n## Two\nSecond.\n")
        self.assertEqual([c.heading for c in chunks], ["One", "Two"])
        self.assertEqual(chunks[1].citation, "a.md > Two")

    def test_empty_section_is_dropped(self):
        self.assertEqual(chunk_markdown("a.md", "## Empty\n"), [])


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index([
            Chunk("a.md", "Fees", "tuition is paid in instalments"),
            Chunk("b.md", "Classes", "classes run on tuesday"),
            Chunk("c.md", "Mentors", "mentors answer questions"),
        ])

    def test_tokenizer_removes_stopwords_and_plurals(self):
        self.assertEqual(tokenize("What are the mentors?"), ["mentor"])

    def test_ranks_matching_chunk_first(self):
        self.assertEqual(self.index.search("tuition instalments")[0].chunk.source, "a.md")

    def test_coverage_counts_unknown_terms_against_the_match(self):
        self.assertLess(self.index.search("tuition spaceship dragon")[0].coverage, 0.75)

    def test_no_match_returns_nothing(self):
        self.assertEqual(self.index.search("spaceship"), [])

    def test_empty_corpus_is_rejected(self):
        with self.assertRaises(ValueError):
            BM25Index([])


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = RagPipeline(DOCS)

    def test_answers_with_citation(self):
        answer = self.pipeline.ask("How much is the tuition?")
        self.assertIn("45,000", answer.text)
        self.assertEqual(answer.citations, ["fees.md > Tuition"])

    def test_abstains_without_calling_the_answerer(self):
        class Explodes:
            def answer(self, question, hits):
                raise AssertionError("must not be called")

        answer = RagPipeline(DOCS, Explodes()).ask("What is the capital of France?")
        self.assertTrue(answer.abstained)


class ClaudeAnswererTest(unittest.TestCase):
    """Checks the request and response handling against a fake client."""

    hits = RagPipeline(DOCS).retrieve("How much is the tuition?")

    def test_builds_cited_documents_and_reads_citations(self):
        client = FakeClient(NS(stop_reason="end_turn", content=[
            NS(type="thinking"),
            NS(type="text", text="45,000 taka.", citations=[NS(document_title="fees.md > Tuition")]),
        ]))
        answer = ClaudeAnswerer(client).answer("How much?", self.hits)
        self.assertEqual(answer.text, "45,000 taka.")
        self.assertEqual(answer.citations, ["fees.md > Tuition"])
        self.assertEqual(client.kwargs["model"], "claude-opus-5-5")
        self.assertTrue(client.kwargs["messages"][0]["content"][0]["citations"]["enabled"])

    def test_refusal_becomes_abstention(self):
        client = FakeClient(NS(stop_reason="refusal", content=[]))
        self.assertTrue(ClaudeAnswerer(client).answer("How much?", self.hits).abstained)


class ExtractiveTest(unittest.TestCase):
    def test_picks_best_sentence(self):
        hits = BM25Index([Chunk("a.md", "H", "Cats sleep. Dogs bark loudly.")]).search("dogs bark")
        self.assertEqual(ExtractiveAnswerer().answer("dogs bark", hits).text, "Dogs bark loudly.")


if __name__ == "__main__":
    unittest.main()
