# RAG Course Assistant

[![CI](https://github.com/asad6426/rag-course-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/asad6426/rag-course-assistant/actions/workflows/ci.yml)

A small retrieval-augmented generation pipeline that answers questions about a training programme, cites its source, and declines when the documents do not cover the question. I use it to teach how RAG works before students reach for a framework, so every step is short enough to read in class.

The course in `docs/` is made up for this project.

## How it works

```text
question -> BM25 search -> enough evidence? -- no --> "I could not find this"
                                |
                               yes
                                v
                     answerer -> answer + citation
```

- **Chunking:** one chunk per Markdown section, keeping the file and heading for citations ([`rag/chunking.py`](rag/chunking.py)).
- **Retrieval:** BM25 written from the formula, with no search library ([`rag/retrieval.py`](rag/retrieval.py)).
- **Abstention:** the top chunk must cover at least half of the question, weighted by IDF. Otherwise no answer is generated ([`rag/pipeline.py`](rag/pipeline.py)).
- **Answering:** an offline extractive baseline, or Claude with the API's document citations ([`rag/answerers.py`](rag/answerers.py)).

## Results

From [`eval/run_eval.py`](eval/run_eval.py) on 24 hand-labelled questions, using the offline answerer:

| Metric | Result |
| --- | --- |
| Correct source ranked first (19 answerable) | 19/19 |
| Mean reciprocal rank | 1.000 |
| Offline answer contains the expected fact | 18/19 |
| Declines an out-of-scope question (5 unanswerable) | 5/5 |

The miss is "Where is LangGraph taught?". Retrieval finds the right section, but "taught" is not in the document, so coverage falls below the threshold and the system declines. Fixing it needs semantic retrieval or synonyms, which is a useful classroom discussion, so I left it visible.

## Limits

- The evaluation set is small, and I wrote it alongside the corpus. The threshold was not tuned on a separate set, so these numbers show that the pipeline works and do not predict accuracy on real documents.
- Retrieval is lexical only. There are no embeddings or vector store.
- The Claude answerer is tested against a fake client. Its request and response handling is covered, but this repository's CI does not call the live API.

## Run it

No packages are needed for the offline pipeline.

```bash
python -m unittest discover -s tests -t .
python -m eval.run_eval
python -m rag "How much is the tuition?"
```

To generate answers with Claude:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python -m rag "How much is the tuition?" --claude
```

The adapter defaults to `claude-opus-5-5` and enables server-side refusal fallbacks.

Built with Claude Code.
