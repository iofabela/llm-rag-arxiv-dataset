# Evaluation notebook

`retrieval_evaluation.py` is a [marimo](https://marimo.io) notebook comparing
the three retrieval strategies (`hybrid_rerank`, `hybrid_only`,
`elasticsearch_rrf`) on a sample of the arXiv dataset — retrieval quality
(hit rate / MRR), LLM-judged relevancy, cost, and latency — plus a composite
score and statistical tests. `eval_lib.py` is its supporting code (question
generation, judging, metrics) and is only imported by this notebook, never by
the production app.

Full documentation is in **[docs/retrieval_evaluation.md](../docs/retrieval_evaluation.md)**:

- how it works and the metrics it computes
- prerequisites and configuration
- how to run it (`make eval` or `uv run marimo edit`)
- the latest results (winner: `hybrid_rerank`)