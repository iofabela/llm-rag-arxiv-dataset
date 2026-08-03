import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # The cross-encoder model is already cached locally from a prior run --
    # skip HF Hub's network version-check on every load (avoids the
    # "unauthenticated requests" warning and an unnecessary network call).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import eval_lib as ev
    from arxiv_rag.es_search import (
        get_es_client,
        index_exists,
        index_to_elasticsearch,
    )
    from arxiv_rag.indexing import get_index
    from arxiv_rag.openai_client import get_client
    from arxiv_rag.rerank import get_cross_encoder

    return (
        ThreadPoolExecutor,
        as_completed,
        ev,
        get_client,
        get_cross_encoder,
        get_es_client,
        get_index,
        index_exists,
        index_to_elasticsearch,
        mo,
        np,
        os,
        pd,
        plt,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Retrieval Strategy Evaluation

    Compares three retrieval strategies for the arXiv RAG pipeline on a sample of
    the dataset:

    1. **hybrid_rerank** -- NumPy/BM25 hybrid search (RRF) + local cross-encoder rerank (current production default)
    2. **hybrid_only** -- NumPy/BM25 hybrid search (RRF) only, no rerank
    3. **elasticsearch_rrf** -- Elasticsearch's native RRF retriever (BM25 + kNN), no rerank

    For each strategy we measure **retrieval quality** (hit rate / MRR against a
    synthetic ground-truth question generated per sampled paper), **answer
    relevancy** (LLM-as-a-judge), **token cost**, and **latency** -- then combine
    them into a single composite score to pick a winner.

    All three strategies share the exact same generation prompt and model; only
    the retrieval step differs, so the comparison isolates retrieval quality.
    """)
    return


@app.cell
def _():
    SAMPLE_SIZE = 200
    TOP_K = 5
    RANDOM_SEED = 42
    # Kept low: the account's gpt-4o-mini rate limit (200k TPM) was getting
    # exhausted at MAX_WORKERS=8, since generation + judge calls (~1.5-2k
    # tokens each) stack up fast across many threads. 3 keeps sustained
    # throughput comfortably under that limit; generate_question /
    # generate_answer / judge_relevancy also retry with backoff on 429s.
    MAX_WORKERS = 3
    return MAX_WORKERS, RANDOM_SEED, SAMPLE_SIZE, TOP_K


@app.cell
def _(get_index, mo):
    index = get_index()
    mo.md(f"Loaded **{len(index.df):,}** papers, embeddings shape `{index.embeddings.shape}`.")
    return (index,)


@app.cell
def _(get_client, get_cross_encoder, get_es_client, mo):
    # Warm up the lazily-constructed singletons (OpenAI client, cross-encoder
    # model, ES client) here in the main thread, before any ThreadPoolExecutor
    # cell below. get_cross_encoder() in particular loads a PyTorch model via
    # HF transformers; racing its first construction across multiple worker
    # threads segfaults (native tokenizer/torch init isn't thread-safe).
    get_client()
    get_cross_encoder()
    get_es_client()
    mo.md("Warmed up OpenAI client, cross-encoder model, and Elasticsearch client.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Elasticsearch setup
    """)
    return


@app.cell
def _(index, index_exists, index_to_elasticsearch, mo):
    try:
        if not index_exists():
            index_to_elasticsearch(index)
            es_message = f"Indexed {len(index.df):,} papers into Elasticsearch."
        else:
            es_message = "Elasticsearch index already present, reusing it."
        es_available = True
    except Exception as exc:  # noqa: BLE001
        es_message = f"Elasticsearch unreachable ({exc}); the elasticsearch_rrf strategy will be skipped."
        es_available = False
    mo.md(f"**Elasticsearch:** {es_message}")
    return (es_available,)


@app.cell
def _(es_available, ev, mo):
    strategy_names = [name for name in ev.RETRIEVAL_STRATEGIES if name != "elasticsearch_rrf" or es_available]
    mo.md(f"Evaluating strategies: **{', '.join(strategy_names)}**")
    return (strategy_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluation sample & ground-truth questions
    """)
    return


@app.cell
def _(RANDOM_SEED, SAMPLE_SIZE, index, mo):
    sample_df = index.df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    mo.md(f"Sampled **{len(sample_df)}** papers (seed={RANDOM_SEED}) as the evaluation set.")
    return (sample_df,)


@app.cell
def _(MAX_WORKERS, ThreadPoolExecutor, as_completed, ev, mo, sample_df):
    questions_by_id = {}
    question_gen_cost = 0.0
    with mo.status.progress_bar(total=len(sample_df), title="Generating eval questions") as _bar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as _pool:
            _futures = {_pool.submit(ev.generate_question, row): row["entry_id"] for _, row in sample_df.iterrows()}
            for _future in as_completed(_futures):
                entry_id = _futures[_future]
                question, usage = _future.result()
                questions_by_id[entry_id] = question
                question_gen_cost += usage.cost
                _bar.update()

    eval_df = sample_df.copy()
    eval_df["question"] = eval_df["entry_id"].map(questions_by_id)
    mo.md(
        f"Generated {len(eval_df)} questions. Question-generation cost: "
        f"**${question_gen_cost:.4f}** (one-time, shared across all strategies)."
    )
    return (eval_df,)


@app.cell(disabled=True, hide_code=True)
def _(mo):
    mo.md("""
    ## Running the evaluation
    """)
    return


@app.cell
def _(
    MAX_WORKERS,
    TOP_K,
    ThreadPoolExecutor,
    as_completed,
    ev,
    eval_df,
    index,
    mo,
    os,
    pd,
    strategy_names,
    time,
):
    tasks = [
        (row.entry_id, row.question, strategy_name)
        for row in eval_df.itertuples()
        for strategy_name in strategy_names
    ]

    eval_results = []
    eval_start = time.time()
    with mo.status.progress_bar(total=len(tasks), title="Running evaluation", subtitle="question x strategy") as _bar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as _pool:
            _futures = {
                _pool.submit(ev.evaluate_one, index, entry_id, question, strategy_name, TOP_K): (entry_id, strategy_name)
                for entry_id, question, strategy_name in tasks
            }
            for _future in as_completed(_futures):
                eval_results.append(_future.result())
                _bar.update()
    eval_elapsed_s = time.time() - eval_start

    results_df = pd.DataFrame([r.__dict__ for r in eval_results])
    os.makedirs("artifacts", exist_ok=True)
    results_df.to_csv("artifacts/eval_results.csv", index=False)

    mo.md(f"Ran {len(results_df)} (question x strategy) evaluations in {eval_elapsed_s / 60:.1f} min.")
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Results
    """)
    return


@app.cell(hide_code=True)
def _(TOP_K, ev, mo, results_df):
    summary_df = ev.summarize(results_df, TOP_K)
    mo.vstack([mo.md("### Per-strategy summary"), mo.ui.table(summary_df)])
    return (summary_df,)


@app.cell
def _(np, plt):
    STRATEGY_COLORS = {
        "hybrid_rerank": "#2a78d6",
        "hybrid_only": "#eb6834",
        "elasticsearch_rrf": "#1baf7a",
    }
    STRATEGY_LABELS = {
        "hybrid_rerank": "Hybrid + Rerank",
        "hybrid_only": "Hybrid only",
        "elasticsearch_rrf": "Elasticsearch RRF",
    }

    def make_comparison_figure(summary, specs, ncols=2, highlight=None):
        """specs: list of (column, title, value_fmt)."""
        n = len(specs)
        ncols = min(ncols, n)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), facecolor="#fcfcfb")
        axes = np.array(axes).reshape(-1)

        strategies = summary["strategy"].tolist()
        colors = [STRATEGY_COLORS.get(s, "#898781") for s in strategies]
        labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

        for ax, (col, title, fmt) in zip(axes, specs):
            values = summary[col].tolist()
            ax.set_facecolor("#fcfcfb")
            edge_colors = ["#0b0b0b" if s == highlight else "none" for s in strategies]
            edge_widths = [2.5 if s == highlight else 0 for s in strategies]
            bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor=edge_colors, linewidth=edge_widths)
            for bar_, value in zip(bars, values):
                ax.text(
                    bar_.get_x() + bar_.get_width() / 2,
                    bar_.get_height(),
                    fmt.format(value),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#0b0b0b",
                )
            ax.set_title(title, color="#0b0b0b", fontsize=12, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#c3c2b7")
            ax.tick_params(colors="#52514e", labelsize=9)
            ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.8)
            ax.set_axisbelow(True)
            top = max(values) if values else 1
            ax.set_ylim(0, top * 1.25 if top > 0 else 1)

        for ax in axes[n:]:
            ax.axis("off")

        fig.tight_layout()
        return fig

    return (make_comparison_figure,)


@app.cell(hide_code=True)
def _(TOP_K, make_comparison_figure, mo, summary_df):
    fig_retrieval = make_comparison_figure(
        summary_df,
        [
            (f"hit_rate@{TOP_K}", f"Hit Rate @ {TOP_K}", "{:.3f}"),
            (f"mrr@{TOP_K}", f"MRR @ {TOP_K}", "{:.3f}"),
        ],
    )
    fig_retrieval.savefig("artifacts/eval_retrieval_metrics.png", dpi=150)
    mo.vstack([mo.md("### Retrieval quality"), fig_retrieval])
    return


@app.cell(hide_code=True)
def _(make_comparison_figure, mo, summary_df):
    fig_cost = make_comparison_figure(
        summary_df,
        [
            ("avg_cost_usd", "Avg Cost / Query (USD)", "${:.5f}"),
            ("total_cost_usd", f"Total Cost, {len(summary_df)} strategies (USD)", "${:.4f}"),
        ],
    )
    fig_cost.savefig("artifacts/eval_cost.png", dpi=150)
    mo.vstack([mo.md("### Token cost (calc_total_price)"), fig_cost])
    return


@app.cell(hide_code=True)
def _(make_comparison_figure, mo, summary_df):
    fig_latency = make_comparison_figure(
        summary_df,
        [
            ("avg_retrieval_latency_s", "Avg Retrieval Latency (s)", "{:.2f}s"),
            ("avg_total_latency_s", "Avg Total Latency (s)", "{:.2f}s"),
        ],
    )
    fig_latency.savefig("artifacts/eval_latency.png", dpi=150)
    mo.vstack([mo.md("### Speed / latency"), fig_latency])
    return


@app.cell(hide_code=True)
def _(make_comparison_figure, mo, summary_df):
    fig_relevancy = make_comparison_figure(
        summary_df,
        [("avg_relevancy", "Avg Answer Relevancy (LLM judge, 1-5)", "{:.2f}")],
        ncols=1,
    )
    fig_relevancy.savefig("artifacts/eval_relevancy.png", dpi=150)
    mo.vstack([mo.md("### Answer relevancy (LLM-as-a-judge)"), fig_relevancy])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Composite score

    Combines all four dimensions into one score per strategy (each metric is
    min-max normalized across strategies, cost/latency inverted so "lower is
    better" scores higher). Adjust the weights to see how the ranking shifts
    under different priorities.
    """)
    return


@app.cell
def _(mo):
    DEFAULT_WEIGHTS = {
        "retrieval": 0.30,
        "relevancy": 0.35,
        "speed": 0.15,
        "cost": 0.20,
    }
    get_weights, set_weights = mo.state(DEFAULT_WEIGHTS)
    return DEFAULT_WEIGHTS, get_weights, set_weights


@app.cell
def _(DEFAULT_WEIGHTS, mo, set_weights):
    reset_button = mo.ui.button(
        label="Reset to defaults",
        on_click=lambda _: set_weights(DEFAULT_WEIGHTS.copy()),
    )
    return (reset_button,)


@app.cell(hide_code=True)
def _(get_weights, mo, reset_button, set_weights):
    weight_retrieval = mo.ui.slider(
        0, 1, value=get_weights()["retrieval"], step=0.05,
        label="Retrieval quality weight",
        on_change=lambda v: set_weights({**get_weights(), "retrieval": v}),
    )
    weight_relevancy = mo.ui.slider(
        0, 1, value=get_weights()["relevancy"], step=0.05,
        label="Relevancy weight",
        on_change=lambda v: set_weights({**get_weights(), "relevancy": v}),
    )
    weight_speed = mo.ui.slider(
        0, 1, value=get_weights()["speed"], step=0.05,
        label="Speed weight",
        on_change=lambda v: set_weights({**get_weights(), "speed": v}),
    )
    weight_cost = mo.ui.slider(
        0, 1, value=get_weights()["cost"], step=0.05,
        label="Cost weight",
        on_change=lambda v: set_weights({**get_weights(), "cost": v}),
    )

    mo.vstack([weight_retrieval, weight_relevancy, weight_speed, weight_cost, reset_button])

    return weight_cost, weight_relevancy, weight_retrieval, weight_speed


@app.cell(hide_code=True)
def _(
    TOP_K,
    ev,
    mo,
    summary_df,
    weight_cost,
    weight_relevancy,
    weight_retrieval,
    weight_speed,
):
    total_weight = weight_retrieval.value + weight_relevancy.value + weight_speed.value + weight_cost.value
    normalized_weights = {
        "retrieval": weight_retrieval.value / total_weight if total_weight else 0,
        "relevancy": weight_relevancy.value / total_weight if total_weight else 0,
        "speed": weight_speed.value / total_weight if total_weight else 0,
        "cost": weight_cost.value / total_weight if total_weight else 0,
    }
    score_df = ev.compute_composite_score(summary_df, TOP_K, normalized_weights)
    winner = score_df.iloc[0]["strategy"]
    mo.vstack(
        [
            mo.md(f"**Winner: `{winner}`** (score {score_df.iloc[0]['score']:.3f})"),
            mo.ui.table(score_df[["strategy", "score"]]),
        ]
    )
    return score_df, winner


@app.cell(hide_code=True)
def _(make_comparison_figure, mo, score_df, winner):
    fig_score = make_comparison_figure(score_df, [("score", "Composite Score", "{:.3f}")], ncols=1, highlight=winner)
    fig_score.savefig("artifacts/eval_composite_score.png", dpi=150)
    mo.vstack([mo.md("### Composite score by strategy"), fig_score])
    return


@app.cell(hide_code=True)
def _(mo, winner):
    mo.md(f"""
    ## Conclusion

    Based on this evaluation, **`{winner}`** is the recommended retrieval strategy
    under the current weights. Adjust the sliders above to see how the ranking
    shifts under different priorities (e.g. cost-sensitive vs quality-sensitive
    deployments).

    To make the production app use it, set in `.env`:

    ```
    RETRIEVAL_STRATEGY={winner}
    ```

    `src/arxiv_rag/rag.py` reads `settings.retrieval_strategy` to pick among the
    strategies defined in `src/arxiv_rag/strategies.py` -- no code changes needed.
    """)
    return


if __name__ == "__main__":
    app.run()
