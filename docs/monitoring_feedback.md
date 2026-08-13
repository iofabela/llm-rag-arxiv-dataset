# Monitoring & feedback

Every conversation and every 👍/👎 rating is persisted in PostgreSQL and
visualized live in a Grafana dashboard. Feedback also gives you a human signal
to combine with the [retrieval evaluation](retrieval_evaluation.md).

## Feedback flow

1. The Streamlit chat UI shows 👍/👎 buttons under each assistant reply
   (`app_streamlit.py` → `render_feedback`).
2. Clicking one sends `POST /feedback` with the conversation's id and a
   `+1`/`-1` score to the FastAPI backend.
3. The API inserts a row into the `feedback` table
   (`src/arxiv_rag/db.py` → `save_feedback`).

If Postgres is unreachable, chat keeps working — logging and feedback are
just silently disabled (`conversation_id` is `null` in the `/chat` response).

## Database schema

Created automatically on API startup (`db.init_db`):

```
conversations (id, question, answer, model, retrieval_strategy,
               prompt_tokens, completion_tokens, total_tokens,
               response_time, cost, timestamp)
feedback      (id, conversation_id → conversations.id, feedback ±1, timestamp)
```

## Grafana dashboard

The `grafana` service auto-provisions a Postgres datasource and the
**"Arxiv RAG - Conversations & Feedback"** dashboard from
[`grafana/provisioning/`](../grafana/provisioning) — no manual setup needed.
Default login `admin`/`admin`.

Its **11 panels** (titles from
[`rag_feedback.json`](../grafana/provisioning/dashboards/rag_feedback.json)):

| Panel | What it shows |
|---|---|
| Total Conversations | Volume of answered questions |
| Avg Response Time (s) | End-to-end latency per conversation |
| Total OpenAI Cost | Cumulative USD spend |
| Avg Tokens / Conversation | Token efficiency |
| Thumbs Up / Thumbs Down | Feedback volume by polarity |
| Feedback Ratio | 👍 / (👍+👎) |
| Conversations by Retrieval Strategy | Which `retrieval_strategy` served each conversation |
| Conversations Over Time | Volume timeline |
| Feedback Over Time | Feedback timeline |
| Recent Conversations | Latest conversations table |

## Running it

```
make monitor          # docker compose up -d rag-postgres grafana
make api              # the API must be running so conversations get logged
```

Then open `http://localhost:3000`.

To modify the dashboard, edit
[`rag_feedback.json`](../grafana/provisioning/dashboards/rag_feedback.json)
and restart the `grafana` container (or edit it live in the Grafana UI and
export the JSON back).

## Screenshots

<!-- TODO(screenshot): add docs/screenshots/grafana_dashboard.png — the "Arxiv RAG - Conversations & Feedback" dashboard in Grafana -->
<!-- TODO(screenshot): add docs/screenshots/streamlit_sources.png — chat reply with the cited-sources expander and 👍/👎 buttons -->

## See also

- [`src/arxiv_rag/db.py`](../src/arxiv_rag/db.py) — Postgres access layer
- [`grafana/provisioning/`](../grafana/provisioning) — datasource + dashboard
- [`docs/api_scalar.md`](api_scalar.md) — the `/feedback` endpoint contract