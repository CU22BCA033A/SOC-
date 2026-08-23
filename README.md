# AI Customer Support Agent

A self-hosted AI customer support agent built on FastAPI, ChromaDB, and Claude (or NVIDIA's
free NIM models — see [Choosing an LLM provider](#choosing-an-llm-provider)). Answers customer
questions by retrieving grounded context from a markdown knowledge base and citing its
sources — it never answers *policy/product* questions from general model knowledge. It'll
still chat normally about anything else (greetings, general-knowledge questions), kept
separate from the strict, cited, policy-grounded path so the two never blur together — with at
most one LLM call per turn either way, so adding that separation doesn't cost extra latency.

**Status: Phase 1 (knowledge base + RAG chat) is built and working end-to-end.** Tool use
(order lookup, refunds), conversation memory summarization, guardrails, and the admin
dashboard are follow-on phases — see [Roadmap](#roadmap) below.

## Architecture

```
User message
   │
   ▼
[Greeting/thanks/farewell?] ── yes → canned reply, no retrieval, no LLM call
   │ no
   ▼
[Retrieval Layer] ── embeds query → searches Chroma → returns top-k chunks + similarity score
   │
   ▼
[Confidence gate] ── top score >= threshold?
   │ no                                          │ yes
   ▼                                              ▼
[LLM] ── one call: answers freely if this      [LLM] ── answers ONLY from the retrieved
 isn't policy-shaped, or returns a sentinel      chunks, with inline citations
 meaning "this needs grounding I don't have"
   │              │
"I don't know,   normal chat answer,
 escalate?"       no citations, no KB claims
   │                │
   └────────┬───────┘
            ▼
[Response + citations (grounded path only) + escalated flag]
   │
   ▼
[Audit log] ── query, retrieved chunks + scores, confidence, answer, latency — every turn
```

The confidence gate is enforced in code (`backend/app/api/chat.py`), not just in the system
prompt — if retrieval doesn't clear `RETRIEVAL_CONFIDENCE_THRESHOLD`, the grounded-answer LLM
call never happens. Below that threshold, a message is either genuinely out of scope for the
knowledge base (routed to escalation) or just general conversation the KB was never going to
cover (routed to a normal, ungrounded chat reply). One LLM call decides *and* answers — the
system prompt asks the model to output a fixed sentinel token instead of an answer when the
question is policy-shaped, so there's no separate classification round-trip adding latency.
Either way, the strict "answer only from retrieved context, cite your sources" path only ever
runs when there's actually retrieved context to ground it in, so a policy question with no
matching documentation can't get an invented answer. **Every turn costs at most one LLM call.**

## Project layout

```
backend/
  app/
    main.py            FastAPI app, CORS, startup
    config.py           Settings (env vars)
    db.py / models.py   SQLAlchemy — conversations, messages, audit logs (SQLite; Postgres-ready)
    schemas.py           Pydantic request/response models
    rag/
      chunking.py        Heading-based markdown chunker (not fixed character counts)
      vectorstore.py      Chroma wrapper — swap in Pinecone/Weaviate here later, nowhere else
      ingest.py           Loads knowledge_base/*.md, chunks, embeds, upserts
    agent/
      prompts.py          System prompt: answer only from context, cite sources, resist
                           instructions embedded in retrieved documents
      claude_client.py    Calls the Claude Messages API, maps SDK errors to a graceful
                           AgentError instead of ever 500-ing the endpoint
    api/
      chat.py             POST /chat
      kb.py               GET /kb/docs
  knowledge_base/         6 seed docs: shipping, returns, account, pricing, troubleshooting, privacy
  scripts/ingest.py        CLI: (re)build the vector index
  tests/                   pytest — chunking + chat API (mocked retrieval/LLM, no network needed)
frontend/
  src/ChatWidget.jsx        Embeddable chat widget (citations, escalation banner, typing indicator)
  src/App.jsx                Demo page that mounts the widget
```

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set an API key for whichever LLM_PROVIDER you're using
# (see "Choosing an LLM provider" below)

uvicorn app.main:app --reload --port 8000
```

The vector index rebuilds automatically from `knowledge_base/*.md` every time the server
starts — no separate ingestion step needed (though `python scripts/ingest.py` still works
standalone if you want to rebuild it without restarting the server). The first run downloads
a small (~80MB) local embedding model (`all-MiniLM-L6-v2`, via Chroma's built-in ONNX
embedding function) — no external API calls and no GPU required, so retrieval works fully
offline once cached.

### Choosing an LLM provider

By default this project calls Claude (`LLM_PROVIDER=anthropic` in `.env`) — get a key at
[console.anthropic.com](https://console.anthropic.com). This is what the grounding/citation
prompt in `app/agent/prompts.py` was written and tested against, so it's the recommended
default.

If you'd rather not pay anything to test it, set `LLM_PROVIDER=nvidia` and `NVIDIA_API_KEY`
(a free key from [build.nvidia.com](https://build.nvidia.com) → API Keys → Generate) to route
chat turns through an NVIDIA-hosted open model (Llama by default — see `NVIDIA_MODEL` in
`.env.example`) instead of Claude. It's a genuinely free option, but expect noticeably weaker
adherence to the "answer only from context, cite sources" instructions than Claude gives —
open models follow strict grounding instructions less reliably. NVIDIA's free-tier catalog
changes over time; if `NVIDIA_MODEL`'s default stops working, check the current model list at
build.nvidia.com and update it.

Both providers plug into the same `/chat` endpoint through `app/agent/router.py` — switching
is a `.env` change, not a code change.

Verify it's up:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/kb/docs
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_BASE defaults to http://localhost:8000
npm run dev
```

Open `http://localhost:5173`.

## Environment variables (backend/.env)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `nvidia` — see "Choosing an LLM provider" above. |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic`. Without it, `/chat` still runs and returns a graceful "not configured" escalation instead of crashing. |
| `AGENT_MODEL` | `claude-sonnet-4-6` | Model used to generate answers (Anthropic provider only). |
| `NVIDIA_API_KEY` | — | Required when `LLM_PROVIDER=nvidia`. Free key from build.nvidia.com. |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA's OpenAI-compatible endpoint. |
| `NVIDIA_MODEL` | `meta/llama-3.1-8b-instruct` | Model used when `LLM_PROVIDER=nvidia`. An 8B model by default for lower latency on the free tier — check build.nvidia.com for current availability, or bump to a 70B model for better quality at the cost of speed. |
| `DATABASE_URL` | `sqlite:///./support_agent.db` | Swap for a `postgresql://...` URL in production — the schema is driver-agnostic. |
| `CHROMA_DIR` | `./chroma_data` | Where the vector index persists on disk. |
| `KNOWLEDGE_BASE_DIR` | `./knowledge_base` | Folder scanned by the ingestion script. |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.35` | Below this cosine-similarity score, the agent escalates instead of answering. |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated origins allowed to call the API from a browser. |

## Adding a new knowledge base doc

Drop a markdown file into `backend/knowledge_base/`, structured with `#`/`##` headings (the
chunker splits on headings, not character counts, so each retrieved chunk is a complete,
citable section — not an arbitrary slice of text). Then re-run:

```bash
python scripts/ingest.py
```

Restart isn't required for the API server — only for a fresh `ingest.py` run to pick up the
new/changed file. Ask a question about it right away; the answer will cite the new doc by its
`#` H1 title.

## Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How long do I have to return an item?", "session_id": "demo-1"}'
```

With a valid `ANTHROPIC_API_KEY`, this returns something like:

```json
{
  "conversation_id": "…",
  "answer": "You have 30 days from delivery to return most items for a full refund, as long as they're unused and in original packaging [Returns & Refunds Policy]. ...",
  "citations": [
    {"doc": "Returns & Refunds Policy", "heading": "Return Window", "score": 0.7612}
  ],
  "escalated": false,
  "escalation_reason": null,
  "latency_ms": 1830
}
```

Asking something outside the knowledge base (e.g. "What's the capital of France?") returns
`escalated: true` with `escalation_reason: "low_retrieval_confidence"` and no LLM call is made
— verified in this repo's tests and via manual `curl` runs during development.

## Testing

```bash
cd backend && source .venv/bin/activate
python -m pytest tests/ -v
```

Tests cover the chunking pipeline directly and the `/chat` endpoint with the vector store and
Claude client mocked, so the suite runs offline with no API key and no embedding-model
download required.

## Deploying

The backend needs a real filesystem it can write to on boot (for the Chroma index) — that
rules out plain Vercel serverless functions, which have no persistent local disk between
invocations. The split that works cleanly: **backend on Render, frontend on Vercel.**

The backend rebuilds its entire vector index from `knowledge_base/*.md` on every process
start (see the `lifespan` handler in `app/main.py`), so it needs no persistent disk at all —
a fresh container with nothing on it becomes fully functional within a few seconds of boot.
SQLite conversation history is the one thing that resets on redeploy on Render's free tier;
that's fine for testing, and swappable for a hosted Postgres later via `DATABASE_URL`.

### 1. Backend → Render

1. In the [Render dashboard](https://dashboard.render.com/), **New → Web Service**, point it
   at this repo.
2. Set:
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free is fine for testing
3. Add environment variables (Render's env var UI, not a file). For Claude (default):
   `ANTHROPIC_API_KEY` (your key — never commit this), `AGENT_MODEL=claude-sonnet-4-6`,
   `CORS_ORIGINS=*` (tighten this in step 3 below once you have a Vercel URL). To use the free
   NVIDIA option instead, set `LLM_PROVIDER=nvidia` and `NVIDIA_API_KEY` in place of the
   Anthropic vars — see "Choosing an LLM provider" above. The rest (`DATABASE_URL`,
   `CHROMA_DIR`, `KNOWLEDGE_BASE_DIR`, `RETRIEVAL_CONFIDENCE_THRESHOLD`) are optional — the
   built-in defaults in `app/config.py` already work on Render as-is.
4. Deploy. Once live, note the public URL — something like
   `https://support-agent-backend.onrender.com`.
5. Verify: `curl https://<your-render-url>/health` and `curl https://<your-render-url>/kb/docs`.

A `render.yaml` Blueprint is also included at the repo root as a shortcut (**New → Blueprint**
instead of **New → Web Service**) — it encodes the same settings so you don't have to type them
in by hand. Use whichever you're more comfortable with; if the Blueprint's YAML schema has
drifted from what your Render account expects, the manual steps above are the reliable fallback.

Render's free tier spins the service down after inactivity; the first request after a while
will be slow (cold start + re-ingestion) and fast after that.

### 2. Frontend → Vercel

1. In the [Vercel dashboard](https://vercel.com/new), import this same GitHub repo.
2. Set **Root Directory** to `frontend` (this repo is a monorepo — the widget lives in a
   subfolder, not at the repo root). `frontend/vercel.json` pins the build command and output
   directory; Vercel's Vite preset handles the rest.
3. Add an environment variable: `VITE_API_BASE` = your Render URL from step 1
   (e.g. `https://support-agent-backend.onrender.com`). Vite bakes `VITE_`-prefixed vars in at
   *build* time, so this must be set before deploying, not after.
4. Deploy. Vercel gives you a URL like `https://your-app.vercel.app`.

### 3. Connect the two

Go back to the Render service's environment variables and set `CORS_ORIGINS` to your Vercel
URL (e.g. `https://your-app.vercel.app`) instead of the wildcard `*` the blueprint ships with
for initial testing — this is what allows the browser-based widget to actually call the API.
Redeploy the backend for the change to take effect, then open the Vercel URL and chat.

## Roadmap

Built so far (**Phase 1**): ingestion pipeline, heading-based chunking, Chroma retrieval,
confidence-gated grounded answering with citations, honest "I don't know" fallback, full
per-turn audit logging, chat widget.

Not yet built — next phases, in the order described in the original project brief:

- **Phase 2 — Tools**: `get_order_status`, `get_account_info` (read-only, auto-run) and
  `create_refund`, `create_support_ticket`, `escalate_to_human` (write actions gated behind a
  "needs confirmation" step in both the API and the chat widget).
- **Phase 3 — Memory**: persist conversation summaries so long threads don't resend full
  transcripts every turn.
- **Phase 4 — Guardrails**: stronger prompt-injection sanitization of retrieved content,
  explicit refusals for legal/medical/abusive content, and per-session rate limiting. (Intent
  routing between policy and general chat already exists — see Architecture above — folded
  into the single answer call rather than a separate classifier pass, to keep latency down.)
- **Phase 5 — Admin dashboard**: a React view over the `retrieval_audit_logs` table already
  being populated — live conversations, escalations, KB doc management, deflection-rate
  analytics.
- **Phase 6 — Polish**: streaming responses, clickable inline citations, broader test coverage.

The database schema, vector store wrapper, and audit logging were all designed with these
phases in mind (e.g. `RetrievalAuditLog` already captures everything the Phase 5 dashboard
needs to query), but only Phase 1's functionality is implemented and tested today.
