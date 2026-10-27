# Insurance Support Assistant

An internal chatbot for an insurance company's customer-support team. While helping a customer on a call or chat, the agent selects that customer and asks questions such as *"Is the policy active?"*, *"What is the waiting period for diabetes?"* or *"What is the status of the claim?"*. The bot answers from the customer's own policy document and their policy and claim records, and shows which policy sections it used.

Everything runs locally: the models run in [Ollama](https://ollama.com), and [Floci](https://floci.io) emulates S3.

## How it works

```
                        ┌────────────── Ingestion ──────────────┐
files/*.pdf ──► S3 ──► SHA-256 check ──► section chunking ──► embeddings ──► ChromaDB
              (Floci)   (skip unchanged)   (by heading font)   (all-minilm)

                        ┌─────────────── Query ─────────────────┐
agent question ──► intent classifier (llama3.2:1b)
                     ├─ rag ──────────────► hybrid search (dense + BM25, RRF) ──► cross-encoder rerank
                     ├─ customer_details ─► Customer API (SQLite, stored in S3)
                     └─ policy_status ────► Customer API
                                   │
                                   ▼
                        answer (llama3.2:latest) + sources
```

- **Ingestion.** PDFs are uploaded to S3. Each one is hashed, and an unchanged document is never re-embedded. Changing the embedding model re-embeds everything. Documents are split at their numbered section and lettered subsection headings, which are detected by font size and weight. Each chunk is at most 256 tokens and is stored in ChromaDB.
- **Retrieval.** Dense search and BM25 keyword search each return their top 20 chunks. Reciprocal rank fusion merges the two lists, and the `ms-marco-MiniLM-L-6-v2` cross-encoder reranks the result down to 5 chunks. Search only covers the selected customer's own policy documents.
- **Routing.** A small model decides whether the question needs the policy document, the customer record, the policy/claim status, or several of these. Any question other than small talk also searches the document.
- **Answering.** The main model answers only from the retrieved text and data, cites the section, and says plainly when the information isn't there, so the agent knows to escalate.

## Project layout

| Path | What it is |
|---|---|
| `ingestion/dms.py` | Uploads `files/*.pdf` to the S3 bucket |
| `ingestion/index.py` | Hash check, chunking, embedding and indexing of everything in the bucket |
| `ingestion/chunker.py` | Section-aware PDF chunking (PyMuPDF) |
| `ingestion/vector_store.py` | Embeddings (Ollama) and the ChromaDB collection |
| `ingestion/hash_tracker.py` | The `vectorstore/manifest.json` hash manifest |
| `rag/intent.py` | Intent classifier / router |
| `rag/retriever.py` | Hybrid search, RRF and reranking |
| `rag/chatbot.py` | Builds the context and streams the answer |
| `customer_api/` | FastAPI customer and policy-status API over SQLite, with sample data |
| `app/main.py` | Chat API and web page for support agents |
| `app/cli.py` | Terminal chat for quick testing |
| `files/` | Policy PDFs (fictional specimens) |
| `vectorstore/` | ChromaDB data and the hash manifest (generated) |

## Requirements

- Python 3.12
- [Ollama](https://ollama.com), with these models pulled:
  ```
  ollama pull llama3.2:1b
  ollama pull llama3.2:latest
  ollama pull all-minilm:l6-v2
  ```
- Docker, to run Floci (and optionally the whole stack)
- [floci-cli](https://floci.io), for running without Docker Compose

## Run locally

```powershell
# 1. Install
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env_example .env

# 2. Start the S3 emulator
floci aws start --detach
floci aws wait

# 3. Upload and index the policy PDFs
.\venv\Scripts\python.exe -m ingestion.dms
.\venv\Scripts\python.exe -m ingestion.index

# 4. Start the customer API (terminal 1)
.\venv\Scripts\python.exe -m uvicorn customer_api.server:app --port 8001

# 5. Start the chat app (terminal 2)
.\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000, choose a customer, and ask a question. The first answer is slower while the models load.

To chat in the terminal instead, run `.\venv\Scripts\python.exe -m app.cli`.

## Run with Docker Compose

Ollama keeps running on the host. The containers reach it at `host.docker.internal:11434`.

```powershell
floci aws stop                      # if floci-cli's instance is running, it holds port 4566
docker compose up -d --build --wait
docker compose run --rm indexer     # upload and index the PDFs
```

Then open http://localhost:8000.

## Configuration

Settings are read from `.env`; see `.env_example` for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `INTENT_MODEL` | `llama3.2:1b` | Intent classifier |
| `MAIN_MODEL` | `llama3.2:latest` | Answer model |
| `EMBEDDING_MODEL` | `all-minilm:l6-v2` | Embeddings (changing it re-embeds all documents) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker |
| `OLLAMA_NUM_CTX` | `8192` | Context window for the answer model |
| `AWS_ENDPOINT_URL` | `http://localhost:4566` | S3 endpoint (Floci) |
| `S3_BUCKET` | `insurance-docs` | Bucket for PDFs and the database |
| `CUSTOMER_API_URL` | `http://localhost:8001` | Customer API used by the chatbot |

## Common tasks

- **Add or update a policy document:** put the PDF in `files/`, then run `ingestion.dms` and `ingestion.index` again. Unchanged documents are skipped. Then add the PDF's file name as the `document_key` of the matching policy in `customer_api/db.py`, because retrieval only searches documents linked to the customer's policies.
- **Reset the customer database:** `python -m customer_api.db` reloads the sample data and uploads it to `s3://insurance-docs/database/insurance.db`. The customer API downloads it from there at startup.
- **Rebuild the vector index:** delete `vectorstore/chroma` and `vectorstore/manifest.json`, then run `ingestion.index`.

## Sample customers

| Customer ID | Name | Situation |
|---|---|---|
| `CUST-88274615` | Rohan Vikram Mehta | Health policy, active (policy PDF available) |
| `MSG-CU-30419277` | Sneha Prakash Iyer | Home policy, active, with a claim under review (policy PDF available) |
| `CUST-10000001` | Priya Sharma | Motor policy in its grace period, premium overdue |
| `CUST-10000002` | Amit Verma | Life policy, lapsed |

All names, policies and documents are fictional.

## Limitations

This is a demo. Before it is used with real customer data:

- There is no login. Put it behind company sign-in (SSO) or a VPN, and log which agent looked up which customer.
- Chat sessions live in memory, so run a single app process.
- The customer database is a read-only SQLite file. Replace it with the real policy system or a proper database.
- Replace Floci with real S3, accessed through an IAM role.
