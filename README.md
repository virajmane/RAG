# Offline RAG Q&A System

Production-ready RAG with local embeddings, deduplication, and hallucination guardrails.

Ask questions about books in data/books/ — fully offline retrieval, only LLM call goes online via OpenRouter.

---

Features

- Offline Embeddings: BAAI/bge-small-en-v1.5 (135 MB, top-tier)
- Incremental Updates: Only new/changed files processed
- Deduplication: Same content, different name → skipped
- Supports: .md, .txt, .pdf
- Audit Trail: See retrieved chunks + exact prompt
- Hallucination-Proof: LLM says "I don't know" if not in context
- Secure: API key from .env

---

Setup

git clone https://github.com/virajmane/RAG.git
cd RAG
cp .env.example .env
# Edit .env with your OpenRouter key
pip install -r requirements.txt

---

Usage

1. Add Books
Place .md, .txt, .pdf files inside: data/books/

2. Build Vector DB
python ingest.py

3. Ask Questions
python query.py

Change QUESTION inside query.py if needed.

---

Project Structure

├── data/books/             ← Your books
├── chroma_db_offline/      ← Vector DB (auto-created)
├── ingest.py               ← Build/update DB
├── query.py                ← Ask questions
├── .env.example            ← Template
└── requirements.txt

---

Example Output

=== FINAL ANSWER ===
Alice

Resume Highlights

- Built offline RAG with bge-small embeddings + Chroma
- Added content-hash deduplication
- Strict anti-hallucination prompt
- Secure .env key handling

---

Future Ideas

- Gradio UI
- FastAPI endpoint
- Docker support
- GPU acceleration
