# RAG Demo — Retrieval-Augmented Generation Web App

A small, from-scratch implementation of Retrieval-Augmented Generation (RAG), built as a learning project to understand how the technique actually works under the hood — no high-level RAG frameworks, just the core pieces wired together by hand.

Upload a `.txt` or `.pdf` document, ask questions about it in a browser UI, and get answers grounded in the document's actual content, with the retrieved source chunks shown alongside the answer for transparency.

## What this demonstrates

- **Embeddings** — converting text into vectors that capture meaning, using a local `sentence-transformers` model (no API cost, runs offline)
- **Chunking** — splitting documents into overlapping pieces so long text can be embedded meaningfully
- **Vector search** — storing chunks in a [Chroma](https://www.trychroma.com/) vector database and retrieving the most relevant ones via cosine similarity
- **Metadata filtering** — scoping retrieval to a specific uploaded document, on top of semantic search
- **Grounded generation** — passing retrieved context to an LLM (via [OpenRouter](https://openrouter.ai/)'s free tier) with an explicit instruction to answer only from the provided context
- **A real web UI** — Flask backend with JSON API routes, vanilla JS frontend using `fetch()`, no frontend framework

## Tech stack

| Piece | Tool |
|---|---|
| Backend | Flask |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector database | Chroma |
| LLM generation | OpenRouter API (free tier) |
| PDF parsing | `pypdf` |
| Frontend | Plain HTML/CSS/JS |

## Setup

1. Clone the repo and install dependencies:

   ```bash
   git clone <your-repo-url>
   cd rag_webapp
   pip install -r requirements.txt
   ```

2. Get a free API key from [OpenRouter](https://openrouter.ai/).

3. Create a `.env` file in the project root (see `.env.example`):

   ```
   OPENROUTER_API_KEY=your-actual-key-here
   ```

4. Run the app:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000/` in your browser.

## Usage

1. Upload a `.txt` or `.pdf` file — it gets chunked, embedded, and stored in a local Chroma database.
2. (Optional) Use the "Limit to document" dropdown to scope your question to one specific uploaded file, rather than searching across everything indexed.
3. Type a question and click Ask.
4. The app retrieves the most relevant chunks, shows them below the answer, and generates a response grounded in that retrieved context.

## Project structure

```
rag_webapp/
├── app.py              # Flask app: RAG pipeline + API routes
├── templates/
│   └── index.html      # Frontend UI
├── requirements.txt
├── .env.example         # Template for required environment variables
├── .gitignore
└── README.md
```

## How it works, end to end

```
Upload document
   │
   ▼
Extract text (.txt read directly / .pdf via pypdf)
   │
   ▼
Split into overlapping chunks
   │
   ▼
Embed each chunk (sentence-transformers)
   │
   ▼
Store in Chroma, tagged with source filename metadata
   │
   ▼
User asks a question
   │
   ▼
Embed the question (same model)
   │
   ▼
Similarity search in Chroma (optionally filtered by source)
   │
   ▼
Top-k chunks inserted into a prompt template
   │
   ▼
LLM (via OpenRouter) generates an answer grounded in retrieved context
```

## Notes

- The embedding model runs entirely locally — no API calls or cost for retrieval, only for the final generation step.
- Uploaded files and the Chroma database are stored locally (`uploads/`, `chroma_webapp_db/`) and are excluded from version control.
- This project prioritizes clarity over production-readiness — it's meant to make every step of RAG visible and understandable, not to be a scalable deployment.