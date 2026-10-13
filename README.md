# GitContext — AI-Powered Codebase Assistant

GitContext is an AI-powered codebase assistant that helps developers understand public GitHub repositories using natural-language questions. It retrieves relevant code, reranks results, and generates answers grounded in source code with file and line references.

## Features

- **GitHub Repository Indexing:** Clone and index a public GitHub repository.
- **Code-Aware Chunking:** Extract Python classes and functions into searchable chunks.
- **Semantic Search:** Retrieve relevant code using sentence-transformer embeddings.
- **Hybrid Retrieval:** Combine vector similarity search with BM25 keyword search.
- **Reranking:** Improve result ordering with a cross-encoder.
- **REST API:** Expose functionality through FastAPI.
- **Grounded Answers:** Use retrieved code as evidence for AI-generated answers.
- **Source References:** Return file paths and line ranges for retrieved code.

## Architecture

```text
Public GitHub Repository
          |
          v
   Repository Indexing
          |
          v
    Code Chunking
          |
          v
   Embedding Generation
          |
          v
   Semantic Search + BM25
          |
          v
   Reciprocal Rank Fusion
          |
          v
   Cross-Encoder Reranking
          |
          v
       FastAPI
          |
          v
   AI Answer + Sources
```

## Tech Stack

- Python 3.12
- GitPython
- AST-based Python code parsing
- Sentence Transformers
- `all-MiniLM-L6-v2` embeddings
- BM25 keyword retrieval
- `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking
- FastAPI and Uvicorn
- Google Gemini API for answer generation
- NumPy

## Project Structure

```text
gitcontext/
├── app/
│   ├── api.py
│   ├── index_repo.py
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed.py
│   ├── search.py
│   ├── hybrid_search.py
│   ├── rerank.py
│   ├── evaluate.py
│   └── pipeline.py
├── data/
│   ├── repository/
│   ├── documents.jsonl
│   ├── chunks.jsonl
│   ├── embeddings.npy
│   └── embedding_metadata.jsonl
├── .env
├── .gitignore
└── README.md
```

## Setup

### 1. Clone this project

```bash
git clone <YOUR-GITCONTEXT-REPOSITORY-URL>
cd gitcontext
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
```

### 3. Install dependencies

Install the packages used by the project:

```bash
python -m pip install GitPython numpy sentence-transformers rank-bm25 fastapi "uvicorn[standard]" python-dotenv google-genai
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Keep this key private. Never commit `.env` to GitHub.

### 5. Start the API

```bash
python -m uvicorn app.api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/index` | Index a public GitHub repository |
| POST | `/search` | Retrieve and rerank relevant code |
| POST | `/ask` | Answer a question using retrieved code |

Check `/docs` for the exact request fields required by each endpoint.

### Example search request

```json
{
  "query": "Where is the Flask application class defined?",
  "top_k": 5
}
```

### Example ask request

```json
{
  "query": "What does the Flask class do?",
  "top_k": 5
}
```

Use these request examples only if they match the schemas currently exposed by your API.

## Evaluation

GitContext includes an evaluation script for testing retrieval quality.

Evaluation goals include:

- Recall@5
- Mean Reciprocal Rank (MRR@5)
- Retrieval latency
- Relevance of retrieved code chunks
- Accuracy of source references

Record measured results before publishing performance claims.

## Current Limitations

- Initial chunking focuses on Python classes and functions.
- Indexing a repository rebuilds its local index.
- Large repositories may require more efficient incremental indexing.
- AI-generated answers depend on the quality of retrieved evidence.
- Retrieval scores are ranking signals, not probabilities of correctness.

## Future Improvements

- Incremental repository indexing
- Support for additional programming languages
- Git history and pull-request indexing
- Automated retrieval evaluation
- A web-based chat interface
- Deployment and monitoring
- Better handling of large repositories

## Security

- Use only repositories you are authorized to access.
- Keep API keys in environment variables.
- Do not commit `.env`, virtual environments, or generated data files.
- Validate repository URLs before cloning.

## Author

Built as a portfolio project to explore code retrieval, reranking, and retrieval-augmented generation (RAG).
