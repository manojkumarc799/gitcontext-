import os
import subprocess
import sys
from app.pipeline import run_pipeline

from urllib.parse import urlparse
import shutil
import re

from git import Repo
from git.exc import GitCommandError



from dotenv import load_dotenv
from google import genai
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
METADATA_PATH = DATA_DIR / "embedding_metadata.jsonl"

app = FastAPI(
    title="GitContext API",
    description="Search a codebase using hybrid retrieval and reranking.",
    version="1.0.0",
)
load_dotenv(BASE_DIR / ".env")

gemini_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_key) if gemini_key else None



# Load data and models once when the API starts.
embeddings = np.load(EMBEDDINGS_PATH)

with METADATA_PATH.open("r", encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file if line.strip()]

if len(chunks) != len(embeddings):
    raise RuntimeError("Embeddings and metadata have different lengths.")

def chunk_text(chunk):
    return (
        f"Path: {chunk.get('file_path', '')}\n"
        f"Name: {chunk.get('name', '')}\n"
        f"Type: {chunk.get('type', '')}\n"
        f"Code:\n{chunk.get('content', '')}"
    )

texts = [chunk_text(chunk) for chunk in chunks]

def tokenize(text):
    return text.lower().replace("_", " ").split()

bm25 = BM25Okapi([tokenize(text) for text in texts])

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

class IndexRequest(BaseModel):
    repository_url: str = Field(min_length=1, max_length=500)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_loaded": len(chunks),
    }


@app.post("/search")
def search(request: SearchRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    # Stage 1: semantic retrieval.
    query_vector = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    semantic_scores = embeddings @ query_vector
    semantic_indices = np.argsort(semantic_scores)[::-1][:20]

    # Stage 2: keyword retrieval.
    keyword_scores = bm25.get_scores(tokenize(query))
    keyword_indices = np.argsort(keyword_scores)[::-1][:20]

    # Stage 3: Reciprocal Rank Fusion.
    rrf_scores = {}

    for rank, index in enumerate(semantic_indices, start=1):
        rrf_scores[int(index)] = (
            rrf_scores.get(int(index), 0.0) + 1 / (60 + rank)
        )

    for rank, index in enumerate(keyword_indices, start=1):
        rrf_scores[int(index)] = (
            rrf_scores.get(int(index), 0.0) + 1 / (60 + rank)
        )

    candidate_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )[:20]

    # Stage 4: Cross-encoder reranking.
    pairs = [(query, texts[index]) for index in candidate_indices]
    rerank_scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_indices, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:request.top_k]

    results = []

    for index, score in ranked:
        chunk = chunks[index]

        results.append({
            "name": chunk.get("name", ""),
            "type": chunk.get("type", ""),
            "file_path": chunk.get("file_path", ""),
            "start_line": chunk.get("start_line"),
            "end_line": chunk.get("end_line"),
            "score": float(score),
            "content": chunk.get("content", ""),
        })

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


@app.post("/ask")
def ask(request: SearchRequest):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing from the .env file.",
        )

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    # Reuse the existing search endpoint's retrieval logic.
    search_response = search(
        SearchRequest(query=query, top_k=request.top_k)
    )

    results = search_response["results"]

    if not results:
        return {
            "question": query,
            "answer": "I could not find relevant code to answer this question.",
            "sources": [],
        }

    # Give the model code and source locations as evidence.
    context_parts = []

    for result in results:
        context_parts.append(
            f"File: {result['file_path']}\n"
            f"Lines: {result['start_line']}-{result['end_line']}\n"
            f"Symbol: {result['name']}\n"
            f"Code:\n{result['content']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    try:
        response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=(
        "You are GitContext, a codebase assistant. "
        "Answer only from the supplied code. "
        "Do not invent behavior. Cite supporting locations "
        "as [file_path:line_number]. If evidence is insufficient, "
        "say so clearly.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved code context:\n{context}"
        ),
        )

        answer = response.text or "The model did not return a text answer."

    
    except Exception as exc:
        print("OpenAI error:", repr(exc))
        raise HTTPException(
            status_code=502,
            detail=f"LLM request failed: {type(exc).__name__}. Check server logs.",
        ) from exc


    sources = [
        {
            "file_path": result["file_path"],
            "start_line": result["start_line"],
            "end_line": result["end_line"],
            "name": result["name"],
        }
        for result in results
    ]

    return {
        "question": query,
        "answer": answer,
        "sources": sources,
    }


@app.post("/index")
def index_repository(request: IndexRequest):
    url = request.repository_url.strip()
    parsed = urlparse(url)

    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise HTTPException(
            status_code=400,
            detail="Provide an HTTPS GitHub repository URL.",
        )

    parts = parsed.path.strip("/").split("/")
    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail="URL must look like https://github.com/owner/repo",
        )

    owner, repo_name = parts
    if not re.fullmatch(r"[A-Za-z0-9-]+", owner) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", repo_name.removesuffix(".git")
    ):
        raise HTTPException(status_code=400, detail="Invalid repository URL.")

    safe_url = f"https://github.com/{owner}/{repo_name.removesuffix('.git')}.git"
    repo_dir = DATA_DIR / "repository"
    temp_dir = DATA_DIR / "repository_new"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        Repo.clone_from(safe_url, temp_dir, depth=1)

        # Keep the current repository until the new clone succeeds.
        backup_dir = DATA_DIR / "repository_old"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        if repo_dir.exists():
            repo_dir.rename(backup_dir)

        temp_dir.rename(repo_dir)

        subprocess.run(
            [sys.executable, str(BASE_DIR / "app" / "pipeline.py")],
            cwd=BASE_DIR,
            check=True,
        )

        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        return {
            "status": "success",
            "repository_url": safe_url,
            "message": "Repository indexed. Restart the API to load the new index.",
        }

    except (GitCommandError, subprocess.CalledProcessError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Repository indexing failed. Check the server terminal.",
        ) from exc

@app.post("/index")
def index_repository(request: IndexRequest):
    repo_url = request.repo_url.strip()

    if not repo_url.startswith("https://github.com/"):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid public GitHub HTTPS URL."
        )

    try:
        # This currently runs the pipeline on the configured repository.
        run_pipeline()

        return {
            "status": "success",
            "message": "Repository indexed successfully.",
            "repo_url": repo_url
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {exc}"
        ) from exc

