
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBEDDINGS_FILE = Path("data/embeddings.npy")
METADATA_FILE = Path("data/embedding_metadata.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
CANDIDATE_K = 20
RRF_K = 60


def tokenize(text):
    """Simple tokenizer for code and natural-language queries."""
    return text.lower().replace("_", " ").split()


def main():
    embeddings = np.load(EMBEDDINGS_FILE)

    with METADATA_FILE.open(encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts do not match.")

    # Include names and paths so exact code identifiers are searchable.
    texts = [
        f"{chunk['name']} {chunk['file_path']} "
        f"{chunk['type']} {chunk['content']}"
        for chunk in chunks
    ]

    bm25 = BM25Okapi([tokenize(text) for text in texts])

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\nAsk a repository question (or 'exit'): ").strip()

        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        query_vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        vector_scores = embeddings @ query_vector
        vector_order = np.argsort(vector_scores)[::-1][:CANDIDATE_K]

        bm25_scores = bm25.get_scores(tokenize(query))
        bm25_order = np.argsort(bm25_scores)[::-1][:CANDIDATE_K]

        # Reciprocal Rank Fusion combines both ranked lists.
        rrf_scores = {}

        for ranking in (vector_order, bm25_order):
            for rank, index in enumerate(ranking, start=1):
                rrf_scores[index] = rrf_scores.get(index, 0.0) + (
                    1.0 / (RRF_K + rank)
                )

        ranked = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )[:TOP_K]

        print("\nTop hybrid-search results:\n")

        for rank, index in enumerate(ranked, start=1):
            chunk = chunks[index]

            print(f"{rank}. {chunk['name']} ({chunk['type']})")
            print(f"   File: {chunk['file_path']}")
            print(
                f"   Lines: {chunk['start_line']}-"
                f"{chunk['end_line']}"
            )
            print(f"   RRF score: {rrf_scores[index]:.5f}")
            print(chunk["content"][:500])
            print("-" * 60)


if __name__ == "__main__":
    main()
