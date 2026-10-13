
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBEDDINGS_FILE = Path("data/embeddings.npy")
METADATA_FILE = Path("data/embedding_metadata.jsonl")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_K = 20
TOP_K = 5
RRF_K = 60


def tokenize(text):
    return text.lower().replace("_", " ").split()


def main():
    embeddings = np.load(EMBEDDINGS_FILE)

    with METADATA_FILE.open(encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts do not match.")

    texts = [
        f"{chunk['name']} {chunk['file_path']} "
        f"{chunk['type']} {chunk['content']}"
        for chunk in chunks
    ]

    bm25 = BM25Okapi([tokenize(text) for text in texts])

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    while True:
        query = input("\nAsk a repository question (or 'exit'): ").strip()

        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        query_vector = embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        vector_scores = embeddings @ query_vector
        vector_order = np.argsort(vector_scores)[::-1][:CANDIDATE_K]

        bm25_scores = bm25.get_scores(tokenize(query))
        bm25_order = np.argsort(bm25_scores)[::-1][:CANDIDATE_K]

        # Combine vector and BM25 rankings with Reciprocal Rank Fusion.
        rrf_scores = {}

        for ranking in (vector_order, bm25_order):
            for rank, index in enumerate(ranking, start=1):
                rrf_scores[index] = rrf_scores.get(index, 0.0) + (
                    1.0 / (RRF_K + rank)
                )

        candidates = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )[:CANDIDATE_K]

        pairs = [
            (query, texts[index])
            for index in candidates
        ]

        print("Reranking candidates...")
        scores = reranker.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:TOP_K]

        print("\nTop reranked code chunks:\n")

        for rank, (index, score) in enumerate(ranked, start=1):
            chunk = chunks[index]

            print(f"{rank}. {chunk['name']} ({chunk['type']})")
            print(f"   File: {chunk['file_path']}")
            print(
                f"   Lines: {chunk['start_line']}-"
                f"{chunk['end_line']}"
            )
            print(f"   Reranker score: {float(score):.4f}")
            print("   Code:")
            print(chunk["content"][:700])
            print("-" * 60)


if __name__ == "__main__":
    main()
