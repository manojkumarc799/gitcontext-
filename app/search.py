
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDINGS_FILE = Path("data/embeddings.npy")
METADATA_FILE = Path("data/embedding_metadata.jsonl")


def main():
    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Run app/embed.py before starting semantic search."
        )

    embeddings = np.load(EMBEDDINGS_FILE)

    with METADATA_FILE.open(encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    if len(embeddings) != len(chunks):
        raise ValueError("Embedding and metadata counts do not match.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    while True:
        query = input("\nAsk about the repository (or type 'exit'): ")

        if query.strip().lower() in {"exit", "quit"}:
            break

        if not query.strip():
            continue

        query_vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = embeddings @ query_vector
        top_indices = np.argsort(scores)[::-1][:5]

        print("\nTop 5 relevant code chunks:\n")

        for rank, index in enumerate(top_indices, start=1):
            chunk = chunks[index]

            print(f"{rank}. {chunk['name']} ({chunk['type']})")
            print(f"   File: {chunk['file_path']}")
            print(
                f"   Lines: {chunk['start_line']}-"
                f"{chunk['end_line']}"
            )
            print(f"   Similarity: {scores[index]:.4f}")
            print("   Code:")
            print(chunk["content"][:700])
            print("-" * 60)


if __name__ == "__main__":
    main()
