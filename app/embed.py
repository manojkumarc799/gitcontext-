
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("data/chunks.jsonl")
EMBEDDINGS_FILE = Path("data/embeddings.npy")
METADATA_FILE = Path("data/embedding_metadata.jsonl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 16


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. Run app/chunk.py first."
        )

    with INPUT_FILE.open(encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not chunks:
        raise ValueError("No code chunks found.")

    print(f"Loaded {len(chunks)} code chunks.")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        f"File: {chunk['file_path']}\n"
        f"Name: {chunk['name']}\n"
        f"Type: {chunk['type']}\n"
        f"Code:\n{chunk['content']}"
        for chunk in chunks
    ]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)

    with METADATA_FILE.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Vectors saved to: {EMBEDDINGS_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    main()
