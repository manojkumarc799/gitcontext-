
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

with (DATA_DIR / "embedding_metadata.jsonl").open(
    encoding="utf-8"
) as f:
    chunks = [json.loads(line) for line in f if line.strip()]

embeddings = np.load(DATA_DIR / "embeddings.npy")

def chunk_text(chunk):
    return (
        f"Path: {chunk.get('file_path', '')}\n"
        f"Name: {chunk.get('name', '')}\n"
        f"Type: {chunk.get('type', '')}\n"
        f"Code:\n{chunk.get('content', '')}"
    )

texts = [chunk_text(c) for c in chunks]

def tokenize(text):
    return text.lower().replace("_", " ").split()

bm25 = BM25Okapi([tokenize(text) for text in texts])

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Expected function names are initial hypotheses.
# Inspect your actual source code to verify them.

tests = [
    {
        "query": "Find the function that encodes data using base64.",
        "expected": "base64_encode",
    },
    {
        "query": "Where is JSON data loaded?",
        "expected": "_CompactJSON.loads",
    },
    {
        "query": "Find the function that dumps JSON data.",
        "expected": "_CompactJSON.dumps",
    },
    {
        "query": "Find the function that decodes base64 data.",
        "expected": "base64_decode",
    },
    {
        "query": "Where is URL-safe payload compression handled?",
        "expected": "URLSafeSerializerMixin",
    },
    {
        "query": "Find the class representing invalid payload data.",
        "expected": "BadPayload",
    },
    {
        "query": "Find the class representing invalid data.",
        "expected": "BadData",
    },
    {
        "query": "Where is the serializer class defined?",
        "expected": "Serializer",
    },
    {
        "query": "Find the function that loads a serialized payload.",
        "expected": "load_payload",
    },
    {
        "query": "Find the function that dumps a serialized payload.",
        "expected": "dump_payload",
    },
]


def retrieve(query, top_k=5):
    query_vector = embedding_model.encode(
        [query], normalize_embeddings=True
    )[0]

    semantic = np.argsort(embeddings @ query_vector)[::-1][:20]
    keyword = np.argsort(bm25.get_scores(tokenize(query)))[::-1][:20]

    rrf = {}
    for rank, idx in enumerate(semantic, start=1):
        rrf[int(idx)] = rrf.get(int(idx), 0) + 1 / (60 + rank)

    for rank, idx in enumerate(keyword, start=1):
        rrf[int(idx)] = rrf.get(int(idx), 0) + 1 / (60 + rank)

    candidates = sorted(rrf, key=rrf.get, reverse=True)[:20]
    scores = reranker.predict([(query, texts[i]) for i in candidates])

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [chunks[i] for i, _ in ranked[:top_k]]

passed = 0

for test in tests:
    results = retrieve(test["query"])
    expected = test["expected"]

    found = any(
        expected.lower() in (
            str(result.get("name", ""))
            + "\n"
            + str(result.get("content", ""))
        ).lower()
        for result in results
    )

    passed += int(found)
    print(f"\nQuery: {test['query']}")
    print(f"Expected: {expected}")
    print(f"Found in top 5: {'PASS' if found else 'CHECK'}")

    for result in results[:3]:
        print(
            f"  {result.get('name')} — "
            f"{result.get('file_path')}:{result.get('start_line')}"
        )

print(f"\nInitial test score: {passed}/{len(tests)}")
print("This is a small smoke test, not a general accuracy benchmark.")
