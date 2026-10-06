"""
rag.py - A minimal, dependency-light RAG (Retrieval Augmented Generation) store.

This is intentionally simple so you can SEE every step of RAG:

    1. CHUNK   - Split a document into small pieces.
    2. EMBED   - Turn each chunk into a vector (list of numbers) via OpenAI.
    3. STORE   - Keep the chunks + their vectors in memory.
    4. SEARCH  - Embed the user's question, then find the most similar chunks
                 using cosine similarity. Return only the top matches.

In a production system you'd swap the in-memory list for a vector database
(FAISS, Chroma, pgvector, Pinecone, etc.), but the core idea is identical.
"""

import os
import re

import numpy as np
from openai import OpenAI


# The embedding model turns text into a 1536-dimension vector. "small" is cheap
# and more than good enough for a tiny FAQ.
EMBEDDING_MODEL = "text-embedding-3-small"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity measures how close two vectors point in the same
    direction (1.0 = identical meaning, 0.0 = unrelated). This is the standard
    way to compare embeddings.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class RagStore:
    """An in-memory vector store for a single text document."""

    def __init__(self, client: OpenAI):
        self.client = client
        self.chunks: list[str] = []          # the raw text pieces
        self.embeddings: list[np.ndarray] = []  # one vector per chunk

    # --- STEP 1: CHUNK ---
    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """
        Split the document into chunks. Our FAQ is naturally organised into
        Q/A blocks separated by blank lines, so we split on blank lines. For a
        generic document you'd use fixed-size overlapping windows instead.
        """
        blocks = re.split(r"\n\s*\n", text.strip())
        return [b.strip() for b in blocks if b.strip()]

    # --- STEP 2: EMBED ---
    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        """Call OpenAI to turn a list of strings into a list of vectors."""
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    # --- STEP 3: STORE (build the index once) ---
    def build_from_file(self, file_path: str) -> None:
        """Read a file, chunk it, embed the chunks, and keep them in memory."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.chunks = self._chunk_text(text)
        if not self.chunks:
            self.embeddings = []
            return

        self.embeddings = self._embed(self.chunks)
        print(f"[RAG] Indexed {len(self.chunks)} chunks from {os.path.basename(file_path)}")

    # --- STEP 4: SEARCH ---
    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.35,
    ) -> list[tuple[str, float]]:
        """
        Embed the query and return the most similar chunks, each paired with its
        similarity score.

        Two filters control what comes back:
          - top_k:     the maximum number of chunks to return.
          - min_score: a relevance cutoff. Chunks scoring below this are dropped,
                       so a question with only one good match returns only that
                       match instead of padding the list with weak results.
        """
        if not self.embeddings:
            return []

        query_vec = self._embed([query])[0]

        scored = [
            (chunk, _cosine_similarity(query_vec, vec))
            for chunk, vec in zip(self.chunks, self.embeddings)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        # Keep only chunks that clear the relevance bar...
        relevant = [pair for pair in scored if pair[1] >= min_score]
        # ...then cap at top_k.
        return relevant[:top_k]
