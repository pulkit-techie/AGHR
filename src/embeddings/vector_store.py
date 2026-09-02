"""
AGHR System — Vector Store (Phase 2.2)
========================================
FAISS and ChromaDB wrappers for storing and searching embeddings.
Supports cosine similarity and L2 distance search with top-K retrieval.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from loguru import logger


class FAISSVectorStore:
    """
    FAISS-based vector store for fast similarity search.

    Supports:
      - Cosine similarity (via normalized vectors + Inner Product)
      - L2 distance
      - Save/load index to disk
    """

    def __init__(self, dimension: int, similarity: str = "cosine"):
        import faiss

        self.dimension = dimension
        self.similarity = similarity
        self.chunks: List[Dict] = []

        if similarity == "cosine":
            # For normalized vectors, Inner Product = Cosine Similarity
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)

        logger.info(f"FAISS store: dim={dimension}, similarity={similarity}")

    def add(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """
        Add embeddings and their corresponding chunk metadata.

        Args:
            embeddings: numpy array of shape (n, dimension).
            chunks: List of chunk dicts with matching order.
        """
        assert len(embeddings) == len(chunks), "Embeddings and chunks length mismatch"
        assert embeddings.shape[1] == self.dimension, "Dimension mismatch"

        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} vectors (total: {self.index.ntotal})")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for top-K most similar chunks.

        Args:
            query_embedding: Query vector of shape (dimension,).
            top_k: Number of results to return.

        Returns:
            List of dicts with 'chunk', 'score', 'rank' keys.
        """
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "rank": rank + 1,
            })

        return results

    def save(self, index_dir: str) -> None:
        """Save FAISS index and chunk metadata to disk."""
        import faiss

        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) → {index_dir}")

    def load(self, index_dir: str) -> None:
        """Load FAISS index and chunk metadata from disk."""
        import faiss

        path = Path(index_dir)
        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.dimension = self.index.d
        logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors) from {index_dir}")


class ChromaVectorStore:
    """
    ChromaDB-based vector store as an alternative to FAISS.

    Provides persistent storage and metadata filtering.
    """

    def __init__(self, collection_name: str = "aghr_medical",
                 persist_dir: str = "data/chroma_db"):
        import chromadb

        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"Chroma store: collection={collection_name}, dir={persist_dir}")

    def add(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """Add embeddings and chunks to ChromaDB."""
        ids = [chunk.get("chunk_id", f"chunk_{i}") for i, chunk in enumerate(chunks)]
        documents = [chunk.get("text", "") for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            meta = {k: str(v) for k, v in chunk.get("metadata", {}).items()}
            meta["doc_id"] = chunk.get("doc_id", "unknown")
            metadatas.append(meta)

        # ChromaDB has batch size limits, so add in batches
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info(f"Added {len(chunks)} vectors to Chroma (total: {self.collection.count()})")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for top-K most similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk": {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                },
                "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.0,
                "rank": i + 1,
            })

        return output


def create_vector_store(backend: str = "faiss", **kwargs):
    """Factory function to create the appropriate vector store."""
    if backend == "faiss":
        return FAISSVectorStore(**kwargs)
    elif backend == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
