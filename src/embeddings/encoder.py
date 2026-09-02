"""
AGHR System — Embedding Encoder (Phase 2.1)
=============================================
Generates dense vector embeddings for text chunks using sentence-transformers.
Supports MiniLM (fast baseline) and BGE (high accuracy).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger
from tqdm import tqdm


class EmbeddingEncoder:
    """
    Text-to-vector embedding encoder using sentence-transformers.

    Supports:
      - all-MiniLM-L6-v2 (384-dim, fast baseline)
      - BAAI/bge-base-en-v1.5 (768-dim, high accuracy)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64, normalize: bool = True, device: str = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Encoder: {model_name} (dim={self.dimension}, device={self.model.device})")

    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        Args:
            texts: List of text strings.
            show_progress: Whether to show progress bar.

        Returns:
            numpy array of shape (n_texts, dimension).
        """
        logger.info(f"Encoding {len(texts)} texts (batch_size={self.batch_size})...")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        logger.info(f"Encoded → shape {embeddings.shape}")
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        # BGE models need "Represent this sentence:" prefix for queries
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self.model.encode(
            [query], normalize_embeddings=self.normalize, convert_to_numpy=True
        )
        return embedding[0]

    def encode_chunks(self, chunks: List[Dict], text_key: str = "text") -> np.ndarray:
        """Encode chunk dicts, extracting text from the specified key."""
        texts = [chunk.get(text_key, "") for chunk in chunks]
        return self.encode_texts(texts)

    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> None:
        """Save embeddings to .npy file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, embeddings)
        logger.info(f"Saved embeddings {embeddings.shape} → {filepath}")

    @staticmethod
    def load_embeddings(filepath: str) -> np.ndarray:
        """Load embeddings from .npy file."""
        embeddings = np.load(filepath)
        logger.info(f"Loaded embeddings {embeddings.shape} from {filepath}")
        return embeddings
