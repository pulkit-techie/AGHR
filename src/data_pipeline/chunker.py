"""
AGHR System — Document Chunker (Phase 1.3)
============================================
Token-based chunking with configurable overlap.
  - Chunk size: 500 tokens (default)
  - Overlap: 80 tokens (default)
  - Preserves document metadata through chunks
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field, asdict

import tiktoken
from loguru import logger
from tqdm import tqdm


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    total_chunks: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DocumentChunker:
    """
    Token-based document chunker with overlap.

    Uses tiktoken for accurate token counting to ensure chunks
    fit within embedding model and LLM context windows.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80,
                 encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        self._chunk_counter = 0

        logger.info(f"Chunker: size={chunk_size}, overlap={chunk_overlap}, enc={encoding_name}")

    def _next_id(self, doc_id: str, idx: int) -> str:
        return f"{doc_id}_CHUNK_{idx:03d}"

    def chunk_text(self, text: str, doc_id: str = "DOC",
                   metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Split text into overlapping token-based chunks.

        Args:
            text: Input text to chunk.
            doc_id: Parent document ID.
            metadata: Metadata to propagate to chunks.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            return [Chunk(
                chunk_id=self._next_id(doc_id, 0),
                doc_id=doc_id,
                text=text.strip(),
                chunk_index=0,
                total_chunks=1,
                token_count=total_tokens,
                metadata=metadata or {},
            )]

        chunks = []
        start = 0
        chunk_idx = 0
        step = self.chunk_size - self.chunk_overlap

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(Chunk(
                    chunk_id=self._next_id(doc_id, chunk_idx),
                    doc_id=doc_id,
                    text=chunk_text,
                    chunk_index=chunk_idx,
                    total_chunks=0,  # updated below
                    token_count=len(chunk_tokens),
                    metadata=metadata or {},
                ))
                chunk_idx += 1

            start += step
            if end >= total_tokens:
                break

        # Update total_chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def chunk_documents(self, documents: List[Dict],
                        text_key: str = "text") -> List[Chunk]:
        """
        Chunk a list of document dicts.

        Args:
            documents: List of document dicts with text field.
            text_key: Key for the text field.

        Returns:
            List of all Chunk objects across all documents.
        """
        all_chunks = []

        for doc in tqdm(documents, desc="Chunking documents"):
            doc_id = doc.get("doc_id", f"DOC_{len(all_chunks):05d}")
            text = doc.get(text_key, "")
            metadata = doc.get("metadata", {})
            metadata["source"] = doc.get("source", "unknown")
            metadata["source_name"] = doc.get("source_name", "unknown")

            chunks = self.chunk_text(text, doc_id=doc_id, metadata=metadata)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunking complete: {len(documents)} docs → {len(all_chunks)} chunks "
            f"(avg {len(all_chunks)/max(len(documents),1):.1f} chunks/doc)"
        )
        return all_chunks

    @staticmethod
    def save_chunks(chunks: List[Chunk], filepath: str) -> None:
        """Save chunks to JSONL file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(chunks)} chunks → {filepath}")

    @staticmethod
    def load_chunks(filepath: str) -> List[Chunk]:
        """Load chunks from JSONL file."""
        chunks = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    chunks.append(Chunk(**data))
        logger.info(f"Loaded {len(chunks)} chunks from {filepath}")
        return chunks

    def get_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Compute chunking statistics."""
        if not chunks:
            return {"total_chunks": 0}

        token_counts = [c.token_count for c in chunks]
        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "unique_docs": len(set(c.doc_id for c in chunks)),
        }
