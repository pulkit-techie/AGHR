"""
AGHR System — Dataset Splitter (Phase 1.4)
============================================
Splits chunked documents and QA pairs into train/val/test sets.
  - Default: 70% train, 15% validation, 15% test
  - Stratified by source document to avoid data leakage
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from loguru import logger


class DatasetSplitter:
    """
    Splits data into train/val/test sets with stratification.

    Ensures chunks from the same document stay in the same split
    to prevent data leakage.
    """

    def __init__(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split_by_document(self, chunks: List[Dict],
                          doc_id_key="doc_id") -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split chunks ensuring all chunks from the same document are in the same split.

        Args:
            chunks: List of chunk dicts.
            doc_id_key: Key for document ID in each chunk.

        Returns:
            Tuple of (train_chunks, val_chunks, test_chunks).
        """
        # Group chunks by document ID
        doc_groups = defaultdict(list)
        for chunk in chunks:
            doc_id = chunk.get(doc_id_key, "unknown")
            doc_groups[doc_id].append(chunk)

        # Shuffle document IDs
        doc_ids = list(doc_groups.keys())
        random.seed(self.seed)
        random.shuffle(doc_ids)

        # Calculate split points
        n = len(doc_ids)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train_ids = set(doc_ids[:train_end])
        val_ids = set(doc_ids[train_end:val_end])
        test_ids = set(doc_ids[val_end:])

        # Assign chunks to splits
        train = [c for did in train_ids for c in doc_groups[did]]
        val = [c for did in val_ids for c in doc_groups[did]]
        test = [c for did in test_ids for c in doc_groups[did]]

        logger.info(
            f"Split: {len(train)} train ({len(train_ids)} docs), "
            f"{len(val)} val ({len(val_ids)} docs), "
            f"{len(test)} test ({len(test_ids)} docs)"
        )
        return train, val, test

    def split_qa_pairs(self, qa_pairs: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split QA pairs randomly (no document-level stratification needed)."""
        random.seed(self.seed)
        shuffled = qa_pairs.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train = shuffled[:train_end]
        val = shuffled[train_end:val_end]
        test = shuffled[val_end:]

        logger.info(f"QA split: {len(train)} train, {len(val)} val, {len(test)} test")
        return train, val, test

    @staticmethod
    def save_splits(train, val, test, output_dir: str, prefix: str = "chunks") -> None:
        """Save train/val/test splits to JSONL files."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        for name, data in [("train", train), ("val", val), ("test", test)]:
            filepath = path / f"{prefix}_{name}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(data)} items → {filepath}")
