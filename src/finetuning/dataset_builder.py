"""
AGHR System — Fine-tuning Dataset Builder (Phase 9.1)
======================================================
Creates instruction-format datasets for QLoRA fine-tuning from
QA pairs + retrieved context.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

from loguru import logger


class DatasetBuilder:
    """
    Builds instruction-tuning datasets in the format:
      { instruction, input (context + question), output (answer) }
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def build_from_qa_pairs(self, qa_path: str, chunks: List[Dict],
                             max_samples: int = 3000) -> List[Dict]:
        """
        Build fine-tuning dataset from QA pairs and chunk context.

        Args:
            qa_path: Path to QA pairs JSON file.
            chunks: List of chunk dicts for context retrieval.
            max_samples: Maximum samples to generate.

        Returns:
            List of instruction-format dicts.
        """
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)

        # Build simple text index for context matching
        chunk_texts = {i: c.get("text", "") for i, c in enumerate(chunks)}

        samples = []
        for qa in qa_pairs[:max_samples]:
            question = qa.get("question", "")
            answer = qa.get("answer", "") or qa.get("long_answer", "")

            if not question or not answer:
                continue

            # Find relevant context by keyword matching
            context = self._find_context(question, chunk_texts)

            sample = {
                "instruction": ("You are a medical QA assistant. Answer the question "
                               "using only the provided context. If the answer is not "
                               "in the context, say 'Not found.'"),
                "input": f"Context: {context[:800]}\nQuestion: {question}",
                "output": answer,
                "text": (f"### Instruction:\n"
                        f"You are a medical QA assistant. Answer the question "
                        f"using only the provided context. If the answer is not "
                        f"in the context, say 'Not found.'\n\n"
                        f"### Input:\n"
                        f"Context: {context[:800]}\n"
                        f"Question: {question}\n\n"
                        f"### Response:\n"
                        f"{answer}"),
            }
            samples.append(sample)

        random.shuffle(samples)
        logger.info(f"Built {len(samples)} fine-tuning samples")
        return samples

    def _find_context(self, question: str, chunk_texts: Dict[int, str],
                       top_k: int = 3) -> str:
        """Simple keyword-based context retrieval for dataset building."""
        q_words = set(question.lower().split())
        scores = []

        for idx, text in chunk_texts.items():
            t_words = set(text.lower().split())
            overlap = len(q_words & t_words)
            scores.append((overlap, idx))

        scores.sort(reverse=True)
        top_indices = [idx for _, idx in scores[:top_k]]
        return "\n".join([chunk_texts[i][:300] for i in top_indices])

    def save_dataset(self, samples: List[Dict], output_path: str) -> None:
        """Save dataset as JSONL."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(samples)} samples → {output_path}")

    def split_dataset(self, samples: List[Dict],
                       train_ratio: float = 0.9) -> tuple:
        """Split into train/eval sets."""
        n = int(len(samples) * train_ratio)
        return samples[:n], samples[n:]
