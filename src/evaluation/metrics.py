"""
AGHR System — Evaluation Metrics (Phase 11)
=============================================
Quantitative evaluation: BLEU, ROUGE, Exact Match, F1,
Retrieval Precision/Recall, and latency benchmarking.
"""

import re
import time
from typing import Dict, List, Tuple
from collections import Counter

from loguru import logger


class MetricsCalculator:
    """Computes standard QA evaluation metrics."""

    def exact_match(self, prediction: str, reference: str) -> float:
        """Exact Match (EM) — 1.0 if normalized strings match."""
        return 1.0 if self._normalize(prediction) == self._normalize(reference) else 0.0

    def f1_score(self, prediction: str, reference: str) -> float:
        """Token-level F1 score."""
        pred_tokens = self._normalize(prediction).split()
        ref_tokens = self._normalize(reference).split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        return 2 * (precision * recall) / (precision + recall)

    def bleu_score(self, prediction: str, reference: str) -> float:
        """BLEU score using nltk."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            pred_tokens = self._normalize(prediction).split()
            ref_tokens = self._normalize(reference).split()
            if not pred_tokens or not ref_tokens:
                return 0.0
            smooth = SmoothingFunction().method1
            return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        except ImportError:
            logger.warning("nltk not installed, skipping BLEU")
            return 0.0

    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """ROUGE-1, ROUGE-2, ROUGE-L scores."""
        try:
            from rouge_score.rouge_scorer import RougeScorer
            scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = scorer.score(reference, prediction)
            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        except ImportError:
            logger.warning("rouge-score not installed, skipping ROUGE")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    def retrieval_precision_at_k(self, retrieved_ids: List[str],
                                  relevant_ids: List[str], k: int = 5) -> float:
        """Precision@K for retrieval evaluation."""
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        relevant_set = set(relevant_ids)
        hits = sum(1 for rid in top_k if rid in relevant_set)
        return hits / len(top_k)

    def retrieval_recall_at_k(self, retrieved_ids: List[str],
                               relevant_ids: List[str], k: int = 5) -> float:
        """Recall@K for retrieval evaluation."""
        top_k = set(retrieved_ids[:k])
        if not relevant_ids:
            return 0.0
        hits = sum(1 for rid in relevant_ids if rid in top_k)
        return hits / len(relevant_ids)

    def evaluate_batch(self, predictions: List[str],
                        references: List[str]) -> Dict[str, float]:
        """Evaluate a batch of predictions against references."""
        assert len(predictions) == len(references)
        n = len(predictions)

        em_scores, f1_scores, bleu_scores = [], [], []
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

        for pred, ref in zip(predictions, references):
            em_scores.append(self.exact_match(pred, ref))
            f1_scores.append(self.f1_score(pred, ref))
            bleu_scores.append(self.bleu_score(pred, ref))
            rouge = self.rouge_scores(pred, ref)
            rouge1_scores.append(rouge["rouge1"])
            rouge2_scores.append(rouge["rouge2"])
            rougeL_scores.append(rouge["rougeL"])

        return {
            "exact_match": sum(em_scores) / n,
            "f1": sum(f1_scores) / n,
            "bleu": sum(bleu_scores) / n,
            "rouge1": sum(rouge1_scores) / n,
            "rouge2": sum(rouge2_scores) / n,
            "rougeL": sum(rougeL_scores) / n,
            "num_samples": n,
        }

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text


class ExperimentRunner:
    """
    Runs comparative experiments across system configurations:
      1. Base LLM (no retrieval)
      2. + Prompt engineering
      3. + RAG (vector only)
      4. + RAG + KG (AGHR hybrid)
      5. + Fine-tuned model
    """

    def __init__(self, metrics_calculator: MetricsCalculator = None):
        self.metrics = metrics_calculator or MetricsCalculator()
        self.results = {}

    def run_experiment(self, name: str, pipeline_fn, test_data: List[Dict]) -> Dict:
        """
        Run a single experiment configuration.

        Args:
            name: Experiment name (e.g., "base", "rag", "aghr").
            pipeline_fn: Function that takes a question and returns an answer.
            test_data: List of {question, answer} dicts.

        Returns:
            Dict with metrics and timing.
        """
        predictions, references, latencies = [], [], []

        for item in test_data:
            question = item["question"]
            reference = item["answer"]

            start = time.time()
            prediction = pipeline_fn(question)
            latency = time.time() - start

            predictions.append(prediction)
            references.append(reference)
            latencies.append(latency)

        metrics = self.metrics.evaluate_batch(predictions, references)
        metrics["avg_latency"] = sum(latencies) / len(latencies)
        metrics["total_latency"] = sum(latencies)
        metrics["experiment"] = name

        self.results[name] = metrics
        logger.info(f"[{name}] EM={metrics['exact_match']:.3f}, "
                     f"F1={metrics['f1']:.3f}, BLEU={metrics['bleu']:.3f}, "
                     f"Latency={metrics['avg_latency']:.3f}s")

        return metrics

    def get_comparison_table(self) -> List[Dict]:
        """Get all results as a comparison table."""
        return list(self.results.values())

    def statistical_test(self, scores_a: List[float],
                          scores_b: List[float]) -> Dict:
        """Paired t-test between two sets of scores."""
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
            return {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
        except ImportError:
            logger.warning("scipy not installed, skipping t-test")
            return {"t_statistic": 0, "p_value": 1.0, "significant": False}
