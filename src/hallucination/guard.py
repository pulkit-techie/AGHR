"""
AGHR System — Hallucination Guard (Phase 10)
==============================================
Multi-layer hallucination prevention:
  1. Grounding check — verify answer is supported by context (NLI)
  2. Confidence scoring — retrieval relevance + answer consistency
  3. Fallback — "Insufficient information" when confidence < threshold
"""

import re
from typing import Dict, List, Optional

from loguru import logger


class GroundingChecker:
    """
    Verifies that generated answers are grounded in the retrieved context.
    Uses NLI (Natural Language Inference) or keyword overlap.
    """

    def __init__(self, use_nli: bool = False, nli_model: str = None):
        self.use_nli = use_nli
        self.nli_pipeline = None

        if use_nli and nli_model:
            try:
                from transformers import pipeline
                self.nli_pipeline = pipeline("text-classification",
                                            model=nli_model, device=-1)
                logger.info(f"NLI grounding model loaded: {nli_model}")
            except Exception as e:
                logger.warning(f"NLI model failed to load: {e}. Using keyword fallback.")
                self.use_nli = False

    def check_grounding(self, answer: str, context: str) -> Dict:
        """
        Check if the answer is grounded in the context.

        Args:
            answer: Generated answer text.
            context: Retrieved context text.

        Returns:
            Dict with is_grounded, grounding_score, method.
        """
        if not answer or not context:
            return {"is_grounded": False, "grounding_score": 0.0, "method": "empty"}

        # Check for explicit "not found" responses
        not_found_phrases = [
            "not found", "insufficient information", "cannot determine",
            "not enough information", "cannot be determined", "no information",
            "not mentioned", "not in the context",
        ]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in not_found_phrases):
            return {"is_grounded": True, "grounding_score": 1.0, "method": "not_found_detected"}

        if self.use_nli and self.nli_pipeline:
            return self._nli_check(answer, context)
        else:
            return self._keyword_check(answer, context)

    def _keyword_check(self, answer: str, context: str) -> Dict:
        """Keyword overlap grounding check."""
        answer_words = set(re.findall(r'\b\w{3,}\b', answer.lower()))
        context_words = set(re.findall(r'\b\w{3,}\b', context.lower()))

        if not answer_words:
            return {"is_grounded": False, "grounding_score": 0.0, "method": "keyword"}

        overlap = len(answer_words & context_words)
        score = overlap / len(answer_words)

        return {
            "is_grounded": score > 0.3,
            "grounding_score": min(score, 1.0),
            "method": "keyword",
            "overlap_words": overlap,
            "total_answer_words": len(answer_words),
        }

    def _nli_check(self, answer: str, context: str) -> Dict:
        """NLI-based grounding check."""
        try:
            result = self.nli_pipeline(
                f"{context[:500]} [SEP] {answer}",
                truncation=True, max_length=512,
            )
            label = result[0]["label"].lower()
            score = result[0]["score"]

            is_grounded = "entail" in label
            return {
                "is_grounded": is_grounded,
                "grounding_score": score if is_grounded else 1.0 - score,
                "method": "nli",
                "nli_label": label,
            }
        except Exception as e:
            logger.warning(f"NLI check failed: {e}")
            return self._keyword_check(answer, context)


class ConfidenceScorer:
    """
    Computes confidence score based on multiple signals:
      - Retrieval relevance (AGHR scores)
      - Grounding score
      - Answer length/quality heuristics
    """

    def __init__(self, weights: Dict = None):
        self.weights = weights or {
            "retrieval": 0.4,
            "grounding": 0.4,
            "quality": 0.2,
        }

    def compute_confidence(self, retrieval_scores: List[float],
                            grounding_result: Dict,
                            answer: str) -> float:
        """
        Compute overall confidence score.

        Args:
            retrieval_scores: List of AGHR scores from top results.
            grounding_result: Output from GroundingChecker.
            answer: Generated answer text.

        Returns:
            Confidence score [0.0, 1.0].
        """
        # Retrieval confidence
        if retrieval_scores:
            retrieval_conf = sum(retrieval_scores) / len(retrieval_scores)
        else:
            retrieval_conf = 0.0

        # Grounding confidence
        grounding_conf = grounding_result.get("grounding_score", 0.0)

        # Quality heuristics
        quality_conf = self._assess_quality(answer)

        # Weighted combination
        confidence = (
            self.weights["retrieval"] * retrieval_conf +
            self.weights["grounding"] * grounding_conf +
            self.weights["quality"] * quality_conf
        )

        return min(max(confidence, 0.0), 1.0)

    def _assess_quality(self, answer: str) -> float:
        """Simple answer quality heuristics."""
        if not answer:
            return 0.0

        score = 0.5  # baseline

        # Penalize very short answers
        if len(answer.split()) < 3:
            score -= 0.2
        # Reward moderate length
        elif 10 <= len(answer.split()) <= 100:
            score += 0.2

        # Penalize repetitive text
        words = answer.lower().split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 0.3

        # Reward structured answers
        if any(marker in answer for marker in ["Reasoning:", "Answer:", "Sources:"]):
            score += 0.1

        return min(max(score, 0.0), 1.0)


class HallucinationGuard:
    """
    Complete hallucination prevention pipeline.
    Combines grounding check + confidence scoring + fallback.
    """

    def __init__(self, confidence_threshold: float = 0.5,
                 fallback_message: str = "Insufficient information to provide a reliable answer.",
                 use_nli: bool = False, nli_model: str = None):
        self.confidence_threshold = confidence_threshold
        self.fallback_message = fallback_message
        self.grounding_checker = GroundingChecker(use_nli=use_nli, nli_model=nli_model)
        self.confidence_scorer = ConfidenceScorer()

    def evaluate(self, answer: str, context: str,
                 retrieval_scores: List[float] = None) -> Dict:
        """
        Evaluate an answer for hallucination risk.

        Args:
            answer: Generated answer.
            context: Retrieved context.
            retrieval_scores: AGHR scores from retrieval.

        Returns:
            Dict with final_answer, confidence, is_reliable, grounding, etc.
        """
        # Step 1: Grounding check
        grounding = self.grounding_checker.check_grounding(answer, context)

        # Step 2: Confidence scoring
        confidence = self.confidence_scorer.compute_confidence(
            retrieval_scores=retrieval_scores or [],
            grounding_result=grounding,
            answer=answer,
        )

        # Step 3: Fallback decision
        is_reliable = confidence >= self.confidence_threshold

        if not is_reliable and not grounding.get("is_grounded", False):
            final_answer = self.fallback_message
        else:
            final_answer = answer

        result = {
            "final_answer": final_answer,
            "original_answer": answer,
            "confidence": round(confidence, 4),
            "is_reliable": is_reliable,
            "grounding": grounding,
            "used_fallback": final_answer == self.fallback_message,
        }

        if not is_reliable:
            logger.warning(f"Low confidence ({confidence:.3f}): answer may be unreliable")

        return result
