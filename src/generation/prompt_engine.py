"""
AGHR System — Prompt Engine (Phase 7)
=======================================
Manages prompt templates for different strategies:
  - Zero-shot, Few-shot, Chain-of-thought
  - Structured output (Reasoning/Answer/Sources/Confidence)
  - Template formatting with context injection
"""

import yaml
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


class PromptEngine:
    """
    Loads and formats prompt templates for LLM generation.
    Supports multiple prompting strategies and structured output parsing.
    """

    def __init__(self, prompts_path: str = "config/prompts.yaml"):
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.templates = yaml.safe_load(f)
        logger.info(f"Loaded {len(self.templates)} prompt templates")

    def format_prompt(self, strategy: str, question: str,
                      context: str = "", vector_context: str = "",
                      graph_context: str = "") -> str:
        """
        Format a prompt using the specified strategy.

        Args:
            strategy: "zero_shot", "few_shot", "chain_of_thought", or "structured"
            question: User's question.
            context: Combined context (for non-structured prompts).
            vector_context: Vector retrieval context (for structured prompt).
            graph_context: Graph retrieval context (for structured prompt).

        Returns:
            Formatted prompt string.
        """
        template = self.templates.get(strategy)
        if not template:
            logger.warning(f"Unknown strategy '{strategy}', falling back to zero_shot")
            template = self.templates.get("zero_shot", "Answer: {question}")

        if strategy == "structured":
            return template.format(
                question=question,
                vector_context=vector_context or context,
                graph_context=graph_context or "No graph context available.",
            )
        else:
            return template.format(
                question=question,
                context=context,
            )

    def format_finetuning_sample(self, question: str, context: str,
                                  answer: str) -> str:
        """Format a sample for QLoRA fine-tuning."""
        template = self.templates.get("finetuning_template", "")
        return template.format(
            question=question,
            context=context,
            answer=answer,
        )

    @staticmethod
    def parse_structured_output(text: str) -> Dict:
        """
        Parse LLM output into structured components.

        Expected format:
            Reasoning: ...
            Final Answer: ...
            Sources: ...
            Confidence: 0.XX

        Returns:
            Dict with reasoning, answer, sources, confidence keys.
        """
        result = {
            "reasoning": "",
            "answer": "",
            "sources": "",
            "confidence": 0.0,
            "raw": text,
        }

        lines = text.strip().split("\n")
        current_key = None

        for line in lines:
            line_stripped = line.strip()
            lower = line_stripped.lower()

            if lower.startswith("reasoning:"):
                current_key = "reasoning"
                result["reasoning"] = line_stripped[len("reasoning:"):].strip()
            elif lower.startswith("final answer:"):
                current_key = "answer"
                result["answer"] = line_stripped[len("final answer:"):].strip()
            elif lower.startswith("sources:"):
                current_key = "sources"
                result["sources"] = line_stripped[len("sources:"):].strip()
            elif lower.startswith("confidence:"):
                current_key = "confidence"
                try:
                    val = line_stripped[len("confidence:"):].strip()
                    result["confidence"] = float(val)
                except ValueError:
                    result["confidence"] = 0.0
            elif current_key and line_stripped:
                result[current_key] = result[current_key] + " " + line_stripped

        # If no structured output was found, use the whole text as the answer
        if not result["answer"]:
            result["answer"] = text.strip()

        return result
