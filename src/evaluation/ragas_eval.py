"""
AGHR System — RAGAS LLM-as-a-Judge Evaluation (Pro Enhancement)
===============================================================
Evaluates generation using advanced RAGAS metrics:
  - Faithfulness (Hallucination detection)
  - Answer Relevancy
  - Context Precision
  - Context Recall
"""

import os
import pandas as pd
from loguru import logger

def run_ragas_evaluation(questions: list, answers: list, contexts: list, ground_truths: list = None):
    """
    Run RAGAS evaluation on a batch of QA pairs.
    Note: Requires OPENAI_API_KEY in environment variables by default.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        logger.error("RAGAS or datasets library not installed.")
        return None

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts, # RAGAS expects a list of list of strings
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy, context_precision]
    if ground_truths:
        metrics.append(context_recall)

    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found. RAGAS requires an LLM to act as a judge. "
                       "Please set the environment variable or configure a local LLM in RAGAS.")

    try:
        logger.info("Starting RAGAS evaluation... this may take some time depending on the API.")
        result = evaluate(
            dataset,
            metrics=metrics,
        )
        df = result.to_pandas()
        
        # Calculate means
        summary = {m.name: df[m.name].mean() for m in metrics}
        logger.info(f"RAGAS Summary: {summary}")
        
        return {
            "summary": summary,
            "dataframe": df
        }
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return None
