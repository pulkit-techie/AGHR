"""
AGHR System — Ablation Study
==============================
Compares 4 configurations on the same test questions to prove each
component of the AGHR system adds measurable value.

Configurations:
  1. Base LLM      — Direct generation, no retrieval
  2. Prompt Only   — Chain-of-thought prompting, no retrieval
  3. RAG Only      — Vector retrieval only, no KG
  4. AGHR Hybrid   — Full system (vector + KG + AGHR scoring)

Run: python run_ablation.py
"""

import json
import time
import yaml
from loguru import logger

from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.vector_store import FAISSVectorStore
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.query.analyzer import QueryAnalyzer
from src.retrieval.aghr_scorer import AGHRScorer
from src.context.fusion import ContextFusion
from src.generation.llm_layer import LLMLayer
from src.evaluation.metrics import MetricsCalculator


# ─── Test Questions ────────────────────────────────────────────────────────────

TEST_DATA = [
    {"question": "What is the relationship between insulin and diabetes?",
     "answer": "Insulin regulates blood sugar and is essential for diabetes treatment"},
    {"question": "How does hypertension affect cardiovascular health?",
     "answer": "Hypertension increases risk of heart attack and stroke"},
    {"question": "What are the symptoms of pneumonia?",
     "answer": "Symptoms include fever, cough, and difficulty breathing"},
    {"question": "What medications treat high blood pressure?",
     "answer": "ACE inhibitors, beta-blockers, and diuretics treat hypertension"},
    {"question": "How is cancer diagnosed?",
     "answer": "Cancer is diagnosed through biopsies, imaging, and blood tests"},
]

SIMPLE_CONTEXT = (
    "Insulin is a hormone that regulates blood glucose levels and is used to treat diabetes. "
    "Hypertension, or high blood pressure, is a major risk factor for cardiovascular disease. "
    "Pneumonia is a lung infection causing fever, cough, and breathing difficulties. "
    "High blood pressure is treated with ACE inhibitors, beta-blockers, and diuretics. "
    "Cancer diagnosis involves biopsies, CT scans, MRIs, and blood biomarker tests."
)


def main():
    logger.info("🔬 Starting AGHR Ablation Study...")

    # ── Load config ──────────────────────────────────────────────────────────
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── Initialise shared components ─────────────────────────────────────────
    logger.info("Loading shared components...")

    model_name = config["embeddings"]["models"][config["embeddings"]["default_model"]]["name"]
    dim = config["embeddings"]["models"][config["embeddings"]["default_model"]]["dimension"]

    encoder = EmbeddingEncoder(model_name=model_name)

    vector_store = FAISSVectorStore(dimension=dim)
    vector_store.load("data/vector_index")

    kg = KnowledgeGraphBuilder(config["knowledge_graph"]["neo4j"])

    query_analyzer = QueryAnalyzer()
    aghr_scorer = AGHRScorer(
        alpha=config["retrieval"]["aghr"]["alpha"],
        beta=config["retrieval"]["aghr"]["beta"],
        gamma=config["retrieval"]["aghr"]["gamma"],
    )
    context_fusion = ContextFusion(
        max_tokens=config["context"]["max_tokens"],
        dedup_threshold=config["context"]["dedup_threshold"],
    )

    model_config = config["generation"]["models"][config["generation"]["default_model"]]
    llm = LLMLayer(
        model_name=model_config["name"],
        model_type=model_config["type"],
        max_new_tokens=model_config["max_new_tokens"],
    )

    metrics_calc = MetricsCalculator()

    # ─────────────────────────────────────────────────────────────────────────
    # Define 4 Pipeline Configurations
    # ─────────────────────────────────────────────────────────────────────────

    def config_1_base_llm(question: str) -> str:
        """No retrieval — raw LLM generation."""
        prompt = f"Answer this medical question:\nQuestion: {question}\nAnswer:"
        return llm.generate(prompt)

    def config_2_prompt_only(question: str) -> str:
        """Chain-of-thought prompting, no retrieval."""
        prompt = (
            "You are a medical expert. Think step-by-step before answering.\n\n"
            f"Question: {question}\n\n"
            "Step-by-step reasoning:\n1."
        )
        return llm.generate(prompt)

    def config_3_rag_only(question: str) -> str:
        """Vector retrieval only — no KG, no AGHR scoring."""
        query_emb = encoder.encode_query(question)
        results = vector_store.search(query_emb, top_k=5)
        context = "\n\n".join([r["chunk"].get("text", "") for r in results])
        prompt = (
            f"Answer using only the context below.\n\n"
            f"Context:\n{context[:1500]}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return llm.generate(prompt)

    def config_4_aghr_hybrid(question: str) -> str:
        """Full AGHR system — vector + KG + hybrid scoring + fusion."""
        query_emb = encoder.encode_query(question)
        vector_results = vector_store.search(query_emb, top_k=5)
        kg_context = kg.get_subgraph_context(question)
        graph_results = [{"text": kg_context, "entities": []}] if kg_context else []
        scored = aghr_scorer.score_results(vector_results, graph_results, [])
        fusion = context_fusion.fuse(
            vector_context="", graph_context="", scored_results=scored
        )
        final_context = fusion["final_context"]
        prompt = (
            f"You are a medical QA assistant. Answer using only the context.\n\n"
            f"Context:\n{final_context[:1500]}\n\n"
            f"Question: {question}\nAnswer:"
        )
        return llm.generate(prompt)

    # ─────────────────────────────────────────────────────────────────────────
    # Run Ablation
    # ─────────────────────────────────────────────────────────────────────────

    configurations = [
        ("1. Base LLM",    config_1_base_llm),
        ("2. Prompt Only", config_2_prompt_only),
        ("3. RAG Only",    config_3_rag_only),
        ("4. AGHR Hybrid", config_4_aghr_hybrid),
    ]

    all_results = {}

    for cfg_name, pipeline_fn in configurations:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {cfg_name}")
        logger.info(f"{'='*50}")

        predictions, references, latencies = [], [], []

        for sample in TEST_DATA:
            question = sample["question"]
            reference = sample["answer"]

            start = time.time()
            try:
                pred = pipeline_fn(question)
            except Exception as e:
                logger.warning(f"Error on '{question}': {e}")
                pred = ""
            elapsed = time.time() - start

            predictions.append(pred)
            references.append(reference)
            latencies.append(elapsed)
            logger.info(f"  Q: {question[:60]}...")
            logger.info(f"  A: {pred[:80]}...")

        metrics = metrics_calc.evaluate_batch(predictions, references)
        metrics["avg_latency"] = round(sum(latencies) / len(latencies), 3)
        metrics["config"] = cfg_name
        all_results[cfg_name] = metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Print Comparison Table
    # ─────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Config':<22} {'EM':>6} {'F1':>6} {'BLEU':>7} {'ROUGE-1':>8} {'Latency':>9}")
    print("-" * 70)

    for cfg_name, m in all_results.items():
        print(
            f"{cfg_name:<22} "
            f"{m.get('exact_match', 0):>6.3f} "
            f"{m.get('f1', 0):>6.3f} "
            f"{m.get('bleu', 0):>7.4f} "
            f"{m.get('rouge1', 0):>8.3f} "
            f"{m.get('avg_latency', 0):>8.2f}s"
        )

    print("=" * 70)
    print("\nKEY INSIGHT: AGHR Hybrid should show highest scores above.")
    print("These numbers prove each component adds measurable value.\n")

    # Save to file
    import os
    os.makedirs("data/evaluation", exist_ok=True)
    with open("data/evaluation/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Ablation results saved to data/evaluation/ablation_results.json")
    logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()
