"""
AGHR System — Evaluation Runner (Step 6)
========================================
Runs the end-to-end evaluation benchmark for the AGHR system.
"""

import os
import json
import yaml
from pathlib import Path
from loguru import logger

# Import AGHR components
from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.vector_store import FAISSVectorStore
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.query.analyzer import QueryAnalyzer
from src.retrieval.aghr_scorer import AGHRScorer
from src.context.fusion import ContextFusion
from src.generation.llm_layer import LLMLayer
from src.evaluation.metrics import MetricsCalculator, ExperimentRunner

def main():
    logger.info("🚀 Starting AGHR Evaluation Benchmark...")

    # 1. Load config
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open("config/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # 2. Initialize components
    logger.info("Loading components...")
    
    model_name = config["embeddings"]["models"][config["embeddings"]["default_model"]]["name"]
    encoder = EmbeddingEncoder(model_name=model_name)
    
    vector_store = FAISSVectorStore(
        dimension=config["embeddings"]["models"][config["embeddings"]["default_model"]]["dimension"]
    )
    vector_store.load("data/vector_index")
    
    kg = KnowledgeGraphBuilder(config["knowledge_graph"]["neo4j"])
    # Load local graph as fallback
    kg.load("data/kg_triples/knowledge_graph.json")

    query_analyzer = QueryAnalyzer()
    
    aghr_scorer = AGHRScorer(
        alpha=config["retrieval"]["aghr"]["alpha"],
        beta=config["retrieval"]["aghr"]["beta"],
        gamma=config["retrieval"]["aghr"]["gamma"]
    )
    
    context_fusion = ContextFusion(
        max_tokens=config["context"]["max_tokens"],
        dedup_threshold=config["context"]["dedup_threshold"]
    )
    
    model_config = config["generation"]["models"][config["generation"]["default_model"]]
    llm = LLMLayer(
        model_name=model_config["name"],
        model_type=model_config["type"],
        max_new_tokens=model_config["max_new_tokens"]
    )

    # 3. Define the pipeline function
    def aghr_pipeline(question: str) -> str:
        # Route query
        route = query_analyzer.analyze(question)
        
        # Get Vector context
        query_emb = encoder.encode_query(question)
        vector_results = vector_store.search(query_emb)
        
        # Get KG context
        kg_context = kg.get_subgraph_context(question)
        
        # Hybrid Scoring
        graph_results = [{"text": kg_context, "entities": []}] if kg_context else []
        scored_results = aghr_scorer.score_results(vector_results, graph_results, [])
        
        # Fuse Context
        fusion_result = context_fusion.fuse(
            vector_context="", 
            graph_context="", 
            scored_results=scored_results
        )
        final_context = fusion_result["final_context"]
        
        # Generate Answer
        prompt = prompts["structured"].format(
            vector_context=final_context,
            graph_context="Combined above.",
            question=question
        )
        answer = llm.generate(prompt)
        return answer

    # 4. Load Evaluation Data
    qa_path = "data/qa_pairs/hotpotqa_qa.json"
    if not os.path.exists(qa_path):
        logger.error(f"Test data not found at {qa_path}. Please run basic RAG first.")
        return
        
    with open(qa_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    # Take a small sample to keep evaluation fast for the demo
    sample_size = min(5, len(test_data))
    test_sample = test_data[:sample_size]
    
    logger.info(f"Running evaluation on {sample_size} samples...")

    # 5. Run Evaluation
    metrics_calc = MetricsCalculator()
    runner = ExperimentRunner(metrics_calc)
    
    results = runner.run_experiment(
        name="AGHR_Hybrid",
        pipeline_fn=aghr_pipeline,
        test_data=test_sample
    )

    # 6. Save and print results
    os.makedirs("data/evaluation", exist_ok=True)
    with open("data/evaluation/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("✅ Evaluation Complete!")
    print("\n" + "="*50)
    print("🏆 FINAL EVALUATION METRICS 🏆")
    print("="*50)
    print(json.dumps(results, indent=2))
    print("="*50)

if __name__ == "__main__":
    main()
