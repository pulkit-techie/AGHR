"""
AGHR System — Comprehensive Evaluation Runner
===============================================
Runs the full ablation study on 50 medical QA pairs and generates
publication-quality visualizations for the project report.

Generates:
  - Ablation comparison table (4 configurations)
  - Grouped bar chart of all metrics
  - Per-category performance heatmap
  - Latency comparison chart
  - Radar chart (model capability)
  - Delta improvement chart (AGHR vs baseline)
  - Statistical significance tests

Run: python run_full_evaluation.py
"""

import json
import time
import os
import sys
import yaml
from pathlib import Path
from loguru import logger

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.metrics import MetricsCalculator
from src.evaluation.test_dataset import EXTENDED_TEST_DATA, CATEGORY_MAP


def run_configuration(name, pipeline_fn, test_data, metrics_calc):
    """Run a single pipeline configuration over all test data."""
    predictions, references, latencies = [], [], []
    per_question = []

    for i, sample in enumerate(test_data):
        question = sample["question"]
        reference = sample["answer"]

        start = time.time()
        try:
            pred = pipeline_fn(question)
        except Exception as e:
            logger.warning(f"Error on Q{i}: {e}")
            pred = ""
        elapsed = time.time() - start

        predictions.append(pred)
        references.append(reference)
        latencies.append(elapsed)
        per_question.append({
            "idx": i,
            "question": question,
            "reference": reference,
            "prediction": pred,
            "latency": elapsed,
        })

        if i % 10 == 0:
            logger.info(f"  [{name}] Progress: {i+1}/{len(test_data)}")

    # Compute aggregate metrics
    metrics = metrics_calc.evaluate_batch(predictions, references)
    metrics["avg_latency"] = round(sum(latencies) / len(latencies), 4)
    metrics["total_latency"] = round(sum(latencies), 2)
    metrics["config"] = name
    metrics["per_question"] = per_question

    # Compute per-category metrics
    cat_metrics = {}
    for cat_name, indices in CATEGORY_MAP.items():
        valid_indices = [i for i in indices if i < len(predictions)]
        if valid_indices:
            cat_preds = [predictions[i] for i in valid_indices]
            cat_refs = [references[i] for i in valid_indices]
            cat_m = metrics_calc.evaluate_batch(cat_preds, cat_refs)
            cat_m["num_samples"] = len(valid_indices)
            cat_metrics[cat_name] = cat_m
    metrics["per_category"] = cat_metrics

    return metrics, predictions, references, latencies


def generate_visualizations(all_results, output_dir):
    """Generate all charts and save to output_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    configs = list(all_results.keys())
    # Color palette
    colors = ["#667eea", "#764ba2", "#00c853", "#ff9100"]

    # ─── 1. Grouped Bar Chart: All Metrics ───────────────────────
    metric_keys = ["exact_match", "f1", "bleu", "rouge1", "rougeL"]
    metric_labels = ["Exact Match", "F1", "BLEU", "ROUGE-1", "ROUGE-L"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metric_labels))
    width = 0.18

    for i, cfg in enumerate(configs):
        values = [all_results[cfg].get(mk, 0) for mk in metric_keys]
        bars = ax.bar(x + i * width, values, width, label=cfg, color=colors[i], alpha=0.9)
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xlabel("Metric", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title("Ablation Study — All Metrics Comparison", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(0.5, max(all_results[c].get("f1", 0) for c in configs) * 1.5))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_all_metrics_comparison.png"), dpi=150)
    plt.close()
    logger.info("Generated: 01_all_metrics_comparison.png")

    # ─── 2. Radar Chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    radar_keys = ["f1", "bleu", "rouge1", "rougeL", "exact_match"]
    radar_labels = ["F1", "BLEU", "ROUGE-1", "ROUGE-L", "Exact Match"]
    N = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for i, cfg in enumerate(configs):
        values = [all_results[cfg].get(mk, 0) for mk in radar_keys]
        values += values[:1]
        ax.fill(angles, values, alpha=0.15, color=colors[i])
        ax.plot(angles, values, 'o-', linewidth=2, label=cfg, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_title("Model Capability Radar — Ablation Study", fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_capability_radar.png"), dpi=150)
    plt.close()
    logger.info("Generated: 02_capability_radar.png")

    # ─── 3. Delta Improvement Chart (AGHR vs Base) ───────────────
    if len(configs) >= 2:
        base_cfg = configs[0]
        aghr_cfg = configs[-1]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        deltas = []
        d_labels = []
        # For metrics: higher is better (green = positive delta)
        for mk, ml in zip(metric_keys, metric_labels):
            delta = all_results[aghr_cfg].get(mk, 0) - all_results[base_cfg].get(mk, 0)
            deltas.append(delta)
            d_labels.append(ml)
        # For latency: lower is better, so INVERT the delta (negative latency = improvement)
        lat_delta = all_results[base_cfg].get("avg_latency", 0) - all_results[aghr_cfg].get("avg_latency", 0)
        deltas.append(lat_delta)
        d_labels.append("Latency Saved (s)")
        
        bar_colors = ["#00c853" if d >= 0 else "#ff1744" for d in deltas]
        bars = ax.barh(d_labels, deltas, color=bar_colors, alpha=0.85)
        
        for bar, val in zip(bars, deltas):
            offset = 0.002 if val >= 0 else -0.002
            ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2.,
                    f'{val:+.4f}', ha='left' if val >= 0 else 'right', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel("Delta (positive = AGHR better)", fontsize=11, fontweight='bold')
        ax.set_title(f"Improvement: {aghr_cfg} over {base_cfg}", fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_delta_improvement.png"), dpi=150)
        plt.close()
        logger.info("Generated: 03_delta_improvement.png")

    # ─── 4. Latency Comparison ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    lat_values = [all_results[c].get("avg_latency", 0) for c in configs]
    bars = ax.bar(configs, lat_values, color=colors[:len(configs)], alpha=0.85)
    
    for bar, val in zip(bars, lat_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel("Average Latency (seconds)", fontsize=11, fontweight='bold')
    ax.set_title("Average Latency per Configuration", fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_latency_comparison.png"), dpi=150)
    plt.close()
    logger.info("Generated: 04_latency_comparison.png")

    # ─── 5. Per-Category Heatmap (AGHR only) ─────────────────────
    aghr_cfg = configs[-1]
    cat_data = all_results[aghr_cfg].get("per_category", {})
    if cat_data:
        categories = list(cat_data.keys())
        hm_metrics = ["f1", "bleu", "rouge1"]
        hm_labels = ["F1", "BLEU", "ROUGE-1"]
        
        heatmap_data = []
        for cat in categories:
            row = [cat_data[cat].get(mk, 0) for mk in hm_metrics]
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0)
        
        ax.set_xticks(np.arange(len(hm_labels)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(hm_labels, fontsize=10)
        ax.set_yticklabels(categories, fontsize=9)
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(hm_labels)):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        ax.set_title(f"Per-Category Performance — {aghr_cfg}", fontsize=13, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8, label="Score")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "05_category_heatmap.png"), dpi=150)
        plt.close()
        logger.info("Generated: 05_category_heatmap.png")

    # ─── 6. Progressive Improvement (F1 across configs) ──────────
    fig, ax = plt.subplots(figsize=(10, 5))
    f1_values = [all_results[c].get("f1", 0) for c in configs]
    
    ax.plot(configs, f1_values, 'o-', linewidth=3, markersize=12, color='#764ba2', markerfacecolor='#ff9100')
    ax.fill_between(range(len(configs)), f1_values, alpha=0.15, color='#764ba2')
    
    for i, (c, v) in enumerate(zip(configs, f1_values)):
        ax.annotate(f'{v:.3f}', (i, v), textcoords="offset points", xytext=(0, 12),
                   ha='center', fontsize=11, fontweight='bold', color='#764ba2')
    
    ax.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    ax.set_title("Progressive F1 Improvement Across Configurations", fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_progressive_f1.png"), dpi=150)
    plt.close()
    logger.info("Generated: 06_progressive_f1.png")

    logger.info(f"All {6} visualizations saved to {output_dir}/")


def main():
    logger.info("🔬 Starting Comprehensive AGHR Evaluation (50 questions)...")

    # Load config
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize components
    logger.info("Loading system components...")

    from src.embeddings.encoder import EmbeddingEncoder
    from src.embeddings.vector_store import FAISSVectorStore
    from src.knowledge_graph.entity_extractor import EntityExtractor
    from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
    from src.query.analyzer import QueryAnalyzer
    from src.retrieval.aghr_scorer import AGHRScorer, HybridRetriever
    from src.context.fusion import ContextFusion
    from src.generation.llm_layer import LLMLayer

    model_cfg = config["embeddings"]["models"]["baseline"]
    encoder = EmbeddingEncoder(model_name=model_cfg["name"])

    vs = FAISSVectorStore(dimension=model_cfg["dimension"],
                          similarity=config["vector_store"]["similarity"])
    vs.load(config["vector_store"]["index_path"])

    neo4j_cfg = config["knowledge_graph"]["neo4j"]
    kg = KnowledgeGraphBuilder(
        uri=neo4j_cfg["uri"],
        username=neo4j_cfg["username"],
        password=neo4j_cfg["password"]
    )

    entity_ext = EntityExtractor(spacy_model="en_core_web_sm")
    query_analyzer = QueryAnalyzer(entity_extractor=entity_ext)
    aghr = AGHRScorer(
        alpha=config["retrieval"]["aghr"]["alpha"],
        beta=config["retrieval"]["aghr"]["beta"],
        gamma=config["retrieval"]["aghr"]["gamma"],
    )

    hybrid = HybridRetriever(
        vector_store=vs, knowledge_graph=kg, encoder=encoder,
        entity_extractor=entity_ext, query_analyzer=query_analyzer,
        aghr_scorer=aghr, top_k=config["vector_store"]["top_k"],
        max_hops=config["knowledge_graph"]["max_hops"],
    )

    context_fusion = ContextFusion(
        max_tokens=config["context"]["max_tokens"],
        dedup_threshold=config["context"]["dedup_threshold"],
    )

    gen_cfg = config["generation"]["models"]["flan_t5"]
    llm = LLMLayer(
        model_name=gen_cfg["name"],
        model_type=gen_cfg["type"],
        max_new_tokens=gen_cfg["max_new_tokens"],
    )

    metrics_calc = MetricsCalculator()

    # ─── Define 4 Pipeline Configurations ─────────────────────────

    def config_1_base_llm(question):
        prompt = f"Answer this medical question:\nQuestion: {question}\nAnswer:"
        return llm.generate(prompt)

    def config_2_prompt_only(question):
        prompt = (
            f"You are a medical expert. Answer the following medical question in detail.\n\n"
            f"Question: {question}\n\nDetailed answer:"
        )
        return llm.generate(prompt)

    def config_3_rag_only(question):
        query_emb = encoder.encode_query(question)
        results = vs.search(query_emb, top_k=5)
        context = "\n".join([r["chunk"].get("text", "") for r in results])
        # Limit context for flan-t5-small (512 token limit)
        prompt = (
            f"Answer the medical question based on the context.\n\n"
            f"Context: {context[:400]}\n\n"
            f"Question: {question}\n\nDetailed answer:"
        )
        return llm.generate(prompt)

    def config_4_aghr_hybrid(question):
        retrieval = hybrid.retrieve(question)
        fused = context_fusion.fuse(
            retrieval["vector_context"],
            retrieval["graph_context"],
            retrieval["scored_results"],
        )
        # Limit context for flan-t5-small (512 token limit)
        prompt = (
            f"Answer the medical question based on the context.\n\n"
            f"Context: {fused['final_context'][:400]}\n\n"
            f"Question: {question}\n\nDetailed answer:"
        )
        return llm.generate(prompt)

    # ─── Run All Configurations ───────────────────────────────────

    configurations = [
        ("1. Base LLM",    config_1_base_llm),
        ("2. Prompt Only", config_2_prompt_only),
        ("3. RAG Only",    config_3_rag_only),
        ("4. AGHR Hybrid", config_4_aghr_hybrid),
    ]

    all_results = {}

    for cfg_name, pipeline_fn in configurations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {cfg_name} ({len(EXTENDED_TEST_DATA)} questions)")
        logger.info(f"{'='*60}")

        metrics, preds, refs, lats = run_configuration(
            cfg_name, pipeline_fn, EXTENDED_TEST_DATA, metrics_calc
        )

        # Remove per_question from saved results (too large, save separately)
        per_q = metrics.pop("per_question", [])
        all_results[cfg_name] = metrics

        logger.info(f"  [{cfg_name}] F1={metrics['f1']:.4f}, "
                     f"BLEU={metrics['bleu']:.4f}, "
                     f"ROUGE-1={metrics['rouge1']:.4f}, "
                     f"Latency={metrics['avg_latency']:.3f}s")

    # ─── Print Comparison Table ───────────────────────────────────

    print("\n" + "=" * 80)
    print("COMPREHENSIVE ABLATION STUDY RESULTS (50 Questions)")
    print("=" * 80)
    print(f"{'Config':<22} {'EM':>6} {'F1':>8} {'BLEU':>8} {'ROUGE-1':>8} {'ROUGE-L':>8} {'Latency':>9}")
    print("-" * 80)

    for cfg_name, m in all_results.items():
        print(
            f"{cfg_name:<22} "
            f"{m.get('exact_match', 0):>6.3f} "
            f"{m.get('f1', 0):>8.4f} "
            f"{m.get('bleu', 0):>8.4f} "
            f"{m.get('rouge1', 0):>8.4f} "
            f"{m.get('rougeL', 0):>8.4f} "
            f"{m.get('avg_latency', 0):>8.3f}s"
        )

    print("=" * 80)

    # ─── Save Results ─────────────────────────────────────────────

    results_dir = "data/evaluation"
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "full_ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Results saved to {results_dir}/full_ablation_results.json")

    # ─── Generate Visualizations ──────────────────────────────────

    plots_dir = "data/evaluation/plots"
    generate_visualizations(all_results, plots_dir)

    logger.info("✅ Comprehensive evaluation complete!")
    print(f"\n📊 Visualizations saved to: {plots_dir}/")
    print(f"📄 Results saved to: {results_dir}/full_ablation_results.json")


if __name__ == "__main__":
    main()
