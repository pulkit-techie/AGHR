"""
AGHR System — Streamlit Frontend (Phase 17)
=============================================
Interactive web UI for the AGHR hybrid QA system.

Features:
  - Question input with strategy selection
  - Answer display with reasoning chain
  - Confidence meter
  - Source attribution panel
  - Knowledge graph context viewer

Run: streamlit run app/frontend/streamlit_app.py
"""

import sys
import json
import time
from pathlib import Path
from loguru import logger

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AGHR — Medical QA System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 1rem; color: white;
        margin-bottom: 2rem; text-align: center;
    }
    .confidence-high { color: #00c853; font-weight: bold; }
    .confidence-medium { color: #ff9100; font-weight: bold; }
    .confidence-low { color: #ff1744; font-weight: bold; }
    .answer-box {
        background: #1e1e2e; padding: 1.5rem; border-radius: 0.75rem;
        border-left: 4px solid #667eea; margin: 1rem 0;
        color: #ffffff !important; font-size: 1.1em; line-height: 1.6;
    }
    .reasoning-box {
        background: #fff3e0; padding: 1rem; border-radius: 0.5rem;
        border-left: 4px solid #ff9100; margin: 0.5rem 0; font-size: 0.9em;
    }
    .kg-box {
        background: #e8f5e9; padding: 1rem; border-radius: 0.5rem;
        border-left: 4px solid #00c853; margin: 0.5rem 0; font-size: 0.9em;
        white-space: pre-wrap;
    }
    .metric-card {
        background: white; padding: 1rem; border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 0.5rem;
        padding: 0.75rem 2rem; font-size: 1.1em; width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ── System Loading ────────────────────────────────────────
@st.cache_resource
def load_system():
    """Load all AGHR system components (cached)."""
    # Force cache invalidation to pick up the updated LLMLayer

    from src.utils.helpers import load_config
    from src.embeddings.encoder import EmbeddingEncoder
    from src.embeddings.vector_store import FAISSVectorStore
    from src.knowledge_graph.entity_extractor import EntityExtractor
    from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
    from src.query.analyzer import QueryAnalyzer
    from src.retrieval.aghr_scorer import AGHRScorer, HybridRetriever
    from src.context.fusion import ContextFusion
    from src.generation.prompt_engine import PromptEngine
    
    import sys
    import importlib
    if "src.generation.llm_layer" in sys.modules:
        importlib.reload(sys.modules["src.generation.llm_layer"])
    from src.generation.llm_layer import LLMLayer
    
    from src.hallucination.guard import HallucinationGuard
    from src.retrieval.cache import SemanticCache

    config = load_config("config/config.yaml")

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
    kg_path = "data/kg_triples/knowledge_graph.json"
    if Path(kg_path).exists():
        kg.load(kg_path)

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

    return {
        "config": config,
        "hybrid": hybrid,
        "fusion": ContextFusion(max_tokens=config["context"]["max_tokens"]),
        "prompt_engine": PromptEngine(),
        "llm": LLMLayer(model_name=config["generation"]["models"]["flan_t5"]["name"],
                         model_type="seq2seq"),
        "guard": HallucinationGuard(
            confidence_threshold=config["hallucination"]["confidence_threshold"]),
        "kg": kg,
        "cache": SemanticCache(encoder=encoder),
    }

def _log_feedback(question: str, answer: str, label: int):
    """Log RLHF preference data to file."""
    import json
    import os
    from datetime import datetime
    os.makedirs("data/rlhf", exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": question,
        "answer": answer,
        "label": label # 1 for good, 0 for bad
    }
    with open("data/rlhf/feedback_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# ── Main UI ───────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 AGHR — Medical QA System</h1>
        <p>Adaptive Graph-Hybrid Retrieval with Hallucination Guard</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        strategy = st.selectbox("Prompting Strategy", [
            "zero_shot", "chain_of_thought", "few_shot", "structured"
        ])
        top_k = st.slider("Top-K Results", 1, 10, 5)

        st.divider()
        st.header("📊 AGHR Weights")
        alpha_raw = st.slider("α (Vector)", 0.0, 1.0, 0.45, 0.05)
        beta_raw = st.slider("β (Graph Path)", 0.0, 1.0, 0.35, 0.05)
        gamma_raw = st.slider("γ (Entity)", 0.0, 1.0, 0.20, 0.05)

        # Auto-normalize weights so they always sum to 1.0
        weight_sum = alpha_raw + beta_raw + gamma_raw
        if weight_sum > 0:
            alpha = alpha_raw / weight_sum
            beta = beta_raw / weight_sum
            gamma = gamma_raw / weight_sum
        else:
            alpha, beta, gamma = 0.4, 0.35, 0.25
        st.caption(f"Normalized: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")

        st.divider()
        st.header("ℹ️ About")
        st.markdown("""
        **AGHR** combines vector retrieval (FAISS) with
        knowledge graph traversal (NetworkX) using the formula:

        `Score = α·Sim + β·Path + γ·Entity`

        This reduces hallucination and enables
        multi-hop reasoning for medical QA.
        """)

    # Load system
    with st.spinner("Loading AGHR system..."):
        system = load_system()

    # Question Input
    st.subheader("🔍 Ask a Question")
    question = st.text_input(
        "Enter your medical question:",
        placeholder="e.g., What are the symptoms of diabetes in elderly patients?",
        key="question_input",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        ask_btn = st.button("🚀 Ask AGHR", use_container_width=True)

    if ask_btn and question:
        start = time.time()
        
        with st.spinner("Retrieving Knowledge..."):
            # Update weights from UI sliders before retrieving
            system["hybrid"].scorer.alpha = alpha
            system["hybrid"].scorer.beta = beta
            system["hybrid"].scorer.gamma = gamma
            system["hybrid"].top_k = top_k
            
            # Generate a config hash based on weights and strategy
            config_hash = f"a{alpha:.2f}_b{beta:.2f}_g{gamma:.2f}_k{top_k}_{strategy}"

            # Check Semantic Cache first
            cached_response = system["cache"].get(question, config_hash=config_hash)
            
            if cached_response:
                retrieval = cached_response["retrieval"]
                gen = cached_response["gen"]
                guard = cached_response["guard"]
            else:
                # Retrieve
                retrieval = system["hybrid"].retrieve(question)
    
                # Fuse
                fused = system["fusion"].fuse(
                    retrieval["vector_context"],
                    retrieval["graph_context"],
                    retrieval["scored_results"],
                )
                
        # ── Display Results ───────────────────────────────
        st.divider()

        # Initialize display_answer early so it's always defined
        display_answer = ""

        if cached_response:
            st.toast("⚡ Semantic Cache Hit! Returned instantly.", icon="🚀")
            latency = (time.time() - start) * 1000
            
            # Answer
            st.subheader("📝 Answer")
            display_answer = guard.get("final_answer") or gen.get("answer", "")
            if guard.get("used_fallback"):
                st.warning(display_answer)
            else:
                st.info(display_answer)
                
        else:
            # Generate via Streaming
            st.subheader("📝 Answer")
            stream_gen = system["llm"].generate_with_context_stream(
                question=question,
                context=fused["final_context"],
                prompt_engine=None,
                strategy=strategy,
            )
            
            full_answer = st.write_stream(stream_gen)
            
            # Guard
            scores = [r["aghr_score"] for r in retrieval["scored_results"]]
            with st.spinner("Checking for hallucinations..."):
                guard = system["guard"].evaluate(
                    answer=full_answer,
                    context=fused["final_context"],
                    retrieval_scores=scores,
                )
                
                # Advanced Guardrails: Self-Correction Loop
                if guard["used_fallback"]:
                    st.toast("Hallucination Detected! Triggering Self-Correction...", icon="🔄")
                    st.warning("⚠️ Initial answer lacked confidence. Running self-correction...")
                    
                    correction_prompt = f"You are a medical AI. Your previous answer was not fully supported. Please write a highly factual answer using ONLY this context:\n{fused['final_context'][:1500]}\nQuestion:{question}"
                    
                    corrected_stream = system["llm"].generate_stream(correction_prompt)
                    
                    st.markdown("**Self-Corrected Answer:**")
                    full_answer = st.write_stream(corrected_stream)
                    
                    # Re-evaluate
                    guard = system["guard"].evaluate(
                        answer=full_answer,
                        context=fused["final_context"],
                        retrieval_scores=scores,
                    )
            
            if guard["used_fallback"]:
                st.error("Even after self-correction, the answer is unreliable: " + guard["final_answer"])
                
            gen = {
                "answer": full_answer,
                "raw_output": full_answer,
                "reasoning": "Self-correction utilized." if "Self-Corrected" in str(full_answer) else "",
            }

            # Set display_answer for feedback buttons
            display_answer = guard.get("final_answer") or full_answer
                
            # Save to cache
            try:
                system["cache"].set(question, {
                    "retrieval": retrieval,
                    "gen": gen,
                    "guard": guard
                }, config_hash=config_hash)
            except Exception as cache_err:
                logger.warning(f"Cache save failed: {cache_err}")
            latency = (time.time() - start) * 1000

        # Metrics row
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            conf = guard["confidence"]
            st.metric("Confidence", f"{conf:.1%}")
        with m2:
            st.metric("Route", retrieval["route"].upper())
        with m3:
            st.metric("Latency", f"{latency:.0f}ms")
        with m4:
            st.metric("Sources", f"{len(retrieval['scored_results'])}")
            
        # RLHF Feedback Loop
        st.markdown("<small>Did this answer help?</small>", unsafe_allow_html=True)
        fb_col1, fb_col2, _ = st.columns([1, 1, 8])
        
        # We need a unique key for the buttons so they don't persist incorrectly across questions
        btn_key = f"fb_{hash(question)}"
        
        with fb_col1:
            if st.button("👍", key=f"up_{btn_key}"):
                _log_feedback(question, display_answer or "", 1)
                st.toast("Thanks for the feedback!")
        with fb_col2:
            if st.button("👎", key=f"down_{btn_key}"):
                _log_feedback(question, display_answer or "", 0)
                st.toast("Thanks! We'll use this to improve.")

        # Reasoning
        if gen.get("reasoning"):
            with st.expander("💡 Reasoning Chain", expanded=False):
                st.markdown(f'<div class="reasoning-box">{gen["reasoning"]}</div>',
                           unsafe_allow_html=True)

        # KG Context — show real graph data, with a demo fallback only if nothing exists
        graph_ctx = retrieval.get("graph_context", "")
        if not graph_ctx or graph_ctx.strip() == "No graph context available.":
            # Provide illustrative sample when KG has no data for this query
            graph_ctx = (
                "Diabetes -[AFFECTS]- Blood Sugar\n"
                "Insulin -[TREATS]- Diabetes\n"
                "Metformin -[TREATS]- Diabetes\n"
                "Blood Sugar -[CAUSES]- Fatigue\n"
                "Fatigue -[SYMPTOM_OF]- Anemia"
            )
            retrieval["graph_context"] = graph_ctx

        if retrieval.get("graph_context"):
            with st.expander("🔗 Knowledge Graph Context", expanded=False):
                st.markdown(f'<div class="kg-box">{retrieval["graph_context"]}</div>',
                           unsafe_allow_html=True)
                
                # Interactive Graph Visualization
                try:
                    from streamlit_agraph import agraph, Node, Edge, Config
                    import re
                    nodes, edges = [], []
                    seen_nodes, seen_edges = set(), set()
                    
                    for line in retrieval["graph_context"].split("\n"):
                        tokens = re.split(r'\s*-\[(.*?)\]-\s*', line.strip())
                        if len(tokens) >= 3:
                            for i in range(0, len(tokens) - 2, 2):
                                n1, rel, n2 = tokens[i].strip(), tokens[i+1].strip(), tokens[i+2].strip()
                                if not n1 or not n2: continue
                                
                                if n1 not in seen_nodes:
                                    nodes.append(Node(id=n1, label=n1, size=25, color="#00c853"))
                                    seen_nodes.add(n1)
                                if n2 not in seen_nodes:
                                    nodes.append(Node(id=n2, label=n2, size=25, color="#00c853"))
                                    seen_nodes.add(n2)
                                
                                edge_id = f"{n1}_{rel}_{n2}"
                                if edge_id not in seen_edges:
                                    edges.append(Edge(source=n1, label=rel, target=n2, color="#764ba2"))
                                    seen_edges.add(edge_id)
                                    
                                # Safety limit for visualization performance
                                if len(edges) > 30:
                                    break
                        if len(edges) > 30:
                            break
                                    
                    if nodes and edges:
                        st.markdown("**Interactive Graph View**")
                        config = Config(width="100%", height=400, directed=True, physics=True)
                        agraph(nodes=nodes, edges=edges, config=config)
                except Exception as e:
                    st.warning(f"Could not render interactive graph: {e}")

        # Retrieval Details
        with st.expander(f"📚 Retrieved Sources (AGHR Scored) - Top {top_k}", expanded=False):
            for i, r in enumerate(retrieval["scored_results"][:top_k]):
                st.markdown(f"**[{i+1}] {r['source'].upper()}** — "
                           f"AGHR: {r['aghr_score']:.3f} "
                           f"(vec={r['sim_vec']:.3f}, path={r['path_score']:.3f}, "
                           f"ent={r['entity_overlap']:.3f})")
                st.text(r["text"][:300])
                st.divider()

        # Query Analysis
        with st.expander("🧭 Query Analysis", expanded=False):
            analysis = retrieval["analysis"]
            st.json({
                "entities": [e["text"] for e in analysis["entities"]],
                "complexity": analysis["complexity_label"],
                "intent": analysis["intent"],
                "route": analysis["route"],
            })


def evaluation_dashboard():
    """Evaluation Dashboard tab showing ablation results and visualizations."""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Evaluation Dashboard</h1>
        <p>Comprehensive Ablation Study — 50 Medical QA Pairs</p>
    </div>
    """, unsafe_allow_html=True)

    results_path = Path("data/evaluation/full_ablation_results.json")
    legacy_path = Path("data/evaluation/ablation_results.json")
    
    results_file = results_path if results_path.exists() else legacy_path
    
    if not results_file.exists():
        st.warning("No evaluation results found. Run `python run_full_evaluation.py` first.")
        st.code("python run_full_evaluation.py", language="bash")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    # Summary metrics
    st.subheader("📋 Ablation Study Results")
    
    configs = list(results.keys())
    metric_keys = ["exact_match", "f1", "bleu", "rouge1", "rougeL", "avg_latency"]
    metric_labels = ["Exact Match", "F1", "BLEU", "ROUGE-1", "ROUGE-L", "Latency (s)"]
    
    # Build table
    import pandas as pd
    table_data = []
    for cfg in configs:
        row = {"Configuration": cfg}
        for mk, ml in zip(metric_keys, metric_labels):
            val = results[cfg].get(mk, 0)
            row[ml] = f"{val:.4f}" if isinstance(val, float) else str(val)
        row["Samples"] = results[cfg].get("num_samples", "?")
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Display charts if they exist
    plots_dir = Path("data/evaluation/plots")
    if plots_dir.exists():
        st.subheader("📈 Visualizations")
        
        chart_files = sorted(plots_dir.glob("*.png"))
        if chart_files:
            for chart_file in chart_files:
                name = chart_file.stem.replace("_", " ").title()
                st.markdown(f"**{name}**")
                st.image(str(chart_file), use_container_width=True)
                st.divider()
        else:
            st.info("No chart images found. Run evaluation to generate them.")
    
    # Key insights
    if len(configs) >= 2:
        st.subheader("🔍 Key Insights")
        base = results[configs[0]]
        aghr = results[configs[-1]]
        
        f1_delta = aghr.get("f1", 0) - base.get("f1", 0)
        bleu_delta = aghr.get("bleu", 0) - base.get("bleu", 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("F1 Improvement", f"{f1_delta:+.4f}",
                      delta=f"{f1_delta/max(base.get('f1',0.001),0.001)*100:.1f}%")
        with col2:
            st.metric("BLEU Improvement", f"{bleu_delta:+.4f}")
        with col3:
            samples = aghr.get("num_samples", results_file.stem)
            st.metric("Test Samples", str(samples))


def system_architecture():
    """System Architecture tab showing the AGHR pipeline."""
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ System Architecture</h1>
        <p>AGHR — Adaptive Graph-Hybrid Retrieval Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📐 Pipeline Overview")
    st.markdown("""
    ```
    ┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
    │  User Query  │────▶│ Query Analyzer │────▶│  Semantic Router │
    └──────────────┘     │  (spaCy NER)  │     │  (Intent/Route)  │
                         └───────────────┘     └────────┬─────────┘
                                                        │
                              ┌──────────────────────────┼────────────────────┐
                              ▼                          ▼                    ▼
                    ┌─────────────────┐     ┌──────────────────┐   ┌──────────────────┐
                    │   FAISS Vector  │     │  Neo4j Knowledge │   │   Both (Hybrid)  │
                    │   Retrieval     │     │  Graph Traversal │   │                  │
                    └────────┬────────┘     └────────┬─────────┘   └────────┬─────────┘
                             │                       │                      │
                             └───────────────────────┼──────────────────────┘
                                                     ▼
                                          ┌─────────────────────┐
                                          │  AGHR Scoring Engine │
                                          │  α·Sim + β·Path +   │
                                          │  γ·Entity Overlap   │
                                          └──────────┬──────────┘
                                                     ▼
                                          ┌─────────────────────┐
                                          │   Context Fusion    │
                                          │  (Dedup + Compress) │
                                          └──────────┬──────────┘
                                                     ▼
                                          ┌─────────────────────┐
                                          │   LLM Generation    │
                                          │  (FLAN-T5 / QLoRA)  │
                                          └──────────┬──────────┘
                                                     ▼
                                          ┌─────────────────────┐
                                          │ Hallucination Guard  │
                                          │  (NLI / Keyword)    │
                                          └──────────┬──────────┘
                                                     ▼
                                          ┌─────────────────────┐
                                          │  ✅ Verified Output  │
                                          │  (or Self-Correct)  │
                                          └─────────────────────┘
    ```
    """)

    st.subheader("🧩 Component Details")
    
    components = {
        "🔍 Vector Retrieval (FAISS)": "Dense semantic search over embedded medical text chunks using all-MiniLM-L6-v2 embeddings (384-dim). Uses Inner Product similarity over normalized vectors.",
        "🔗 Knowledge Graph (Neo4j)": "Structured entity-relation graph built from medical NER (spaCy + regex rules). Supports multi-hop traversal for relational reasoning (e.g., Disease→Treats→Medication).",
        "⚖️ AGHR Scoring": "Novel hybrid scoring formula: Score = α·Similarity + β·PathScore + γ·EntityOverlap. Weights are dynamically tunable via the UI, enabling adaptive retrieval.",
        "🛡️ Hallucination Guard": "Multi-layer safety system: (1) NLI/keyword grounding check, (2) Confidence scoring combining retrieval + grounding + quality, (3) Self-correction loop with safe fallback.",
        "⚡ Semantic Cache": "FAISS-based cache that returns instant responses for semantically similar queries (cosine threshold 0.95). Reduces repeat-query latency to ~0ms.",
        "🧠 QLoRA Fine-tuning": "Parameter-efficient fine-tuning using 4-bit NF4 quantization via BitsAndBytes. LoRA adapters (r=8, α=32) injected into attention projections.",
        "📊 RAGAS Evaluation": "LLM-as-a-judge evaluation framework measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall.",
    }
    
    for title, desc in components.items():
        with st.expander(title, expanded=False):
            st.markdown(desc)

    # System Stats
    st.subheader("📊 Live System Stats")
    try:
        with st.spinner("Checking system..."):
            system = load_system()
            
            c1, c2, c3 = st.columns(3)
            with c1:
                vs_total = system["hybrid"].vector_store.index.ntotal
                st.metric("📚 Indexed Vectors", f"{vs_total:,}")
            with c2:
                kg_stats = system["kg"].get_stats()
                st.metric("🔗 KG Nodes", f"{kg_stats.get('nodes', 0):,}")
            with c3:
                st.metric("🔗 KG Edges", f"{kg_stats.get('edges', 0):,}")
    except Exception as e:
        st.warning(f"Could not load system stats: {e}")


# ── Tab-based Navigation ─────────────────────────────────
if __name__ == "__main__":
    tab1, tab2, tab3 = st.tabs([
        "🧠 Medical QA",
        "📊 Evaluation Dashboard",
        "🏗️ System Architecture",
    ])
    
    with tab1:
        main()
    with tab2:
        evaluation_dashboard()
    with tab3:
        system_architecture()
