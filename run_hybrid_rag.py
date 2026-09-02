"""
AGHR System — Hybrid RAG Pipeline Demo (Macro-Phase B)
=======================================================
End-to-end script demonstrating the full hybrid retrieval system:
  1. Load existing chunks + FAISS index (from Phase A)
  2. Build Knowledge Graph (entity + relation extraction)
  3. Query analysis + routing
  4. AGHR hybrid scoring
  5. Context fusion
  6. LLM generation with FLAN-T5

Run: python run_hybrid_rag.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def main():
    from src.utils.logger import setup_logger
    from src.utils.helpers import load_config

    setup_logger()
    config = load_config("config/config.yaml")
    logger.info("=" * 60)
    logger.info("AGHR System — Hybrid RAG Pipeline (Phase A+B)")
    logger.info("=" * 60)

    # ── Load existing data from Phase A ──────────────────
    logger.info("\n📂 Loading Phase A artifacts...")

    from src.data_pipeline.chunker import DocumentChunker
    from src.embeddings.encoder import EmbeddingEncoder
    from src.embeddings.vector_store import FAISSVectorStore

    # Load chunks
    chunks_path = "data/chunks/chunks_train.jsonl"
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    logger.info(f"Loaded {len(chunks)} train chunks")

    # Load encoder
    model_cfg = config["embeddings"]["models"]["baseline"]
    encoder = EmbeddingEncoder(
        model_name=model_cfg["name"],
        batch_size=config["embeddings"]["batch_size"],
        normalize=config["embeddings"]["normalize"],
    )

    # Load FAISS index
    vector_store = FAISSVectorStore(
        dimension=model_cfg["dimension"],
        similarity=config["vector_store"]["similarity"],
    )
    vector_store.load(config["vector_store"]["index_path"])

    # ── Phase 3: Build Knowledge Graph ────────────────────
    logger.info("\n🧠 PHASE 3 — Knowledge Graph Construction")

    from src.knowledge_graph.entity_extractor import EntityExtractor
    from src.knowledge_graph.relation_extractor import RelationExtractor
    from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder

    # Use a subset for speed (first 500 chunks)
    kg_chunks = chunks[:500]

    # Entity extraction
    logger.info("Extracting entities...")
    entity_extractor = EntityExtractor(spacy_model="en_core_web_sm", use_medical_rules=True)
    entity_results = entity_extractor.extract_from_documents(kg_chunks)

    unique_entities = entity_extractor.get_unique_entities(entity_results)
    logger.info(f"Top entities: {[e['text'] for e in unique_entities[:10]]}")

    # Relation extraction
    logger.info("Extracting relations...")
    relation_extractor = RelationExtractor(spacy_model="en_core_web_sm")
    relation_results = relation_extractor.extract_from_documents(
        kg_chunks, entity_results=entity_results
    )

    all_triples = relation_extractor.get_all_triples(relation_results)
    logger.info(f"Sample triples: {all_triples[:5]}")

    # Build graph
    logger.info("Building knowledge graph...")
    kg = KnowledgeGraphBuilder()
    kg.build_from_relation_results(relation_results)

    kg_stats = kg.get_stats()
    logger.info(f"KG stats: {json.dumps(kg_stats, indent=2)}")

    # Save KG
    kg_path = "data/kg_triples/knowledge_graph.json"
    kg.save(kg_path)

    # ── Phase 4: Query Analysis ───────────────────────────
    logger.info("\n🧭 PHASE 4 — Query Understanding")

    from src.query.analyzer import QueryAnalyzer

    query_analyzer = QueryAnalyzer(entity_extractor=entity_extractor)

    # ── Phase 5: AGHR Hybrid Retrieval ────────────────────
    logger.info("\n🚀 PHASE 5 — AGHR Hybrid Retrieval")

    from src.retrieval.aghr_scorer import AGHRScorer, HybridRetriever
    from src.context.fusion import ContextFusion

    aghr_scorer = AGHRScorer(
        alpha=config["retrieval"]["aghr"]["alpha"],
        beta=config["retrieval"]["aghr"]["beta"],
        gamma=config["retrieval"]["aghr"]["gamma"],
    )

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        knowledge_graph=kg,
        encoder=encoder,
        entity_extractor=entity_extractor,
        query_analyzer=query_analyzer,
        aghr_scorer=aghr_scorer,
        top_k=config["vector_store"]["top_k"],
        max_hops=config["knowledge_graph"]["max_hops"],
    )

    context_fusion = ContextFusion(
        max_tokens=config["context"]["max_tokens"],
        dedup_threshold=config["context"]["dedup_threshold"],
    )

    # ── Test queries ──────────────────────────────────────
    test_queries = [
        "What causes heart disease?",
        "How does diabetes affect the kidney?",
        "What is the treatment for hypertension?",
        "Who directed the movie Leviathan?",
        "What are symptoms of depression in elderly?",
    ]

    for query in test_queries:
        logger.info(f"\n{'='*50}")
        logger.info(f"🔎 Query: {query}")

        # Retrieve
        start_time = time.time()
        result = hybrid_retriever.retrieve(query)
        retrieval_time = time.time() - start_time

        # Analysis
        analysis = result["analysis"]
        logger.info(f"   Route: {result['route']} | "
                     f"Complexity: {analysis['complexity_label']} | "
                     f"Entities: {[e['text'] for e in analysis['entities']]}")

        # Fuse context
        fused = context_fusion.fuse(
            result["vector_context"],
            result["graph_context"],
            result["scored_results"],
        )

        # Show top results
        for r in result["scored_results"][:3]:
            logger.info(f"   [{r['source']}] AGHR={r['aghr_score']:.4f} "
                         f"(vec={r['sim_vec']:.3f}, path={r['path_score']:.3f}, "
                         f"ent={r['entity_overlap']:.3f}) | "
                         f"{r['text'][:80]}...")

        logger.info(f"   ⏱️ Retrieval time: {retrieval_time:.3f}s")
        logger.info(f"   📊 Fusion: {fused['stats']}")

    # ── Generation with FLAN-T5 ───────────────────────────
    logger.info("\n🤖 PHASE 7 — LLM Generation (FLAN-T5)")

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from src.utils.helpers import load_prompts

        prompts = load_prompts("config/prompts.yaml")
        model_name = config["generation"]["models"]["flan_t5"]["name"]

        logger.info(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Test structured generation
        query = "What causes heart disease?"
        result = hybrid_retriever.retrieve(query)
        fused = context_fusion.fuse(
            result["vector_context"],
            result["graph_context"],
            result["scored_results"],
        )

        prompt = prompts["chain_of_thought"].format(
            context=fused["final_context"][:1500],
            question=query,
        )

        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.3, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"\n📋 Question: {query}")
        logger.info(f"📝 Answer: {answer}")
        logger.info(f"🔗 Route: {result['route']}")
        scores = [f"{r['aghr_score']:.3f}" for r in result['scored_results'][:3]]
        logger.info(f"📊 AGHR Scores: {scores}")

    except Exception as e:
        logger.warning(f"Generation skipped: {e}")

    # ── Summary ───────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✅ Hybrid RAG Pipeline Complete!")
    logger.info(f"   🧠 KG: {kg_stats['nodes']} nodes, {kg_stats['edges']} edges")
    logger.info(f"   🔢 Unique entities: {len(unique_entities)}")
    logger.info(f"   📐 Triples: {len(all_triples)}")
    logger.info(f"   🚀 AGHR weights: α={aghr_scorer.alpha}, β={aghr_scorer.beta}, γ={aghr_scorer.gamma}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
