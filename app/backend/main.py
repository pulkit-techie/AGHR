"""
AGHR System — FastAPI Backend (Phase 17)
==========================================
REST API for the AGHR hybrid QA system.

Endpoints:
  POST /api/query  — ask a question, get structured answer
  GET  /api/health — health check
  GET  /api/stats  — system statistics
"""

import sys
import json
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

app = FastAPI(
    title="AGHR — Adaptive Graph-Hybrid Retrieval",
    description="Hybrid RAG + Knowledge Graph QA system with hallucination guard",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ─────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="User's question", min_length=3)
    strategy: str = Field("chain_of_thought", description="Prompting strategy")
    top_k: int = Field(5, description="Number of retrieval results")

class QueryResponse(BaseModel):
    question: str
    answer: str
    reasoning: str = ""
    sources: str = ""
    confidence: float = 0.0
    route: str = ""
    is_reliable: bool = True
    used_fallback: bool = False
    retrieval_scores: List[float] = []
    latency_ms: float = 0.0
    kg_context: str = ""

class HealthResponse(BaseModel):
    status: str = "healthy"
    components: Dict[str, str] = {}

# ── Global System Components ──────────────────────────────

_system = {}

def get_system():
    """Lazy-initialize system components."""
    if _system:
        return _system

    from src.utils.helpers import load_config
    from src.embeddings.encoder import EmbeddingEncoder
    from src.embeddings.vector_store import FAISSVectorStore
    from src.knowledge_graph.entity_extractor import EntityExtractor
    from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
    from src.query.analyzer import QueryAnalyzer
    from src.retrieval.aghr_scorer import AGHRScorer, HybridRetriever
    from src.context.fusion import ContextFusion
    from src.generation.prompt_engine import PromptEngine
    from src.generation.llm_layer import LLMLayer
    from src.hallucination.guard import HallucinationGuard

    config = load_config("config/config.yaml")

    # Encoder
    model_cfg = config["embeddings"]["models"]["baseline"]
    encoder = EmbeddingEncoder(model_name=model_cfg["name"])

    # Vector store
    vs = FAISSVectorStore(dimension=model_cfg["dimension"],
                          similarity=config["vector_store"]["similarity"])
    vs.load(config["vector_store"]["index_path"])

    # Knowledge graph
    kg = KnowledgeGraphBuilder()
    kg_path = "data/kg_triples/knowledge_graph.json"
    if Path(kg_path).exists():
        kg.load(kg_path)

    # Components
    entity_ext = EntityExtractor(spacy_model="en_core_web_sm")
    query_analyzer = QueryAnalyzer(entity_extractor=entity_ext)
    aghr_scorer = AGHRScorer(
        alpha=config["retrieval"]["aghr"]["alpha"],
        beta=config["retrieval"]["aghr"]["beta"],
        gamma=config["retrieval"]["aghr"]["gamma"],
    )

    hybrid = HybridRetriever(
        vector_store=vs, knowledge_graph=kg, encoder=encoder,
        entity_extractor=entity_ext, query_analyzer=query_analyzer,
        aghr_scorer=aghr_scorer,
        top_k=config["vector_store"]["top_k"],
        max_hops=config["knowledge_graph"]["max_hops"],
    )

    _system.update({
        "config": config,
        "encoder": encoder,
        "vector_store": vs,
        "kg": kg,
        "hybrid_retriever": hybrid,
        "context_fusion": ContextFusion(max_tokens=config["context"]["max_tokens"]),
        "prompt_engine": PromptEngine(),
        "llm": LLMLayer(model_name=config["generation"]["models"]["flan_t5"]["name"],
                         model_type="seq2seq"),
        "guard": HallucinationGuard(
            confidence_threshold=config["hallucination"]["confidence_threshold"]),
    })

    return _system

# ── Endpoints ─────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        components={
            "vector_store": "loaded" if _system.get("vector_store") else "not loaded",
            "knowledge_graph": "loaded" if _system.get("kg") else "not loaded",
            "llm": "loaded" if _system.get("llm") else "not loaded",
        },
    )

@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    system = get_system()
    start = time.time()

    try:
        # Retrieve
        retrieval = system["hybrid_retriever"].retrieve(request.question)

        # Fuse context
        fused = system["context_fusion"].fuse(
            retrieval["vector_context"],
            retrieval["graph_context"],
            retrieval["scored_results"],
        )

        # Generate
        gen_result = system["llm"].generate_with_context(
            question=request.question,
            context=fused["final_context"],
            prompt_engine=system["prompt_engine"],
            strategy=request.strategy,
        )

        # Hallucination guard
        scores = [r["aghr_score"] for r in retrieval["scored_results"]]
        guard_result = system["guard"].evaluate(
            answer=gen_result["answer"],
            context=fused["final_context"],
            retrieval_scores=scores,
        )

        latency = (time.time() - start) * 1000

        return QueryResponse(
            question=request.question,
            answer=guard_result["final_answer"],
            reasoning=gen_result.get("reasoning", ""),
            sources=gen_result.get("sources", ""),
            confidence=guard_result["confidence"],
            route=retrieval["route"],
            is_reliable=guard_result["is_reliable"],
            used_fallback=guard_result["used_fallback"],
            retrieval_scores=scores[:5],
            latency_ms=round(latency, 1),
            kg_context=retrieval.get("graph_context", ""),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
def get_stats():
    system = get_system()
    kg_stats = system["kg"].get_stats() if system.get("kg") else {}
    return {
        "vector_store_size": system["vector_store"].index.ntotal if system.get("vector_store") else 0,
        "knowledge_graph": kg_stats,
        "model": system["config"]["generation"]["models"]["flan_t5"]["name"],
        "aghr_weights": system["config"]["retrieval"]["aghr"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
