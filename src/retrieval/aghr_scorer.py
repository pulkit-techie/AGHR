"""
AGHR System — AGHR Hybrid Scorer (Phase 5.3) ⭐ KEY NOVELTY
==============================================================
Adaptive Graph-Hybrid Retrieval scoring formula:

  Score = α·Sim_vec + β·Path_score + γ·Entity_overlap

Where:
  α = vector similarity weight (default 0.40)
  β = graph path score weight (default 0.35)
  γ = entity overlap weight (default 0.25)
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple

from loguru import logger


class AGHRScorer:
    """
    Core AGHR hybrid scoring engine.

    Combines three signals:
      1. Vector similarity (cosine) from FAISS/Chroma
      2. Graph path score from KG traversal
      3. Entity overlap between query and retrieved context
    """

    def __init__(self, alpha: float = 0.40, beta: float = 0.35, gamma: float = 0.25):
        # Soft normalization instead of hard assert (supports dynamic UI weight tuning)
        weight_sum = alpha + beta + gamma
        if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"AGHR weights sum to {weight_sum:.4f}, auto-normalizing to 1.0")
            alpha, beta, gamma = alpha / weight_sum, beta / weight_sum, gamma / weight_sum
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        logger.info(f"AGHR Scorer: α={self.alpha:.3f}, β={self.beta:.3f}, γ={self.gamma:.3f}")

    def compute_score(self, sim_vec: float, path_score: float,
                      entity_overlap: float) -> float:
        """
        Compute AGHR hybrid score.

        Args:
            sim_vec: Vector similarity score [0, 1].
            path_score: Graph path relevance score [0, 1].
            entity_overlap: Entity overlap ratio [0, 1].

        Returns:
            Combined AGHR score [0, 1].
        """
        score = (self.alpha * sim_vec +
                 self.beta * path_score +
                 self.gamma * entity_overlap)
        return min(max(score, 0.0), 1.0)

    def score_results(self, vector_results: List[Dict],
                      graph_results: List[Dict],
                      query_entities: List[str]) -> List[Dict]:
        """
        Score and rank combined vector + graph retrieval results.

        Args:
            vector_results: Results from vector retrieval [{chunk, score, rank}].
            graph_results: Results from graph retrieval [{text, path_score, entities}].
            query_entities: Entities extracted from the query.

        Returns:
            Sorted list of scored results.
        """
        scored = []
        seen_texts = set()

        # Score vector results
        for vr in vector_results:
            text = vr["chunk"].get("text", "")
            text_key = text[:100].lower()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            sim_vec = self._normalize_score(vr.get("score", 0))
            path_score = self._compute_path_score_for_text(text, graph_results)
            entity_overlap = self._compute_entity_overlap(text, query_entities)

            aghr_score = self.compute_score(sim_vec, path_score, entity_overlap)

            scored.append({
                "text": text,
                "source": "vector",
                "sim_vec": sim_vec,
                "path_score": path_score,
                "entity_overlap": entity_overlap,
                "aghr_score": aghr_score,
                "metadata": vr["chunk"].get("metadata", {}),
            })

        # Score graph results
        for gr in graph_results:
            text = gr.get("text", "")
            text_key = text[:100].lower()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            sim_vec = self._compute_sim_for_graph_result(gr, vector_results)
            path_score = self._normalize_score(gr.get("path_score", 0))
            entity_overlap = self._compute_entity_overlap(text, query_entities)

            aghr_score = self.compute_score(sim_vec, path_score, entity_overlap)

            scored.append({
                "text": text,
                "source": "graph",
                "sim_vec": sim_vec,
                "path_score": path_score,
                "entity_overlap": entity_overlap,
                "aghr_score": aghr_score,
                "metadata": gr.get("metadata", {}),
            })

        # Sort by AGHR score (descending)
        scored.sort(key=lambda x: x["aghr_score"], reverse=True)

        logger.info(f"AGHR scoring: {len(scored)} results scored "
                     f"(top score: {scored[0]['aghr_score']:.4f})" if scored else
                     "AGHR scoring: 0 results")

        return scored

    def _normalize_score(self, score: float) -> float:
        """Normalize score to [0, 1] range."""
        return min(max(float(score), 0.0), 1.0)

    def _compute_entity_overlap(self, text: str, query_entities: List[str]) -> float:
        """
        Compute entity overlap between text and query entities.

        Returns:
            Overlap ratio [0, 1].
        """
        if not query_entities:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for ent in query_entities if ent.lower() in text_lower)
        return matches / len(query_entities)

    def _compute_path_score_for_text(self, text: str,
                                      graph_results: List[Dict]) -> float:
        """
        Compute path relevance score for a text based on graph results.
        Checks if the text contains entities mentioned in graph paths.
        """
        if not graph_results:
            return 0.0

        text_lower = text.lower()
        max_score = 0.0

        for gr in graph_results:
            # Check entity overlap with graph entities
            entities = gr.get("entities", [])
            if entities:
                overlap = sum(1 for e in entities if e.lower() in text_lower)
                score = overlap / max(len(entities), 1)
                max_score = max(max_score, score)

            # Boost if graph path text matches
            graph_text = gr.get("text", "").lower()
            if graph_text:
                words = set(graph_text.split())
                text_words = set(text_lower.split())
                if words:
                    word_overlap = len(words & text_words) / len(words)
                    max_score = max(max_score, word_overlap * 0.5)

        return min(max_score, 1.0)

    def _compute_sim_for_graph_result(self, graph_result: Dict,
                                       vector_results: List[Dict]) -> float:
        """Estimate vector similarity for a graph result by text matching."""
        if not vector_results:
            return 0.0

        gr_text = graph_result.get("text", "").lower()
        max_sim = 0.0

        for vr in vector_results:
            vr_text = vr["chunk"].get("text", "").lower()
            # Simple word overlap as proxy for similarity
            gr_words = set(gr_text.split())
            vr_words = set(vr_text.split())
            if gr_words:
                overlap = len(gr_words & vr_words) / max(len(gr_words | vr_words), 1)
                max_sim = max(max_sim, overlap)

        return min(max_sim, 1.0)


class HybridRetriever:
    """
    Combines vector retrieval, graph retrieval, and AGHR scoring
    into a single retrieval pipeline.
    """

    def __init__(self, vector_store, knowledge_graph, encoder,
                 entity_extractor, query_analyzer, aghr_scorer,
                 top_k: int = 5, max_hops: int = 2):
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.encoder = encoder
        self.entity_extractor = entity_extractor
        self.query_analyzer = query_analyzer
        self.scorer = aghr_scorer
        self.top_k = top_k
        self.max_hops = max_hops

    def retrieve(self, query: str) -> Dict:
        """
        Full hybrid retrieval pipeline.

        Args:
            query: User question.

        Returns:
            Dict with analysis, vector_results, graph_results, scored_results, context.
        """
        # Step 1: Analyze query
        analysis = self.query_analyzer.analyze(query)
        route = analysis["route"]

        # Step 2: Vector retrieval
        vector_results = []
        if route in ("vector", "hybrid"):
            query_emb = self.encoder.encode_query(query)
            vector_results = self.vector_store.search(query_emb, top_k=self.top_k)

        # Step 3: Graph retrieval
        graph_results = []
        query_entity_texts = [e["text"] for e in analysis["entities"]]

        if route in ("graph", "hybrid") and self.kg:
            graph_context = self.kg.get_subgraph_context(
                query_entity_texts, max_hops=self.max_hops
            )
            if graph_context and graph_context != "No graph context available.":
                graph_results.append({
                    "text": graph_context,
                    "path_score": 0.7,
                    "entities": query_entity_texts,
                    "source": "knowledge_graph",
                })

        # Step 4: AGHR scoring
        scored = self.scorer.score_results(vector_results, graph_results, query_entity_texts)

        # Step 5: Build final context
        top_results = scored[:self.top_k]
        vector_context = "\n\n".join(
            [r["text"] for r in top_results if r["source"] == "vector"]
        )
        graph_context_text = "\n\n".join(
            [r["text"] for r in top_results if r["source"] == "graph"]
        )

        return {
            "query": query,
            "analysis": analysis,
            "route": route,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "scored_results": scored[:self.top_k],
            "vector_context": vector_context,
            "graph_context": graph_context_text,
            "combined_context": (vector_context + "\n\n" + graph_context_text).strip(),
        }
