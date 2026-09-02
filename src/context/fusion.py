"""
AGHR System — Context Fusion Engine (Phase 6)
===============================================
Merges vector + graph contexts into a single optimized context:
  - Deduplication (semantic similarity threshold)
  - Re-ranking by relevance
  - Compression to fit token budget
"""

import re
from typing import Dict, List

from loguru import logger


class ContextFusion:
    """
    Fuses vector and graph retrieval contexts into a single
    optimized context window for LLM generation.
    """

    def __init__(self, max_tokens: int = 2048, dedup_threshold: float = 0.85):
        self.max_tokens = max_tokens
        self.dedup_threshold = dedup_threshold

    def fuse(self, vector_context: str, graph_context: str,
             scored_results: List[Dict] = None) -> Dict:
        """
        Fuse vector and graph contexts.

        Args:
            vector_context: Text from vector retrieval.
            graph_context: Text from KG traversal.
            scored_results: AGHR-scored results for re-ranking.

        Returns:
            Dict with final_context, stats, and components.
        """
        # Step 1: Collect all context segments
        segments = []

        if scored_results:
            for r in scored_results:
                segments.append({
                    "text": r["text"],
                    "score": r.get("aghr_score", 0),
                    "source": r.get("source", "unknown"),
                })
        else:
            if vector_context:
                for para in vector_context.split("\n\n"):
                    if para.strip():
                        segments.append({"text": para.strip(), "score": 0.5, "source": "vector"})
            if graph_context:
                for para in graph_context.split("\n\n"):
                    if para.strip():
                        segments.append({"text": para.strip(), "score": 0.5, "source": "graph"})

        if not segments:
            return {"final_context": "", "stats": {"segments": 0}}

        # Step 2: Deduplicate
        deduped = self._deduplicate(segments)

        # Step 3: Re-rank by score
        deduped.sort(key=lambda x: x["score"], reverse=True)

        # Step 4: Compress to token budget
        final_segments = self._compress(deduped)

        # Step 5: Assemble final context
        final_context = "\n\n".join([s["text"] for s in final_segments])

        stats = {
            "original_segments": len(segments),
            "after_dedup": len(deduped),
            "final_segments": len(final_segments),
            "vector_segments": sum(1 for s in final_segments if s["source"] == "vector"),
            "graph_segments": sum(1 for s in final_segments if s["source"] == "graph"),
            "total_chars": len(final_context),
        }

        logger.info(f"Context fusion: {stats['original_segments']} → "
                     f"{stats['final_segments']} segments")

        return {
            "final_context": final_context,
            "segments": final_segments,
            "stats": stats,
        }

    def _deduplicate(self, segments: List[Dict]) -> List[Dict]:
        """Remove near-duplicate segments using word overlap."""
        if not segments:
            return []

        unique = [segments[0]]

        for seg in segments[1:]:
            is_dup = False
            seg_words = set(seg["text"].lower().split())

            for existing in unique:
                existing_words = set(existing["text"].lower().split())
                if not seg_words or not existing_words:
                    continue

                # Jaccard similarity
                intersection = len(seg_words & existing_words)
                union = len(seg_words | existing_words)
                similarity = intersection / max(union, 1)

                if similarity > self.dedup_threshold:
                    is_dup = True
                    # Keep the higher-scored one
                    if seg["score"] > existing["score"]:
                        unique.remove(existing)
                        unique.append(seg)
                    break

            if not is_dup:
                unique.append(seg)

        return unique

    def _compress(self, segments: List[Dict]) -> List[Dict]:
        """Compress segments to fit within token budget."""
        result = []
        total_chars = 0
        char_budget = self.max_tokens * 4  # rough char-to-token ratio

        for seg in segments:
            text = seg["text"]
            if total_chars + len(text) > char_budget:
                remaining = char_budget - total_chars
                if remaining > 100:
                    seg = seg.copy()
                    seg["text"] = text[:remaining] + "..."
                    result.append(seg)
                break
            result.append(seg)
            total_chars += len(text)

        return result
