"""
AGHR System — Query Analyzer & Classifier (Phase 4)
=====================================================
Analyzes queries to extract intent, entities, complexity,
and routes to the appropriate retrieval strategy.
"""

import re
from typing import Dict, List, Tuple

from loguru import logger


class QueryAnalyzer:
    """
    Analyzes user queries to determine:
      - Extracted entities
      - Query complexity (simple/moderate/complex)
      - Intent type (factual/relational/complex)
      - Routing decision (vector/graph/hybrid)
    """

    def __init__(self, entity_extractor=None):
        self.entity_extractor = entity_extractor

        # Relational keywords indicate graph-based queries
        self.relational_keywords = {
            "cause", "causes", "caused", "effect", "affects", "affect",
            "lead", "leads", "leading", "result", "results",
            "treat", "treats", "treatment", "treatments",
            "prevent", "prevents", "prevention",
            "relate", "relates", "related", "relationship", "relation",
            "connect", "connection", "link", "linked",
            "risk", "factor", "increase", "decrease",
            "symptom", "symptoms", "sign", "signs",
            "diagnose", "diagnosis", "manage", "management",
        }

        # Multi-hop indicators
        self.multihop_patterns = [
            r"\b(?:how\s+does\s+.+\s+(?:affect|cause|lead|relate))",
            r"\b(?:what\s+is\s+the\s+(?:relationship|connection|link))",
            r"\b(?:if.+then|because|therefore|consequently)",
            r"\b(?:through|via|by\s+means\s+of|chain|path|pathway)",
        ]

    def analyze(self, query: str) -> Dict:
        """
        Full query analysis pipeline.

        Args:
            query: User's question.

        Returns:
            Dict with entities, complexity, intent, routing, etc.
        """
        # Extract entities
        entities = self._extract_entities(query)

        # Determine complexity
        complexity = self._assess_complexity(query, entities)

        # Classify intent
        intent = self._classify_intent(query, entities, complexity)

        # Determine routing
        route = self._route_query(intent, complexity)

        result = {
            "query": query,
            "entities": entities,
            "entity_count": len(entities),
            "complexity": complexity,
            "complexity_label": self._complexity_label(complexity),
            "intent": intent,
            "route": route,
        }

        logger.debug(f"Query analysis: complexity={complexity}, intent={intent}, route={route}")
        return result

    def _extract_entities(self, query: str) -> List[Dict]:
        """Extract entities from the query."""
        if self.entity_extractor:
            return self.entity_extractor.extract_entities(query)
        # Fallback: simple capitalized word extraction
        entities = []
        words = query.split()
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean[0].isupper() and len(clean) > 2:
                entities.append({"text": clean, "type": "UNKNOWN", "source": "simple"})
        return entities

    def _assess_complexity(self, query: str, entities: List[Dict]) -> int:
        """
        Assess query complexity on a 1-3 scale.
        1 = simple (factual, single entity)
        2 = moderate (relational, 2 entities)
        3 = complex (multi-hop, 3+ entities or multi-hop indicators)
        """
        score = 1
        query_lower = query.lower()

        # Entity count factor
        if len(entities) >= 3:
            score = max(score, 3)
        elif len(entities) >= 2:
            score = max(score, 2)

        # Relational keyword factor
        relational_count = sum(1 for kw in self.relational_keywords if kw in query_lower)
        if relational_count >= 2:
            score = max(score, 3)
        elif relational_count >= 1:
            score = max(score, 2)

        # Multi-hop pattern factor
        for pattern in self.multihop_patterns:
            if re.search(pattern, query_lower):
                score = 3
                break

        # Question word complexity
        if any(q in query_lower for q in ["how does", "why does", "what is the relationship"]):
            score = max(score, 2)
        if any(q in query_lower for q in ["how does", "explain the chain", "what pathway"]):
            score = max(score, 3)

        return score

    def _classify_intent(self, query: str, entities: List[Dict], complexity: int) -> str:
        """Classify query intent: factual, relational, or complex."""
        query_lower = query.lower()

        # Check for relational intent
        has_relational = any(kw in query_lower for kw in self.relational_keywords)

        if complexity >= 3:
            return "complex"
        elif has_relational or complexity == 2:
            return "relational"
        else:
            return "factual"

    def _route_query(self, intent: str, complexity: int) -> str:
        """Determine retrieval route based on intent and complexity."""
        routing = {
            "factual": "vector",
            "relational": "hybrid",  # Changed from "graph" to ensure vector fallback
            "complex": "hybrid",
        }
        return routing.get(intent, "hybrid")

    @staticmethod
    def _complexity_label(score: int) -> str:
        return {1: "simple", 2: "moderate", 3: "complex"}.get(score, "unknown")
