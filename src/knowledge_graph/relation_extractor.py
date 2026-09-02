"""
AGHR System — Relation Extractor (Phase 3.2)
==============================================
Extracts entity relationships from text using:
  1. spaCy dependency parsing for SVO triples
  2. Co-occurrence based relation inference
  3. Pattern-based medical relation extraction
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from loguru import logger
from tqdm import tqdm


# Medical relation patterns
MEDICAL_RELATION_PATTERNS = [
    # (pattern, relation_type, entity1_group, entity2_group)
    (r"(\w[\w\s]+?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w[\w\s]+)", "CAUSES", 1, 2),
    (r"(\w[\w\s]+?)\s+(?:treats?|cures?|heals?)\s+(\w[\w\s]+)", "TREATS", 1, 2),
    (r"(\w[\w\s]+?)\s+(?:prevents?|reduces?\s+risk\s+of)\s+(\w[\w\s]+)", "PREVENTS", 1, 2),
    (r"(\w[\w\s]+?)\s+(?:is\s+a\s+symptom\s+of|symptoms?\s+include)\s+(\w[\w\s]+)", "SYMPTOM_OF", 1, 2),
    (r"(\w[\w\s]+?)\s+(?:is\s+associated\s+with|linked\s+to|related\s+to)\s+(\w[\w\s]+)", "ASSOCIATED_WITH", 1, 2),
    (r"(\w[\w\s]+?)\s+(?:is\s+diagnosed\s+by|diagnosed\s+using)\s+(\w[\w\s]+)", "DIAGNOSED_BY", 1, 2),
    (r"(\w[\w\s]+?)\s+(?:is\s+managed\s+by|managed\s+with|treated\s+with)\s+(\w[\w\s]+)", "MANAGED_BY", 1, 2),
    (r"(\w[\w\s]+?)\s+(?:increases?\s+risk\s+of|risk\s+factor\s+for)\s+(\w[\w\s]+)", "RISK_FACTOR_FOR", 1, 2),
    (r"(\w[\w\s]+?)\s+(?:is\s+part\s+of|belongs?\s+to)\s+(\w[\w\s]+)", "PART_OF", 1, 2),
    (r"(\w[\w\s]+?)\s+(?:affects?|impacts?)\s+(\w[\w\s]+)", "AFFECTS", 1, 2),
]


class RelationExtractor:
    """
    Extracts relations between entities using multiple strategies:
      1. Dependency parsing (SVO triples)
      2. Co-occurrence in same sentence
      3. Pattern-based medical relations
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        import spacy
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Compile patterns
        self.relation_patterns = [
            (re.compile(p, re.IGNORECASE), rel, g1, g2)
            for p, rel, g1, g2 in MEDICAL_RELATION_PATTERNS
        ]

    def extract_svo_triples(self, text: str) -> List[Dict]:
        """
        Extract Subject-Verb-Object triples using spaCy dependency parsing.

        Returns:
            List of {subject, relation, object} dicts.
        """
        doc = self.nlp(text[:50000])
        triples = []

        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subject = None
                obj = None

                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = self._get_span_text(child)
                    if child.dep_ in ("dobj", "attr", "prep"):
                        if child.dep_ == "prep":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    obj = self._get_span_text(grandchild)
                        else:
                            obj = self._get_span_text(child)

                if subject and obj and len(subject) > 1 and len(obj) > 1:
                    triples.append({
                        "entity_1": subject,
                        "relation": token.lemma_.upper(),
                        "entity_2": obj,
                        "source": "dependency_parse",
                    })

        return triples

    def extract_pattern_relations(self, text: str) -> List[Dict]:
        """Extract relations using medical pattern matching."""
        relations = []
        for pattern, rel_type, g1, g2 in self.relation_patterns:
            for match in pattern.finditer(text):
                e1 = match.group(g1).strip()
                e2 = match.group(g2).strip()
                if len(e1) > 1 and len(e2) > 1 and len(e1) < 50 and len(e2) < 50:
                    relations.append({
                        "entity_1": e1,
                        "relation": rel_type,
                        "entity_2": e2,
                        "source": "pattern_match",
                    })
        return relations

    def extract_cooccurrence_relations(self, text: str,
                                        entities: List[Dict]) -> List[Dict]:
        """
        Extract relations based on entity co-occurrence within the same sentence.

        If two entities appear in the same sentence, they are likely related.
        """
        doc = self.nlp(text[:50000])
        relations = []

        for sent in doc.sents:
            sent_text = sent.text.lower()
            sent_entities = []

            for ent in entities:
                if ent["text"].lower() in sent_text:
                    sent_entities.append(ent)

            # Create pairs from co-occurring entities
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    e1, e2 = sent_entities[i], sent_entities[j]
                    if e1["text"].lower() != e2["text"].lower():
                        relations.append({
                            "entity_1": e1["text"],
                            "entity_1_type": e1["type"],
                            "relation": "CO_OCCURS_WITH",
                            "entity_2": e2["text"],
                            "entity_2_type": e2["type"],
                            "source": "cooccurrence",
                        })

        return relations

    def extract_all_relations(self, text: str,
                               entities: List[Dict] = None) -> List[Dict]:
        """
        Run all relation extraction methods and combine results.

        Args:
            text: Input text.
            entities: Pre-extracted entities (for co-occurrence).

        Returns:
            Combined deduplicated list of relations.
        """
        all_relations = []

        # SVO triples
        svo = self.extract_svo_triples(text)
        all_relations.extend(svo)

        # Pattern-based
        patterns = self.extract_pattern_relations(text)
        all_relations.extend(patterns)

        # Co-occurrence (if entities provided)
        if entities:
            cooc = self.extract_cooccurrence_relations(text, entities)
            all_relations.extend(cooc)

        # Deduplicate
        seen = set()
        unique = []
        for rel in all_relations:
            key = (
                rel["entity_1"].lower(),
                rel["relation"],
                rel["entity_2"].lower(),
            )
            if key not in seen:
                seen.add(key)
                unique.append(rel)

        return unique

    def extract_from_documents(self, documents: List[Dict],
                                entity_results: List[Dict] = None,
                                text_key: str = "text") -> List[Dict]:
        """
        Extract relations from all documents.

        Args:
            documents: List of document/chunk dicts.
            entity_results: Entity extraction results (for co-occurrence).
            text_key: Key for text field.

        Returns:
            List of {doc_id, relations: [...]} dicts.
        """
        # Build entity lookup by doc_id
        entity_lookup = {}
        if entity_results:
            for er in entity_results:
                entity_lookup[er["doc_id"]] = er["entities"]

        results = []
        total_rels = 0

        for doc in tqdm(documents, desc="Extracting relations"):
            text = doc.get(text_key, "")
            doc_id = doc.get("doc_id", doc.get("chunk_id", "unknown"))
            entities = entity_lookup.get(doc_id, [])

            relations = self.extract_all_relations(text, entities)
            total_rels += len(relations)

            results.append({
                "doc_id": doc_id,
                "relations": relations,
                "relation_count": len(relations),
            })

        logger.info(f"Relation extraction: {total_rels} relations from {len(documents)} docs")
        return results

    def get_all_triples(self, relation_results: List[Dict]) -> List[Dict]:
        """Flatten all relation results into a single list of triples."""
        triples = []
        seen = set()
        for result in relation_results:
            for rel in result["relations"]:
                key = (rel["entity_1"].lower(), rel["relation"], rel["entity_2"].lower())
                if key not in seen:
                    seen.add(key)
                    triples.append(rel)

        logger.info(f"Total unique triples: {len(triples)}")
        return triples

    @staticmethod
    def _get_span_text(token) -> str:
        """Get the full noun phrase text for a token."""
        if hasattr(token, 'subtree'):
            return " ".join([t.text for t in token.subtree
                           if t.pos_ in ("NOUN", "PROPN", "ADJ", "DET", "COMPOUND")])
        return token.text
