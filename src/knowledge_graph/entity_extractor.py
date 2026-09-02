"""
AGHR System — Entity Extractor (Phase 3.1)
============================================
Extracts named entities from text using spaCy + custom medical entity rules.
Supports both general NER and domain-specific (medical) entity types.
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter

from loguru import logger
from tqdm import tqdm


# Medical entity patterns for rule-based extraction
MEDICAL_PATTERNS = {
    "DISEASE": [
        r"\b(?:diabetes|hypertension|cancer|asthma|arthritis|alzheimer|parkinson|"
        r"stroke|pneumonia|tuberculosis|malaria|hepatitis|anemia|epilepsy|"
        r"osteoporosis|dementia|depression|anxiety|obesity|influenza|covid|"
        r"heart\s+disease|heart\s+failure|kidney\s+disease|liver\s+disease|"
        r"blood\s+pressure|coronary\s+artery|atrial\s+fibrillation)\b"
    ],
    "SYMPTOM": [
        r"\b(?:fever|cough|headache|fatigue|nausea|vomiting|dizziness|"
        r"chest\s+pain|shortness\s+of\s+breath|back\s+pain|joint\s+pain|"
        r"swelling|inflammation|bleeding|numbness|weakness|insomnia|"
        r"weight\s+loss|weight\s+gain|blurred\s+vision|confusion)\b"
    ],
    "TREATMENT": [
        r"\b(?:surgery|therapy|chemotherapy|radiation|dialysis|transplant|"
        r"rehabilitation|physiotherapy|immunotherapy|vaccination|"
        r"blood\s+transfusion|bypass|angioplasty|ventilation)\b"
    ],
    "MEDICATION": [
        r"\b(?:aspirin|insulin|metformin|ibuprofen|paracetamol|amoxicillin|"
        r"atorvastatin|lisinopril|amlodipine|omeprazole|metoprolol|"
        r"levothyroxine|prednisone|warfarin|heparin|morphine)\b"
    ],
    "BODY_PART": [
        r"\b(?:heart|lung|liver|kidney|brain|stomach|intestine|pancreas|"
        r"spleen|thyroid|bone|muscle|blood|artery|vein|nerve|spine|"
        r"colon|bladder|prostate|uterus|ovary|breast)\b"
    ],
}


class EntityExtractor:
    """
    Hybrid entity extraction using spaCy NER + medical pattern rules.

    Combines:
      1. spaCy transformer NER for general entities
      2. Regex-based rules for domain-specific medical entities
      3. Entity normalization and deduplication
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", use_medical_rules: bool = True):
        import spacy

        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"{spacy_model} not found, downloading en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.use_medical_rules = use_medical_rules

        # Compile medical patterns
        self.medical_patterns = {}
        if use_medical_rules:
            for ent_type, patterns in MEDICAL_PATTERNS.items():
                combined = "|".join(patterns)
                self.medical_patterns[ent_type] = re.compile(combined, re.IGNORECASE)

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using spaCy + medical rules.

        Args:
            text: Input text.

        Returns:
            List of entity dicts: {text, type, start, end, source}
        """
        entities = []
        seen = set()

        # 1. spaCy NER
        doc = self.nlp(text[:100000])  # Limit for performance
        for ent in doc.ents:
            key = (ent.text.lower().strip(), ent.label_)
            if key not in seen and len(ent.text.strip()) > 1:
                entities.append({
                    "text": ent.text.strip(),
                    "type": self._map_spacy_label(ent.label_),
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": "spacy",
                })
                seen.add(key)

        # 2. Medical pattern matching
        if self.use_medical_rules:
            for ent_type, pattern in self.medical_patterns.items():
                for match in pattern.finditer(text):
                    matched_text = match.group().strip()
                    key = (matched_text.lower(), ent_type)
                    if key not in seen and len(matched_text) > 1:
                        entities.append({
                            "text": matched_text,
                            "type": ent_type,
                            "start": match.start(),
                            "end": match.end(),
                            "source": "medical_rules",
                        })
                        seen.add(key)

        return entities

    def extract_from_documents(self, documents: List[Dict],
                                text_key: str = "text") -> List[Dict]:
        """
        Extract entities from a list of documents.

        Args:
            documents: List of document/chunk dicts.
            text_key: Key for the text field.

        Returns:
            List of dicts: {doc_id, entities: [...]}
        """
        results = []
        all_entity_count = Counter()

        for doc in tqdm(documents, desc="Extracting entities"):
            text = doc.get(text_key, "")
            doc_id = doc.get("doc_id", doc.get("chunk_id", "unknown"))
            entities = self.extract_entities(text)

            for ent in entities:
                all_entity_count[ent["type"]] += 1

            results.append({
                "doc_id": doc_id,
                "entities": entities,
                "entity_count": len(entities),
            })

        logger.info(f"Entity extraction: {sum(all_entity_count.values())} entities "
                     f"from {len(documents)} docs")
        logger.info(f"Entity types: {dict(all_entity_count)}")
        return results

    def normalize_entity(self, text: str) -> str:
        """Normalize entity text for deduplication."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def get_unique_entities(self, entity_results: List[Dict]) -> List[Dict]:
        """Get deduplicated list of all unique entities across documents."""
        unique = {}
        for result in entity_results:
            for ent in result["entities"]:
                key = (self.normalize_entity(ent["text"]), ent["type"])
                if key not in unique:
                    unique[key] = {
                        "text": ent["text"],
                        "normalized": key[0],
                        "type": ent["type"],
                        "count": 0,
                        "doc_ids": [],
                    }
                unique[key]["count"] += 1
                unique[key]["doc_ids"].append(result["doc_id"])

        entities = sorted(unique.values(), key=lambda x: x["count"], reverse=True)
        logger.info(f"Unique entities: {len(entities)}")
        return entities

    @staticmethod
    def _map_spacy_label(label: str) -> str:
        """Map spaCy NER labels to our entity types."""
        mapping = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "DATE": "DATE",
            "CARDINAL": "NUMBER",
            "MONEY": "NUMBER",
            "PERCENT": "NUMBER",
            "PRODUCT": "PRODUCT",
            "EVENT": "EVENT",
            "WORK_OF_ART": "WORK",
            "NORP": "GROUP",
            "FAC": "FACILITY",
        }
        return mapping.get(label, label)
