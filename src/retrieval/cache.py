"""
AGHR System — Semantic Cache (Pro Enhancement)
==============================================
Provides a semantic caching layer to instantly return
responses for semantically similar queries.

Features:
  - FAISS-based semantic similarity matching
  - Exact string match for true 0ms latency
  - Numpy-safe serialization for robust persistence
"""

import numpy as np
import faiss
import json
import os
import copy
from loguru import logger
from typing import Dict, Optional, Any, List


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and other non-serializable objects to native Python types."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # Fallback: try to convert to string
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


class SemanticCache:
    def __init__(self, encoder, threshold: float = 0.95, save_path: str = "data/cache"):
        self.encoder = encoder
        self.threshold = threshold
        self.save_path = save_path
        
        # Attempt to get dimension from encoder
        try:
            self.dimension = encoder.dimension
        except AttributeError:
            self.dimension = 384 # Default for MiniLM
            
        self.index = faiss.IndexFlatIP(self.dimension) # Inner product for Cosine Sim
        self.cache_data = {} # idx -> dict of cached response
        self.exact_match_cache = {} # str -> dict of cached response (for 0ms latency)
        self._next_idx = 0
        
        os.makedirs(self.save_path, exist_ok=True)
        self.load()

    def get(self, query: str, config_hash: str = "") -> Optional[Dict[str, Any]]:
        """Retrieve from cache if a semantically similar query exists."""
        # 1. Check exact string match first for TRUE 0ms latency
        normalized_query = query.lower().strip()
        exact_key = f"{normalized_query}_{config_hash}"
        if exact_key in self.exact_match_cache:
            logger.info(f"Exact string Cache HIT for query: '{query}' with config: '{config_hash}'")
            return copy.deepcopy(self.exact_match_cache[exact_key])
            
        if self._next_idx == 0:
            return None
            
        # Encode query
        emb = self.encoder.encode_query(query)
        if len(emb.shape) == 1:
            emb = emb.reshape(1, -1)
            
        # Search
        scores, indices = self.index.search(emb, 1)
        
        if len(scores) > 0 and len(scores[0]) > 0:
            best_score = scores[0][0]
            best_idx = indices[0][0]
            
            if best_score >= self.threshold and best_idx in self.cache_data:
                cached_data = self.cache_data[best_idx]
                # Check if config matches
                if cached_data.get("config_hash", "") == config_hash:
                    logger.info(f"Cache HIT for query: '{query}' (score: {best_score:.3f})")
                    return copy.deepcopy(cached_data)
                
        logger.debug(f"Cache MISS for query: '{query}'")
        return None
        
    def set(self, query: str, response_data: Dict[str, Any], config_hash: str = ""):
        """Store response in semantic cache. All data is made JSON-safe before storing."""
        # Make a deep copy and sanitize all numpy types
        safe_data = _make_json_safe(response_data)
        safe_data["config_hash"] = config_hash
        
        # 1. Store in exact match cache
        normalized_query = query.lower().strip()
        exact_key = f"{normalized_query}_{config_hash}"
        self.exact_match_cache[exact_key] = safe_data
        
        # Encode query
        emb = self.encoder.encode_query(query)
        if len(emb.shape) == 1:
            emb = emb.reshape(1, -1)
            
        # Store in semantic cache
        self.index.add(emb)
        self.cache_data[self._next_idx] = safe_data
        self._next_idx += 1
        
        # Save periodically or immediately
        self.save()
        
    def save(self):
        """Persist cache to disk."""
        try:
            faiss.write_index(self.index, os.path.join(self.save_path, "cache.index"))
            # cache_data is already json-safe from set()
            with open(os.path.join(self.save_path, "cache_data.json"), "w") as f:
                json.dump(self.cache_data, f, default=str)
        except Exception as e:
            logger.warning(f"Could not save semantic cache: {e}")
            
    def load(self):
        """Load cache from disk."""
        idx_path = os.path.join(self.save_path, "cache.index")
        data_path = os.path.join(self.save_path, "cache_data.json")
        
        if os.path.exists(idx_path) and os.path.exists(data_path):
            try:
                self.index = faiss.read_index(idx_path)
                with open(data_path, "r") as f:
                    raw_data = json.load(f)
                    self.cache_data = {int(k): v for k, v in raw_data.items()}
                    
                # Rebuild exact match cache from loaded data
                # Since we don't store the query string in cache_data currently, 
                # we will just start exact_match_cache empty for loaded data.
                # Future queries will populate it.
                self.exact_match_cache = {}
                
                self._next_idx = len(self.cache_data)
                self.dimension = self.index.d
                logger.info(f"Loaded Semantic Cache with {self._next_idx} entries.")
            except Exception as e:
                logger.warning(f"Could not load semantic cache, starting fresh: {e}")
    
    def clear(self):
        """Clear all cached entries."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.cache_data = {}
        self.exact_match_cache = {}
        self._next_idx = 0
        self.save()
        logger.info("Semantic cache cleared.")
