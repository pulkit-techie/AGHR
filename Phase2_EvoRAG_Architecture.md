# Phase 2 Architecture: EvoRAG (Evolutionary RAG)
## Future Work & Self-Learning Graph Integration

While the current AGHR system successfully demonstrates a robust **Static Hybrid Retrieval** architecture (combining vector semantics with a pre-built Neo4j medical graph), our immediate roadmap involves transitioning to a **Dynamic, Self-Evolving GraphRAG System (EvoRAG)**.

This document outlines the proposed Phase 2 architecture, which can be presented during your Viva as the ultimate goal of your research.

---

### 1. The Core Innovation: The Learning Trigger
Standard RAG systems, including our current Phase 1 implementation, are "read-only." If a user asks a question about a medical concept that exists in the Vector DB but is completely missing from the Neo4j Graph, the system falls back to pure vector retrieval.

**EvoRAG** introduces a write-back learning loop. 
If the Neo4j graph returns a `Miss` (0.000 path score), the system triggers **Learning Mode**:
1. It retrieves the context via Vector Search.
2. It generates the answer via FLAN-T5.
3. The **Hallucination Guard (DeBERTa)** evaluates the answer.
4. **If the answer is SAFE (high confidence)**: The system automatically extracts the new entities and relationships from the verified context and **writes them directly into Neo4j**.

### 2. High-Level EvoRAG Flow

```text
User Query
   ↓
Query Analyzer (Entity & Intent Extraction)
   ↓
Graph Check (Neo4j)
   ↓
IF Found → Normal Hybrid AGHR Retrieval
IF Not Found → Trigger Learning Mode
   ↓
[Learning Mode]
Dual Retrieval (FAISS + Graph)
   ↓
Context Fusion
   ↓
LLM Generation (FLAN-T5)
   ↓
Hallucination Guard (DeBERTa NLI)
   ↓
IF Safe (>85% Confidence) → Generate Cypher Query → Store Knowledge in Graph
   ↓
Re-Retrieve (Updated Graph)
   ↓
Final Answer + Dynamic Graph Updates
```

### 3. Architecture Layers (Simplified View)

* **Layer 1: Input Intelligence**: Query parsing and LLM-based query rewriting to expand medical acronyms.
* **Layer 2: Knowledge Access**: FAISS (Dense Vectors), BM25 (Sparse Keywords), and Neo4j (Structural Triples).
* **Layer 3: Reasoning Core**: Dynamic AGHR ranking. Instead of static UI sliders, a lightweight classifier automatically adjusts `α, β, γ` weights based on query intent.
* **Layer 4: Generation**: FLAN-T5 or TinyLlama generation strictly grounded in fused context.
* **Layer 5: Validation**: DeBERTa cross-encoder hallucination guarding.
* **Layer 6: Learning Loop ⭐**: The write-back mechanism that allows Neo4j to continuously grow its edge connections based on successful, verified RAG retrievals.

### 4. Summary Statement for Presentation
> *"Our Phase 1 implementation proves that Adaptive Graph-Hybrid Retrieval (AGHR) can mathematically outperform standard RAG by penalizing hallucinated or structurally disconnected text. Our Phase 2 goal is **EvoRAG**, a self-evolving architecture that dynamically learns from unseen queries. By integrating our Hallucination Guard as a gatekeeper, EvoRAG will safely write new verified medical connections back into the Neo4j graph in real-time, allowing the system to get mathematically smarter with every question asked."*
