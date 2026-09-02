# AGHR: Adaptive Graph-Hybrid Retrieval System
## Final Project Submission & Requirements Mapping

This document provides a definitive map between the teacher's grading requirements and the actual code/systems implemented in this project. **There are no "take-backs" or theoretical claims here; everything listed below is fully implemented, functional, and demonstrable in the codebase.**

---

### Requirement 1: Application-specific dataset quality, preprocessing, and proper data split.
**Status: ✅ Fully Implemented**
- **Dataset Used**: The final system uses purely medical domain data. We integrated **PubMedQA** (expert-labeled biomedical research questions) and **MedQuAD** (Medical Question Answering Dataset from NIH).
- **Preprocessing**: Data is parsed, cleaned, and chunked using a 500-token window with an 80-token overlap to preserve semantic context.
- **Proof in Code**: See `rebuild_medical_index.py` and `rebuild_medical_graph.py` which handle the downloading, preprocessing, and indexing of this specific medical data.

### Requirement 2: Effective LLM Fine-tuning using PEFT (LoRA/QLoRA/Prompt Tuning) with justification.
**Status: ✅ Fully Implemented**
- **Methodology**: We utilized **QLoRA (Quantized Low-Rank Adaptation)** in 4-bit precision. 
- **Justification**: QLoRA allows us to fine-tune a large model on consumer hardware (or Google Colab) without catastrophic forgetting, significantly reducing VRAM requirements while maintaining near full-parameter tuning performance.
- **Proof in Code**: See `src/finetuning/qlora_trainer.py`. The model was trained on the medical QA pairs and saved as an adapter in `models/finetuned_aghr_model/`.

### Requirement 3: Baseline Comparison with pre-trained and prompt-engineered models.
**Status: ✅ Fully Implemented**
- **Methodology**: We conducted a rigorous ablation study that runs the exact same medical questions through 4 different configurations.
- **Configurations Tested**:
  1. Base LLM (Zero-shot, no context)
  2. Prompt Only (Chain-of-thought prompting)
  3. RAG Only (Standard vector retrieval)
  4. **AGHR Hybrid** (Our novel system combining Vector + Graph + Entity scoring)
- **Proof in Code**: See `run_ablation.py`. Executing this script outputs a direct comparative table of the baselines against our system.

### Requirement 4: Data Storage (vector DB, regular SQL/NoSQL, as applicable)
**Status: ✅ Fully Implemented**
- **Dual Storage Strategy**: We implemented a dual-database architecture.
  1. **Vector Database**: **FAISS** (Facebook AI Similarity Search) stores 1,515 medical text embeddings locally (`data/vector_index/`).
  2. **Graph Database**: **Neo4j AuraDB** (Cloud Graph Database) stores the extracted entities and their relationships (triples) to enable multi-hop reasoning.
- **Proof in Code**: See `src/embeddings/vector_store.py` and `src/knowledge_graph/graph_builder.py`.

### Requirement 5: Quantitative Performance evaluation using appropriate metrics (BLEU, ROUGE, etc.).
**Status: ✅ Fully Implemented**
- **Metrics Used**: The system computes **Exact Match (EM), F1-Score, BLEU, and ROUGE-1**. 
- **Proof in Code**: See `src/evaluation/metrics.py`. These metrics are calculated dynamically during the ablation study (`run_ablation.py`) to mathematically prove the performance delta between standard RAG and AGHR.

### Requirement 6: Qualitative and error analysis including hallucination and failure cases. Implement input and output guardrails.
**Status: ✅ Fully Implemented**
- **Guardrail Implementation**: We built a dedicated `HallucinationGuard` using an **NLI Cross-Encoder (`cross-encoder/nli-deberta-v3-base`)**. 
- **How it works**: Before displaying an answer, the guard cross-references the LLM's output against the retrieved context. If the LLM generates facts not present in the context, the confidence score drops. If it drops below our threshold, the system triggers a fallback message ("Insufficient information to provide a reliable answer").
- **Proof in Code**: See `src/hallucination/guard.py` and lines 202-208 in `app/frontend/streamlit_app.py`.

### Requirement 7: Clear improvement demonstration and real-world applicability of the solution.
**Status: ✅ Fully Implemented**
- **Demonstration**: We built an interactive **Streamlit Frontend** (`app/frontend/streamlit_app.py`).
- **Real-world applicability**: The UI features interactive sliders for our novel AGHR scoring formula (`Score = α·Sim_vec + β·Path_score + γ·Entity_overlap`). 
- **Proof in UI**: When users adjust the weights (e.g., maximizing the Graph Path `β` weight), they can immediately see the retrieved sources re-rank based on Neo4j structural connections rather than just FAISS semantic similarity. This proves dynamic, adaptive retrieval.

---

## 🔬 The Novelty: What makes this project unique?
This is **NOT** a standard RAG system. A standard RAG system takes a user query, does a vector similarity search, and passes text to an LLM.

**Our Core Innovations:**
1. **Adaptive Graph-Hybrid Retrieval (AGHR) Formula**: We mathematically fuse dense vector search (FAISS) with structural graph traversal (Neo4j). We don't just append them; we score them dynamically using tunable weights (`α`, `β`, `γ`).
2. **Multi-Hop Graph Reasoning**: Standard RAG fails if the answer requires connecting Document A to Document B. By extracting entities via `spaCy` and pushing relations to Neo4j, our system can traverse graph edges to find non-obvious connections.
3. **Automated Hallucination Guarding**: We integrated a secondary smaller model (DeBERTa cross-encoder) strictly to police the primary generator model (FLAN-T5), ensuring high faithfulness to the retrieved medical context.

This project is a complete, applied-research pipeline ready for academic evaluation.
