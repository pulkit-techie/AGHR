# 🧠 AGHR — Adaptive Graph-Hybrid Retrieval System
## Complete Project Documentation

> **Domain:** Healthcare / Medical Question Answering  
> **Research Claim:** AGHR improves QA accuracy by ~25–30% and reduces hallucination vs standard RAG.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [User-Level Features](#3-user-level-features)
4. [Backend Services & Modules](#4-backend-services--modules)
5. [Data Pipeline (Phase 1)](#5-data-pipeline-phase-1)
6. [Embedding & Vector Search (Phase 2)](#6-embedding--vector-search-phase-2)
7. [Knowledge Graph Engine (Phase 3)](#7-knowledge-graph-engine-phase-3)
8. [Query Intelligence (Phase 4)](#8-query-intelligence-phase-4)
9. [AGHR Hybrid Retrieval (Phase 5)](#9-aghr-hybrid-retrieval-phase-5)
10. [Context Fusion (Phase 6)](#10-context-fusion-phase-6)
11. [Prompt Engineering & LLM Generation (Phases 7-8)](#11-prompt-engineering--llm-generation-phases-7-8)
12. [QLoRA Fine-Tuning (Phase 9)](#12-qlora-fine-tuning-phase-9)
13. [Hallucination Guard (Phase 10)](#13-hallucination-guard-phase-10)
14. [Evaluation & Metrics (Phase 11)](#14-evaluation--metrics-phase-11)
15. [Deployment Layer (Phase 17)](#15-deployment-layer-phase-17)
16. [Pro Features](#16-pro-features)
17. [Technology Stack](#17-technology-stack)
18. [Project Structure](#18-project-structure)

---

## 1. Project Overview

**AGHR (Adaptive Graph-Hybrid Retrieval)** is an advanced Retrieval-Augmented Generation (RAG) system purpose-built for medical question answering. Unlike standard RAG systems that rely solely on vector similarity search, AGHR introduces a novel **tri-signal scoring formula** that mathematically fuses:

- **Dense Vector Search** (FAISS) for semantic similarity
- **Knowledge Graph Traversal** (Neo4j AuraDB) for structural/relational reasoning
- **Entity Overlap Analysis** for precision matching

The system addresses three critical problems in medical AI:

| Problem | AGHR Solution |
|---------|---------------|
| **Hallucination** | NLI-based Hallucination Guard with Self-Correction Loop |
| **Single-hop Limitation** | Multi-hop graph traversal (up to 3 hops) via Neo4j |
| **Lack of Explainability** | Interactive Knowledge Graph visualization + source attribution |

---

## 2. System Architecture

### High-Level Pipeline

```
User Question
      │
      ▼
┌─────────────────┐
│  Query Analyzer  │ ← Entity extraction, complexity scoring, intent classification
└────────┬────────┘
         │ Routes to: VECTOR / GRAPH / HYBRID
         ▼
┌────────────────────────────────────────────┐
│           Dual-Stream Retrieval            │
│  ┌──────────────┐    ┌──────────────────┐  │
│  │  FAISS Index  │    │  Neo4j AuraDB    │  │
│  │  (1,515 docs) │    │  (483 nodes,     │  │
│  │  Cosine Sim   │    │   629 edges)     │  │
│  └──────┬───────┘    └────────┬─────────┘  │
│         └──────────┬──────────┘             │
└────────────────────┼────────────────────────┘
                     ▼
          ┌─────────────────────┐
          │   AGHR Scorer ⭐     │
          │ Score = α·Vec +     │
          │   β·Path + γ·Entity │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │   Context Fusion    │ ← Deduplication, re-ranking, compression
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │   LLM Generation    │ ← FLAN-T5 / TinyLlama with token streaming
          │   + Prompt Engine   │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │ Hallucination Guard │ ← NLI cross-encoder + confidence scoring
          │ + Self-Correction   │
          └──────────┬──────────┘
                     ▼
            Final Answer + Confidence
            + Sources + KG Visualization
```

### Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│                 Streamlit Frontend               │
│  (Interactive UI, KG Viz, Weight Tuning, RLHF)  │
│                  Port 8501/8502                  │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────┐
│              FastAPI Backend                     │
│  POST /api/query  GET /api/health               │
│  GET /api/stats                                 │
│                  Port 8000                       │
└─────────────────────┬───────────────────────────┘
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
  ┌─────────┐   ┌──────────┐   ┌──────────┐
  │  FAISS  │   │  Neo4j   │   │ HF Models│
  │ (Local) │   │ (Cloud)  │   │ (Local)  │
  └─────────┘   └──────────┘   └──────────┘
```

---

## 3. User-Level Features

### 🔍 Question Answering
- Type any medical question in natural language
- System automatically determines the best retrieval strategy (Vector, Graph, or Hybrid)
- Answers are streamed token-by-token in real-time (like ChatGPT)

### 📊 AGHR Weight Tuning
- Interactive sliders in the sidebar let you adjust `α` (Vector), `β` (Graph Path), and `γ` (Entity) weights in real-time
- Immediately see how different weight configurations change retrieval results and answer quality

### 🔗 Interactive Knowledge Graph
- Expand the "Knowledge Graph Context" section to see entity relationships
- Powered by `streamlit-agraph` for a fully interactive, draggable, zoomable graph visualization
- Nodes represent medical entities; edges represent relationships (CAUSES, TREATS, CO_OCCURS_WITH, etc.)

### 📈 Real-Time Metrics Dashboard
- **Confidence Score** — How grounded the answer is (0–100%)
- **Route** — Which retrieval path was used (VECTOR / GRAPH / HYBRID)
- **Latency** — End-to-end response time in milliseconds
- **Sources** — Number of retrieved source documents

### 🛡️ Hallucination Warnings
- If the system detects low confidence, it displays an orange/red warning
- Triggers an automatic **Self-Correction Loop** to regenerate a more factual answer
- If still unreliable after correction, shows a safe fallback message

### 💡 Reasoning Chain
- Expandable section showing the model's chain-of-thought reasoning
- Helps users understand *why* the system arrived at its answer

### 📚 Source Attribution
- Full list of retrieved source documents with their individual AGHR scores
- Breakdown showing vector similarity, graph path score, and entity overlap per source

### 🔧 Prompting Strategy Selector
- Choose between: `zero_shot`, `chain_of_thought`, `few_shot`, `structured`
- Each strategy changes how the LLM is prompted, affecting answer style and depth

### 👍👎 RLHF Feedback Loop
- Thumbs up / thumbs down buttons on every answer
- Feedback is logged to `data/rlhf/feedback_log.jsonl` for future model improvement

### ⚡ Semantic Caching
- Previously asked questions return instantly (0ms latency)
- Uses both exact string matching and FAISS-based semantic similarity matching

### 🔄 Top-K Control
- Slider to control how many source documents are retrieved (1–10)

---

## 4. Backend Services & Modules

### FastAPI REST API (`app/backend/main.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Submit a question, receive structured answer with confidence, sources, KG context |
| `/api/health` | GET | Health check — reports status of vector store, KG, and LLM components |
| `/api/stats` | GET | System statistics — vector store size, KG node/edge counts, model info, AGHR weights |

**Request Schema (POST /api/query):**
```json
{
  "question": "What causes diabetes?",
  "strategy": "chain_of_thought",
  "top_k": 5
}
```

**Response Schema:**
```json
{
  "question": "...",
  "answer": "...",
  "reasoning": "...",
  "confidence": 0.72,
  "route": "hybrid",
  "is_reliable": true,
  "used_fallback": false,
  "retrieval_scores": [0.85, 0.72, ...],
  "latency_ms": 1234.5,
  "kg_context": "Diabetes -[CO_OCCURS_WITH]- Insulin ..."
}
```

### Streamlit Frontend (`app/frontend/streamlit_app.py`)
- Premium gradient-styled UI with custom CSS
- `@st.cache_resource` for one-time model loading
- Real-time token streaming via `st.write_stream`
- Interactive graph via `streamlit-agraph`

---

## 5. Data Pipeline (Phase 1)

### 5.1 Data Collection (`src/data_pipeline/collector.py`)
Multi-source ingestion engine supporting:
- **PDF Extraction** — via `pdfplumber` (primary) / `PyPDF2` (fallback)
- **Web Scraping** — via `trafilatura` for clean text extraction from URLs
- **HuggingFace Datasets** — `HotpotQA` (multi-hop reasoning) and `PubMedQA` (medical domain)
- **Local Files** — `.txt`, `.csv`, `.json` file loading

All documents are stored as structured `Document` dataclass objects with metadata.

### 5.2 Text Cleaning (`src/data_pipeline/cleaner.py`)
- Unicode normalization (NFKC)
- HTML tag removal
- URL sanitization
- Special character filtering
- **Exact deduplication** via MD5 hashing
- **Fuzzy deduplication** via MinHash LSH (datasketch)
- Minimum text length filtering (50 chars)

### 5.3 Chunking (`src/data_pipeline/chunker.py`)
- **Token-based chunking** using `tiktoken` (cl100k_base encoding)
- Default: **500 tokens per chunk, 80 token overlap**
- Preserves document metadata through chunks
- Each chunk gets a unique ID: `{DOC_ID}_CHUNK_{INDEX}`

### 5.4 Dataset Splitting (`src/data_pipeline/splitter.py`)
- **70/15/15** train/val/test split
- **Stratified by document ID** — all chunks from the same document stay in the same split to prevent data leakage
- Separate splitting logic for QA pairs

---

## 6. Embedding & Vector Search (Phase 2)

### 6.1 Embedding Encoder (`src/embeddings/encoder.py`)
- **Primary Model:** `all-MiniLM-L6-v2` (384-dim, fast baseline)
- **High-Accuracy Model:** `BAAI/bge-base-en-v1.5` (768-dim)
- Batch encoding with progress bars
- L2 normalization for cosine similarity
- BGE-specific query prefix handling

### 6.2 Vector Store (`src/embeddings/vector_store.py`)

**FAISS Implementation:**
- `IndexFlatIP` for cosine similarity (normalized vectors)
- `IndexFlatL2` for L2 distance
- Save/load to disk (`index.faiss` + `chunks.json`)
- Currently indexes **1,515 medical text chunks**

**ChromaDB Implementation (Alternative):**
- Persistent storage with metadata filtering
- Batch insertion with 5,000-item batches
- Built-in cosine similarity via HNSW

---

## 7. Knowledge Graph Engine (Phase 3)

### 7.1 Entity Extraction (`src/knowledge_graph/entity_extractor.py`)
**Hybrid approach combining:**
1. **spaCy NER** — General named entity recognition
2. **Medical Pattern Rules** — Regex-based extraction for:
   - DISEASE (diabetes, cancer, hypertension, etc.)
   - SYMPTOM (fever, cough, chest pain, etc.)
   - TREATMENT (surgery, chemotherapy, dialysis, etc.)
   - MEDICATION (aspirin, insulin, metformin, etc.)
   - BODY_PART (heart, lung, liver, brain, etc.)

### 7.2 Relation Extraction (`src/knowledge_graph/relation_extractor.py`)
**Three extraction strategies:**
1. **Dependency Parsing** — SVO (Subject-Verb-Object) triples via spaCy
2. **Pattern Matching** — 10+ medical relation patterns:
   - CAUSES, TREATS, PREVENTS, SYMPTOM_OF, ASSOCIATED_WITH, DIAGNOSED_BY, MANAGED_BY, RISK_FACTOR_FOR, PART_OF, AFFECTS
3. **Co-occurrence** — Entities in the same sentence → `CO_OCCURS_WITH` relation

### 7.3 Graph Builder (`src/knowledge_graph/graph_builder.py`)
- **Cloud Database:** Neo4j AuraDB (production)
- **Batch insertion** using Cypher `UNWIND` queries grouped by relationship type
- Multi-hop traversal via parameterized Cypher queries (up to 3 hops)
- Local JSON snapshot export for offline visualization
- **Current Stats:** 483 nodes, 629 edges

---

## 8. Query Intelligence (Phase 4)

### Query Analyzer (`src/query/analyzer.py`)

**Entity Extraction** — Extracts medical entities from the user's question

**Complexity Assessment (1-3 scale):**
| Level | Label | Criteria |
|-------|-------|----------|
| 1 | Simple | Single entity, factual question |
| 2 | Moderate | 2 entities or relational keywords present |
| 3 | Complex | 3+ entities, multi-hop patterns detected |

**Intent Classification:**
- `factual` → Direct fact lookup
- `relational` → Relationship between entities
- `complex` → Multi-hop reasoning required

**Routing Decision:**
| Intent | Route |
|--------|-------|
| Factual | `vector` (FAISS only) |
| Relational | `hybrid` (FAISS + Neo4j) |
| Complex | `hybrid` (FAISS + Neo4j) |

---

## 9. AGHR Hybrid Retrieval (Phase 5) ⭐ KEY NOVELTY

### The AGHR Scoring Formula (`src/retrieval/aghr_scorer.py`)

```
Score = α · Sim_vec + β · Path_score + γ · Entity_overlap
```

| Weight | Signal | Description | Default |
|--------|--------|-------------|---------|
| α | Vector Similarity | Cosine similarity from FAISS | 0.45 |
| β | Graph Path Score | Relevance from KG traversal paths | 0.35 |
| γ | Entity Overlap | Query entity presence in retrieved text | 0.20 |

**Key behaviors:**
- Weights auto-normalize to sum to 1.0
- Deduplicates results across vector and graph streams
- Sorts by AGHR score descending
- Returns top-K results with full score breakdown

### Hybrid Retriever (`src/retrieval/aghr_scorer.py` — `HybridRetriever` class)
Orchestrates the full retrieval pipeline:
1. Query Analysis → route determination
2. Vector retrieval (if route includes vector)
3. Graph retrieval via `get_subgraph_context` (if route includes graph)
4. AGHR scoring of combined results
5. Context assembly

---

## 10. Context Fusion (Phase 6)

### Context Fusion Engine (`src/context/fusion.py`)

**Pipeline:**
1. **Collect** — Gather segments from scored results
2. **Deduplicate** — Jaccard similarity threshold (0.85) to remove near-duplicates; keeps higher-scored version
3. **Re-rank** — Sort by AGHR score
4. **Compress** — Fit within token budget (2,048 tokens × 4 chars ≈ 8,192 char limit)
5. **Assemble** — Join segments into final context string

---

## 11. Prompt Engineering & LLM Generation (Phases 7-8)

### Prompt Engine (`src/generation/prompt_engine.py`)
Supports four prompting strategies loaded from `config/prompts.yaml`:

| Strategy | Description |
|----------|-------------|
| `zero_shot` | Direct question, no examples |
| `few_shot` | Includes example QA pairs for in-context learning |
| `chain_of_thought` | Asks model to reason step-by-step before answering |
| `structured` | Requests formatted output: Reasoning → Answer → Sources → Confidence |

Also handles fine-tuning sample formatting (Instruction/Input/Response format).

### LLM Layer (`src/generation/llm_layer.py`)

**Supported Models:**
| Model | Type | Use Case |
|-------|------|----------|
| `google/flan-t5-small` | Seq2Seq | Fast inference, low memory |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Causal | Higher quality, more VRAM |

**Key features:**
- **Token Streaming** — `TextIteratorStreamer` for real-time token output
- **Beam Search** — `num_beams=3` for higher quality answers
- **Context Limiting** — Auto-truncates context for small models (400 tokens) vs large (1,500 tokens)
- **Min Token Enforcement** — `min_new_tokens=10` prevents single-word answers
- Optional 4-bit quantization via `BitsAndBytesConfig`

---

## 12. QLoRA Fine-Tuning (Phase 9)

### Dataset Builder (`src/finetuning/dataset_builder.py`)
- Builds instruction-format datasets from QA pairs + retrieved context
- Format: `### Instruction → ### Input → ### Response`
- Keyword-based context retrieval for dataset augmentation

### QLoRA Trainer (`src/finetuning/qlora_trainer.py`)

**Configuration:**
| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 8 |
| LoRA Alpha | 32 |
| Dropout | 0.05 |
| Target Modules | q, v (T5) / q_proj, v_proj (Llama) |
| Quantization | 4-bit NF4 |
| Compute Dtype | bfloat16 |
| Epochs | 3 |
| Batch Size | 4 |
| Learning Rate | 2e-4 |

**Pipeline:**
1. Load base model with 4-bit quantization (GPU) or standard (CPU)
2. Apply LoRA adapters via PEFT
3. Tokenize dataset with padding and labels
4. Train using HuggingFace `Trainer` (version-safe)
5. Save LoRA adapter weights to `models/finetuned/`

---

## 13. Hallucination Guard (Phase 10)

### Three-Layer Defense (`src/hallucination/guard.py`)

**Layer 1 — Grounding Checker:**
- **Keyword Method:** Calculates word overlap between answer and context (threshold: 0.3)
- **NLI Method (Optional):** Uses `cross-encoder/nli-deberta-v3-base` for natural language inference
- Detects "not found" phrases as valid grounded responses

**Layer 2 — Confidence Scorer:**
Weighted combination of three signals:
| Signal | Weight | Description |
|--------|--------|-------------|
| Retrieval | 0.4 | Average AGHR scores of top results |
| Grounding | 0.4 | Grounding checker score |
| Quality | 0.2 | Answer length, repetition, structure heuristics |

**Layer 3 — Self-Correction Loop (UI):**
1. If confidence < threshold → trigger self-correction
2. Re-prompt LLM with stricter instructions and trimmed context
3. Re-evaluate with Hallucination Guard
4. If still unreliable → show safe fallback message

---

## 14. Evaluation & Metrics (Phase 11)

### Metrics Calculator (`src/evaluation/metrics.py`)

| Category | Metrics |
|----------|---------|
| **Generation** | Exact Match, F1 (token-level), BLEU (smoothed), ROUGE-1, ROUGE-2, ROUGE-L |
| **Retrieval** | Precision@K, Recall@K |
| **Statistical** | Paired t-test for significance (p < 0.05) |
| **Performance** | Average latency, total latency |

### Ablation Study (`run_ablation.py`)
Compares 4 system configurations on the same test set:

| Config | Description |
|--------|-------------|
| Base LLM | Zero-shot, no retrieval |
| + Prompt | Chain-of-thought prompting |
| + RAG | Standard vector retrieval only |
| + AGHR | Full hybrid system (Vector + Graph + Entity) |

### RAGAS Evaluation (`src/evaluation/ragas_eval.py`)
Advanced LLM-as-a-Judge metrics:
- **Faithfulness** — Is the answer supported by the context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Are the retrieved contexts relevant?
- **Context Recall** — Are all necessary contexts retrieved?

---

## 15. Deployment Layer (Phase 17)

### FastAPI Backend
- CORS-enabled REST API
- Lazy-loading of ML models (loaded on first request)
- Pydantic request/response validation
- Global singleton pattern for system components
- Runs on port 8000

### Streamlit Frontend
- Premium gradient-styled UI with custom CSS
- Cached model loading (`@st.cache_resource`)
- Real-time token streaming
- Interactive knowledge graph visualization
- RLHF feedback logging
- Runs on port 8501/8502

---

## 16. Pro Features

### ⚡ Semantic Caching (`src/retrieval/cache.py`)
- **Dual-layer cache:**
  - Exact string match → 0ms latency
  - FAISS semantic similarity search (threshold: 0.95) → near-instant
- Numpy-safe JSON serialization for robust persistence
- Disk persistence (`data/cache/cache.index` + `cache_data.json`)
- Config-hash awareness (different settings = different cache entries)

### 🔄 Token Streaming
- Uses `TextIteratorStreamer` from HuggingFace Transformers
- Runs generation in a background thread
- Yields tokens as they're produced
- Integrated with Streamlit's `st.write_stream`

### 📊 RLHF Feedback Collection
- Thumbs up/down on every answer
- Logged as JSONL with timestamp, question, answer, and label
- Ready for future preference-based fine-tuning

### 🔗 Interactive Knowledge Graph Visualization
- Powered by `streamlit-agraph`
- Dynamic node/edge rendering from Neo4j query results
- Physics-enabled layout for organic graph arrangement
- Safety limit of 30 edges to prevent UI overload

---

## 17. Technology Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.10+ |
| **Frontend** | Streamlit, streamlit-agraph |
| **Backend API** | FastAPI, Uvicorn |
| **Vector DB** | FAISS (primary), ChromaDB (alternative) |
| **Graph DB** | Neo4j AuraDB (cloud) |
| **Embeddings** | Sentence-Transformers (MiniLM / BGE) |
| **LLM** | FLAN-T5, TinyLlama (via HuggingFace Transformers) |
| **Fine-tuning** | PEFT (LoRA), BitsAndBytes (4-bit), TRL |
| **NLP** | spaCy (NER + dependency parsing) |
| **Tokenization** | tiktoken (cl100k_base) |
| **Evaluation** | NLTK (BLEU), rouge-score, RAGAS, SciPy |
| **Data** | PubMedQA, MedQuAD, HotpotQA |
| **Logging** | Loguru |

---

## 18. Project Structure

```
aghr-system/
├── config/
│   ├── config.yaml              # Central configuration (all parameters)
│   └── prompts.yaml             # Prompt templates for all strategies
├── data/
│   ├── raw/                     # Raw collected documents
│   ├── chunks/                  # Tokenized text chunks
│   ├── qa_pairs/                # QA pairs from datasets
│   ├── kg_triples/              # Knowledge graph JSON snapshot
│   ├── vector_index/            # FAISS index + chunk metadata
│   ├── cache/                   # Semantic cache persistence
│   └── rlhf/                    # User feedback logs
├── src/
│   ├── data_pipeline/
│   │   ├── collector.py         # Multi-source data ingestion
│   │   ├── cleaner.py           # Text cleaning + deduplication
│   │   ├── chunker.py           # Token-based chunking
│   │   └── splitter.py          # Train/val/test splitting
│   ├── embeddings/
│   │   ├── encoder.py           # Sentence-transformer encoding
│   │   └── vector_store.py      # FAISS + ChromaDB stores
│   ├── knowledge_graph/
│   │   ├── entity_extractor.py  # spaCy + medical rule NER
│   │   ├── relation_extractor.py # SVO + pattern + co-occurrence
│   │   └── graph_builder.py     # Neo4j AuraDB interface
│   ├── query/
│   │   └── analyzer.py          # Intent, complexity, routing
│   ├── retrieval/
│   │   ├── aghr_scorer.py       # ⭐ AGHR formula + HybridRetriever
│   │   └── cache.py             # Semantic caching layer
│   ├── context/
│   │   └── fusion.py            # Dedup, re-rank, compress
│   ├── generation/
│   │   ├── prompt_engine.py     # Template management
│   │   └── llm_layer.py         # Model loading + streaming
│   ├── finetuning/
│   │   ├── dataset_builder.py   # Instruction-format datasets
│   │   └── qlora_trainer.py     # QLoRA training pipeline
│   ├── hallucination/
│   │   └── guard.py             # Grounding + confidence + fallback
│   ├── evaluation/
│   │   ├── metrics.py           # BLEU, ROUGE, F1, EM
│   │   ├── ragas_eval.py        # RAGAS LLM-as-Judge
│   │   └── test_dataset.py      # Test data management
│   └── utils/
│       └── helpers.py           # Config loading, logging setup
├── app/
│   ├── frontend/
│   │   └── streamlit_app.py     # Interactive web UI
│   └── backend/
│       └── main.py              # FastAPI REST API
├── models/                      # Fine-tuned model adapters
├── results/                     # Evaluation metrics & plots
├── run_basic_rag.py             # Quick RAG demo script
├── run_hybrid_rag.py            # Full hybrid RAG demo
├── run_ablation.py              # Ablation study runner
├── run_evals.py                 # Evaluation runner
├── rebuild_medical_index.py     # Rebuild FAISS index from medical data
├── rebuild_medical_graph.py     # Rebuild Neo4j graph from medical data
├── requirements.txt             # Python dependencies
└── README.md                    # Project README
```

---

> **This project is a complete, applied-research pipeline that goes far beyond a standard RAG system. It implements adaptive retrieval, multi-hop reasoning, hallucination prevention, and professional-grade deployment — ready for academic evaluation and real-world medical QA applications.**
