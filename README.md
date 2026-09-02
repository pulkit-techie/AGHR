# AGHR — Adaptive Graph-Hybrid Retrieval System

> **Research Project**: Hybrid RAG + Knowledge Graph system for medical QA with hallucination reduction, multi-hop reasoning, and explainable answers.

## 🎯 Research Claim

> *AGHR improves QA accuracy by ~25–30% and reduces hallucination vs standard RAG.*

## 🏥 Domain

**Healthcare / Medical QA** (Senior Citizen Health Focus)

## 🏗️ Architecture

```
Query → [Query Analyzer] → [Classifier]
                              ↓
                    ┌─────────┼─────────┐
                    ↓         ↓         ↓
              [Vector DB] [Knowledge] [Hybrid]
              [FAISS]     [Graph]     [Both]
                    ↓         ↓         ↓
                    └─────────┼─────────┘
                              ↓
                      [AGHR Scoring]
                     α·Sim + β·Path + γ·Entity
                              ↓
                      [Context Fusion]
                              ↓
                      [LLM Generation]
                      (FLAN-T5 / TinyLlama)
                              ↓
                   [Hallucination Guard]
                              ↓
                     Reasoning + Answer
                     + Sources + Confidence
```

## 📦 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

### 2. Run Basic RAG Demo (Macro-Phase A)

```bash
python run_basic_rag.py
```

This will:
- Download HotpotQA dataset (pilot: 500 samples)
- Clean and chunk documents
- Generate embeddings (MiniLM)
- Build FAISS vector index
- Test retrieval + generation with FLAN-T5

### 3. Configuration

Edit `config/config.yaml` to customize:
- Chunk size, overlap, top-K
- Embedding model selection
- AGHR scoring weights (α, β, γ)
- LLM model choice
- Neo4j connection settings

## 📁 Project Structure

```
aghr-system/
├── config/           # Configuration files
├── data/             # Raw, processed, chunks, QA pairs
├── src/
│   ├── data_pipeline/    # Phase 1: Collection, cleaning, chunking
│   ├── embeddings/       # Phase 2: Encoding, vector store
│   ├── knowledge_graph/  # Phase 3: Entity/relation extraction, Neo4j
│   ├── query/            # Phase 4: Query analysis, classification
│   ├── retrieval/        # Phase 5: Vector, graph, AGHR hybrid
│   ├── context/          # Phase 6: Fusion, compression
│   ├── generation/       # Phase 7-8: Prompts, LLM layer
│   ├── finetuning/       # Phase 9: QLoRA training
│   ├── hallucination/    # Phase 10: Grounding, confidence
│   ├── evaluation/       # Phase 11-15: Metrics, experiments
│   ├── explainability/   # Phase 16: Reasoning chains
│   └── utils/            # Logging, helpers
├── app/              # Phase 17: FastAPI + Streamlit
├── results/          # Metrics, plots, reports
└── run_basic_rag.py  # Quick demo script
```

## 📊 Evaluation Metrics

| Category | Metrics |
|----------|---------|
| **Generation** | BLEU, ROUGE, Exact Match, F1 |
| **Retrieval** | Precision@K, Recall@K |
| **Advanced** | Faithfulness, Answer Relevancy, Context Recall |
| **Performance** | Latency, Token Cost |

## 🔬 Experiments

| System | Description |
|--------|-------------|
| Base LLM | No retrieval, direct generation |
| + Prompt | With chain-of-thought prompting |
| + RAG | Vector retrieval augmented |
| + RAG+KG | Hybrid AGHR system |
| + Fine-tuned | QLoRA fine-tuned model |

## 📄 License

Academic research project.
