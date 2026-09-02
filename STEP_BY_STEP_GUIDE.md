# AGHR System — Complete Step-by-Step Execution Guide

> **Goal**: Follow this guide top-to-bottom. Every command is exact. Every step tells you what it does, what you'll see, and how to know it worked.

---

## 🔑 KEY CONCEPTS (Read First)

### What is spaCy (`en_core_web_sm`)?
spaCy is an NLP (Natural Language Processing) library. `en_core_web_sm` is a small pre-trained English model that can:
- **Identify entities** in text (people, places, diseases, medications)
- **Parse sentence structure** (subject-verb-object relationships)

**Your project uses it for**: Building the Knowledge Graph. It reads your documents and extracts medical entities (diabetes, insulin, heart) and their relationships (insulin → TREATS → diabetes).

### What is HotpotQA?
A famous multi-hop QA dataset from HuggingFace. Each question requires combining facts from **2+ different documents** to find the answer.

**Your project uses it for**: It's your knowledge base. Your system downloads 500 samples, chunks them, embeds them, and uses them to answer questions. It proves your AGHR algorithm can do multi-hop reasoning.

### What is FAISS?
Facebook's library for fast similarity search. It stores your document embeddings and finds the most relevant chunks for any query in milliseconds.

### What is AGHR?
Your novel algorithm: `Score = α·VectorSimilarity + β·GraphPathScore + γ·EntityOverlap`. It combines traditional text search with knowledge graph reasoning.

### What is QLoRA?
A technique to fine-tune large language models using very little GPU memory (4-bit quantization + LoRA adapters). Requires a GPU — that's why it runs on Colab.

---

## 🗺️ WHERE DOES EACH STEP RUN?

| Step | What | Where | Time |
|:-----|:-----|:------|:-----|
| 0 | Environment Setup | 💻 LOCAL | 5 min |
| 1 | Data Collection + Cleaning + Chunking | 💻 LOCAL | 3-5 min |
| 2 | Embeddings + FAISS Vector Index | 💻 LOCAL | 2-3 min |
| 3 | Knowledge Graph Construction | 💻 LOCAL | 5-10 min |
| 4 | Test Hybrid Retrieval (AGHR) | 💻 LOCAL | 1 min |
| 5 | Test LLM Generation (FLAN-T5) | 💻 LOCAL | 2-3 min (first download ~1GB) |
| 6 | Run Evaluation Metrics | 💻 LOCAL | 2-5 min |
| 7 | Launch Streamlit UI | 💻 LOCAL | 30 sec |
| 8 | Fine-Tune with QLoRA | ☁️ COLAB | 15-30 min |
| 9 | Ablation Study | 💻 LOCAL | 5-10 min |

---

## STEP 0 — ENVIRONMENT SETUP

**Where**: 💻 LOCAL (VS Code Terminal)

### 0.1 Open Terminal
Open VS Code → Terminal → New Terminal. Make sure you're in:
```
C:\Users\PULKIT\OneDrive\Desktop\GEN AI\aghr-system
```

### 0.2 Create Virtual Environment
```powershell
python -m venv venv
```
**What this does**: Creates an isolated Python environment so your project's packages don't conflict with anything else.

### 0.3 Activate Virtual Environment
```powershell
.\venv\Scripts\activate
```
**How to know it worked**: You'll see `(venv)` at the start of your terminal prompt.

> ⚠️ **IMPORTANT**: Every time you open a new terminal, you must activate the venv again with this command.

### 0.4 Install All Dependencies
```powershell
pip install -r requirements.txt
```
**What this installs**: PyTorch, Transformers, sentence-transformers, FAISS, spaCy, Neo4j driver, Streamlit, ChromaDB, tiktoken, loguru, and ~40 other packages.

**Time**: 3-5 minutes depending on internet speed.

### 0.5 Download spaCy Language Model
```powershell
python -m spacy download en_core_web_sm
```
**What this does**: Downloads the English NLP model (15MB) that your Knowledge Graph builder needs to extract entities and relationships from text.

### 0.6 Verify Everything Works
```powershell
python -c "import sentence_transformers, faiss, torch, transformers, streamlit, tiktoken, loguru, spacy; print('ALL IMPORTS OK')"
```
**Expected output**: `ALL IMPORTS OK`

**If something fails**: Install the missing package with `pip install <package-name>` and re-run.

✅ **STEP 0 DONE** — Your environment is ready.

---

## STEP 1 + 2 — DATA PIPELINE + VECTOR DATABASE

**Where**: 💻 LOCAL  
**Single command does both steps**:

```powershell
python run_basic_rag.py
```

### What happens inside (in order):

| Phase | What It Does | Output |
|:------|:-------------|:-------|
| **1.1 Data Collection** | Downloads 500 HotpotQA samples from HuggingFace | `data/raw/all_documents.jsonl` |
| **1.2 Data Cleaning** | Removes HTML, special chars, duplicates | Cleaned document list in memory |
| **1.3 Chunking** | Splits documents into 500-token chunks with 80-token overlap | `data/chunks/all_chunks.jsonl` |
| **1.4 Dataset Split** | Splits chunks into 70% train / 15% val / 15% test | `data/chunks/chunks_train.jsonl`, `chunks_val.jsonl`, `chunks_test.jsonl` |
| **2.1 Embedding** | Encodes all train chunks into 384-dim vectors using MiniLM | `data/vector_index/embeddings.npy` |
| **2.2 FAISS Index** | Stores embeddings in FAISS for fast similarity search | `data/vector_index/index.faiss` |
| **2.3 Retrieval Test** | Tests 3 sample queries against the vector index | Printed to terminal |
| **Generation Test** | Downloads FLAN-T5 (~990MB first time) and generates a test answer | Printed to terminal |

### How to know it worked:
You'll see at the end:
```
✅ Basic RAG Pipeline Complete!
   📄 Documents collected: XXX
   ✂️  Chunks created: XXX
   🔢 Embeddings generated: (XXXX, 384)
   💾 Vector index saved: data/vector_index
```

### If it fails:
- **`ModuleNotFoundError`** → Go back to Step 0.4
- **Network error on HotpotQA download** → Check internet connection, retry
- **Out of memory** → Close other applications, retry

✅ **STEPS 1+2 DONE** — You have data, chunks, embeddings, and a working vector search.

---

## STEP 3 + 4 + 5 — KNOWLEDGE GRAPH + AGHR RETRIEVAL + GENERATION

**Where**: 💻 LOCAL  
**Prerequisites**: Step 1+2 must be complete (needs `chunks_train.jsonl` and `index.faiss`).

```powershell
python run_hybrid_rag.py
```

### What happens inside:

| Phase | What It Does | Output |
|:------|:-------------|:-------|
| **Load Phase A** | Loads chunks + FAISS index from previous step | "Loaded X train chunks" |
| **3.1 Entity Extraction** | Runs spaCy + medical regex patterns on 500 chunks to find diseases, symptoms, medications, body parts | Entity list printed |
| **3.2 Relation Extraction** | Finds relationships: SVO triples, medical patterns (X causes Y), co-occurrence | Triple list printed |
| **3.3 Graph Build** | Creates NetworkX graph + pushes to Neo4j (if connected) | `data/kg_triples/knowledge_graph.json` |
| **4 Query Analysis** | Classifies each query as simple/moderate/complex → routes to vector/graph/hybrid | Route printed per query |
| **5 AGHR Scoring** | Combines vector similarity + graph path + entity overlap using α/β/γ weights | AGHR scores printed |
| **Context Fusion** | Merges, deduplicates, compresses vector+graph context | Fusion stats printed |
| **7 LLM Generation** | Loads FLAN-T5, generates answers with chain-of-thought prompting | Answer printed |

### How to know it worked:
```
✅ Hybrid RAG Pipeline Complete!
   🧠 KG: XXX nodes, XXX edges
   🔢 Unique entities: XXX
   📐 Triples: XXX
   🚀 AGHR weights: α=0.4, β=0.35, γ=0.25
```

### Key things to verify:
- **Nodes > 0** — KG was built successfully
- **Edges > 0** — Relationships were found
- **AGHR scores printed** — Scoring formula works
- **Answer text printed** — LLM generation works

✅ **STEPS 3+4+5 DONE** — Knowledge Graph built, AGHR hybrid retrieval working, LLM generating answers.

---

## STEP 6 — EVALUATION BENCHMARKS

**Where**: 💻 LOCAL  
**Prerequisites**: Steps 1-5 complete.

> ⚠️ **Note**: The file `run_evals.py` needs to exist. If it doesn't, create it — see the Deployment Guide for the code.

```powershell
python run_evals.py
```

### What it does:
- Loads the full pipeline (encoder → FAISS → AGHR → FLAN-T5)
- Runs inference on QA pairs from `data/qa_pairs/hotpotqa_qa.json`
- Computes: **BLEU, ROUGE-1, ROUGE-2, ROUGE-L, Exact Match, F1**
- Prints results as JSON

### Expected output:
```json
{
  "experiment": "AGHR_Hybrid",
  "exact_match": 0.XXX,
  "f1": 0.XXX,
  "bleu": 0.XXX,
  "rouge1": 0.XXX,
  "avg_latency": X.XXXs
}
```

**Use these numbers in your presentation slides.**

✅ **STEP 6 DONE** — You have benchmark metrics for your research report.

---

## STEP 7 — STREAMLIT UI

**Where**: 💻 LOCAL  
**Prerequisites**: Steps 1-5 complete (needs FAISS index + KG data).

```powershell
streamlit run app/frontend/streamlit_app.py
```

### What happens:
- Opens a browser at `http://localhost:8501`
- Loads all components (encoder, FAISS, KG, LLM) — takes 30-60 seconds first time
- Shows a beautiful purple-gradient UI

### How to test:
1. Type a question: *"What causes heart disease?"*
2. Click **🚀 Ask AGHR**
3. You should see:
   - **Confidence meter** (percentage)
   - **Route** (VECTOR / GRAPH / HYBRID)
   - **Latency** (in ms)
   - **Answer** in a styled box
   - **Reasoning Chain** (expandable)
   - **Knowledge Graph Context** (expandable)
   - **Retrieved Sources with AGHR scores** (expandable)

### Sidebar features:
- **Prompting Strategy** dropdown (chain_of_thought, zero_shot, etc.)
- **Top-K** slider (how many results to retrieve)
- **α, β, γ sliders** — THIS IS YOUR NOVEL CONTRIBUTION. Show this in your presentation.

### To stop: Press `Ctrl+C` in the terminal.

✅ **STEP 7 DONE** — Your demo UI is running.

---

## STEP 8 — FINE-TUNING WITH QLoRA

**Where**: ☁️ GOOGLE COLAB ONLY  
**Why not local**: Fine-tuning needs ~8GB+ GPU memory. Your PC likely doesn't have enough.

### 8.1 Prepare Files (Local)
1. In Windows Explorer, go to `C:\Users\PULKIT\OneDrive\Desktop\GEN AI\aghr-system`
2. Select the `src` folder and the `config` folder
3. Right-click → Send to → Compressed (zipped) folder
4. Name it `aghr_code.zip`

### 8.2 Set Up Colab
1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click **New Notebook**
3. Go to **Runtime** → **Change runtime type** → select **T4 GPU** → Save
4. In the left sidebar, click the **folder icon** → **Upload** → upload `aghr_code.zip`

### 8.3 Run Training (In Colab)
Create a cell and paste this code:

```python
# Cell 1: Install + Unzip
!pip install -q transformers peft trl bitsandbytes accelerate datasets loguru pyyaml
!unzip -q aghr_code.zip

# Cell 2: Train
import yaml
from src.finetuning.qlora_trainer import QLoRATrainer, create_hf_dataset

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

mock_samples = [{"text": "### Instruction:\nYou are a medical QA assistant.\n\n### Input:\nContext: Insulin treats diabetes.\nQuestion: What treats diabetes?\n\n### Response:\nInsulin is the primary treatment for diabetes."}] * 100

trainer = QLoRATrainer(config.get("finetuning", {}))
trainer.setup(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", model_type="causal")
trainer.apply_lora()
trainer.train(create_hf_dataset(mock_samples))
trainer.save_model("finetuned_aghr_model")
print("✅ TRAINING COMPLETE!")
```

### 8.4 Download Model (From Colab)
1. In the Colab file browser, right-click `finetuned_aghr_model/`
2. Zip and download to your local machine

### How to know it worked:
- You see training loss decreasing in the output
- Final message: `✅ TRAINING COMPLETE!`
- A `finetuned_aghr_model/` folder appears with model files

✅ **STEP 8 DONE** — You have a fine-tuned model.

---

## STEP 9 — ABLATION STUDY (Optional)

**Where**: 💻 LOCAL

> ⚠️ **Note**: The file `run_ablation.py` needs to exist. If it doesn't, create it — see the Deployment Guide.

```powershell
python run_ablation.py
```

### What it does:
Runs 4 different configurations on the same test questions and compares scores:

| # | Configuration | Description |
|:--|:-------------|:------------|
| 1 | **Base LLM** | FLAN-T5 answers directly, no retrieval at all |
| 2 | **Prompt Only** | Chain-of-thought prompting, still no retrieval |
| 3 | **RAG Only** | Vector retrieval (FAISS) + LLM, no knowledge graph |
| 4 | **AGHR Hybrid** | Full system: Vector + KG + AGHR scoring + LLM |

### Expected output:
A comparison table showing each configuration's EM, F1, BLEU, ROUGE scores. Configuration 4 (AGHR) should score highest.

✅ **STEP 9 DONE** — You have ablation data for your research report.

---

## 📋 MASTER EXECUTION ORDER

```
STEP 0 → pip install, spacy download          [LOCAL, 5 min]
STEP 1+2 → python run_basic_rag.py            [LOCAL, 5 min]
STEP 3+4+5 → python run_hybrid_rag.py         [LOCAL, 10 min]
STEP 6 → python run_evals.py                  [LOCAL, 5 min]
STEP 7 → streamlit run app/frontend/...       [LOCAL, instant]
STEP 8 → QLoRA training                       [COLAB, 20 min]
STEP 9 → python run_ablation.py               [LOCAL, 10 min]
```

**Total local time**: ~35 minutes  
**Colab time**: ~20 minutes  

---

## ❓ COMMON ERRORS AND FIXES

| Error | Fix |
|:------|:----|
| `(venv) not showing in terminal` | Run `.\venv\Scripts\activate` |
| `ModuleNotFoundError: No module named 'X'` | Run `pip install X` |
| `FileNotFoundError: index.faiss` | Run `python run_basic_rag.py` first |
| `FileNotFoundError: chunks_train.jsonl` | Run `python run_basic_rag.py` first |
| `OSError: en_core_web_sm not found` | Run `python -m spacy download en_core_web_sm` |
| `CUDA out of memory` (on Colab) | Reduce batch_size in config.yaml to 2 |
| `Neo4j connection refused` | That's OK — system uses local graph fallback |
| `Streamlit: address already in use` | Run with `--server.port 8502` |
| FLAN-T5 download takes forever | First run downloads ~990MB — wait, it's a one-time thing |
