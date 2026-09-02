"""
AGHR System — Basic RAG Pipeline Demo
=======================================
End-to-end script that demonstrates Macro-Phase A:
  1. Collect data (HotpotQA + PubMedQA)
  2. Clean text
  3. Chunk documents
  4. Generate embeddings
  5. Store in FAISS
  6. Retrieve relevant chunks for a query
  7. Generate answer using FLAN-T5

Run: python run_basic_rag.py
"""

import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def main():
    """Run the complete basic RAG pipeline."""

    # ── Setup ─────────────────────────────────────────────
    from src.utils.logger import setup_logger
    from src.utils.helpers import load_config

    setup_logger()
    config = load_config("config/config.yaml")
    logger.info("=" * 60)
    logger.info("AGHR System — Basic RAG Pipeline Demo")
    logger.info("=" * 60)

    # ── Phase 1.1: Data Collection ────────────────────────
    logger.info("\n📦 PHASE 1.1 — Data Collection")
    from src.data_pipeline.collector import DataCollector

    collector = DataCollector(output_dir=config["data"]["raw_dir"])

    # Collect HotpotQA (pilot: first 500 samples for speed)
    logger.info("Loading HotpotQA (pilot)...")
    collector.collect_hotpotqa(split="train[:500]")

    # Save raw documents
    raw_path = collector.save_documents()
    stats = collector.get_stats()
    logger.info(f"Collection stats: {json.dumps(stats, indent=2)}")

    # ── Phase 1.2: Data Cleaning ──────────────────────────
    logger.info("\n🧹 PHASE 1.2 — Data Cleaning")
    from src.data_pipeline.cleaner import TextCleaner

    doc_dicts = [d.to_dict() for d in collector.documents]
    cleaner = TextCleaner(min_text_length=30)
    cleaned_docs = cleaner.clean_documents(doc_dicts)
    logger.info(f"After cleaning: {len(cleaned_docs)} documents")

    # ── Phase 1.3: Chunking ──────────────────────────────
    logger.info("\n✂️ PHASE 1.3 — Document Chunking")
    from src.data_pipeline.chunker import DocumentChunker

    chunker = DocumentChunker(
        chunk_size=config["data"]["chunk_size"],
        chunk_overlap=config["data"]["chunk_overlap"],
    )
    chunks = chunker.chunk_documents(cleaned_docs)
    chunk_stats = chunker.get_stats(chunks)
    logger.info(f"Chunk stats: {json.dumps(chunk_stats, indent=2)}")

    # Save chunks
    chunks_path = config["data"]["chunks_dir"] + "/all_chunks.jsonl"
    chunker.save_chunks(chunks, chunks_path)

    # ── Phase 1.4: Dataset Split ─────────────────────────
    logger.info("\n📊 PHASE 1.4 — Dataset Split")
    from src.data_pipeline.splitter import DatasetSplitter

    splitter = DatasetSplitter(
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
    )

    chunk_dicts = [c.to_dict() for c in chunks]
    train_chunks, val_chunks, test_chunks = splitter.split_by_document(chunk_dicts)
    splitter.save_splits(train_chunks, val_chunks, test_chunks,
                         output_dir=config["data"]["chunks_dir"], prefix="chunks")

    # ── Phase 2.1: Embedding Generation ──────────────────
    logger.info("\n🔢 PHASE 2.1 — Embedding Generation")
    from src.embeddings.encoder import EmbeddingEncoder

    model_cfg = config["embeddings"]["models"]["baseline"]
    encoder = EmbeddingEncoder(
        model_name=model_cfg["name"],
        batch_size=config["embeddings"]["batch_size"],
        normalize=config["embeddings"]["normalize"],
    )

    # Use train chunks for the vector store
    train_texts = [c["text"] for c in train_chunks]
    embeddings = encoder.encode_texts(train_texts)
    encoder.save_embeddings(embeddings, config["vector_store"]["index_path"] + "/embeddings.npy")

    # ── Phase 2.2: Vector Store ──────────────────────────
    logger.info("\n💾 PHASE 2.2 — Vector Store Setup")
    from src.embeddings.vector_store import FAISSVectorStore

    vector_store = FAISSVectorStore(
        dimension=model_cfg["dimension"],
        similarity=config["vector_store"]["similarity"],
    )
    vector_store.add(embeddings, train_chunks)
    vector_store.save(config["vector_store"]["index_path"])

    # ── Phase 2.3: Retrieval Test ─────────────────────────
    logger.info("\n🔍 PHASE 2.3 — Retrieval Test")

    test_queries = [
        "What are the symptoms of diabetes in elderly patients?",
        "How does hypertension affect heart disease?",
        "What treatments are available for arthritis?",
    ]

    for query in test_queries:
        logger.info(f"\n🔎 Query: {query}")
        query_emb = encoder.encode_query(query)
        results = vector_store.search(query_emb, top_k=config["vector_store"]["top_k"])

        for r in results[:3]:
            score = r["score"]
            text_preview = r["chunk"]["text"][:150] + "..."
            logger.info(f"  [{r['rank']}] Score: {score:.4f} | {text_preview}")

    # ── Basic Generation (FLAN-T5) ────────────────────────
    logger.info("\n🤖 Basic Generation Test (FLAN-T5)")
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = config["generation"]["models"]["flan_t5"]["name"]
        logger.info(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Test with first query
        query = test_queries[0]
        query_emb = encoder.encode_query(query)
        results = vector_store.search(query_emb, top_k=3)
        context = "\n\n".join([r["chunk"]["text"] for r in results])

        prompt = (
            f"Answer the question using only the provided context.\n\n"
            f"Context:\n{context[:1500]}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"\n📋 Question: {query}")
        logger.info(f"📝 Answer: {answer}")
        logger.info(f"📚 Sources: {len(results)} chunks retrieved")

    except Exception as e:
        logger.warning(f"Generation test skipped (model loading failed): {e}")
        logger.info("Run 'pip install transformers torch' to enable generation.")

    # ── Summary ───────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("✅ Basic RAG Pipeline Complete!")
    logger.info(f"   📄 Documents collected: {stats['total_documents']}")
    logger.info(f"   ✂️  Chunks created: {chunk_stats['total_chunks']}")
    logger.info(f"   🔢 Embeddings generated: {embeddings.shape}")
    logger.info(f"   💾 Vector index saved: {config['vector_store']['index_path']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
