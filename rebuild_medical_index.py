"""
Rebuild FAISS index with REAL medical data.
Downloads PubMedQA + MedQuAD and re-indexes everything.
Run: python rebuild_medical_index.py
"""

import json
import numpy as np
from pathlib import Path
from loguru import logger

def main():
    logger.info("Downloading real medical datasets...")

    from datasets import load_dataset
    from src.embeddings.encoder import EmbeddingEncoder
    from src.embeddings.vector_store import FAISSVectorStore

    chunks = []

    # ── 1. PubMedQA ──────────────────────────────────────────────────────────
    try:
        logger.info("Loading PubMedQA...")
        pubmed = load_dataset("pubmed_qa", "pqa_labeled", split="train[:500]", trust_remote_code=True)
        for item in pubmed:
            # Combine context sentences into a paragraph
            context = " ".join(item.get("context", {}).get("sentences", [[]])[0][:5]) if item.get("context") else ""
            question = item.get("question", "")
            answer = str(item.get("final_decision", "")) + ". " + item.get("long_answer", "")

            if context:
                chunks.append({"text": context, "source": "pubmed_context", "doc_id": f"pubmed_{len(chunks)}"})
            if question and answer:
                chunks.append({"text": f"Q: {question} A: {answer}", "source": "pubmed_qa", "doc_id": f"pubmed_qa_{len(chunks)}"})

        logger.info(f"PubMedQA: {len(chunks)} chunks")
    except Exception as e:
        logger.warning(f"PubMedQA failed: {e}")

    # ── 2. MedQuAD ───────────────────────────────────────────────────────────
    try:
        logger.info("Loading MedQuAD...")
        medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train[:1000]")
        for item in medquad:
            q = item.get("Question", "")
            a = item.get("Answer", "")
            if q and a:
                text = f"Medical Question: {q}\nAnswer: {a}"
                chunks.append({"text": text, "source": "medquad", "doc_id": f"medquad_{len(chunks)}"})
        logger.info(f"After MedQuAD: {len(chunks)} total chunks")
    except Exception as e:
        logger.warning(f"MedQuAD failed: {e}")

    # ── 3. Hardcoded core medical facts (always included) ────────────────────
    medical_facts = [
        "Smoking is the leading cause of preventable cancer deaths. Tobacco smoke contains over 70 known carcinogens. Long-term smoking dramatically increases risk of lung cancer, throat cancer, mouth cancer, and esophageal cancer.",
        "Diabetes mellitus is a chronic disease characterized by high blood sugar levels. Type 1 diabetes is caused by lack of insulin production. Type 2 diabetes is caused by insulin resistance. Symptoms include frequent urination, excessive thirst, and fatigue.",
        "Hypertension (high blood pressure) is a major risk factor for heart disease and stroke. Normal blood pressure is below 120/80 mmHg. Treatment includes lifestyle changes, ACE inhibitors, beta-blockers, and diuretics.",
        "Cancer is a disease of uncontrolled cell growth. Common types include lung, breast, prostate, and colorectal cancer. Diagnosis uses biopsies, imaging (MRI, CT), and blood tests. Treatment options include surgery, chemotherapy, radiation, and immunotherapy.",
        "COVID-19 is caused by the SARS-CoV-2 coronavirus. Symptoms include fever, dry cough, fatigue, and loss of smell/taste. Vaccines are highly effective at preventing severe disease.",
        "Pneumonia is an infection of the lungs causing air sacs to fill with fluid. Symptoms are fever, chills, cough with phlegm, and difficulty breathing. Treatment is antibiotics for bacterial pneumonia.",
        "Insulin is a hormone produced by the pancreas that allows cells to use glucose for energy. Diabetic patients either cannot produce insulin (Type 1) or cannot use it effectively (Type 2).",
        "Alzheimer's disease is the most common form of dementia. It causes memory loss, confusion, and behavioral changes. It is caused by accumulation of amyloid plaques and tau tangles in the brain.",
        "Asthma is a chronic respiratory condition causing airway inflammation and narrowing. Symptoms include wheezing, shortness of breath, and chest tightness. Treated with bronchodilators and corticosteroids.",
        "Heart attack (myocardial infarction) occurs when blood flow to a part of the heart is blocked. Symptoms include chest pain, shortness of breath, and pain radiating to the arm. Treatment includes aspirin, clot-busting drugs, and angioplasty.",
        "Antibiotics are medications that kill or inhibit the growth of bacteria. They are NOT effective against viral infections like the flu or common cold. Overuse leads to antibiotic resistance.",
        "Depression is a mental health disorder characterized by persistent sadness, loss of interest, and fatigue. It is treated with antidepressants, psychotherapy, and lifestyle changes.",
        "HIV (Human Immunodeficiency Virus) attacks the immune system. Without treatment it progresses to AIDS. Antiretroviral therapy (ART) can suppress the virus and allow normal life expectancy.",
        "Cholesterol is a fatty substance in the blood. High LDL cholesterol increases risk of heart disease. Statins are medications used to lower cholesterol levels.",
        "Vaccines work by teaching the immune system to recognize pathogens. They are the most effective way to prevent infectious diseases like measles, polio, and COVID-19.",
    ]

    for i, fact in enumerate(medical_facts):
        chunks.append({"text": fact, "source": "medical_facts", "doc_id": f"fact_{i}"})

    logger.info(f"Total medical chunks: {len(chunks)}")

    if not chunks:
        logger.error("No chunks collected! Cannot rebuild index.")
        return

    # ── 4. Encode all chunks ──────────────────────────────────────────────────
    logger.info("Loading embedding model...")
    encoder = EmbeddingEncoder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    texts = [c["text"] for c in chunks]
    logger.info(f"Encoding {len(texts)} chunks...")
    embeddings = encoder.encode_texts(texts, show_progress=True)

    # ── 5. Build new FAISS index ──────────────────────────────────────────────
    logger.info("Building FAISS index...")
    store = FAISSVectorStore(dimension=384, similarity="cosine")
    store.add(embeddings, chunks)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    Path("data/vector_index").mkdir(parents=True, exist_ok=True)
    store.save("data/vector_index")
    logger.info(f"Saved new medical FAISS index with {store.index.ntotal} vectors!")
    logger.info("DONE! Restart Streamlit to use the new medical index.")


if __name__ == "__main__":
    main()
