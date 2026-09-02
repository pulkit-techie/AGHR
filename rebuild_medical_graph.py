"""
Wipe old Wikipedia data and build a REAL Medical Knowledge Graph in Neo4j.
Run: python rebuild_medical_graph.py
"""

from loguru import logger
from datasets import load_dataset
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.relation_extractor import RelationExtractor
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.utils.helpers import load_config

def main():
    logger.info("Initializing Graph Building Pipeline...")
    config = load_config("config/config.yaml")
    
    neo4j_cfg = config["knowledge_graph"]["neo4j"]
    kg = KnowledgeGraphBuilder(
        uri=neo4j_cfg["uri"],
        username=neo4j_cfg["username"],
        password=neo4j_cfg["password"]
    )
    entity_ext = EntityExtractor(spacy_model="en_core_web_sm")
    relation_ext = RelationExtractor()

    # ── 1. WIPE OLD WIKIPEDIA GRAPH ──────────────────────────────────────────
    logger.warning("WIPING OLD NEO4J DATABASE (Say goodbye to Cuban Marimba Band!)")
    if kg.driver:
        with kg.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j database is now completely empty.")
    else:
        logger.error("Could not connect to Neo4j. Is it running?")
        return

    # ── 2. GATHER MEDICAL DATA ───────────────────────────────────────────────
    texts = []
    
    # Core hardcoded facts
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
        "Vaccines work by teaching the immune system to recognize pathogens. They are the most effective way to prevent infectious diseases like measles, polio, and COVID-19."
    ]
    texts.extend(medical_facts)

    # A subset of MedQuAD (taking just 50 to keep processing fast)
    try:
        medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train[:50]")
        for item in medquad:
            q = item.get("Question", "")
            a = item.get("Answer", "")
            if q and a:
                texts.append(f"{q} {a}")
    except Exception as e:
        logger.warning(f"Could not load MedQuAD: {e}")

    logger.info(f"Processing {len(texts)} medical texts for Knowledge Graph...")

    # ── 3. EXTRACT RELATIONS & PUSH TO NEO4J ─────────────────────────────────
    all_relations = []
    for i, text in enumerate(texts):
        if i % 10 == 0:
            logger.info(f"Extracting relations: {i}/{len(texts)}")
            
        entities = entity_ext.extract_entities(text)
        relations = relation_ext.extract_all_relations(text, entities)
        
        if relations:
            all_relations.append({
                "doc_id": f"med_{i}",
                "relations": relations
            })

    logger.info("Pushing extracted relations to Neo4j AuraDB...")
    kg.build_from_relation_results(all_relations)
    
    # Save a local JSON snapshot for Streamlit visualization
    save_path = "data/kg_triples/knowledge_graph.json"
    logger.info(f"Saving local graph snapshot to {save_path}...")
    kg.save(save_path)

    logger.info("DONE! Medical Knowledge Graph is now live in Neo4j and locally.")

if __name__ == "__main__":
    main()
