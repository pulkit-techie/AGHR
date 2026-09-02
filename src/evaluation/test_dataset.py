"""
AGHR System — Extended Evaluation Dataset (50 Medical QA Pairs)
================================================================
Curated medical QA test set for robust evaluation.
Each entry has a question and a gold-standard reference answer.
"""

EXTENDED_TEST_DATA = [
    # --- General Medicine (10) ---
    {"question": "What is the relationship between insulin and diabetes?",
     "answer": "Insulin regulates blood sugar levels; in Type 1 diabetes the body cannot produce insulin, and in Type 2 diabetes the body becomes resistant to insulin."},
    {"question": "How does hypertension affect cardiovascular health?",
     "answer": "Hypertension increases the risk of heart attack, stroke, heart failure, and kidney disease by damaging blood vessels."},
    {"question": "What are the symptoms of pneumonia?",
     "answer": "Symptoms include fever, cough with phlegm, shortness of breath, chest pain, fatigue, and chills."},
    {"question": "What medications treat high blood pressure?",
     "answer": "ACE inhibitors, beta-blockers, calcium channel blockers, and diuretics are commonly used to treat hypertension."},
    {"question": "How is cancer diagnosed?",
     "answer": "Cancer is diagnosed through biopsies, imaging tests like CT and MRI, blood tests for tumor markers, and physical examinations."},
    {"question": "What causes Type 2 diabetes?",
     "answer": "Type 2 diabetes is caused by a combination of genetic factors, obesity, sedentary lifestyle, and insulin resistance."},
    {"question": "What is the treatment for asthma?",
     "answer": "Asthma is treated with inhaled corticosteroids for long-term control and short-acting bronchodilators for quick relief."},
    {"question": "How does obesity affect health?",
     "answer": "Obesity increases the risk of diabetes, heart disease, hypertension, stroke, certain cancers, and joint problems."},
    {"question": "What are the symptoms of a heart attack?",
     "answer": "Symptoms include chest pain or pressure, shortness of breath, pain in the arm or jaw, nausea, and cold sweats."},
    {"question": "What is the role of cholesterol in heart disease?",
     "answer": "High LDL cholesterol leads to plaque buildup in arteries, causing atherosclerosis and increasing the risk of heart attack and stroke."},

    # --- Pharmacology (8) ---
    {"question": "What are the side effects of metformin?",
     "answer": "Common side effects include nausea, diarrhea, stomach pain, and in rare cases, lactic acidosis."},
    {"question": "How does aspirin prevent heart attacks?",
     "answer": "Aspirin inhibits platelet aggregation by blocking cyclooxygenase, reducing blood clot formation in coronary arteries."},
    {"question": "What is the mechanism of action of ACE inhibitors?",
     "answer": "ACE inhibitors block the angiotensin-converting enzyme, preventing the conversion of angiotensin I to angiotensin II, thereby reducing blood pressure."},
    {"question": "What are the risks of long-term corticosteroid use?",
     "answer": "Long-term use can cause osteoporosis, weight gain, diabetes, cataracts, adrenal suppression, and increased infection risk."},
    {"question": "How do statins lower cholesterol?",
     "answer": "Statins inhibit HMG-CoA reductase, an enzyme in the liver that is critical for cholesterol production, thereby lowering LDL levels."},
    {"question": "What is the difference between ibuprofen and paracetamol?",
     "answer": "Ibuprofen is an anti-inflammatory pain reliever (NSAID) while paracetamol (acetaminophen) reduces pain and fever but has no anti-inflammatory effect."},
    {"question": "What drugs are used to treat epilepsy?",
     "answer": "Antiepileptic drugs include valproate, carbamazepine, lamotrigine, levetiracetam, and phenytoin."},
    {"question": "How does insulin therapy work for diabetic patients?",
     "answer": "Insulin therapy replaces or supplements the body's natural insulin to regulate blood glucose levels in diabetic patients."},

    # --- Cardiology (6) ---
    {"question": "What is atrial fibrillation?",
     "answer": "Atrial fibrillation is an irregular, often rapid heart rhythm originating in the atria that increases the risk of stroke and heart failure."},
    {"question": "How is heart failure classified?",
     "answer": "Heart failure is classified by the New York Heart Association into four functional classes based on severity of symptoms during physical activity."},
    {"question": "What is the relationship between smoking and cardiovascular disease?",
     "answer": "Smoking damages blood vessel walls, promotes atherosclerosis, raises blood pressure, and significantly increases the risk of heart attack and stroke."},
    {"question": "What is an angioplasty procedure?",
     "answer": "Angioplasty is a minimally invasive procedure where a balloon catheter opens blocked coronary arteries, often with stent placement."},
    {"question": "What causes congestive heart failure?",
     "answer": "Congestive heart failure is caused by conditions that weaken or damage the heart muscle, including coronary artery disease, hypertension, and valve disorders."},
    {"question": "How does exercise benefit cardiovascular health?",
     "answer": "Exercise strengthens the heart muscle, improves blood circulation, lowers blood pressure, reduces cholesterol, and decreases the risk of heart disease."},

    # --- Respiratory (5) ---
    {"question": "What is the difference between COPD and asthma?",
     "answer": "COPD involves progressive, irreversible airflow limitation usually caused by smoking, while asthma involves reversible airway obstruction with inflammation and bronchospasm."},
    {"question": "How is tuberculosis diagnosed?",
     "answer": "Tuberculosis is diagnosed through chest X-rays, sputum culture, tuberculin skin test (Mantoux), and interferon-gamma release assays."},
    {"question": "What causes pulmonary embolism?",
     "answer": "Pulmonary embolism is caused by a blood clot, usually from deep vein thrombosis in the legs, that travels to and blocks arteries in the lungs."},
    {"question": "What are the treatments for chronic obstructive pulmonary disease?",
     "answer": "COPD treatments include bronchodilators, inhaled corticosteroids, oxygen therapy, pulmonary rehabilitation, and smoking cessation."},
    {"question": "How does COVID-19 affect the respiratory system?",
     "answer": "COVID-19 causes inflammation and damage to lung tissue, leading to pneumonia, acute respiratory distress syndrome, and impaired gas exchange."},

    # --- Neurology (5) ---
    {"question": "What are the early symptoms of Alzheimer's disease?",
     "answer": "Early symptoms include memory loss, difficulty with familiar tasks, confusion about time and place, and changes in mood and personality."},
    {"question": "How does Parkinson's disease affect the brain?",
     "answer": "Parkinson's disease involves the progressive loss of dopamine-producing neurons in the substantia nigra, leading to tremor, rigidity, and bradykinesia."},
    {"question": "What is a stroke and what causes it?",
     "answer": "A stroke occurs when blood supply to part of the brain is interrupted, caused by either a blood clot (ischemic) or bleeding (hemorrhagic)."},
    {"question": "What treatments are available for epilepsy?",
     "answer": "Epilepsy treatments include antiepileptic medications, ketogenic diet, vagus nerve stimulation, and in refractory cases, surgical resection."},
    {"question": "How does multiple sclerosis affect the nervous system?",
     "answer": "Multiple sclerosis causes the immune system to attack the myelin sheath around nerve fibers, disrupting communication between the brain and body."},

    # --- Endocrinology (4) ---
    {"question": "What is hypothyroidism and how is it treated?",
     "answer": "Hypothyroidism is underproduction of thyroid hormones, causing fatigue and weight gain. It is treated with synthetic levothyroxine replacement."},
    {"question": "How does the pancreas regulate blood sugar?",
     "answer": "The pancreas produces insulin to lower blood sugar and glucagon to raise it, maintaining glucose homeostasis."},
    {"question": "What are the complications of uncontrolled diabetes?",
     "answer": "Complications include neuropathy, nephropathy, retinopathy, cardiovascular disease, diabetic foot ulcers, and ketoacidosis."},
    {"question": "What is Cushing's syndrome?",
     "answer": "Cushing's syndrome results from prolonged exposure to excess cortisol, causing weight gain, high blood pressure, skin changes, and muscle weakness."},

    # --- Gastroenterology (4) ---
    {"question": "What causes gastroesophageal reflux disease (GERD)?",
     "answer": "GERD is caused by the weakening of the lower esophageal sphincter, allowing stomach acid to flow back into the esophagus."},
    {"question": "How is inflammatory bowel disease different from irritable bowel syndrome?",
     "answer": "IBD (Crohn's and ulcerative colitis) involves chronic inflammation of the digestive tract, while IBS is a functional disorder without structural damage."},
    {"question": "What is cirrhosis of the liver?",
     "answer": "Cirrhosis is late-stage scarring of the liver caused by chronic liver diseases like hepatitis and alcohol abuse, leading to liver failure."},
    {"question": "What are the symptoms of appendicitis?",
     "answer": "Symptoms include sudden pain around the navel that shifts to the lower right abdomen, nausea, vomiting, fever, and loss of appetite."},

    # --- Oncology (4) ---
    {"question": "What is the difference between chemotherapy and radiation therapy?",
     "answer": "Chemotherapy uses drugs to kill cancer cells systemically, while radiation therapy uses high-energy beams to target and destroy cancer cells locally."},
    {"question": "How does immunotherapy treat cancer?",
     "answer": "Immunotherapy boosts the body's immune system to recognize and attack cancer cells, using checkpoint inhibitors, CAR-T cells, or cancer vaccines."},
    {"question": "What are the risk factors for breast cancer?",
     "answer": "Risk factors include family history, BRCA gene mutations, age, obesity, hormone replacement therapy, and alcohol consumption."},
    {"question": "How is leukemia treated?",
     "answer": "Leukemia is treated with chemotherapy, targeted therapy, radiation, stem cell transplant, and in some cases immunotherapy."},

    # --- Multi-hop Reasoning (4) ---
    {"question": "How does diabetes lead to kidney disease?",
     "answer": "High blood sugar from diabetes damages the blood vessels in the kidneys' filtering units (nephrons), leading to diabetic nephropathy and eventual kidney failure."},
    {"question": "What is the pathway from hypertension to stroke?",
     "answer": "Hypertension damages arterial walls, promotes atherosclerosis and weakens blood vessels in the brain, which can lead to ischemic or hemorrhagic stroke."},
    {"question": "How does smoking cause lung cancer?",
     "answer": "Carcinogens in tobacco smoke damage DNA in lung cells, causing mutations that lead to uncontrolled cell growth and malignant tumor formation."},
    {"question": "How does obesity lead to sleep apnea?",
     "answer": "Excess fat deposits around the upper airway in obese individuals cause airway obstruction during sleep, leading to obstructive sleep apnea."},
]

# Category metadata for analysis
CATEGORY_MAP = {
    "General Medicine": list(range(0, 10)),
    "Pharmacology": list(range(10, 18)),
    "Cardiology": list(range(18, 24)),
    "Respiratory": list(range(24, 29)),
    "Neurology": list(range(29, 34)),
    "Endocrinology": list(range(34, 38)),
    "Gastroenterology": list(range(38, 42)),
    "Oncology": list(range(42, 46)),
    "Multi-hop Reasoning": list(range(46, 50)),
}
