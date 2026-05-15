import random
import csv
import json

# ----------------------------
# 1. Disease knowledge base
# ----------------------------
DISEASE_BANK = [
    {"name": "Type 2 diabetes mellitus", "icd10": "E11.9", "concept_id": 201826,
     "symptoms": ["polyuria", "polydipsia", "fatigue", "blurred vision"]},

    {"name": "Essential hypertension", "icd10": "I10", "concept_id": 320128,
     "symptoms": ["headache", "dizziness", "asymptomatic"]},

    {"name": "Acute bronchitis", "icd10": "J20.9", "concept_id": 254761,
     "symptoms": ["cough", "sputum", "low-grade fever"]},

    {"name": "Asthma", "icd10": "J45.909", "concept_id": 317009,
     "symptoms": ["wheezing", "shortness of breath", "chest tightness"]},

    {"name": "Urinary tract infection", "icd10": "N39.0", "concept_id": 132797,
     "symptoms": ["dysuria", "frequency", "suprapubic pain"]},

    {"name": "Hyperlipidemia", "icd10": "E78.5", "concept_id": 432867,
     "symptoms": ["asymptomatic", "elevated cholesterol"]},

    {"name": "Gastroesophageal reflux disease", "icd10": "K21.9", "concept_id": 314658,
     "symptoms": ["heartburn", "regurgitation", "chest discomfort"]},

    {"name": "Depression", "icd10": "F32.9", "concept_id": 440383,
     "symptoms": ["low mood", "anhedonia", "fatigue"]},

    {"name": "Pneumonia", "icd10": "J18.9", "concept_id": 255848,
     "symptoms": ["fever", "cough", "shortness of breath"]},

    {"name": "Obesity", "icd10": "E66.9", "concept_id": 432851,
     "symptoms": ["weight gain", "fatigue", "asymptomatic"]}
]

NOISE_SYMPTOMS = [
    "mild nausea", "sleep disturbance", "normal appetite",
    "general weakness", "fatigue"
]

# ----------------------------
# 2. Sampling logic
# ----------------------------
def sample_conditions():
    n = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    return random.sample(DISEASE_BANK, n)

def generate_patient():
    conditions = sample_conditions()

    symptoms = []
    diagnoses = []

    for c in conditions:
        symptoms += random.sample(c["symptoms"], k=min(2, len(c["symptoms"])))
        diagnoses.append(c)

    symptoms += random.sample(NOISE_SYMPTOMS, 1)

    return {
        "age": random.randint(18, 85),
        "sex": random.choice(["male", "female"]),
        "symptoms": list(set(symptoms)),
        "diagnoses": diagnoses
    }

# ----------------------------
# 3. Clinical note generator
# ----------------------------
def generate_note(p):
    dx_text = ", ".join([d["name"] for d in p["diagnoses"]])

    return (
        f"CHIEF COMPLAINT: {', '.join(p['symptoms'])}. "
        f"HPI: {p['age']}-year-old {p['sex']} with {', '.join(p['symptoms'])}. "
        f"ASSESSMENT: Likely {dx_text}. "
        f"PLAN: Further evaluation recommended."
    )

# ----------------------------
# 4. Generate dataset
# ----------------------------
def generate_dataset(n=50):
    dataset = []

    for i in range(n):
        p = generate_patient()
        note = generate_note(p)

        dataset.append({
            "note_id": i,
            "age": p["age"],
            "sex": p["sex"],
            "note": note,
            "gold_diagnoses": [d["name"] for d in p["diagnoses"]],
            "gold_icd10": [d["icd10"] for d in p["diagnoses"]],
            "gold_concept_ids": [d["concept_id"] for d in p["diagnoses"]]
        })

    return dataset

# ----------------------------
# 5. Save CSV
# ----------------------------
def save_csv(data, filename="synthetic_clinical_notes_50.csv"):
    keys = data[0].keys()

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

# ----------------------------
# 6. Save JSON
# ----------------------------
def save_json(data, filename="synthetic_clinical_notes_50.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ----------------------------
# 7. Run everything
# ----------------------------
if __name__ == "__main__":
    data = generate_dataset(50)

    save_csv(data)
    save_json(data)

    print("Generated 50 synthetic clinical notes")
    print("Saved synthetic_clinical_notes_50.csv")
    print("Saved synthetic_clinical_notes_50.json")