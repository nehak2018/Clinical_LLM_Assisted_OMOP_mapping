DATA = [
    {"note": "Patient has diabetes and hypertension", "gold_icd": ["E11.9", "I10"]},
    {"note": "No evidence of asthma", "gold_icd": []},
    {"note": "Chest pain rule out myocardial infarction", "gold_icd": ["I21.9"]},
]

ICD_TO_OMOP = {
    "E11.9": 201826,
    "I10": 320128,
    "I21.9": 4329847,
}