from transformers import pipeline

def load_model():
    return pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple"
    )