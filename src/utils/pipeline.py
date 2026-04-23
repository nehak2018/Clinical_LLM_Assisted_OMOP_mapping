from src.llms.engine import run_model
from data.sample import ICD_TO_OMOP
from src.utils.metrics import evaluate

def map_to_omop(icds):
    return [ICD_TO_OMOP[i] for i in icds if i in ICD_TO_OMOP]

def run_pipeline(text, gold_icd, models, hf_pipeline):
    results = {}

    gold_omop = map_to_omop(gold_icd)

    for m in models:
        preds = run_model(m, text, hf_pipeline)
        mapped = map_to_omop(preds)
        results[m] = (preds, evaluate(mapped, gold_omop))

    return results