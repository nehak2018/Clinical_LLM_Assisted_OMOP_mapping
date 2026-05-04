import re
import requests
from config.settings import LLM_CONFIG, PROMPT
from data.sample import ICD_TO_OMOP

# simple mapping
TEXT_TO_ICD = {
    "diabetes": "E11.9",
    "hypertension": "I10",
    "myocardial infarction": "I21.9",
    "chest pain": "I21.9",
}

# -------------------------
# RULE
# -------------------------
def rule_model(text):
    t = text.lower()
    if "no evidence" in t:
        return []

    out = []
    if "diabetes" in t:
        out.append("E11.9")
    if "hypertension" in t:
        out.append("I10")
    if "chest pain" in t:
        out.append("I21.9")

    return list(set(out))


# -------------------------
# HF
# -------------------------
def hf_model(text, hf_pipeline):
    entities = hf_pipeline(text)
    terms = [e["word"].lower() for e in entities]

    icds = []
    for term in terms:
        for k, v in TEXT_TO_ICD.items():
            if k in term:
                icds.append(v)

    return list(set(icds))


# -------------------------
# OLLAMA
# -------------------------
def call_ollama(model, text):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": PROMPT + "\n\n" + text,
                "stream": False
            },
            timeout=120
        )
        output = r.json()["response"]
        return list(set(re.findall(r"[A-Z]\d{1,2}\.?\d*", output)))
    except:
        return []


# -------------------------
# UNIVERSAL RUNNER
# -------------------------
def run_model(name, text, hf_pipeline=None):
    config = LLM_CONFIG[name]
    t = config["type"]

    if t == "rule":
        return rule_model(text)

    elif t == "hf":
        return hf_model(text, hf_pipeline)

    elif t == "ollama":
        return call_ollama(config["model"], text)

    return []