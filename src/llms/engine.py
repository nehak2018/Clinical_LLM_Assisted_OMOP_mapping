import re
import requests
from config.settings import PROMPT

PROMPT_old = """
Extract clinical diagnoses from the note.
Return ONLY ICD-10 codes as a Python list.
Example: ["E11.9", "I10"]
"""

PROMPT = """
You are a clinical coding assistant.

Task:
Extract ALL clinical diagnoses and ICD-10 codes from notes.

STRICT RULES:
- Include ALL conditions (not just one)
- Return ICD-10 codes
- Use most specific codes (e.g., E11.9, I10)
- Only use diagnoses clearly mentioned

Return format:
["I10", "I21.9"]

If uncertain, return empty list.
"""

PROMPT_getiing_but_hallucinating = """
You are a clinical coding assistant.

Extract ALL clinical diagnoses and ICD-10 codes from notes.

Rules:
- Include ALL conditions (not just one)
- Return ICD-10 codes
- Use most specific codes (e.g., E11.9, I10)
- No explanation

Output:
["E11.9", "I10"]
"""

PROMPT_Giving_only_one_condition ="""
You are a clinical coding assistant.

Extract ICD-10 codes from the note.


STRICT RULES:
- Return ONLY valid ICD-10 codes
- Codes MUST include decimals where applicable (e.g., E11.9, not E11)
- Do NOT return category codes like E11 or I10 without specificity
- No explanation

Output format:
["E11.9", "I10"]
"""





TEXT_TO_ICD = {
    "diabetes": "E11.9",
    "hypertension": "I10",
    "myocardial infarction": "I21.9",
    "chest pain": "I21.9",
}

def rule_based(text):
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


def hf_model(text, hf_pipeline):
    entities = hf_pipeline(text)
    terms = [e["word"].lower() for e in entities]

    icds = []
    for term in terms:
        for k, v in TEXT_TO_ICD.items():
            if k in term:
                icds.append(v)

    return list(set(icds))

""" 
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
        return [] """


def call_ollama(model, note):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": PROMPT + "\n\n" + note,
                "stream": False
            },
            timeout=120
        )

        output = response.json().get("response", "")

        # Extract ICD codes
        codes = re.findall(r"[A-Z]\d{1,2}(?:\.\d+)?", output)

        codes = sorted(list(set(codes)))

        return list(set(codes)), output

    except Exception as e:
        return [], str(e)