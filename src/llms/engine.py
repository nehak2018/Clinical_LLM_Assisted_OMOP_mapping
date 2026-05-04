import re
import requests
from config.settings import PROMPT



# =========================
# PROMPT
# =========================
PROMPT_for_non_grounded = """
You are an expert clinical coding and OMOP mapping assistant.

Task:
Extract ALL clinically relevant diagnoses from the clinical note and return:
1. Diagnosis
2. ICD-10 code
3. OMOP Concept ID (best match)

Instructions:
- Include ALL diagnoses explicitly stated or strongly implied
- Include chronic diseases, current diagnoses, and past medical history
- Use MOST SPECIFIC ICD-10 code possible
- Include multiple diagnoses when present
- Infer likely diagnoses when clinically supported
- Return ONLY a Python-style list of dictionaries
- No explanation
- No markdown
- No commentary

Output Example:
[
 {"diagnosis":"Type 2 diabetes mellitus","icd10":"E11.9","omop_concept_id":201826},
 {"diagnosis":"Essential hypertension","icd10":"I10","omop_concept_id":320128}
]

Clinical Note:
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



def call_ollama(model, note):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": PROMPT_for_non_grounded + "\n\n" + note,
                "stream": False
            },
            timeout=120
        )

        output = response.json().get("response", "")

        # Extract ICD codes
        #codes = re.findall(r"[A-Z]\d{1,2}(?:\.\d+)?", output)
        #codes = sorted(list(set(codes)))
        #return list(set(codes)), output
        return output
    except Exception as e:
        return [], str(e)


def parse_llm_output(raw_output):
    """
    Parse LLM output into structured OMOP records.

    Handles:
    - raw string JSON
    - JSON embedded inside text
    - tuple outputs (codes, text)
    - minor formatting issues
    """

    # -----------------------------
    # 1. Handle tuple input safely
    # -----------------------------
    if isinstance(raw_output, tuple):
        raw_output = raw_output[1]  # take LLM text only

    if raw_output is None:
        return []

    raw_output = str(raw_output)

    try:
        # -----------------------------------------
        # 2. Extract JSON array from messy output
        # -----------------------------------------
        start = raw_output.find("[")
        end = raw_output.rfind("]") + 1

        if start == -1 or end == 0:
            raise ValueError("No JSON array found")

        json_str = raw_output[start:end]

        # Fix common LLM formatting issues
        json_str = json_str.replace("'", '"')

        data = json.loads(json_str)

        # -----------------------------------------
        # 3. Validate structure
        # -----------------------------------------
        if isinstance(data, dict):
            data = [data]

        return data

    except Exception:
        # -----------------------------------------
        # 4. Fallback regex extraction
        # -----------------------------------------
        diagnoses = re.findall(r'"diagnosis"\s*:\s*"([^"]+)"', raw_output)
        icds = re.findall(r'"icd10"\s*:\s*"([A-Z]\d{1,2}(?:\.\d+)?)"', raw_output)
        omops = re.findall(r'"omop_concept_id"\s*:\s*(\d+)', raw_output)

        results = []

        for i in range(min(len(diagnoses), len(icds), len(omops))):
            results.append({
                "diagnosis": diagnoses[i],
                "icd10": icds[i],
                "omop_concept_id": int(omops[i])
            })

        return results


'''

# =========================
# PARSER
# =========================
def parse_llm_output(raw_output):
    """
    Attempts to extract list of dictionaries from LLM output.
    Falls back to regex if formatting is messy.
    """
    try:
        # Extract probable list block
        start = raw_output.find("[")
        end = raw_output.rfind("]") + 1

        parsed_text = raw_output[start:end]

        # Convert single quotes to double quotes if needed
        parsed_text = parsed_text.replace("'", '"')

        data = json.loads(parsed_text)

        return data

    except:
        # Fallback regex extraction
        diagnoses = re.findall(r'"diagnosis":\s*"([^"]+)"', raw_output)
        icds = re.findall(r'"icd10":\s*"([A-Z]\d{2}(?:\.\d+)?)"', raw_output)
        omops = re.findall(r'"omop_concept_id":\s*(\d+)', raw_output)

        results = []

        for i in range(min(len(diagnoses), len(icds), len(omops))):
            results.append({
                "diagnosis": diagnoses[i],
                "icd10": icds[i],
                "omop_concept_id": int(omops[i])
            })

        return results


Backup
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
'''
