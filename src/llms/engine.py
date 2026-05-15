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

def call_ollama_grounded(model, PROMPT_for_grounded):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": PROMPT_for_grounded,
                "stream": False
            },
            timeout=120
        )
        # IMPORTANT: show real error if request fails
        response.raise_for_status()

        data = response.json()
        output = data.get("response")

        if output is None:
            return f"ERROR: No response field. Full response: {data}"

        return output.strip()
        #return response.json().get("response", "").strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

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


# def run_llama_extract_condition(model, note):
#     PROMPT_for_grounded = f"""
#     Extract the primary diagnosis from this clinical note:
#     {note}
#     Return only the condition phrase.
#     """
#     return call_ollama_grounded(model, PROMPT_for_grounded)

def run_llama_extract_condition(model, note):
    PROMPT_for_grounded = f"""
    You are an information extraction system.

    TASK:
    Extract ONLY the medical condition or primary diagnosis mentioned in the clinical note.

    RULES:
    - Output MUST be a single medical concept or short phrase
    - DO NOT include explanations
    - DO NOT include sentences
    - DO NOT include phrases like "possible", "likely", "could be", "based on context"
    - DO NOT add punctuation or quotes
    - If multiple conditions exist, return ONLY the primary one
    - If no clear diagnosis exists, return: UNKNOWN

    CLINICAL NOTE:
    {note}

    OUTPUT:
    """
    return call_ollama_grounded(model, PROMPT_for_grounded)



def run_llama_select_icd(model, note, candidates):
    PROMPT_for_grounded_icd = f"""
    You are a clinical coding assistant specialized in ICD-10-CM coding.

    Clinical Note:
    {note}

    Candidate ICD codes:
    {candidates}

    Task:
    Select EXACTLY ONE ICD-10-CM candidate FROM THE PROVIDED LIST ONLY.

    Rules:
    - Prefer most specific clinically supported and primary code
    - You MUST choose ONLY from the candidates list above
    - You MUST select an ICD-10-CM code (format: letter + digits, e.g. E11.9, I10, J18.9)
    - Do NOT select SNOMED codes (SNOMED codes are long numeric strings like 1030411000000101)
    - Copy icd, concept_id, concept_name, standard_concept_id, standard_concept_name EXACTLY from the candidate 
    - Do NOT invent or modify any field
    - Do NOT use external knowledge

    For the reason field:
    - Explain in 1 sentence WHY this ICD code best matches the clinical note
    - Reference specific clinical terms from the note (e.g. "Note mentions type 2 diabetes without complications")

    For the brief_justification field:
    - Explain in 1 sentence WHY you rejected the other candidates
    - Reference what made them less appropriate (e.g. "E11.21 rejected as note has no mention of diabetic kidney disease")


    Return ONLY valid JSON, no extra text:

    {{
    "icd": "",
    "concept_id": "",
    "concept_name":"",
    "standard_concept_id":"",
    "standard_concept_name":"",
    "reason": ""
    "brief_justification" :""
    }}
    """
    return call_ollama_grounded(model, PROMPT_for_grounded_icd)


# def run_llama_select_icd(model, note, candidates):
#     PROMPT_for_grounded_icd = f"""
#     You are a clinical coding assistant specialized in ICD-10-CM coding.

#     Clinical Note:
#     {note}

#     Candidate ICD codes:
#     {candidates}

#     Task:
#     Select EXACTLY ONE candidate FROM THE PROVIDED LIST ONLY.
#     Select EXACTLY ONE ICD-10-CM candidate FROM THE PROVIDED LIST ONLY.

#     Rules:
#     - You MUST choose ONLY from candidates
#     - Prefer most specific clinically supported code
#     - Copy ICD code EXACTLY from candidate
#     - Copy concept_id EXACTLY from candidate
#     - Copy concept_name EXACTLY from candidate
#     - Copy standard_concept_id EXACTLY from candidate
#     - Copy standard_concept_name EXACTLY from candidate
#     - Do NOT invent or modify concept_id
#     - Do NOT use external knowledge


#     Return ONLY valid JSON:

#     {{
#     "icd": "",
#     "concept_id": "",
#     "concept_name":"",
#     "standard_concept_id":"",
#     "standard_concept_name":"",
#     "reason": ""
#     "brief justification" :""
#     }}
#     """
#     return call_ollama_grounded(model, PROMPT_for_grounded_icd)



# def run_llama_select_icd(model, note, candidates):
#     PROMPT_for_grounded_icd = f"""
#     You are a clinical coding assistant.
#     Clinical Note:
#     {note}
#     Candidate ICD codes:
#     {candidates}
#     Select ONLY ONE ICD code from the list.
#     Return:
#     ICD code + concept_id + brief justification
#     """
#     return call_ollama_grounded(model, PROMPT_for_grounded_icd)
