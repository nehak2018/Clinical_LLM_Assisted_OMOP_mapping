MODEL_REGISTRY = {
    # -------- HF ----------
    "RoBERTa (HF)": {"type": "hf"},

    # -------- LOCAL ----------
    "Llama 3 (8B)": {"type": "ollama", "model": "llama3"},
    "Mistral 7B": {"type": "ollama", "model": "mistral"},

    # -------- API (stub for now) ----------
    "GPT-4": {"type": "api"},
    "Claude 3 Opus": {"type": "api"},
    "Gemini Advanced": {"type": "api"},

    # -------- REFERENCE ONLY ----------
    "Claude 2.0": {"type": "reference", "score": 0.998},
    "PaLM 2": {"type": "reference", "score": 0.996},
}


LLM_CONFIG = {
    "Rule-Based": {"type": "rule"},
    "HF": {"type": "hf"},
    "Llama 3.2": {"type": "ollama", "model": "llama3.2"},
    "Qwen3": {"type": "ollama", "model": "qwen"},
    "Phi-4-mini": {"type": "ollama", "model": "phi"},
}

PROMPT = """
Extract clinical diagnoses from the note.
Return ONLY ICD-10 codes as a Python list.
Example: ["E11.9", "I10"]
"""






'''
LLM_CONFIG = {
    "Rule-Based": {"type": "rule"},
    "HF (Bio NER)": {"type": "hf"},
    "Llama 3.2": {"type": "ollama", "model": "llama3.2"},
    "Qwen3": {"type": "ollama", "model": "qwen"},
    "Phi-4-mini": {"type": "ollama", "model": "phi"},
}

PROMPT = """
Extract clinical diagnoses from the note.
Return ONLY ICD-10 codes as a Python list.
Example: ["E11.9", "I10"]
"""
'''