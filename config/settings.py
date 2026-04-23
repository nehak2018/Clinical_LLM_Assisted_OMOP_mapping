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