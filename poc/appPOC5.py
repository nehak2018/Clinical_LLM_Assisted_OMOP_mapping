import streamlit as st
import pandas as pd
import time
import re
import requests

# Hugging Face
from transformers import pipeline

st.set_page_config(page_title="OMOP Multi-LLM Benchmark", layout="wide")
st.title("🧠 OMOP Multi-LLM Benchmark (HF + Local LLMs via Ollama)")

# =========================================================
# 🔧 CONFIG
# =========================================================
LLM_CONFIG = {
    "Rule-Based": {"type": "rule"},
    "HF (Bio NER)": {"type": "hf"},
    "Llama 3.2": {"type": "ollama", "model": "llama3.2"},
    "Qwen3": {"type": "ollama", "model": "qwen"},
    "Phi-4-mini": {"type": "ollama", "model": "phi"},
}

MODEL_COST = {
    "Rule-Based": 0.0,
    "HF (Bio NER)": 0.0,
    "Llama 3.2": 0.0,
    "Qwen3": 0.0,
    "Phi-4-mini": 0.0,
}

PROMPT = """
Extract clinical diagnoses from the note.
Return ONLY ICD-10 codes as a Python list.
Example: ["E11.9", "I10"]
"""

# =========================================================
# 📦 SAMPLE DATA (replace later with your dataset)
# =========================================================
DATA = [
    {"note": "Patient has diabetes and hypertension", "gold_icd": ["E11.9", "I10"]},
    {"note": "No evidence of asthma", "gold_icd": []},
    {"note": "Chest pain rule out myocardial infarction", "gold_icd": ["I21.9"]},
    {"note": "Patient takes metformin for high blood sugar", "gold_icd": ["E11.9"]},
]

# Demo mapping (replace with real OMOP/Athena later)
ICD_TO_OMOP = {
    "E11.9": 201826,
    "I10": 320128,
    "I21.9": 4329847,
}

# HF text → ICD bridge (simple; extend as needed)
TEXT_TO_ICD = {
    "diabetes": "E11.9",
    "hypertension": "I10",
    "myocardial infarction": "I21.9",
    "chest pain": "I21.9",
}

# =========================================================
# 🤗 LOAD HF MODEL (cached)
# =========================================================
@st.cache_resource
def load_hf_model():
    return pipeline(
        "ner",
        model="d4data/biomedical-ner-all",
        aggregation_strategy="simple"
    )

hf_pipeline = load_hf_model()

# =========================================================
# 🧠 EXTRACTION METHODS
# =========================================================
def rule_based(text):
    t = text.lower()
    if "no evidence" in t or "denies" in t:
        return []
    out = []
    if "diabetes" in t or "metformin" in t or "high blood sugar" in t:
        out.append("E11.9")
    if "hypertension" in t:
        out.append("I10")
    if "myocardial infarction" in t or "mi" in t or "chest pain" in t:
        out.append("I21.9")
    return list(set(out))


def hf_model(text):
    entities = hf_pipeline(text)
    terms = [e.get("word", "").lower() for e in entities]

    icds = []
    for term in terms:
        for key, code in TEXT_TO_ICD.items():
            if key in term:
                icds.append(code)

    return list(set(icds))


# =========================================================
# 🤖 LOCAL LLM via OLLAMA
# =========================================================
def call_ollama(model_name, text):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": PROMPT + "\n\n" + text,
                "stream": False
            },
            timeout=120
        )
        r.raise_for_status()
        output = r.json().get("response", "")

        # Extract ICD-10 patterns like E11.9, I10, I21.9
        codes = re.findall(r"[A-Z]\d{1,2}(?:\.\d+)?", output)
        return list(set(codes))

    except Exception:
        return []


# =========================================================
# 🔌 UNIVERSAL RUNNER
# =========================================================
def run_model(name, text):
    cfg = LLM_CONFIG[name]
    t = cfg["type"]

    if t == "rule":
        return rule_based(text)
    elif t == "hf":
        return hf_model(text)
    elif t == "ollama":
        return call_ollama(cfg["model"], text)

    return []


def run_all_models(text, selected_models):
    outputs, times, costs = {}, {}, {}
    for name in selected_models:
        start = time.time()
        outputs[name] = run_model(name, text)
        times[name] = round(time.time() - start, 4)
        costs[name] = MODEL_COST.get(name, 0.0)
    return outputs, times, costs


# =========================================================
# 🔄 MAPPING + METRICS
# =========================================================
def map_to_omop(icds):
    return [ICD_TO_OMOP[i] for i in icds if i in ICD_TO_OMOP]


def evaluate(pred, gold):
    pred, gold = set(pred), set(gold)
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


# =========================================================
# 🎛️ UI
# =========================================================
st.sidebar.header("Controls")

selected_models = st.sidebar.multiselect(
    "Select Models",
    list(LLM_CONFIG.keys()),
    default=list(LLM_CONFIG.keys())
)

mode = st.sidebar.radio("Mode", ["Interactive", "Batch"])

selected_note = st.sidebar.selectbox(
    "Sample Note",
    [d["note"] for d in DATA]
)

custom_note = st.sidebar.text_area("Or enter custom note")

note = custom_note if custom_note else selected_note

# =========================================================
# ▶️ RUN
# =========================================================
if st.button("Run Benchmark"):

    if mode == "Interactive":
        st.subheader("📄 Clinical Note")
        st.write(note)

        gold_icd = next(d["gold_icd"] for d in DATA if d["note"] == selected_note)
        gold_omop = map_to_omop(gold_icd)

        outputs, times, costs = run_all_models(note, selected_models)

        cols = st.columns(len(selected_models))
        results = []

        for i, model in enumerate(selected_models):
            icd_preds = outputs[model]
            omop_preds = map_to_omop(icd_preds)
            p, r, f1 = evaluate(omop_preds, gold_omop)

            with cols[i]:
                st.markdown(f"### {model}")
                st.write("ICD:", icd_preds)
                st.write("OMOP:", omop_preds)
                st.write("Precision, Recall, F1:", (p, r, f1))
                st.write("⏱ Time (s):", times[model])
                st.write("💰 Cost:", costs[model])

            results.append({
                "Model": model,
                "Precision": p,
                "Recall": r,
                "F1": f1,
                "Time": times[model],
                "Cost": costs[model],
            })

        st.divider()
        st.subheader("📊 Comparison")
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.bar_chart(df.set_index("Model")[["F1"]])

        st.subheader("⚠️ Error Analysis")
        for model in selected_models:
            if set(outputs[model]) != set(gold_icd):
                st.write(f"❌ {model} mismatch")
                st.write("Predicted:", outputs[model])
                st.write("Expected:", gold_icd)

    else:
        st.subheader("📦 Batch Evaluation (All Notes)")

        summary = []

        for model in selected_models:
            all_p, all_r, all_f1 = [], [], []
            total_time, total_cost = 0.0, 0.0

            for row in DATA:
                text = row["note"]
                gold_icd = row["gold_icd"]
                gold_omop = map_to_omop(gold_icd)

                start = time.time()
                icd_preds = run_model(model, text)
                elapsed = time.time() - start

                omop_preds = map_to_omop(icd_preds)
                p, r, f1 = evaluate(omop_preds, gold_omop)

                all_p.append(p)
                all_r.append(r)
                all_f1.append(f1)
                total_time += elapsed
                total_cost += MODEL_COST.get(model, 0.0)

            summary.append({
                "Model": model,
                "Precision": sum(all_p)/len(all_p),
                "Recall": sum(all_r)/len(all_r),
                "F1": sum(all_f1)/len(all_f1),
                "Avg Time (s)": total_time/len(DATA),
                "Total Cost": total_cost
            })

        df = pd.DataFrame(summary)
        st.dataframe(df)
        st.bar_chart(df.set_index("Model")[["F1"]])