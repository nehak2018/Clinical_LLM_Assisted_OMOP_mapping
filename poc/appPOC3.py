import streamlit as st
import pandas as pd
import time
from typing import List, Dict

st.set_page_config(page_title="OMOP Multi-LLM Benchmark", layout="wide")
st.title("🧠 OMOP Multi-LLM/ HF/ Rulebased Mapping Benchmark")

# =========================================================
# 🔧 CONFIG: Add/remove models here (universal layer)
# =========================================================
LLM_CONFIG = {
    "Rule-Based": {"type": "rule"},
    "HF (simulated)": {"type": "hf"},
    "GPT (OpenAI stub)": {"type": "openai"},
    "Claude (Anthropic stub)": {"type": "anthropic"},
    "Mistral (local stub)": {"type": "local"},
    "LLaMA (local stub)": {"type": "local"},
}

# Optional per-call cost estimates (dummy; replace with real if using APIs)
MODEL_COST = {
    "GPT (OpenAI stub)": 0.002,
    "Claude (Anthropic stub)": 0.0015,
    "Mistral (local stub)": 0.0,
    "LLaMA (local stub)": 0.0,
    "HF (simulated)": 0.0,
    "Rule-Based": 0.0,
}

# =========================================================
# 📦 SAMPLE DATA (replace with your dataset / MIMIC later)
# =========================================================
DATA = [
    {"note": "Patient has diabetes and hypertension", "gold_icd": ["E11.9", "I10"]},
    {"note": "No evidence of asthma", "gold_icd": []},
    {"note": "Chest pain rule out myocardial infarction", "gold_icd": ["I21.9"]},
    {"note": "Patient takes metformin for high blood sugar", "gold_icd": ["E11.9"]},
]

# ICD → OMOP (demo). Replace with Athena lookup later.
ICD_TO_OMOP = {
    "E11.9": 201826,   # Type 2 Diabetes
    "I10": 320128,     # Hypertension
    "I21.9": 4329847,  # Myocardial Infarction
}

# =========================================================
# 🧠 EXTRACTION METHODS
# =========================================================
def rule_based(text: str) -> List[str]:
    t = text.lower()
    out = []
    if "no evidence" in t or "denies" in t:
        return []  # simple negation
    if "diabetes" in t or "metformin" in t or "high blood sugar" in t:
        out.append("E11.9")
    if "hypertension" in t:
        out.append("I10")
    if "myocardial infarction" in t or "mi" in t or "chest pain" in t:
        out.append("I21.9")
    return list(set(out))


def hf_model(text: str) -> List[str]:
    # Placeholder for BioBERT/ClinicalBERT NER
    return rule_based(text)


# ======== LLM STUBS (plug real APIs here later) ===========
def call_openai(text: str) -> List[str]:
    # Replace with real OpenAI call
    return rule_based(text)


def call_claude(text: str) -> List[str]:
    # Replace with real Anthropic call
    return rule_based(text)


def call_local_model(text: str, model_name: str) -> List[str]:
    # Replace with local inference (e.g., Ollama)
    return rule_based(text)


# =========================================================
# 🔌 UNIVERSAL LLM ROUTER
# =========================================================
def run_llm(model_name: str, cfg: Dict, text: str) -> List[str]:
    t = cfg.get("type")

    if t == "rule":
        return rule_based(text)
    elif t == "hf":
        return hf_model(text)
    elif t == "openai":
        return call_openai(text)
    elif t == "anthropic":
        return call_claude(text)
    elif t == "local":
        return call_local_model(text, model_name)
    else:
        return []


def run_selected_models(text: str, selected_models: List[str]):
    outputs = {}
    times = {}
    costs = {}

    for name in selected_models:
        start = time.time()
        preds = run_llm(name, LLM_CONFIG[name], text)
        elapsed = round(time.time() - start, 4)

        outputs[name] = preds
        times[name] = elapsed
        costs[name] = MODEL_COST.get(name, 0.0)

    return outputs, times, costs


# =========================================================
# 🔄 MAPPING + METRICS
# =========================================================
def map_to_omop(icd_list: List[str]) -> List[int]:
    return [ICD_TO_OMOP[i] for i in icd_list if i in ICD_TO_OMOP]


def evaluate(pred: List[int], gold: List[int]):
    pred_set, gold_set = set(pred), set(gold)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


# =========================================================
# 🎛️ UI: INPUT
# =========================================================
st.sidebar.header("⚙️ Controls")

selected_models = st.sidebar.multiselect(
    "Select Models",
    list(LLM_CONFIG.keys()),
    default=list(LLM_CONFIG.keys()),
)

mode = st.sidebar.radio("Mode", ["Interactive (single note)", "Batch (all notes)"])

selected_note = st.sidebar.selectbox(
    "Choose sample note", [d["note"] for d in DATA]
)
custom_note = st.sidebar.text_area("Or enter your own note")

note = custom_note if custom_note else selected_note

# =========================================================
# ▶️ RUN
# =========================================================
if st.button("Run Benchmark"):

    # =======================
    # 🔹 INTERACTIVE MODE
    # =======================
    if mode == "Interactive (single note)":
        st.subheader("📄 Clinical Note")
        st.write(note)

        gold_icd = next(d["gold_icd"] for d in DATA if d["note"] == selected_note)
        gold_omop = map_to_omop(gold_icd)

        outputs, times, costs = run_selected_models(note, selected_models)

        st.subheader("🔍 Model Outputs")
        cols = st.columns(len(selected_models))

        results = []

        for i, model_name in enumerate(selected_models):
            icd_preds = outputs[model_name]
            omop_preds = map_to_omop(icd_preds)
            score = evaluate(omop_preds, gold_omop)

            with cols[i]:
                st.markdown(f"### {model_name}")
                st.write("ICD:", icd_preds)
                st.write("OMOP:", omop_preds)
                st.write("Precision, Recall, F1:", score)
                st.write("⏱ Time:", times[model_name], "sec")
                st.write("💰 Cost:", costs[model_name])

            results.append({
                "Model": model_name,
                "Precision": score[0],
                "Recall": score[1],
                "F1": score[2],
                "Time": times[model_name],
                "Cost": costs[model_name]
            })

        st.divider()
        st.subheader("📊 Comparison")
        df = pd.DataFrame(results)
        st.dataframe(df)
        st.bar_chart(df.set_index("Model")[["F1"]])

        # Error analysis
        st.subheader("⚠️ Error Analysis")
        for model_name in selected_models:
            if set(outputs[model_name]) != set(gold_icd):
                st.write(f"❌ {model_name} mismatch")
                st.write("Predicted:", outputs[model_name])
                st.write("Expected:", gold_icd)

    # =======================
    # 🔹 BATCH MODE
    # =======================
    else:
        st.subheader("📦 Batch Evaluation (All Notes)")

        summary = []

        for model_name in selected_models:
            all_prec, all_rec, all_f1 = [], [], []
            total_time, total_cost = 0.0, 0.0

            for row in DATA:
                note = row["note"]
                gold_icd = row["gold_icd"]
                gold_omop = map_to_omop(gold_icd)

                start = time.time()
                icd_preds = run_llm(model_name, LLM_CONFIG[model_name], note)
                elapsed = time.time() - start

                omop_preds = map_to_omop(icd_preds)
                p, r, f1 = evaluate(omop_preds, gold_omop)

                all_prec.append(p)
                all_rec.append(r)
                all_f1.append(f1)
                total_time += elapsed
                total_cost += MODEL_COST.get(model_name, 0.0)

            summary.append({
                "Model": model_name,
                "Precision": sum(all_prec)/len(all_prec),
                "Recall": sum(all_rec)/len(all_rec),
                "F1": sum(all_f1)/len(all_f1),
                "Avg Time": total_time/len(DATA),
                "Total Cost": total_cost
            })

        df = pd.DataFrame(summary)
        st.dataframe(df)
        st.bar_chart(df.set_index("Model")[["F1"]])