import streamlit as st
import pandas as pd
import time

# Hugging Face
from transformers import pipeline

st.set_page_config(page_title="OMOP Multi-LLM Benchmark", layout="wide")
st.title("🧠 OMOP Multi-LLM Benchmark (with Hugging Face)")

# =========================================================
# 🔧 CONFIG
# =========================================================
LLM_CONFIG = {
    "Rule-Based": {"type": "rule"},
    "HF (Bio NER)": {"type": "hf"},
    "GPT (stub)": {"type": "openai"},
    "Claude (stub)": {"type": "anthropic"},
    "Mistral (stub)": {"type": "local"},
}

MODEL_COST = {
    "GPT (stub)": 0.002,
    "Claude (stub)": 0.0015,
    "Mistral (stub)": 0.0,
    "HF (Bio NER)": 0.0,
    "Rule-Based": 0.0,
}

# =========================================================
# 📦 SAMPLE DATA
# =========================================================
DATA = [
    {"note": "Patient has diabetes and hypertension", "gold_icd": ["E11.9", "I10"]},
    {"note": "No evidence of asthma", "gold_icd": []},
    {"note": "Chest pain rule out myocardial infarction", "gold_icd": ["I21.9"]},
]

ICD_TO_OMOP = {
    "E11.9": 201826,
    "I10": 320128,
    "I21.9": 4329847,
}

TEXT_TO_ICD = {
    "diabetes": "E11.9",
    "hypertension": "I10",
    "myocardial infarction": "I21.9",
    "chest pain": "I21.9",
}

# =========================================================
# 🤗 LOAD HUGGING FACE MODEL (cached)
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
    if "diabetes" in t:
        out.append("E11.9")
    if "hypertension" in t:
        out.append("I10")
    if "chest pain" in t or "mi" in t:
        out.append("I21.9")

    return list(set(out))


def hf_model(text):
    entities = hf_pipeline(text)

    terms = [e["word"].lower() for e in entities]

    icds = []
    for term in terms:
        for key in TEXT_TO_ICD:
            if key in term:
                icds.append(TEXT_TO_ICD[key])

    return list(set(icds))


# ---- LLM STUBS ----
def call_openai(text):
    return rule_based(text)


def call_claude(text):
    return rule_based(text)


def call_local(text):
    return rule_based(text)


# =========================================================
# 🔌 UNIVERSAL RUNNER
# =========================================================
def run_model(name, text):
    t = LLM_CONFIG[name]["type"]

    if t == "rule":
        return rule_based(text)
    elif t == "hf":
        return hf_model(text)
    elif t == "openai":
        return call_openai(text)
    elif t == "anthropic":
        return call_claude(text)
    elif t == "local":
        return call_local(text)

    return []


def run_all_models(text, selected_models):
    outputs, times, costs = {}, {}, {}

    for name in selected_models:
        start = time.time()
        outputs[name] = run_model(name, text)
        times[name] = round(time.time() - start, 4)
        costs[name] = MODEL_COST.get(name, 0)

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
            score = evaluate(omop_preds, gold_omop)

            with cols[i]:
                st.markdown(f"### {model}")
                st.write("ICD:", icd_preds)
                st.write("OMOP:", omop_preds)
                st.write("Precision, Recall, F1:", score)
                st.write("⏱ Time:", times[model])
                st.write("💰 Cost:", costs[model])

            results.append({
                "Model": model,
                "F1": score[2],
                "Time": times[model]
            })

        df = pd.DataFrame(results)
        st.divider()
        st.subheader("📊 Comparison")
        st.dataframe(df)
        st.bar_chart(df.set_index("Model"))

    else:
        st.subheader("📦 Batch Evaluation")

        summary = []

        for model in selected_models:
            f1_scores = []
            total_time = 0

            for row in DATA:
                note = row["note"]
                gold_icd = row["gold_icd"]
                gold_omop = map_to_omop(gold_icd)

                start = time.time()
                preds = run_model(model, note)
                total_time += time.time() - start

                f1_scores.append(evaluate(map_to_omop(preds), gold_omop)[2])

            summary.append({
                "Model": model,
                "Avg F1": sum(f1_scores)/len(f1_scores),
                "Avg Time": total_time/len(DATA)
            })

        df = pd.DataFrame(summary)
        st.dataframe(df)
        st.bar_chart(df.set_index("Model"))