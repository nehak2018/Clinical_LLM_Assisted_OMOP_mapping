import streamlit as st
import pandas as pd
import time

st.set_page_config(page_title="OMOP Multi-LLM Benchmark", layout="wide")

st.title("🧠 Multi-LLM OMOP Mapping Benchmark")

# -----------------------------
# SAMPLE DATA
# -----------------------------
data = [
    {"note": "Patient has diabetes and hypertension", "gold_icd": ["E11.9", "I10"]},
    {"note": "No evidence of asthma", "gold_icd": []},
    {"note": "Chest pain rule out myocardial infarction", "gold_icd": ["I21.9"]},
]

icd_to_omop = {
    "E11.9": 201826,
    "I10": 320128,
    "I21.9": 4329847
}

# -----------------------------
# BASE MODEL CLASS
# -----------------------------
class BaseModel:
    def extract(self, text):
        raise NotImplementedError


# -----------------------------
# MODELS
# -----------------------------
class RuleModel(BaseModel):
    def extract(self, text):
        text = text.lower()
        out = []
        if "diabetes" in text:
            out.append("E11.9")
        if "hypertension" in text:
            out.append("I10")
        if "myocardial infarction" in text or "chest pain" in text:
            out.append("I21.9")
        return out


class HFModel(BaseModel):
    def extract(self, text):
        # simulate HF model
        return RuleModel().extract(text)


class GPTModel(BaseModel):
    def extract(self, text):
        # simulate GPT (better negation handling)
        text = text.lower()
        if "no evidence" in text or "denies" in text:
            return []
        return RuleModel().extract(text)


class MistralModel(BaseModel):
    def extract(self, text):
        # simulate another LLM
        return GPTModel().extract(text)


# -----------------------------
# MODEL ROUTER
# -----------------------------
models = {
    "Rule-Based": RuleModel(),
    "HF Model": HFModel(),
    "GPT (LLM)": GPTModel(),
    "Mistral (LLM)": MistralModel()
}


def run_all_models(text):
    outputs = {}
    times = {}

    for name, model in models.items():
        start = time.time()
        outputs[name] = model.extract(text)
        times[name] = round(time.time() - start, 4)

    return outputs, times


# -----------------------------
# MAPPING + EVALUATION
# -----------------------------
def map_to_omop(icds):
    return [icd_to_omop[i] for i in icds if i in icd_to_omop]


def evaluate(pred, gold):
    pred = set(pred)
    gold = set(gold)

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


# -----------------------------
# UI INPUT
# -----------------------------
st.sidebar.header("Input")

selected_note = st.sidebar.selectbox(
    "Choose sample note", [d["note"] for d in data]
)

custom_note = st.sidebar.text_area("Or enter your own note")

note = custom_note if custom_note else selected_note

st.subheader("📄 Clinical Note")
st.write(note)

# -----------------------------
# RUN PIPELINE
# -----------------------------
if st.button("Run Benchmark"):

    gold_icd = next(d["gold_icd"] for d in data if d["note"] == selected_note)
    gold_omop = map_to_omop(gold_icd)

    outputs, times = run_all_models(note)

    results = []

    st.subheader("🔍 Model Outputs")

    cols = st.columns(len(models))

    for i, (model_name, icd_preds) in enumerate(outputs.items()):
        omop_preds = map_to_omop(icd_preds)
        score = evaluate(omop_preds, gold_omop)

        with cols[i]:
            st.markdown(f"### {model_name}")
            st.write("ICD:", icd_preds)
            st.write("OMOP:", omop_preds)
            st.write("Precision, Recall, F1:", score)
            st.write("⏱ Time:", times[model_name], "sec")

        results.append({
            "Model": model_name,
            "Precision": score[0],
            "Recall": score[1],
            "F1": score[2],
            "Time": times[model_name]
        })

    # -----------------------------
    # COMPARISON TABLE
    # -----------------------------
    st.divider()
    st.subheader("📊 Comparison")

    df = pd.DataFrame(results)

    st.dataframe(df)

    st.bar_chart(df.set_index("Model")[["F1"]])

    # -----------------------------
    # ERROR ANALYSIS
    # -----------------------------
    st.subheader("⚠️ Error Analysis")

    for model_name, icd_preds in outputs.items():
        if set(icd_preds) != set(gold_icd):
            st.write(f"❌ {model_name} mismatch")
            st.write("Predicted:", icd_preds)
            st.write("Expected:", gold_icd)