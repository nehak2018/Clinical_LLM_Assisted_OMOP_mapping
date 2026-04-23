import streamlit as st
import pandas as pd

st.set_page_config(page_title="OMOP Mapping Demo", layout="wide")

st.title("🧠 LLM vs NLP vs HF for OMOP Mapping")

# -----------------------------
# SAMPLE DATA
# -----------------------------
data = [
    {
        "note": "Patient has diabetes and hypertension",
        "gold_icd": ["E11.9", "I10"]
    },
    {
        "note": "No evidence of asthma",
        "gold_icd": []
    },
    {
        "note": "Chest pain rule out myocardial infarction",
        "gold_icd": ["I21.9"]
    }
]

icd_to_omop = {
    "E11.9": 201826,
    "I10": 320128,
    "I21.9": 4329847
}

# -----------------------------
# MODELS
# -----------------------------
def rule_based(note):
    note = note.lower()
    out = []
    if "diabetes" in note:
        out.append("E11.9")
    if "hypertension" in note:
        out.append("I10")
    if "myocardial infarction" in note or "chest pain" in note:
        out.append("I21.9")
    return out


def llm_model(note):
    note = note.lower()
    if "no evidence" in note or "denies" in note:
        return []
    return rule_based(note)


def hf_model(note):
    return rule_based(note)


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
# PROCESS BUTTON
# -----------------------------
if st.button("Run Mapping"):

    # Ground truth
    gold_icd = next(d["gold_icd"] for d in data if d["note"] == selected_note)
    gold_omop = map_to_omop(gold_icd)

    # Predictions
    rule_icd = rule_based(note)
    llm_icd = llm_model(note)
    hf_icd = hf_model(note)

    rule_omop = map_to_omop(rule_icd)
    llm_omop = map_to_omop(llm_icd)
    hf_omop = map_to_omop(hf_icd)

    # Evaluation
    rule_score = evaluate(rule_omop, gold_omop)
    llm_score = evaluate(llm_omop, gold_omop)
    hf_score = evaluate(hf_omop, gold_omop)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Rule-Based")
        st.write("ICD:", rule_icd)
        st.write("OMOP:", rule_omop)
        st.write("Precision, Recall, F1:", rule_score)

    with col2:
        st.subheader("HF Model")
        st.write("ICD:", hf_icd)
        st.write("OMOP:", hf_omop)
        st.write("Precision, Recall, F1:", hf_score)

    with col3:
        st.subheader("LLM")
        st.write("ICD:", llm_icd)
        st.write("OMOP:", llm_omop)
        st.write("Precision, Recall, F1:", llm_score)

    st.divider()

    st.subheader("📊 Comparison")

    df = pd.DataFrame({
        "Model": ["Rule", "HF", "LLM"],
        "Precision": [rule_score[0], hf_score[0], llm_score[0]],
        "Recall": [rule_score[1], hf_score[1], llm_score[1]],
        "F1": [rule_score[2], hf_score[2], llm_score[2]]
    })

    st.dataframe(df)

    st.bar_chart(df.set_index("Model"))