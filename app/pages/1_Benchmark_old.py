import streamlit as st
import pandas as pd
from src.utils.mapping import map_to_omop
from src.utils.metrics import evaluate
from src.llms.engine import parse_llm_output
import json

st.title("📈 Benchmark")
if "results" not in st.session_state or not st.session_state.results:
    st.warning("Run models first.")
    st.stop()

rows = []

for model, data in st.session_state.results.items():

    # -----------------------------
    # SAFE PARSING
    # -----------------------------
    raw = data["raw_preds"]

    if isinstance(raw, str):
        raw = json.loads(raw)

    icd_valid_map = {
        x["icd10"]: x["icd_valid"]
        for x in data["icd_validation"]
    }

    grounded_map = {
        x["icd"]: x["omop_concept_id"]
        for x in data["grounded_results"]
    }

    # -----------------------------
    # BUILD ROWS
    # -----------------------------
    for item in raw:

        icd = item["icd10"]

        llm_omop = item.get("omop_concept_id")
        grounded_omop = grounded_map.get(icd)

        rows.append({
            #"Note":
            "Model": model,
            "Diagnosis": item["diagnosis"],
            "LLM ICD": icd,
            "LLM OMOP": llm_omop,
            "Grounded OMOP": grounded_omop,
            "Correct?": "✔" if llm_omop == grounded_omop else "❌",
            "ICD Valid?": "✔" if icd_valid_map.get(icd) else "❌"
        })

# -----------------------------
# DATAFRAME
# -----------------------------
df = pd.DataFrame(rows)

# -----------------------------
# DISPLAY
# -----------------------------
st.subheader("📊 OMOP Grounded vs Non-Grounded Comparison")

st.dataframe(df, use_container_width=True)
