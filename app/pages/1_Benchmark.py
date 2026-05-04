import streamlit as st
import pandas as pd

from src.utils.mapping import map_to_omop
from src.utils.metrics import evaluate

st.title("📈 Benchmark")

if "results" not in st.session_state or not st.session_state.results:
    st.warning("Run models first.")
    st.stop()

gold_icd = ["E11.9", "I10"]
gold_omop = map_to_omop(gold_icd)

rows = []

for model, preds in st.session_state.results.items():
    omop_preds = map_to_omop(preds[0])
    p, r, f1 = evaluate(omop_preds, gold_omop)

    rows.append({
        "Model": model,
        "Precision": p,
        "Recall": r,
        "F1": f1
    })

df = pd.DataFrame(rows)

st.dataframe(df)
st.bar_chart(df.set_index("Model")[["F1"]])