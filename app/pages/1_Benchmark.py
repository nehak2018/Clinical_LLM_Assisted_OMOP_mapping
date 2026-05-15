import streamlit as st
import pandas as pd
from src.utils.mapping import map_to_omop
from src.utils.metrics import evaluate
from src.llms.engine import parse_llm_output
import json
from db.get_results import load_table

st.set_page_config(page_title="Benchmark", layout="wide")
st.title("📈 Benchmark")
rows = []

# -----------------------------
# DATAFRAME
# -----------------------------
#df = pd.DataFrame(rows)

df = load_table()

# -----------------------------
# DISPLAY
# -----------------------------
st.subheader("📊 OMOP Grounded vs Non-Grounded Comparison")
st.dataframe(df, use_container_width=True)
