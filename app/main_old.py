import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
#from data.sample import DATA
#from config.settings import LLM_CONFIG
from src.utils.pipeline import run_pipeline
from src.utils.hf_loader import load_model
from data import sample
from config import settings
from src import utils

st.set_page_config(layout="wide")
st.title("🧠 OMOP Multi-LLM Benchmark")
DATA = sample.DATA
LLM_CONFIG = settings.LLM_CONFIG

# load HF once
hf_pipeline = load_model()

models = st.sidebar.multiselect(
    "Select Models",
    list(LLM_CONFIG.keys()),
    default=list(LLM_CONFIG.keys())
)
note = st.selectbox("Choose Note", [d["note"] for d in DATA])

selected_note = st.sidebar.selectbox(
    "Choose Sample Note",
    [d["note"] for d in DATA]
)


# Add textbox (THIS IS WHAT YOU ARE MISSING)
custom_note = st.sidebar.text_area(
    "Or paste your clinical note here",
    height=150
)

# Use custom note if provided
note = custom_note if custom_note.strip() != "" else selected_note

if note in [d["note"] for d in DATA]:
    gold = next(d["gold_icd"] for d in DATA if d["note"] == note)
else:
    gold = []

if st.button("Run"):

    gold = next(d["gold_icd"] for d in DATA if d["note"] == note)

    results = run_pipeline(note, gold, models, hf_pipeline)

    cols = st.columns(len(models))
    summary = []

    for i, m in enumerate(models):
        preds, score = results[m]

        with cols[i]:
            st.subheader(m)
            st.write("ICD:", preds)
            st.write("Score:", score)

        summary.append({
            "Model": m,
            "F1": score[2]
        })

    df = pd.DataFrame(summary)
    st.bar_chart(df.set_index("Model"))