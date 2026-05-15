# ==========================================================
# GROUNDed vs NON-GROUNDED OMOP COMPARISON DASHBOARD
# OHDSI Showcase - Core Comparison Table

#Lets build 1. 1. Grounded vs Non-Grounded Comparison Dashboard (MOST IMPORTANT) 
#| Diagnosis | LLM ICD | LLM OMOP | Grounded OMOP | Match? | Valid ICD? | | --------- | ------- | -------- | ------------- | ------ | ---------- |

# ==========================================================

import pandas as pd
import streamlit as st

# ----------------------------------------------------------
# EXPECTED INPUTS:
# ----------------------------------------------------------
# parsed_preds = LLM raw output
# Example:
# [
#   {"diagnosis":"Type 2 diabetes mellitus", "icd10":"E11.9", "omop_concept_id":201826},
#   {"diagnosis":"Essential hypertension", "icd10":"I10", "omop_concept_id":320128}
# ]
#
# grounded_results = Athena grounded output
# Example:
# [
#   {"icd":"E11.9", "omop_concept_id":201826},
#   {"icd":"I10", "omop_concept_id":320128}
# ]
# ----------------------------------------------------------

if "results" not in st.session_state or not st.session_state.results:
    st.warning("Run models first.")
    st.stop()
    
# -----------------------------
# BUILD GROUNDING LOOKUP
# -----------------------------

rows = []

for model, preds in st.session_state.results.items():
    rows.append({
        "Model": model
    })

df = pd.DataFrame(rows)
st.dataframe(df)