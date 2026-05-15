# ==========================================================
# GROUNDed vs NON-GROUNDED OMOP COMPARISON DASHBOARD
# OHDSI Showcase - Core Comparison Table
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


# -----------------------------
# SAFETY CHECKS
# -----------------------------
if not parsed_preds:
    st.warning("No LLM predictions available.")
    st.stop()

if not grounded_results:
    st.warning("No grounded OMOP results available.")
    st.stop()


# -----------------------------
# BUILD GROUNDING LOOKUP
# -----------------------------
grounded_lookup = {}

for g in grounded_results:
    grounded_lookup[g["icd"]] = g["omop_concept_id"]


# -----------------------------
# BUILD DASHBOARD TABLE
# -----------------------------
comparison_rows = []

for item in parsed_preds:

    diagnosis = item.get("diagnosis", "")
    llm_icd = item.get("icd10", "")
    llm_omop = item.get("omop_concept_id", None)

    grounded_omop = grounded_lookup.get(llm_icd, None)

    # ICD validity (assumes is_valid_icd exists)
    valid_icd = is_valid_icd(llm_icd, concept_df)

    # Match check
    match = llm_omop == grounded_omop

    comparison_rows.append({
        "Diagnosis": diagnosis,
        "LLM ICD": llm_icd,
        "LLM OMOP": llm_omop,
        "Grounded OMOP": grounded_omop,
        "Match?": "✔" if match else "❌",
        "Valid ICD?": "✔" if valid_icd else "❌"
    })


comparison_df = pd.DataFrame(comparison_rows)


# ==========================================================
# KPI METRICS
# ==========================================================
total_cases = len(comparison_df)

valid_icd_count = (comparison_df["Valid ICD?"] == "✔").sum()
match_count = (comparison_df["Match?"] == "✔").sum()

valid_icd_rate = round((valid_icd_count / total_cases) * 100, 2)
match_rate = round((match_count / total_cases) * 100, 2)
grounded_corrections = total_cases - match_count


# ==========================================================
# DASHBOARD UI
# ==========================================================
st.title("🔥 Grounded vs Non-Grounded OMOP Comparison Dashboard")

# KPI ROW
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Diagnoses", total_cases)

with col2:
    st.metric("Valid ICD Rate", f"{valid_icd_rate}%")

with col3:
    st.metric("LLM vs Grounded Match Rate", f"{match_rate}%")


# Secondary KPI
st.metric("Grounded Corrections Needed", grounded_corrections)


# ==========================================================
# MAIN TABLE
# ==========================================================
st.subheader("📊 Diagnosis-Level Comparison")

st.dataframe(
    comparison_df,
    use_container_width=True
)


# ==========================================================
# OPTIONAL FILTERS
# ==========================================================
st.subheader("🚨 Potential Hallucinations / Mismatches")

mismatch_df = comparison_df[
    (comparison_df["Match?"] == "❌") |
    (comparison_df["Valid ICD?"] == "❌")
]

if not mismatch_df.empty:
    st.dataframe(mismatch_df, use_container_width=True)
else:
    st.success("No mismatches or invalid ICDs detected.")


# ==========================================================
# DOWNLOAD
# ==========================================================
csv = comparison_df.to_csv(index=False)

st.download_button(
    label="📥 Download Comparison CSV",
    data=csv,
    file_name="grounded_vs_nongrounded_omop.csv",
    mime="text/csv"
)