import sys
import os
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.llms.engine import rule_based, hf_model, call_ollama, parse_llm_output, run_llama_extract_condition, run_llama_select_icd
from src.utils.hf_loader import load_model
from src.utils.mapping import map_to_omop
from src.utils.athena_lookup import lookup_icd_to_omop
from src.utils.athena_validator import build_icd_index, is_valid_icd
from db.save_results import save_result
from src.utils.athena_retriever import AthenaRetriever

@st.cache_resource
def load_retriever():
    return AthenaRetriever(
        concept_path="../data/athena/CONCEPT.csv",
        concept_relationship_path="../data/athena/CONCEPT_RELATIONSHIP.csv",
        use_embeddings=False
    )

retriever = load_retriever()

# Load ONCE
concept_df = pd.read_csv(
    "../data/athena/CONCEPT.csv",
    sep="\t",
    usecols=["concept_code", "vocabulary_id", "invalid_reason"],
    low_memory=False
)
icd_set = build_icd_index(concept_df)

st.set_page_config(layout="wide")
st.title("🧠 OMOP Multi-LLM Playground")

# Session state
if "results" not in st.session_state:
    st.session_state.results = {}

# Load HF once
hf_pipeline = load_model()

# =========================
# INPUT
# =========================
note = st.text_area("📄 Paste Clinical Note", height=150)

def run_all(note, hf_pipeline):
    models = [
        "Rule-Based",
        "HF",
        "Llama 3.2",
        "Qwen3",
        "Phi-4-mini"
    ]

    results = {}

    for m in models:
        results[m] = run_model(m, note, hf_pipeline)

    return results

def run_model(name, text):
    
    if name == "Rule-Based":
        return rule_based(text)

    elif name == "HF":
        return hf_model(text)

    elif name == "Llama 3.2":
        return call_ollama("llama3.2", text)

    elif name == "Qwen3":
        return call_ollama("qwen", text)

    elif name == "Phi-4-mini":
        return call_ollama("phi", text)

    # API placeholders
    elif name == "GPT-4":
        st.warning("GPT-4 not configured (API required)")
        return []

    elif name == "Claude 3":
        st.warning("Claude not configured (API required)")
        return []

    # Reference models
    elif name == "Claude 2.0":
        return ["(reference benchmark only)"]

    return []

# =========================
# RUN ALL WITH PROGRESS
# =========================
def run_all_models(note, progress_bar, status_text):
    models = {
        "Rule-Based": lambda: rule_based(note),
        "HF": lambda: hf_model(note, hf_pipeline),
        "Llama 3.2": lambda: call_ollama("llama3.2", note),
        "Qwen3": lambda: call_ollama("qwen", note),
        "Phi-4-mini": lambda: call_ollama("phi", note),
    }

    results = {}
    total = len(models)
    done = 0

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fn): name for name, fn in models.items()}

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except:
                result = []

            results[name] = result
            st.session_state.results[name] = result

            done += 1
            progress_bar.progress(done / total)
            status_text.text(f"Completed: {name} ({done}/{total})")

    return results


# =========================
# BUTTONS
# =========================
st.subheader("🚀 Run Models")

if st.button("⚡ Run All Models"):
    progress = st.progress(0)
    status = st.empty()
    run_all_models(note, progress, status)
    progress.empty()
    status.empty()
    st.success("All models completed!")

col3, col4, col5, col6, col7, col8 = st.columns(6)
with col3:
    if st.button("Run Llama 3.2"):
        st.session_state.results["Llama 3.2"] = run_model("Llama 3.2", note)

with col4:
    if st.button("Run Qwen"):
        #st.session_state.results["Qwen3"] = call_ollama("qwen", note)
        st.session_state.results["Qwen3"] = run_model("Qwen3", note, hf_pipeline)

with col5:
    if st.button("Run Phi"):
        #st.session_state.results["Phi-4-mini"] = call_ollama("phi", note)
        st.session_state.results["Phi-4-mini"] = run_model("Phi-4-mini", note, hf_pipeline)

with col6:
    if st.button("Run Mistral"):
        #st.session_state.results["Phi-4-mini"] = call_ollama("phi", note)
        st.session_state.results["Mistral"] = run_model("Mistral", note, hf_pipeline)


with col7:
    if st.button("Run Meditron"):
        #st.session_state.results["Phi-4-mini"] = call_ollama("phi", note)
        st.session_state.results["Phi-4-mini"] = run_model("Phi-4-mini", note, hf_pipeline)

with col8:
    if st.button("Run BioMistral variants"):
        #st.session_state.results["Phi-4-mini"] = call_ollama("phi", note)
        st.session_state.results["Phi-4-mini"] = run_model("Phi-4-mini", note, hf_pipeline)


# =========================
# RESULTS
# =========================
st.divider()
st.subheader("📊 Outputs")

for model, preds in st.session_state.results.items():
    st.markdown(f"### {model}")
    st.write("Raw output:", preds)

    parsed = parse_llm_output(preds)
    st.subheader("📊 Extracted NON Grounded Diagnoses + ICD + OMOP")
    if parsed:
        df = pd.DataFrame(parsed)
        st.dataframe(
            df.rename(columns={
            "diagnosis": "Diagnosis",
            "icd10": "ICD-10 Code",
            "omop_concept_id": "OMOP Concept ID"
            }),
        use_container_width=False
        )
    
    #---------------------------------------------

    st.subheader("📊 Extracted Grounded Diagnoses + ICD + OMOP")
    st.write("Grounded pipeline:")
    
    #---------------------------------------------
    #Athena ICD validations
    # predsForValidation = json.loads(preds)
    # icd_list_Validation = [x["icd10"] for x in predsForValidation]
    # validated = []
    # st.write("Athena Validation: Is this ICD real?")
    # for icd in icd_list_Validation:
    #     validated.append({
    #         "icd10": icd,
    #         "valid": is_valid_icd(icd, icd_set)
    #     })
    # st.write(validated)

    # note_id = save_result(
    #     model=model,
    #     note_text=note,
    #     raw_preds= preds,
    #     icd_validation=validated,
    #     grounded_results=validated
    # )
    # st.write("Saved")
    # st.write(note_id)

    #----------------------------------------------------
    #New grounded pipeline : 
    condition = run_llama_extract_condition(note)
    st.write("condition:", condition)
    
    candidates = retriever.grounded_candidates( condition, top_k=10, method="keyword")
    st.write("candidates:", candidates)
    
    #candidate_text = retriever.format_candidates_for_llm(candidates)
    #final = run_llama_select_icd( note=note, candidates=candidate_text)
    #st.write("Selected ICD:", final)
    #st.write("OMOP Concept:", final)
    #---------------------------------------------
    #Athena Concept ID lookup neeed to check below
    # predsForLookup = json.loads(preds)
    # icd_list = [x["icd10"] for x in predsForLookup]
    # #st.write("OMOP icd_list for look up:", icd_list)
    # omop_results = lookup_icd_to_omop(icd_list)
    # st.write("OMOP Results Concepts from Athena:")
    # st.json(omop_results)
    
    # note_id = save_result(
    #     model=model,
    #     note_text=note,
    #     raw_preds="Test",
    #     icd_validation="Test",
    #     grounded_results="test"
    # )
    # print(f"Saved Note ID: {note_id}")
    
    #-------------------------------------------------
    #note_id = len(st.session_state.results[model]) + 1
    # st.session_state.results[model].append({
    #     "note_id": note_id,
    #     "note": note_text,
    #     "model" : model,
    #     "raw_preds": preds,
    #     "icd_validation": validated,
    #     "grounded_results": omop_results
    # })

    # """ 
    # st.session_state.results[model] = {
    #     "note_id": note_id,
    #     "note": note,
    #     "model" : model,
    #     "raw_preds": preds,
    #     "icd_validation": validated,
    #     "grounded_results": omop_results
    # }
    # """

    #---------------------------------------------
    #st.write("SESSION STATE:")
    #st.json(st.session_state.results)

    #---------------------------------------------
    # concept_df = pd.read_csv("../data/athena/CONCEPT.csv", sep="\t", low_memory=True)
    # predsForValidation = json.loads(preds)
    # icd_list_Validation = [x["icd10"] for x in predsForValidation]
    # validated = []
    # st.write("Athena Validation: Is this ICD real?")
    # for icd in icd_list_Validation:
    #     result = {
    #     "icd10": icd,
    #     "icd_valid": is_valid_icd(icd, concept_df)
    #     }
    #     validated.append(result)
    # st.write(validated)