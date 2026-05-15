import sys
import os
import pandas as pd
import json
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.llms.engine import rule_based, hf_model, call_ollama, parse_llm_output, run_llama_extract_condition, run_llama_select_icd
from src.utils.hf_loader import load_model
from src.utils.mapping import map_to_omop
from src.utils.athena_lookup import lookup_icd_to_omop
from src.utils.athena_validator import build_icd_index, is_valid_icd
from db.save_results import save_result
from src.utils.athena_retriever import AthenaRetriever
# ===================================================================================================================

@st.cache_resource
def load_retriever():
    return AthenaRetriever(
        concept_path="../data/athena/CONCEPT.csv",
        concept_relationship_path="../data/athena/CONCEPT_RELATIONSHIP.csv",
        use_embeddings=False
    )

retriever = load_retriever()

# ===================================================================================================================
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
    note_id = save_result(
         model=model,
         note_text=note,
         raw_preds= preds,
         icd_validation=parsed,
         grounded_results=parsed
     )
    st.write("Saved")
    st.write(note_id)
    #---------------------------------------------
    st.subheader("📊 Extracted Grounded Diagnoses + ICD + OMOP")
    st.write("Grounded pipeline:")
    #----------------------------------------------------
    #New grounded pipeline : 
    #condition = run_llama_extract_condition(note)
    condition = run_llama_extract_condition("llama3.2", note)
    st.write("condition:", condition)

    # Step 1: retrieve all candidates (no cut yet)
    candidates = retriever.grounded_candidates(condition, method="keyword")
    st.write("Candidates retrieved:", len(candidates), "rows")
    st.write("candidates retrived:", candidates)
    
    ranked_candidates = retriever.rank_candidates( query=condition, candidates_df=candidates, note=note, top_k=200)
    st.write("Full Ranked :", len(ranked_candidates), "rows")
    st.write("Full Ranked candidates:", ranked_candidates)

    # Only show ICD10CM to the selector LLM
    top_candidates = ranked_candidates[ ranked_candidates["vocabulary_id"] == "ICD10CM"].head(10)
    
    # candidatesWithSnomed = retriever.add_standard_mappings(top_candidates)
    # st.write("candidates after snomed:", candidatesWithSnomed)
    # candidate_text = retriever.format_candidates_for_llm(candidatesWithSnomed)
    # st.write("candidate_text:", candidate_text)
    # final = run_llama_select_icd( "llama3.2", note=note, candidates=candidate_text)
    # st.write("Selected ICD:", final)

    # Guard: if empty, don't call LLM at all
    if top_candidates.empty:
        st.warning(
            f"No ICD10CM candidates found for: '{condition}'. "
            "Try rephrasing or check Athena vocabulary coverage."
        )
    else:
        candidatesWithSnomed = retriever.add_standard_mappings(top_candidates)
        candidate_text = retriever.format_candidates_for_llm(candidatesWithSnomed)
        st.write("candidate_text:", candidate_text)
        final = run_llama_select_icd("llama3.2", note=note, candidates=candidate_text)
        st.write("Selected ICD:", final)
