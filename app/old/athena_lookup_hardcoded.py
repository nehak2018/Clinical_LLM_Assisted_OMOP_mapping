import pandas as pd
import csv

# Load once
#concept_df = pd.read_csv("../data/athena/CONCEPT.csv", sep="\t", low_memory=False)


def lookup_icd_to_omop(icd_codes):

    print("This is array ========================================================")
    print(icd_codes)
    # -----------------------------
    # SAFE CSV SETTINGS
    # -----------------------------
    csv.field_size_limit(10**7)
    # -----------------------------
    # LOAD ATHENA TABLES
    # -----------------------------
    concept_df = pd.read_csv("../data/athena/CONCEPT.csv", sep="\t", low_memory=True)
    relationship_df = pd.read_csv("../data/athena/CONCEPT_RELATIONSHIP.csv",  sep="\t", usecols=["concept_id_1", "concept_id_2", "relationship_id"], low_memory=True)
    
    # Clean column names
    concept_df.columns = concept_df.columns.str.strip()
    relationship_df.columns = relationship_df.columns.str.strip()

    # -----------------------------
    # HARD-CODED ICD
    # -----------------------------
    
    icd_code = "I10"
    print("\nSTEP 1: Searching ICD in CONCEPT table...")
    source = concept_df[
        (concept_df["concept_code"].astype(str).str.upper() == icd_code) &
        (concept_df["vocabulary_id"].astype(str).str.contains("ICD", na=False)) &
        (concept_df["invalid_reason"].isna() | concept_df["invalid_reason"].isnull())
    ]
    print("Source matches:")
    print(source[["concept_id", "concept_name", "vocabulary_id"]])
    if source.empty:
        print("ICD not found in CONCEPT table")
        exit()
    source_id = int(source.iloc[0]["concept_id"])
    print("\nSource concept_id:", source_id)

    # -----------------------------
    # STEP 2: LOOKUP RELATIONSHIP
    # -----------------------------
    print("\nSTEP 2: Searching CONCEPT_RELATIONSHIP (Maps to)...")
    rel = relationship_df[
        (relationship_df["concept_id_1"] == source_id) &
        (relationship_df["relationship_id"] == "Maps to")
    ]
    print("Relationship matches:")
    print(rel.head())
    if rel.empty:
        print("No 'Maps to' relationship found")
        exit()

    target_id = int(rel.iloc[0]["concept_id_2"])
    print("\nTarget concept_id:", target_id)

    # -----------------------------
    # STEP 3: FINAL OMOP CONCEPT
    # -----------------------------
    print("\nSTEP 3: Fetching OMOP concept...")
    target = concept_df[ concept_df["concept_id"] == target_id]
    print(target[[
        "concept_id",
        "concept_name",
        "vocabulary_id",
        "standard_concept"
    ]])
    if target.empty:
        print("Target concept not found")
        exit()

    print("\nFINAL RESULT")
    results = {
        "icd": icd_code,
        "source_concept_id": source_id,
        "omop_concept_id": target_id,
        "concept_name": target.iloc[0]["concept_name"],
        "vocabulary": target.iloc[0]["vocabulary_id"]
    }  



    return results