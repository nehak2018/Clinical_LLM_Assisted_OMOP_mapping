import pandas as pd
import csv
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
print("BASE_DIR:", BASE_DIR)
ATHENA_DIR = BASE_DIR / "data" / "athena"
print("BASE_DIR:", ATHENA_DIR)


# --------------------------------------------------
# LOAD + CACHE ATHENA TABLES ONCE
# --------------------------------------------------
@lru_cache(maxsize=1)
def load_athena_indexes():
    csv.field_size_limit(10**7)

    print("Loading Athena tables (one-time)...")

    # Load only needed columns
    #concept_df = pd.read_csv("../data/athena/CONCEPT.csv", sep="\t", low_memory=True)
    concept_path = ATHENA_DIR / "CONCEPT.csv"
    concept_df = pd.read_csv(concept_path,
        sep="\t",
        low_memory=False,
        usecols=[
            "concept_id",
            "concept_code",
            "concept_name",
            "vocabulary_id",
            "standard_concept",
            "invalid_reason"
        ]
    )

    relationship_path = ATHENA_DIR / "CONCEPT_RELATIONSHIP.csv"
    relationship_df = pd.read_csv(relationship_path,
        sep="\t",
        low_memory=False,
        usecols=["concept_id_1", "concept_id_2", "relationship_id"]
    )

    # Clean columns
    concept_df.columns = concept_df.columns.str.strip()
    relationship_df.columns = relationship_df.columns.str.strip()

    # --------------------------------------------------
    # STEP 1: ICD SOURCE INDEX
    # Keep only valid ICD concepts
    # --------------------------------------------------
    icd_df = concept_df[
        (concept_df["vocabulary_id"].astype(str).str.contains("ICD", na=False)) &
        (concept_df["invalid_reason"].isna())
    ].copy()

    icd_df["concept_code"] = icd_df["concept_code"].astype(str).str.upper()

    # ICD code -> source concept_id
    icd_to_source = dict(
        zip(icd_df["concept_code"], icd_df["concept_id"])
    )

    # --------------------------------------------------
    # STEP 2: MAPS TO INDEX
    # source concept_id -> standard concept_id
    # --------------------------------------------------
    maps_to_df = relationship_df[
        relationship_df["relationship_id"] == "Maps to"
    ]

    source_to_target = dict(
        zip(maps_to_df["concept_id_1"], maps_to_df["concept_id_2"])
    )

    # --------------------------------------------------
    # STEP 3: TARGET CONCEPT INDEX
    # concept_id -> concept details
    # --------------------------------------------------
    concept_details = concept_df.set_index("concept_id")[
        ["concept_name", "vocabulary_id", "standard_concept"]
    ].to_dict("index")

    print("Athena indexes ready.")

    return icd_to_source, source_to_target, concept_details


# --------------------------------------------------
# FAST LOOKUP FUNCTION
# --------------------------------------------------
def lookup_icd_to_omop(icd_codes):
    icd_to_source, source_to_target, concept_details = load_athena_indexes()
    
    print("This is array ========================================================")
    print(icd_codes)
    results = []

    # Normalize input
    icd_codes = [str(code).upper().strip() for code in icd_codes]

    for icd_code in icd_codes:

        # STEP 1: ICD -> source concept
        source_id = icd_to_source.get(icd_code)

        if source_id is None:
            results.append({
                "icd": icd_code,
                "error": "ICD not found"
            })
            continue

        # STEP 2: source -> standard OMOP
        target_id = source_to_target.get(source_id)

        if target_id is None:
            results.append({
                "icd": icd_code,
                "source_concept_id": source_id,
                "error": "No Maps to relationship"
            })
            continue

        # STEP 3: target details
        target = concept_details.get(target_id)

        if target is None:
            results.append({
                "icd": icd_code,
                "source_concept_id": source_id,
                "omop_concept_id": target_id,
                "error": "Target concept missing"
            })
            continue

        results.append({
            "icd": icd_code,
            "source_concept_id": int(source_id),
            "omop_concept_id": int(target_id),
            "concept_name": target["concept_name"],
            "vocabulary": target["vocabulary_id"],
            "standard_concept": target["standard_concept"]
        })

    return results


'''
def lookup_icd_to_omop(icd_codes):
    results = []
    print("This is array ========================================================")
    print(icd_codes)
    return results

'''