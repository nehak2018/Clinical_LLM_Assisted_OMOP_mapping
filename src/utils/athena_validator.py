import pandas as pd

def build_icd_index(concept_df: pd.DataFrame) -> set:
    """
    Precompute valid ICD codes for O(1) validation.
    """
    icd_df = concept_df[
        concept_df["vocabulary_id"].astype(str).str.contains("ICD10", na=False)
        & concept_df["invalid_reason"].isna()
    ]

    return set(icd_df["concept_code"].astype(str).str.upper())



def is_valid_icd(icd_code: str, icd_set: set) -> bool:
    return icd_code.upper().strip() in icd_set