import pandas as pd


def is_valid_icd(icd_code: str, concept_df: pd.DataFrame) -> bool:
    match = concept_df[
        (concept_df["concept_code"] == icd_code) &
        (concept_df["vocabulary_id"].str.contains("ICD10", na=False)) &
        (concept_df["invalid_reason"].isna())
    ]
    return not match.empty

validated_preds = []