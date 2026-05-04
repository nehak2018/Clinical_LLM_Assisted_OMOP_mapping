import pandas as pd

# Load once
concept_df = pd.read_csv("data/athena/CONCEPT.csv", sep="\t", low_memory=False)

def lookup_icd_to_omop(icd_codes):
    results = []

    for code in icd_codes:

        matches = concept_df[
            (concept_df["concept_code"] == code) &
            (concept_df["vocabulary_id"].str.contains("ICD", na=False))
        ]

        if not matches.empty:
            top = matches.iloc[0]

            results.append({
                "icd_code": code,
                "concept_id": top["concept_id"],
                "concept_name": top["concept_name"],
                "vocabulary": top["vocabulary_id"]
            })

    return results