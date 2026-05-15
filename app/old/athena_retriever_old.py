"""
athena_retriever.py
-----------------------------------
Plug-and-play Athena / OMOP vocabulary retriever for:
1. Exact ICD validation
2. Text condition retrieval (keyword)
3. Semantic retrieval (optional if sentence-transformers installed)
4. ICD -> Standard OMOP concept mapping
5. Candidate generation for TRUE grounded LLM pipelines

Author: OHDSI / OMOP Benchmark Pipeline
"""
from __future__ import annotations
from pathlib import Path
import os
import re
import warnings
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from rapidfuzz.fuzz import ratio

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


class AthenaRetriever:
    """
    Athena / OMOP Vocabulary Retriever

    Required files:
    - CONCEPT.csv
    - CONCEPT_RELATIONSHIP.csv

    Optional:
    - CONCEPT_SYNONYM.csv
    """

    def __init__(
        self,
        concept_path: str,
        concept_relationship_path: str,
        concept_synonym_path: Optional[str] = None,
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.concept_df = pd.read_csv(concept_path, sep="\t", low_memory=False)
        self.rel_df = pd.read_csv(concept_relationship_path, sep="\t", low_memory=False)

        self.synonym_df = None
        if concept_synonym_path and os.path.exists(concept_synonym_path):
            self.synonym_df = pd.read_csv(concept_synonym_path, sep="\t", low_memory=False)

        # Keep valid concepts only
        self.concept_df = self.concept_df[self.concept_df["invalid_reason"].isna()].copy()

        # Normalize
        self.concept_df["concept_name_lower"] = (
            self.concept_df["concept_name"].fillna("").str.lower()
        )

        # Embeddings
        self.use_embeddings = use_embeddings and HAS_EMBEDDINGS
        self.embedding_model = None

        if use_embeddings and not HAS_EMBEDDINGS:
            warnings.warn(
                "sentence-transformers not installed. Falling back to keyword retrieval."
            )

        if self.use_embeddings:
            self.embedding_model = SentenceTransformer(embedding_model)
            self._build_embeddings()

    # ---------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------

    @staticmethod
    def normalize_text(text: str) -> str:
        if text is None:
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _build_embeddings(self):
        print("Building concept embeddings... (this may take time)")
        self.concept_df["embedding"] = self.concept_df["concept_name"].apply(
            lambda x: self.embedding_model.encode(str(x))
        )

    # ---------------------------------------------------------
    # ICD Validation
    # ---------------------------------------------------------

    def validate_icd(
        self,
        icd_code: str,
        vocabulary: str = "ICD10CM"
    ) -> Optional[Dict[str, Any]]:
        """
        Validate ICD exists in Athena
        """
        matches = self.concept_df[
            (self.concept_df["concept_code"] == icd_code) &
            (self.concept_df["vocabulary_id"] == vocabulary)
        ]

        if matches.empty:
            return None

        row = matches.iloc[0]
        return {
            "concept_id": row["concept_id"],
            "concept_name": row["concept_name"],
            "concept_code": row["concept_code"],
            "vocabulary_id": row["vocabulary_id"],
            "standard_concept": row["standard_concept"]
        }

    # ---------------------------------------------------------
    # ICD -> Standard OMOP Mapping
    # ---------------------------------------------------------

    def map_to_standard(self, icd_code: str) -> Optional[Dict[str, Any]]:
        """
        Maps ICD10CM code -> Standard concept using 'Maps to'
        """
        source = self.validate_icd(icd_code)

        if not source:
            return None

        source_id = source["concept_id"]

        rel = self.rel_df[
            (self.rel_df["concept_id_1"] == source_id) &
            (self.rel_df["relationship_id"] == "Maps to")
        ]

        if rel.empty:
            return None

        standard_id = rel.iloc[0]["concept_id_2"]

        standard = self.concept_df[
            self.concept_df["concept_id"] == standard_id
        ]

        if standard.empty:
            return None

        row = standard.iloc[0]

        return {
            "source_icd": icd_code,
            "source_concept_id": source_id,
            "standard_concept_id": row["concept_id"],
            "standard_concept_name": row["concept_name"],
            "standard_code": row["concept_code"],
            "standard_vocabulary": row["vocabulary_id"]
        }

    # ---------------------------------------------------------
    # Keyword Retrieval
    # ---------------------------------------------------------

    def retrieve_keyword(
        self,
        condition: str,
        #top_k: int = 200,
        vocabularies: List[str] = ["ICD10CM", "SNOMED"]
    ) -> pd.DataFrame:
        """
        Retrieve candidate concepts via keyword search
        """
        query = self.normalize_text(condition)

        results = self.concept_df[
            (self.concept_df["vocabulary_id"].isin(vocabularies)) &
            #(self.concept_df["domain_id"] == "Condition") &
            #(self.concept_df["concept_name_lower"].str.contains(query, na=False))
            (self.concept_df["concept_name_lower"].apply(
                lambda x: all(
                    word in str(x)
                    for word in query.split()
                    )
                )   
            )
        ]

        return results[
            [
                "concept_id",
                "concept_name",
                "concept_code",
                "vocabulary_id",
                "standard_concept"
            ]
        ] #.head(top_k)

    # ---------------------------------------------------------
    # Semantic Retrieval
    # ---------------------------------------------------------

    def retrieve_semantic(
        self,
        condition: str,
        top_k: int = 10,
        vocabularies: List[str] = ["ICD10CM", "SNOMED"]
    ) -> pd.DataFrame:
        """
        Semantic retrieval using embeddings
        """
        if not self.use_embeddings:
            raise RuntimeError(
                "Semantic retrieval unavailable. Initialize with use_embeddings=True"
            )

        q_emb = self.embedding_model.encode(condition)

        df = self.concept_df[
            self.concept_df["vocabulary_id"].isin(vocabularies)
        ].copy()

        df["score"] = df["embedding"].apply(
            lambda x: float(np.dot(x, q_emb))
        )

        df = df.sort_values("score", ascending=False)

        return df[
            [
                "concept_id",
                "concept_name",
                "concept_code",
                "vocabulary_id",
                "standard_concept",
                "score"
            ]
        ].head(top_k)

    # ---------------------------------------------------------
    # Grounded Candidate Retrieval
    # ---------------------------------------------------------

    def grounded_candidates(
        self,
        condition: str,
        top_k: int = 200,
        method: str = "keyword"
    ) -> pd.DataFrame:
        """
        Main retrieval function for TRUE grounding
        """
        if method == "semantic":
            return self.retrieve_semantic(condition, top_k=top_k)
        
        # No top_k here — return ALL matches, let rank_candidates do the cut
        return self.retrieve_keyword(condition)
        
    # ---------------------------------------------------------
    # Prompt Formatter for LLM
    # ---------------------------------------------------------

@staticmethod
def format_candidates_for_llm(candidates_df: pd.DataFrame) -> str:
    lines = []
    for i, row in enumerate(candidates_df.itertuples(index=False), start=1):
        standard = ""
        if (
            hasattr(row, "standard_concept_id")
            and row.standard_concept_id is not None
            and not (isinstance(row.standard_concept_id, float) and pd.isna(row.standard_concept_id))
        ):
            standard = (
                f" → OMOP standard_concept_id={int(row.standard_concept_id)}"
                f" ({row.standard_concept_name})"
            )

        lines.append(
            f"{i}. {row.concept_code} | {row.concept_name} | "
            f"{row.vocabulary_id} | concept_id={row.concept_id}{standard}"
        )
    return "\n".join(lines)

    # @staticmethod
    # def format_candidates_for_llm(candidates_df: pd.DataFrame) -> str:
    #     """
    #     Formats Athena candidates for constrained LLM prompt
    #     """
    #     lines = []

    #     for i, row in enumerate(candidates_df.itertuples(index=False), start=1):
    #         lines.append(
    #             f"{i}. {row.concept_code} | {row.concept_name} | "
    #             f"{row.vocabulary_id} | concept_id={row.concept_id}"
    #         )

    #     return "\n".join(lines)

    def add_standard_mappings(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        For each candidate ICD/source concept,
        append OMOP standard mapping columns.
        """

        if candidates_df.empty:
            return candidates_df

        standard_ids = []
        standard_names = []
        standard_vocabularies = []

        for _, row in candidates_df.iterrows():
            icd_code = row["concept_code"]

            mapping = self.map_to_standard(icd_code)

            if mapping:
                standard_ids.append(mapping["standard_concept_id"])
                standard_names.append(mapping["standard_concept_name"])
                standard_vocabularies.append(mapping["standard_vocabulary"])
            else:
                standard_ids.append(None)
                standard_names.append(None)
                standard_vocabularies.append(None)

        enriched = candidates_df.copy()

        enriched["standard_concept_id"] = standard_ids
        enriched["standard_concept_name"] = standard_names
        enriched["standard_vocabulary"] = standard_vocabularies

        return enriched


    def rank_candidates(
        self,
        query: str,
        candidates_df: pd.DataFrame,
        note: str = None,
        top_k: int = 200
    ) -> pd.DataFrame:
        """
        Generalized ranking for Athena candidates.
        Goals:
        - Prefer exact/base disease matches
        - Prefer uncomplicated / unspecified when note lacks complications
        - Penalize overly specific complication variants
        - Prefer ICD10CM slightly over SNOMED for ICD selection
        """

        if candidates_df is None or candidates_df.empty:
            return candidates_df

        query = self.normalize_text(query)
        query_tokens = set(query.split())
        note_text = self.normalize_text(note) if note else ""
        note_tokens = set(note_text.split())

        def score_row(row):
            name = self.normalize_text(row["concept_name"])
            name_tokens = set(name.split())

            # -------------------------------------------------
            # 1. Lexical similarity
            # -------------------------------------------------
            fuzzy_score = ratio(query, name)

            # -------------------------------------------------
            # 2. Query token coverage
            # Candidate should contain MOST query meaning
            # -------------------------------------------------
            token_coverage = (
                len(query_tokens & name_tokens) /
                max(len(query_tokens), 1)
            ) * 100
            
            # -------------------------------------------------
            # 2. Token overlap
            # -------------------------------------------------
            # overlap_score = (
            #     len(query_tokens & name_tokens) /
            #     max(len(query_tokens), 1)
            # ) * 50

            # # -------------------------------------------------
            # 3. Unsupported token penalty
            # Tokens candidate adds that are absent from BOTH
            # query and note = likely over-specific
            # -------------------------------------------------
            unsupported_tokens = name_tokens - query_tokens - note_tokens
            unsupported_penalty = len(unsupported_tokens) * 12
            
            # -------------------------------------------------
            # 4. Simplicity / brevity bonus
            # Shorter clinically valid candidates rank higher
            # -------------------------------------------------
            extra_token_count = max(0, len(name_tokens) - len(query_tokens))
            brevity_bonus = max(0, 60 - (extra_token_count * 10))

            # -------------------------------------------------
            # 5. Exact/base concept boost
            # VERY IMPORTANT
            # -------------------------------------------------
            exact_bonus = 0

            # Exact phrase
            if name == query:
                exact_bonus += 200

            # Candidate begins with query and adds little
            elif name.startswith(query):
                exact_bonus += 80

            # All query tokens fully covered
            elif query_tokens.issubset(name_tokens):
                exact_bonus += 40


            # -------------------------------------------------
            # 3. Extra modifiers penalty
            # More words usually = more subtype specificity
            # -------------------------------------------------
            #extra_tokens = len(name_tokens - query_tokens)
            #modifier_penalty = extra_tokens * 8

            # -------------------------------------------------
            # 4. Vocabulary bonus
            # -------------------------------------------------
            vocab_bonus = 0

            if row["vocabulary_id"] == "ICD10CM":
                vocab_bonus += 15 #10
            elif row["vocabulary_id"] == "SNOMED":
                vocab_bonus += 8 #5

            # -------------------------------------------------
            # 7. Default unspecified bonus
            # Generic non-hardcoded wording
            # -------------------------------------------------
            default_bonus = 0

            if "without" in name_tokens:
                default_bonus += 40

            if "unspecified" in name_tokens:
                default_bonus += 25


            # -------------------------------------------------
            # 5. Generic default / uncomplicated logic
            # -------------------------------------------------
            # default_bonus = 0

            # # Prefer default ICD variants
            # if "without complications" in name:
            #     default_bonus += 40

            # if "unspecified" in name:
            #     default_bonus += 25

            # # Penalize complication-heavy "with ..."
            # if " with " in f" {name} ":
            #     default_bonus -= 25

            # -------------------------------------------------
            # 6. Exact match boost
            # -------------------------------------------------
            # exact_bonus = 0

            # if name == query:
            #     exact_bonus += 60

            # elif name.startswith(query):
            #     exact_bonus += 30

            # -------------------------------------------------
            # FINAL SCORE
            # -------------------------------------------------
            final_score = (
                fuzzy_score
                + token_coverage
                + brevity_bonus
                + exact_bonus
                + vocab_bonus
                + default_bonus
                - unsupported_penalty
            )
            
            # -------------------------------------------------
            # FINAL SCORE
            # -------------------------------------------------
            # final_score = (
            #     fuzzy_score
            #     + overlap_score
            #     + vocab_bonus
            #     + default_bonus
            #     + exact_bonus
            #     - modifier_penalty
            # )

            return final_score

        ranked = candidates_df.copy()

        ranked["rank_score"] = ranked.apply(
            score_row,
            axis=1
        )

        return ranked.sort_values(
            by="rank_score",
            ascending=False
        )