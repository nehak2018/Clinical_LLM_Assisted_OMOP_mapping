"""
validate_icd_semantic.py

ICD semantic validation module (no clinician required)

Combines:
1. Embedding-based ICD ↔ text similarity
2. LLM-based clinical consistency judge
3. Hybrid scoring for ICD correctness vs input text

Designed for OMOP / Athena + LLM pipelines.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

# -----------------------------
# Embedding utilities
# -----------------------------

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed")
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        return np.array(self.model.encode(text))


# -----------------------------
# LLM Judge
# -----------------------------

class LLMJudge:
    """
    Expects a callable LLM client with .chat(prompt) -> str
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def judge(self, text: str, icd_code: str, icd_desc: str) -> Dict[str, Any]:
        prompt = f"""
You are a clinical coding validator.

Task: Determine if ICD code correctly matches clinical text.

Return JSON only:
{{
  "match": true/false,
  "reason": "short reason"
}}

Text:
{text}

ICD:
{icd_code}
Description:
{icd_desc}
"""
        response = self.llm.chat(prompt)

        try:
            import json
            return json.loads(response)
        except Exception:
            return {"match": None, "reason": "parse_error", "raw": response}


# -----------------------------
# ICD Semantic Validator
# -----------------------------

@dataclass
class ICDValidationResult:
    icd_code: str
    embedding_score: float
    llm_match: Optional[bool]
    llm_reason: str
    final_score: float
    decision: str


class ICDSemanticValidator:
    def __init__(self, embedding_model: EmbeddingModel, llm_judge: LLMJudge):
        self.embedder = embedding_model
        self.judge = llm_judge

    def validate(
        self,
        text: str,
        icd_code: str,
        icd_description: str,
        weights: Dict[str, float] = None
    ) -> ICDValidationResult:

        if weights is None:
            weights = {
                "embedding": 0.5,
                "llm": 0.5
            }

        # -----------------------------
        # 1. Embedding similarity
        # -----------------------------
        text_emb = self.embedder.encode(text)
        icd_emb = self.embedder.encode(icd_description)
        emb_score = cosine_similarity(text_emb, icd_emb)

        # normalize to 0–1
        emb_score = max(0.0, min(1.0, (emb_score + 1) / 2))

        # -----------------------------
        # 2. LLM judge
        # -----------------------------
        llm_result = self.judge.judge(text, icd_code, icd_description)
        llm_match = llm_result.get("match", None)
        llm_reason = llm_result.get("reason", "")

        llm_score = 1.0 if llm_match is True else 0.0 if llm_match is False else 0.5

        # -----------------------------
        # 3. Final score
        # -----------------------------
        final_score = (
            weights["embedding"] * emb_score +
            weights["llm"] * llm_score
        )

        # -----------------------------
        # 4. Decision
        # -----------------------------
        if final_score >= 0.75:
            decision = "ACCEPT"
        elif final_score >= 0.5:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        return ICDValidationResult(
            icd_code=icd_code,
            embedding_score=emb_score,
            llm_match=llm_match,
            llm_reason=llm_reason,
            final_score=final_score,
            decision=decision
        )


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    class DummyLLM:
        def chat(self, prompt: str) -> str:
            return '{"match": true, "reason": "semantic overlap detected"}'

    embedder = EmbeddingModel()
    llm = LLMJudge(DummyLLM())

    validator = ICDSemanticValidator(embedder, llm)

    result = validator.validate(
        text="Type 2 diabetes with neuropathy",
        icd_code="E11.9",
        icd_description="Type 2 diabetes mellitus without complications"
    )

    print(result)
