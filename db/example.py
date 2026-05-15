# save_result.py
# Uses your existing db_config.py
# Make sure db_config.py is in the same project folder

import json
import pandas as pd
from typing import Optional

from db_config import get_connection


# =========================
# NOTE ID GENERATOR
# =========================
def get_next_note_id(model: str) -> int:
    """
    Auto-increment note_id per model.
    Example:
      llama -> 1,2,3
      gpt -> 1,2,3
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT COALESCE(MAX(note_id), 0) + 1
        FROM note_results
        WHERE model = ?
        """,
        (model,)
    )

    next_id = cur.fetchone()[0]

    conn.close()

    return next_id


# =========================
# SAVE RESULT
# =========================
def save_result(
    model: str,
    note_text: str,
    preds,
    validated,
    grounded
) -> int:
    """
    Save one processed clinical note result to SQLite.

    Parameters:
        model (str): Model name (e.g. llama, GPT)
        note_text (str): Original note text
        preds (list/dict): Raw LLM predictions
        validated (list/dict): ICD validation output
        grounded (list/dict): OMOP grounded results

    Returns:
        int: note_id
    """

    note_id = get_next_note_id(model)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO note_results (
            note_id,
            model,
            note_text,
            raw_preds,
            icd_validation,
            grounded_results
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        note_id,
        model,
        note_text,
        json.dumps(preds),
        json.dumps(validated),
        json.dumps(grounded)
    ))

    conn.commit()
    conn.close()

    return note_id


# =========================
# LOAD RESULTS
# =========================
def load_results(model: Optional[str] = None) -> pd.DataFrame:
    """
    Load all saved results or filter by model.

    Parameters:
        model (str, optional): Filter by model

    Returns:
        pandas.DataFrame
    """

    conn = get_connection()

    if model:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM note_results
            WHERE model = ?
            ORDER BY note_id
            """,
            conn,
            params=(model,)
        )
    else:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM note_results
            ORDER BY model, note_id
            """,
            conn
        )

    conn.close()

    return df


# =========================
# GET SINGLE NOTE
# =========================
def get_note_by_id(model: str, note_id: int):
    """
    Fetch one note by model + note_id.
    """

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT *
        FROM note_results
        WHERE model = ? AND note_id = ?
    """, (model, note_id))

    row = cur.fetchone()

    conn.close()

    return row


# =========================
# DELETE NOTE
# =========================
def delete_note(model: str, note_id: int):
    """
    Delete one note by model + note_id.
    """

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM note_results
        WHERE model = ? AND note_id = ?
    """, (model, note_id))

    conn.commit()
    conn.close()


# =========================
# RESET DATABASE
# =========================
def reset_database():
    """
    Delete all rows from note_results.
    Use carefully.
    """

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("DELETE FROM note_results")

    conn.commit()
    conn.close()


# =========================
# EXAMPLE TEST
# =========================
if __name__ == "__main__":

    sample_note = "Patient has type 2 diabetes mellitus and essential hypertension."

    sample_preds = [
        {
            "diagnosis": "Type 2 diabetes mellitus",
            "icd10": "E11.9",
            "omop_concept_id": 201826
        },
        {
            "diagnosis": "Essential hypertension",
            "icd10": "I10",
            "omop_concept_id": 320128
        }
    ]

    sample_validated = [
        {"icd10": "E11.9", "valid": True},
        {"icd10": "I10", "valid": True}
    ]

    sample_grounded = [
        {"icd10": "E11.9", "standard_concept_id": 201826},
        {"icd10": "I10", "standard_concept_id": 320128}
    ]

    # Save test note
    note_id = save_result(
        model="llama",
        note_text=sample_note,
        preds=sample_preds,
        validated=sample_validated,
        grounded=sample_grounded
    )

    print(f"Saved Note ID: {note_id}")

    # Load all llama notes
    df = load_results("llama")

    print(df)