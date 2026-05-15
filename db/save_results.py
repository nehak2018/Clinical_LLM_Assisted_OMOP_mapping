# save_result.py
# Uses db_config.py instead of duplicating DB logic

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_config import get_connection
import json


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
        "SELECT COALESCE(MAX(note_id), 0) + 1 FROM note_results WHERE model = ?",
        (model,)
    )
    next_id = cur.fetchone()[0]
    conn.close()
    return next_id


def save_result(model, note_text, raw_preds, icd_validation, grounded_results):
    """
    Save one processed note result to SQLite.
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
        json.dumps(raw_preds),
        json.dumps(icd_validation),
        json.dumps(grounded_results)
    ))

    conn.commit()
    conn.close()

    return note_id