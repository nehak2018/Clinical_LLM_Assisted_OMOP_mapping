import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db_config import get_connection
import json


def load_table():
    conn = get_connection()

    df = pd.read_sql_query("SELECT * FROM note_results",conn)

    conn.close()
    return df