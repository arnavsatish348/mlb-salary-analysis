"""Load the Lahman Baseball Database tables into a local SQLite database."""
from pathlib import Path
import sqlite3

import pandas as pd
import pylahman

TABLES = ["salaries", "batting", "appearances", "people"]
DATA_DIR = Path(pylahman.__file__).parent / "data"


def build_database(db_path: Path) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        for table in TABLES:
            df = pd.read_parquet(DATA_DIR / f"{table.capitalize()}.parquet")
            df.to_sql(table, conn, if_exists="replace", index=False)
    return db_path


def load_features(db_path: Path, sql_path: Path) -> pd.DataFrame:
    query = Path(sql_path).read_text()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
