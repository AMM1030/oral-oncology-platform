"""
Load the synthetic CSVs into a SQLite database.

Run:
    python src/load_to_sqlite.py

Output: data/processed/oral_oncology.db
"""

import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "processed" / "oral_oncology.db"
SCHEMA = ROOT / "sql" / "schema.sql"

TABLES = {
    "patients":       "patients.csv",
    "dental_visits":  "dental_visits.csv",
    "clinical_notes": "clinical_notes.csv",
    "referrals":      "referrals.csv",
    "pathology":      "pathology.csv",
    "appointments":   "appointments.csv",
    "outcomes":       "outcomes.csv",
}


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Create the database file and run the schema.
    print(f"Creating database: {DB_PATH.relative_to(ROOT)}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    with open(SCHEMA, "r") as f:
        conn.executescript(f.read())

    # 2. Load each CSV into its table using pandas.
    for table, csv_name in TABLES.items():
        df = pd.read_csv(DATA_RAW / csv_name)
        df.to_sql(table, conn, if_exists="append", index=False)
        print(f"  loaded {table:<16} {len(df):>6,} rows")

    conn.commit()

    # 3. Sanity check: count rows in each table from the DB itself.
    print("\n--- Row counts read back from DB ---")
    for table in TABLES:
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<16} {n:>6,}")

    conn.close()
    print(f"\nDone. Database at: {DB_PATH}")


if __name__ == "__main__":
    main()