"""
Layer 6.2 - Data Profiling

Computes a baseline data-quality profile for an ingested DataFrame:
- missing values per column (count + %)
- duplicate row count
- per-column dtype, unique value count
- basic numeric stats (min/max/mean) for numeric columns

This profile is saved as JSON so it can later be compared against the
profile of the cleaned dataset (Layer 6.9 - Evaluation).

Run: python src/profiling/profile_data.py
"""
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from load_data import ingest  # noqa: E402

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "titanic.csv")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "titanic_profile_before.json")


def profile(df: pd.DataFrame) -> dict:
    n_rows = len(df)

    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)

    columns_report = {}
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "missing_count": int(missing[col]),
            "missing_pct": float(missing_pct[col]),
            "unique_count": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["min"] = float(df[col].min()) if df[col].notna().any() else None
            col_info["max"] = float(df[col].max()) if df[col].notna().any() else None
            col_info["mean"] = float(df[col].mean()) if df[col].notna().any() else None
        columns_report[col] = col_info

    report = {
        "n_rows": n_rows,
        "n_columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "overall_missing_cells": int(missing.sum()),
        "overall_missing_pct": float((missing.sum() / (n_rows * df.shape[1]) * 100).round(2)),
        "columns": columns_report,
    }
    return report


if __name__ == "__main__":
    df = ingest(RAW_PATH)
    report = profile(df)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== PROFILING REPORT (before cleaning) ===")
    print(f"Rows: {report['n_rows']}, Columns: {report['n_columns']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Overall missing cells: {report['overall_missing_cells']} ({report['overall_missing_pct']}%)")
    print("\nPer-column missing values:")
    for col, info in report["columns"].items():
        if info["missing_count"] > 0:
            print(f"  - {col}: {info['missing_count']} missing ({info['missing_pct']}%)")
    print(f"\nFull report saved to: {REPORT_PATH}")