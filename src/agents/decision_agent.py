"""
Engine Decision (Section 6.4 / 7.1 / 7.3 of the proposal).

"Based on the profiling report, Engine Decision chooses the appropriate
processing path." The implementation is a dataset-agnostic router driven
by column dtype + cardinality, matching the same underlying policy the
Titanic script encoded by hand:

  - numeric column, missing values      -> RULE   (median imputation)
  - low-cardinality categorical, missing -> LLM    (semantic imputation -
                                                     same idea as the
                                                     Titanic `Embarked`
                                                     example in Section 6.5)
  - high-cardinality categorical         -> RULE / manual_review (identifiers,
                                             free text - guessing these with
                                             an LLM is not meaningful)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


class DecisionAgent:
    name = "Engine Decision"

    def __init__(self, high_cardinality_ratio: float = 0.5):
        self.high_cardinality_ratio = high_cardinality_ratio

    def run(self, df: pd.DataFrame, profiler_report: Dict) -> Tuple[List[Dict], List[Dict]]:
        trace: List[Dict] = []
        decisions: List[Dict] = []

        flagged = profiler_report.get("_flagged_columns", [])
        trace.append({
            "agent": self.name,
            "thought": f"Route each flagged column to rule-based or semantic (LLM) "
                       f"handling based on dtype and cardinality. Flagged: {flagged or 'none'}.",
        })

        n_rows = len(df) or 1
        for col in flagged:
            missing = int(df[col].isna().sum())
            if pd.api.types.is_numeric_dtype(df[col]):
                route, action = "rule", "impute_median"
                reason = "numeric column -> deterministic median imputation"
            else:
                cardinality_ratio = df[col].nunique(dropna=True) / n_rows
                if cardinality_ratio <= self.high_cardinality_ratio:
                    route, action = "llm", "semantic_impute"
                    reason = (f"categorical, low cardinality ratio "
                              f"({cardinality_ratio:.3f} <= {self.high_cardinality_ratio}) "
                              f"-> needs semantic reasoning")
                else:
                    route, action = "rule", "mode_or_manual_review"
                    reason = (f"categorical, high cardinality ratio "
                              f"({cardinality_ratio:.3f} > {self.high_cardinality_ratio}) "
                              f"-> identifier-like, not a good LLM target")

            decisions.append({
                "column": col,
                "missing_count": missing,
                "route": route,
                "action": action,
                "reason": reason,
            })
            trace.append({
                "agent": self.name,
                "action": f"route('{col}') -> {route}:{action}",
                "observation": reason,
            })

        return decisions, trace
