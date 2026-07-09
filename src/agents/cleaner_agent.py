"""
Agent Cleaner (Section 6.5 / 7.1 / 7.3 of the proposal).

"If semantic analysis is required, Agent Cleaner activates and produces
suggested corrections." Generalizes src/llm_layer/semantic_cleaner.py
(which was hardcoded to Titanic's `Embarked` column) into a dataset-agnostic
few-shot prompt: build context from the *other* column values in the same
row, restrict the answer to the column's known allowed values when the
cardinality is low, and ask the model to self-report a confidence score
(0-100) alongside its answer -- this confidence is what Agent Reviewer checks
against `agents.score_confidence_threshold` in config.yaml.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from llm_client import MockLLMClient, OllamaClient  # noqa: E402


def _build_prompt(df: pd.DataFrame, row_idx, target_col: str, allowed_values: Optional[List[str]]) -> str:
    row = df.loc[row_idx]
    context_cols = [c for c in df.columns if c != target_col][:6]
    context_lines = "\n".join(f"- {c}: {row[c]}" for c in context_cols if pd.notna(row[c]))

    allowed_clause = ""
    if allowed_values:
        allowed_clause = f"Allowed values: [{', '.join(map(str, allowed_values))}]\n"

    return f"""You are a data-cleaning assistant. A value is missing in the column
'{target_col}'. Use the other fields in this row to infer the most likely value.
{allowed_clause}
Row context:
{context_lines}

Respond with ONLY a compact JSON object, no explanation, in the exact form:
{{"value": <your best guess>, "confidence": <0-100 integer>}}
"""


def _parse_response(raw: str) -> Tuple[Optional[str], int]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, 0
    try:
        obj = json.loads(match.group(0))
        value = obj.get("value")
        confidence = int(obj.get("confidence", 0))
        return (str(value) if value is not None else None), confidence
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, 0


class CleanerAgent:
    name = "Agent Cleaner"

    def __init__(self, llm_client, max_rows_per_column: int = 20):
        self.llm_client = llm_client
        self.max_rows_per_column = max_rows_per_column

    def run(self, df: pd.DataFrame, decisions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        trace: List[Dict] = []
        suggestions: List[Dict] = []

        llm_decisions = [d for d in decisions if d["route"] == "llm"]
        if not llm_decisions:
            trace.append({"agent": self.name, "thought": "No columns routed to LLM - nothing to do."})
            return suggestions, trace

        backend = "MockLLMClient (offline)" if isinstance(self.llm_client, MockLLMClient) else "Ollama"
        trace.append({
            "agent": self.name,
            "thought": f"{len(llm_decisions)} column(s) need semantic imputation. "
                       f"Using LLM backend: {backend}.",
        })

        for decision in llm_decisions:
            col = decision["column"]
            missing_idx = df[df[col].isna()].index.tolist()[: self.max_rows_per_column]

            allowed_values = None
            uniques = df[col].dropna().unique().tolist()
            if 1 < len(uniques) <= 20:
                allowed_values = [str(v) for v in uniques]

            for idx in missing_idx:
                prompt = _build_prompt(df, idx, col, allowed_values)
                try:
                    raw = self.llm_client.generate(prompt)
                except Exception as exc:  # network / server errors -> degrade gracefully
                    trace.append({"agent": self.name, "observation": f"LLM call failed for row {idx}: {exc}"})
                    continue

                value, confidence = _parse_response(raw)
                suggestions.append({
                    "column": col,
                    "row_index": int(idx),
                    "raw_response": raw.strip()[:300],
                    "suggested_value": value,
                    "confidence": confidence,
                    "allowed_values": allowed_values,
                })

            trace.append({
                "agent": self.name,
                "action": f"semantic_impute('{col}') on {len(missing_idx)} row(s)",
                "observation": f"Produced {len([s for s in suggestions if s['column'] == col])} suggestion(s).",
            })

        return suggestions, trace
