import json
import os
from typing import Dict, List, Tuple

import pandas as pd

VALID_EMBARKED_VALUES = ["S", "C", "Q"]


class ReviewerAgent:
    """Agent Reviewer (Section 6.6 / 7.1 / 7.3 of the proposal).

    "Agent Reviewer validates the output and applies valid changes." Generic
    version of the logic below: accepts an Agent Cleaner suggestion only if
    (a) the confidence score is at/above `score_confidence_threshold`
    (config/config.yaml -> agents.score_confidence_threshold) AND
    (b) the value is within the column's allowed set, when one was provided.
    Accepted corrections are applied in-place to the DataFrame; every
    decision (accepted or rejected) is kept in the returned report for the
    audit trail Section 7.2 requires.
    """

    name = "Agent Reviewer"

    def __init__(self, confidence_threshold: int = 70):
        self.confidence_threshold = confidence_threshold

    def run(self, df: pd.DataFrame, suggestions: List[Dict]) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
        trace: List[Dict] = []
        results: List[Dict] = []

        if not suggestions:
            trace.append({"agent": self.name, "thought": "No Agent Cleaner suggestions to review."})
            return df, results, trace

        trace.append({
            "agent": self.name,
            "thought": f"Reviewing {len(suggestions)} suggestion(s) against "
                       f"confidence threshold {self.confidence_threshold}.",
        })

        df = df.copy()
        for s in suggestions:
            value = s["suggested_value"]
            confidence = s.get("confidence", 0)
            allowed = s.get("allowed_values")

            passes_confidence = confidence >= self.confidence_threshold
            passes_domain = (value in allowed) if allowed else (value is not None)
            accepted = bool(passes_confidence and passes_domain and value is not None)

            if accepted:
                df.at[s["row_index"], s["column"]] = value

            results.append({
                **s,
                "passes_confidence": passes_confidence,
                "passes_domain_check": passes_domain,
                "accepted": accepted,
            })

        accepted_count = sum(1 for r in results if r["accepted"])
        trace.append({
            "agent": self.name,
            "action": "apply_accepted_changes()",
            "observation": f"{accepted_count}/{len(results)} corrections accepted and applied "
                           f"(threshold={self.confidence_threshold}).",
        })
        return df, results, trace


def review_llm_decisions(llm_log_path, output_path):
    with open(llm_log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    review_results = []

    for entry in log:
        predicted = entry["predicted_embarked"]
        source = entry["source"]

        is_valid_category = predicted in VALID_EMBARKED_VALUES

        is_clean_response = len(entry["raw_llm_response"]) < 20

        accepted = is_valid_category and is_clean_response

        review_results.append({
            **entry,
            "is_valid_category": is_valid_category,
            "is_clean_response": is_clean_response,
            "accepted_by_reviewer": accepted
        })

        status = "✅ ACCEPTED" if accepted else "❌ REJECTED"
        print(f"ردیف {entry['row_index']}: {status} (پیش‌بینی: {predicted})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(review_results, f, indent=2, ensure_ascii=False)

    accepted_count = sum(1 for r in review_results if r["accepted_by_reviewer"])
    print(f"\n✅ {accepted_count} از {len(review_results)} تصمیم پذیرفته شد.")
    print(f"گزارش کامل در {output_path} ذخیره شد.")

    return review_results


if __name__ == "__main__":
    review_llm_decisions(
        llm_log_path="reports/titanic_llm_log.json",
        output_path="reports/titanic_reviewer_report.json"
    )