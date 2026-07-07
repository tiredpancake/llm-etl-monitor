import json
import os

VALID_EMBARKED_VALUES = ["S", "C", "Q"]


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