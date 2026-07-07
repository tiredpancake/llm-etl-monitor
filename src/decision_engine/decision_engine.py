import json
import os

# سیاست مسیریابی: هر نوع قانون به کدام مسیر می‌رود
ROUTING_POLICY = {
    "uniqueness": "rule",
    "not_null": "rule",
    "range_check": "rule",
    "missing_values": "conditional",  # بسته به نوع ستون تصمیم می‌گیریم
    "schema_mismatch": "llm",
    "semantic_error": "llm",
}

# ستون‌های عددی که مقدار گمشده‌شان با میانه پر می‌شود (Rule-Based)
NUMERIC_IMPUTE_COLUMNS = ["Age", "Fare"]

# ستون‌های دسته‌ای که مقدار گمشده‌شان با کمک LLM یا mode تحلیل می‌شود
CATEGORICAL_COLUMNS = ["Embarked", "Cabin"]


def decide_route(rule_name, column, num_failed):
    """برای هر قانون شکست‌خورده، مسیر پردازش را مشخص می‌کند."""
    base_route = ROUTING_POLICY.get(rule_name, "llm")

    if base_route == "conditional":
        if column in NUMERIC_IMPUTE_COLUMNS:
            return "rule", "impute_median"
        elif column in CATEGORICAL_COLUMNS:
            return "llm", "semantic_impute"
        else:
            return "llm", "manual_review"

    if base_route == "rule":
        return "rule", "auto_fix"

    return "llm", "semantic_correction"


def build_decisions(validation_report_path, output_path):
    with open(validation_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    decisions = {"dataset": report["dataset"], "decisions": []}

    for rule_result in report["rules"]:
        if rule_result["success"]:
            continue  # قانون رعایت شده، نیازی به اقدام نیست

        route, action = decide_route(
            rule_result["rule"],
            rule_result["column"],
            rule_result["num_failed"]
        )

        decisions["decisions"].append({
            "rule": rule_result["rule"],
            "column": rule_result["column"],
            "num_failed": rule_result["num_failed"],
            "route": route,
            "action": action
        })

        print(f"[ROUTE: {route.upper()}] {rule_result['rule']} on '{rule_result['column']}' "
              f"({rule_result['num_failed']} rows) -> action: {action}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)

    print(f"\n✅ تصمیمات در {output_path} ذخیره شد.")
    return decisions


if __name__ == "__main__":
    build_decisions(
        validation_report_path="reports/titanic_validation_report.json",
        output_path="reports/titanic_decisions.json"
    )