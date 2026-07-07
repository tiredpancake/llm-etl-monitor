import pandas as pd
import json
import os

def apply_actions(raw_data_path, decisions_path, output_path):
    df = pd.read_csv(raw_data_path)

    with open(decisions_path, "r", encoding="utf-8") as f:
        decisions = json.load(f)

    log = []

    for decision in decisions["decisions"]:
        column = decision["column"]
        action = decision["action"]
        route = decision["route"]

        # فقط اکشن‌های مسیر rule را همینجا اجرا می‌کنیم
        # اکشن‌های llm در مرحله بعد با Ollama انجام می‌شوند
        if route != "rule":
            continue

        if action == "impute_median":
            median_value = df[column].median()
            missing_count = df[column].isnull().sum()
            df[column] = df[column].fillna(median_value)
            log.append({
                "column": column,
                "action": action,
                "filled_count": int(missing_count),
                "fill_value": float(median_value)
            })
            print(f"✅ ستون '{column}': {missing_count} مقدار گمشده با میانه ({median_value}) پر شد.")

        elif action == "auto_fix":
            # برای uniqueness / not_null / range_check که success=True بودند نیازی به اکشن نیست
            # این شاخه برای مواردی است که واقعاً auto_fix نیاز دارند (مثلاً حذف تکراری‌ها)
            pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ داده پردازش‌شده (جزئی) در {output_path} ذخیره شد.")
    print(f"تعداد مقادیر گمشده باقی‌مانده در کل داده: {df.isnull().sum().sum()}")

    return df, log


if __name__ == "__main__":
    apply_actions(
        raw_data_path="data/raw/titanic.csv",
        decisions_path="reports/titanic_decisions.json",
        output_path="data/processed/titanic_after_rules.csv"
    )