import pandas as pd
import json
import os

# ۱. خواندن دیتاست
df = pd.read_csv("data/raw/titanic.csv")

report = {"dataset": "titanic", "rules": []}

def add_rule_result(rule_name, column, success, bad_indices):
    report["rules"].append({
        "rule": rule_name,
        "column": column,
        "success": bool(success),
        "num_failed": len(bad_indices),
        "failed_row_indices": bad_indices[:20]  # فقط نمونه، همه رو ذخیره نمی‌کنیم
    })
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"[{status}] {rule_name} on '{column}' -> {len(bad_indices)} bad rows")

# قانون ۱: PassengerId باید یکتا باشد
dupes = df[df.duplicated(subset=["PassengerId"], keep=False)].index.tolist()
add_rule_result("uniqueness", "PassengerId", len(dupes) == 0, dupes)

# قانون ۲: Pclass نباید خالی باشد
nulls_pclass = df[df["Pclass"].isnull()].index.tolist()
add_rule_result("not_null", "Pclass", len(nulls_pclass) == 0, nulls_pclass)

# قانون ۳: Sex نباید خالی باشد
nulls_sex = df[df["Sex"].isnull()].index.tolist()
add_rule_result("not_null", "Sex", len(nulls_sex) == 0, nulls_sex)

# قانون ۴: Age باید بین ۰ تا ۱۲۰ باشد (اگر مقدار داشته باشد)
invalid_age = df[(df["Age"].notnull()) & ((df["Age"] < 0) | (df["Age"] > 120))].index.tolist()
add_rule_result("range_check", "Age", len(invalid_age) == 0, invalid_age)

# قانون ۵: مقادیر گمشده در Age (اینها نیاز به تصمیم‌گیری دارند - رد به لایه بعد)
missing_age = df[df["Age"].isnull()].index.tolist()
add_rule_result("missing_values", "Age", len(missing_age) == 0, missing_age)

# ۲. ذخیره گزارش برای استفاده در Decision Engine (مرحله بعد)
os.makedirs("reports", exist_ok=True)
with open("reports/titanic_validation_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n✅ گزارش کامل در reports/titanic_validation_report.json ذخیره شد.")