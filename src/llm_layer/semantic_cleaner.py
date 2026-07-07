import requests
import json
import pandas as pd
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


def call_ollama(prompt, model=MODEL_NAME, temperature=0.1):
    """یک درخواست ساده به Ollama می‌فرستد و متن پاسخ را برمی‌گرداند."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        },
        timeout=120
    )
    if response.status_code != 200:
        print("متن خطای سرور Ollama:", response.text)
    response.raise_for_status()
    return response.json()["response"]

def build_embarked_prompt(row):
    """پرامپت Few-Shot برای پیشنهاد مقدار Embarked گمشده."""
    prompt = f"""تو یک دستیار پاکسازی داده هستی. باید مقدار گمشده‌ی ستون Embarked (بندر سوار شدن مسافر) را حدس بزنی.
مقادیر ممکن فقط این سه حرف هستند: S (Southampton), C (Cherbourg), Q (Queenstown).

مثال ۱:
Pclass: 1, Fare: 80.0, Sex: female -> پاسخ: S

مثال ۲:
Pclass: 3, Fare: 7.5, Sex: male -> پاسخ: S

حالا این رکورد را بررسی کن و فقط یک حرف (S یا C یا Q) به‌عنوان پاسخ بده، بدون هیچ توضیح اضافه:
Pclass: {row['Pclass']}, Fare: {row['Fare']}, Sex: {row['Sex']}
پاسخ:"""
    return prompt


def clean_value(raw_response, valid_values):
    """خروجی مدل را پاکسازی می‌کند و فقط یک مقدار معتبر برمی‌گرداند."""
    raw_response = raw_response.strip().upper()
    for v in valid_values:
        if v in raw_response:
            return v
    return None  


def clean_missing_embarked(raw_data_path, output_data_path, log_path):
    df = pd.read_csv(raw_data_path)
    missing_mask = df["Embarked"].isnull()
    missing_indices = df[missing_mask].index.tolist()

    print(f"تعداد رکوردهای دارای Embarked خالی: {len(missing_indices)}")

    log = []

    for idx in missing_indices:
        row = df.loc[idx]
        prompt = build_embarked_prompt(row)
        raw_response = call_ollama(prompt)
        predicted = clean_value(raw_response, valid_values=["S", "C", "Q"])

        # اگر مدل پاسخ معتبر نداد، fallback به mode (رایج‌ترین مقدار)
        if predicted is None:
            predicted = df["Embarked"].mode()[0]
            source = "fallback_mode"
        else:
            source = "llm"

        df.at[idx, "Embarked"] = predicted

        log.append({
            "row_index": int(idx),
            "Pclass": int(row["Pclass"]),
            "Fare": float(row["Fare"]),
            "Sex": row["Sex"],
            "raw_llm_response": raw_response.strip(),
            "predicted_embarked": predicted,
            "source": source
        })

        print(f"ردیف {idx}: پیش‌بینی = {predicted} (منبع: {source})")

    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df.to_csv(output_data_path, index=False)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"\n✅ داده نهایی در {output_data_path} ذخیره شد.")
    print(f"✅ لاگ تصمیمات LLM در {log_path} ذخیره شد.")

    return df, log



if __name__ == "__main__":
    clean_missing_embarked(
        raw_data_path="data/processed/titanic_after_rules.csv",
        output_data_path="data/processed/titanic_after_llm.csv",
        log_path="reports/titanic_llm_log.json"
    )