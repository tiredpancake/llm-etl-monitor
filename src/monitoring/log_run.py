import mlflow
import json
import pandas as pd
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("titanic_etl_pipeline")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_pipeline_run():
    with mlflow.start_run(run_name="titanic_run_1"):

        mlflow.log_param("dataset", "titanic")
        mlflow.log_param("llm_model", "mistral")
        mlflow.log_param("routing_policy", "hybrid_rule_llm")

        raw_df = pd.read_csv("data/raw/titanic.csv")
        total_cells_before = raw_df.size
        missing_before = raw_df.isnull().sum().sum()
        completeness_before = 100 * (1 - missing_before / total_cells_before)
        mlflow.log_metric("completeness_before", completeness_before)

        final_df = pd.read_csv("data/processed/titanic_after_llm.csv")
        total_cells_after = final_df.size
        missing_after = final_df.isnull().sum().sum()
        completeness_after = 100 * (1 - missing_after / total_cells_after)
        mlflow.log_metric("completeness_after", completeness_after)

        dqis = 100 * (completeness_after - completeness_before) / completeness_before
        mlflow.log_metric("dqis_score", dqis)

        duplicate_count = raw_df.duplicated(subset=["PassengerId"]).sum()
        duplicate_rate = 100 * duplicate_count / len(raw_df)
        mlflow.log_metric("duplicate_rate", duplicate_rate)

        reviewer_report = load_json("reports/titanic_reviewer_report.json")
        total_llm_decisions = len(reviewer_report)
        accepted_decisions = sum(1 for r in reviewer_report if r["accepted_by_reviewer"])
        error_correction_rate = 100 * accepted_decisions / total_llm_decisions if total_llm_decisions > 0 else 0
        mlflow.log_metric("llm_total_decisions", total_llm_decisions)
        mlflow.log_metric("llm_accepted_decisions", accepted_decisions)
        mlflow.log_metric("error_correction_rate", error_correction_rate)

        mlflow.log_artifact("reports/titanic_validation_report.json")
        mlflow.log_artifact("reports/titanic_decisions.json")
        mlflow.log_artifact("reports/titanic_llm_log.json")
        mlflow.log_artifact("reports/titanic_reviewer_report.json")
        mlflow.log_artifact("data/processed/titanic_after_llm.csv")

        print("✅ اجرا با موفقیت در MLflow ثبت شد.")
        print(f"Completeness قبل: {completeness_before:.2f}%")
        print(f"Completeness بعد: {completeness_after:.2f}%")
        print(f"DQIS: {dqis:.2f}%")
        print(f"Error Correction Rate: {error_correction_rate:.2f}%")


if __name__ == "__main__":
    log_pipeline_run()