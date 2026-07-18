"""
Generic, config-driven ETL cleaning + monitoring pipeline.

This is the dataset-agnostic version of the per-dataset Titanic scripts. It
reads a dataset entry from `config/config.yaml` and runs the full flow:

    ingest -> profile(before) -> rule-based clean -> profile(after)
           -> data-quality metrics -> drift detection -> save reports
           -> MLflow logging (degrades gracefully if no server)

Only the *deterministic* (Rule-Based) half of the Hybrid Router is executed
here: whitespace/format normalization, missing-value imputation and exact
duplicate removal. The semantic (LLM) correction step is intentionally left as
a hook (`--with-llm` is a no-op placeholder) so this runs with no GPU / no
Ollama. Run `src/llm_layer/semantic_cleaner.py` separately for the LLM pass.

Usage:
    python -m src.pipelines.run_pipeline --dataset adult_income
    python -m src.pipelines.run_pipeline --dataset titanic --no-mlflow
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import pandas as pd
import yaml

# Reuse the generic profiler (Layer 6.2) and drift detector (Layer 8.3).
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from profiling.profile_data import profile  # noqa: E402
from monitoring.drift_detection import detect_drift, save_report as save_drift_report  # noqa: E402

DEFAULT_NA_TOKENS = ["?", "NA", "N/A", "null", "None", "", "-"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Layer 6.1 - Ingestion (generic)
# ---------------------------------------------------------------------------
def ingest(path: str, na_tokens: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Trim surrounding whitespace on every string cell (fixes "noise" like
    # " Bachelors" -> "Bachelors") *before* deciding what counts as missing.
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()
    # Normalize the various missing-value markers to real NaN.
    df = df.replace({t: pd.NA for t in na_tokens})
    return df


# ---------------------------------------------------------------------------
# Layers 6.3 + 6.7 - Rule-based validation & transformation (generic)
# ---------------------------------------------------------------------------
def rule_based_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict]:
    """Deterministic cleaning: drop exact duplicates, impute missing values
    (numeric -> median, categorical -> mode). Returns the cleaned frame and a
    log describing every action, for auditability (Section 8.2)."""
    log: Dict = {"duplicates_removed": 0, "imputations": []}
    cleaned = df.copy()

    # 1) Exact duplicate rows.
    dup_mask = cleaned.duplicated(keep="first")
    log["duplicates_removed"] = int(dup_mask.sum())
    cleaned = cleaned[~dup_mask].reset_index(drop=True)

    # 2) Missing-value imputation.
    for col in cleaned.columns:
        missing = int(cleaned[col].isna().sum())
        if missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            fill = cleaned[col].median()
            strategy = "median"
        else:
            mode = cleaned[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else ""
            strategy = "mode"
        cleaned[col] = cleaned[col].fillna(fill)
        log["imputations"].append({
            "column": col,
            "strategy": strategy,
            "filled_count": missing,
            "fill_value": str(fill),
        })
    return cleaned, log


# ---------------------------------------------------------------------------
# Layer 6.9 - Data-quality metrics
# ---------------------------------------------------------------------------
def _completeness(df: pd.DataFrame) -> float:
    total = df.size
    return 100.0 * (1 - df.isna().sum().sum() / total) if total else 100.0


def quality_metrics(before: pd.DataFrame, after: pd.DataFrame, dups_removed: int) -> Dict:
    comp_before = _completeness(before)
    comp_after = _completeness(after)
    dqis = 100.0 * (comp_after - comp_before) / comp_before if comp_before else 0.0
    dup_rate_before = 100.0 * dups_removed / len(before) if len(before) else 0.0
    return {
        "completeness_before": round(comp_before, 4),
        "completeness_after": round(comp_after, 4),
        "dqis_score": round(dqis, 4),
        "duplicate_rate_before": round(dup_rate_before, 4),
        "duplicate_rate_after": round(100.0 * after.duplicated().sum() / len(after), 4) if len(after) else 0.0,
    }


# ---------------------------------------------------------------------------
# MLflow logging (graceful)
# ---------------------------------------------------------------------------
def log_to_mlflow(dataset: str, metrics: Dict, drift: Dict, cfg: dict,
                  report_paths: List[str]) -> bool:
    try:
        import mlflow
    except ImportError:
        print("[pipeline] mlflow not installed - skipping MLflow logging.")
        return False
    try:
        mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "http://localhost:5000"))
        mlflow.set_experiment(cfg.get("mlflow", {}).get("experiment_name", "llm-etl-monitoring"))
        with mlflow.start_run(run_name=f"{dataset}_pipeline"):
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("routing_policy", "hybrid_rule_llm")
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_metric("drift_columns_drifted", drift["columns_drifted"])
            mlflow.log_metric("drift_detected", int(drift["drift_detected"]))
            for p in report_paths:
                if os.path.exists(p):
                    mlflow.log_artifact(p)
        return True
    except Exception as exc:  # pragma: no cover - depends on server availability
        print(f"[pipeline] MLflow logging skipped ({type(exc).__name__}: {exc}).")
        return False


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(dataset: str, cfg: dict, use_mlflow: bool = True) -> Dict:
    ds_cfg = cfg["datasets"][dataset]
    path = ds_cfg["path"]
    na_tokens = ds_cfg.get("na_tokens", DEFAULT_NA_TOKENS)

    print(f"=== PIPELINE: {dataset} ===")
    raw = ingest(path, na_tokens)
    print(f"Ingested: {raw.shape[0]} rows x {raw.shape[1]} cols")

    profile_before = profile(raw)
    cleaned, clean_log = rule_based_clean(raw)
    profile_after = profile(cleaned)
    metrics = quality_metrics(raw, cleaned, clean_log["duplicates_removed"])

    # Drift detection (Layer 8.3): treat the data as two sequential batches
    # (first half = reference, second half = current) — the runtime-monitoring
    # scenario where batches arrive over time.
    mid = len(cleaned) // 2
    drift = detect_drift(cleaned.iloc[:mid], cleaned.iloc[mid:], dataset_name=dataset)

    # Persist artifacts.
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    paths = {
        "profile_before": f"reports/{dataset}_profile_before.json",
        "profile_after": f"reports/{dataset}_profile_after.json",
        "clean_log": f"reports/{dataset}_clean_log.json",
        "metrics": f"reports/{dataset}_metrics.json",
        "drift": f"reports/{dataset}_drift_report.json",
        "processed": f"data/processed/{dataset}_cleaned.csv",
    }
    for key, obj in (("profile_before", profile_before), ("profile_after", profile_after),
                     ("clean_log", clean_log), ("metrics", metrics)):
        with open(paths[key], "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    save_drift_report(drift, paths["drift"])
    cleaned.to_csv(paths["processed"], index=False)

    print(f"Duplicates removed:   {clean_log['duplicates_removed']}")
    print(f"Completeness:         {metrics['completeness_before']}% -> {metrics['completeness_after']}%")
    print(f"DQIS score:           {metrics['dqis_score']}%")
    print(f"Duplicate rate:       {metrics['duplicate_rate_before']}% -> {metrics['duplicate_rate_after']}%")
    print(f"Drift (batch split):  {drift['columns_drifted']}/{drift['columns_checked']} columns -> {drift['drifted_columns']}")
    print(f"Artifacts saved under reports/ and data/processed/")

    if use_mlflow:
        log_to_mlflow(dataset, metrics, drift, cfg,
                      [paths["metrics"], paths["drift"], paths["clean_log"], paths["processed"]])

    return {"metrics": metrics, "drift": drift, "clean_log": clean_log, "paths": paths}


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven ETL cleaning + monitoring pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset key in config/config.yaml (e.g. adult_income)")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset not in cfg.get("datasets", {}):
        parser.error(f"Unknown dataset '{args.dataset}'. Known: {list(cfg.get('datasets', {}))}")
    run(args.dataset, cfg, use_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
