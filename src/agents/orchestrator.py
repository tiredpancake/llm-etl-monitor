"""
Multi-agent orchestrator - Section 7.3 of the proposal ("جریان تعامل عامل‌ها").

Implements the exact interaction flow described there:

  1. Agent Profiler analyzes the data and produces a JSON quality report.
  2. Engine Decision picks the processing path based on that report.
  3. If semantic analysis is needed, Agent Cleaner activates and produces
     suggested corrections.
  4. Agent Reviewer validates the output and applies valid changes.
  5. Every step, decision, and result is logged to MLflow for audit,
     traceability, and performance analysis.

This is a hand-rolled ReAct-style loop (each agent emits Thought / Action /
Observation trace entries) rather than a LangChain/CrewAI AgentExecutor,
because LangChain's 1.x agent API changed its whole executor model and
CrewAI pulls in a large, version-fragile dependency tree; a direct
implementation of the same four-agent flow is more robust for a course
project and easier to grade/debug, while remaining a drop-in target to wrap
with LangChain/CrewAI tools later (each Agent class below is already a plain
`run(...)`-style callable, i.e. tool-shaped).

Usage:
    python -m src.agents.orchestrator --dataset titanic
    python -m src.agents.orchestrator --dataset titanic --mock-llm --no-mlflow
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import pandas as pd
import yaml

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipelines"))

from llm_client import get_llm_client, MockLLMClient  # noqa: E402
from profiler_agent import ProfilerAgent  # noqa: E402
from decision_agent import DecisionAgent  # noqa: E402
from cleaner_agent import CleanerAgent  # noqa: E402
from reviewer_agent import ReviewerAgent  # noqa: E402
from run_pipeline import ingest as generic_ingest, rule_based_clean, quality_metrics, load_config  # noqa: E402


def _completeness(df: pd.DataFrame) -> float:
    total = df.size
    return 100.0 * (1 - df.isna().sum().sum() / total) if total else 100.0


def run_agentic_pipeline(dataset: str, cfg: dict, use_mlflow: bool = True,
                          force_mock_llm: bool = False) -> Dict:
    ds_cfg = cfg["datasets"][dataset]
    na_tokens = ds_cfg.get("na_tokens", ["?", "NA", "N/A", "null", "None", "", "-"])

    full_trace: List[Dict] = []
    print(f"=== AGENTIC PIPELINE: {dataset} ===")

    raw = generic_ingest(ds_cfg["path"], na_tokens)
    print(f"Ingested: {raw.shape[0]} rows x {raw.shape[1]} cols")

    # --- Step 1: Agent Profiler ---
    profiler = ProfilerAgent()
    profile_report, trace = profiler.run(raw, dataset)
    full_trace += trace
    print(f"[{profiler.name}] flagged columns: {profile_report['_flagged_columns']}")

    # --- Step 2: Engine Decision ---
    decision_agent = DecisionAgent(
        high_cardinality_ratio=cfg.get("decision_engine", {}).get("high_cardinality_ratio", 0.5)
    )
    decisions, trace = decision_agent.run(raw, profile_report)
    full_trace += trace
    for d in decisions:
        print(f"[{decision_agent.name}] {d['column']} -> {d['route']}:{d['action']} ({d['reason']})")

    # --- Deterministic pass: apply the RULE-routed decisions immediately ---
    df = raw.copy()
    rule_cols = [d["column"] for d in decisions if d["route"] == "rule"]
    dup_mask = df.duplicated(keep="first")
    duplicates_removed = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)
    for col in rule_cols:
        if col not in df.columns or df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "")

    # --- Step 3: Agent Cleaner (only if any column was routed to LLM) ---
    llm_client, is_real_ollama = get_llm_client(cfg, prefer_mock=force_mock_llm)
    cleaner = CleanerAgent(
        llm_client=llm_client,
        max_rows_per_column=cfg.get("decision_engine", {}).get("max_rows_per_llm_batch", 20),
    )
    suggestions, trace = cleaner.run(df, decisions)
    full_trace += trace
    print(f"[{cleaner.name}] {len(suggestions)} suggestion(s) generated "
          f"(backend: {'Ollama' if is_real_ollama else 'MockLLMClient - offline'})")

    # --- Step 4: Agent Reviewer ---
    reviewer = ReviewerAgent(
        confidence_threshold=cfg.get("agents", {}).get("score_confidence_threshold", 70)
    )
    df, review_results, trace = reviewer.run(df, suggestions)
    full_trace += trace
    accepted = sum(1 for r in review_results if r["accepted"])
    print(f"[{reviewer.name}] {accepted}/{len(review_results)} corrections accepted")

    # Any LLM-routed column still missing after review (rejected/low-confidence)
    # falls back to rule-based mode imputation so completeness still resolves.
    for d in decisions:
        if d["route"] == "llm" and df[d["column"]].isna().sum() > 0:
            mode = df[d["column"]].mode(dropna=True)
            df[d["column"]] = df[d["column"]].fillna(mode.iloc[0] if not mode.empty else "")

    metrics = quality_metrics(raw, df, duplicates_removed)
    error_correction_rate = 100.0 * accepted / len(review_results) if review_results else None

    # --- Step 5: persist + MLflow logging (all decisions/results, per Section 7.2) ---
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    paths = {
        "profile": f"reports/{dataset}_agentic_profile.json",
        "decisions": f"reports/{dataset}_agentic_decisions.json",
        "suggestions": f"reports/{dataset}_agentic_suggestions.json",
        "review": f"reports/{dataset}_agentic_review.json",
        "trace": f"reports/{dataset}_agentic_trace.json",
        "metrics": f"reports/{dataset}_agentic_metrics.json",
        "processed": f"data/processed/{dataset}_agentic_cleaned.csv",
    }
    profile_report.pop("_flagged_columns", None)  # not JSON-critical, drop internal marker duplicate
    for key, obj in (("profile", profile_report), ("decisions", decisions),
                     ("suggestions", suggestions), ("review", review_results),
                     ("trace", full_trace), ("metrics", metrics)):
        with open(paths[key], "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    df.to_csv(paths["processed"], index=False)

    print(f"Duplicates removed:   {duplicates_removed}")
    print(f"Completeness:         {metrics['completeness_before']}% -> {metrics['completeness_after']}%")
    if error_correction_rate is not None:
        print(f"Error correction rate: {error_correction_rate:.2f}%")
    print("Artifacts saved under reports/ and data/processed/")

    if use_mlflow:
        _log_to_mlflow(dataset, cfg, metrics, decisions, review_results,
                       error_correction_rate, paths)

    return {
        "metrics": metrics,
        "decisions": decisions,
        "review_results": review_results,
        "error_correction_rate": error_correction_rate,
        "paths": paths,
    }


def _log_to_mlflow(dataset: str, cfg: dict, metrics: Dict, decisions: List[Dict],
                   review_results: List[Dict], error_correction_rate, paths: Dict) -> bool:
    try:
        import mlflow
    except ImportError:
        print("[orchestrator] mlflow not installed - skipping MLflow logging.")
        return False
    try:
        mlflow.set_tracking_uri(cfg.get("mlflow", {}).get("tracking_uri", "http://localhost:5000"))
        mlflow.set_experiment(cfg.get("mlflow", {}).get("experiment_name", "llm-etl-monitoring") + "-agentic")
        with mlflow.start_run(run_name=f"{dataset}_agentic_pipeline"):
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("routing_policy", "multi_agent_react")
            mlflow.log_param("llm_routed_columns", [d["column"] for d in decisions if d["route"] == "llm"])
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_metric("llm_total_decisions", len(review_results))
            mlflow.log_metric("llm_accepted_decisions", sum(1 for r in review_results if r["accepted"]))
            if error_correction_rate is not None:
                mlflow.log_metric("error_correction_rate", error_correction_rate)
            for p in paths.values():
                if os.path.exists(p):
                    mlflow.log_artifact(p)
        return True
    except Exception as exc:  # pragma: no cover - depends on server availability
        print(f"[orchestrator] MLflow logging skipped ({type(exc).__name__}: {exc}).")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent (Profiler/Decision/Cleaner/Reviewer) ETL pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset key in config/config.yaml")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--mock-llm", action="store_true",
                        help="Force the offline MockLLMClient even if Ollama is reachable "
                             "(useful for CI / testing without a GPU).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset not in cfg.get("datasets", {}):
        parser.error(f"Unknown dataset '{args.dataset}'. Known: {list(cfg.get('datasets', {}))}")

    run_agentic_pipeline(args.dataset, cfg, use_mlflow=not args.no_mlflow, force_mock_llm=args.mock_llm)


if __name__ == "__main__":
    main()
