"""
Layer 8.3 - Data Drift Detection

Detects distribution drift between a *reference* batch and a *current* batch
of tabular data, BEFORE the data reaches any downstream model (as argued in
Section 8.3 of the proposal: drift is caught at the ETL layer, not after model
deployment).

Design goals:
- Dataset-agnostic: works on any two DataFrames with the same schema, so it is
  reused across Titanic / Adult Income / OpenML / synthetic datasets.
- Numeric columns  -> Kolmogorov-Smirnov test (+ KL divergence on histograms).
- Categorical columns -> Population Stability Index (PSI) + Jensen-Shannon
  divergence on category frequencies.
- Identifier-like columns (Name, Ticket, Cabin, PassengerId, etc. - i.e. any
  non-numeric column whose unique-value ratio exceeds
  `ID_LIKE_RATIO_THRESHOLD`) are excluded from drift checks: PSI/KL are not
  statistically meaningful when almost every value is unique, and including
  them produces false-positive drift on every run regardless of whether real
  distributional drift occurred. This mirrors the same
  `high_cardinality_ratio` logic already used in decision_agent.py.
- Optional MLflow logging that degrades gracefully when no tracking server is
  running (so the module is usable offline / in CI).

A column is flagged as drifted when:
- numeric:      KS p-value < `ks_pvalue_threshold`   (default 0.05), or
- categorical:  PSI >= `psi_threshold`               (default 0.2, the common
                "significant population shift" cut-off).

Usage (compare two CSVs):
    python -m src.monitoring.drift_detection \
        --reference data/raw/titanic.csv \
        --current   data/processed/titanic_after_llm.csv

Usage (self-contained demo that *simulates* drift on a single dataset, useful
for the evaluation / defense when you don't have a second real batch yet):
    python -m src.monitoring.drift_detection --demo data/raw/titanic.csv
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# --- tunable thresholds (mirrors config/config.yaml where relevant) ---
KS_PVALUE_THRESHOLD = 0.05     # numeric: p < threshold => distributions differ
PSI_THRESHOLD = 0.20           # categorical: PSI >= 0.2 => significant shift
ID_LIKE_RATIO_THRESHOLD = 0.5  # columns with unique_ratio above this (e.g. Name,
                                # Ticket, Cabin, PassengerId) are identifier-like;
                                # PSI/KL are not statistically meaningful for them
                                # and produce false-positive drift, so they are
                                # excluded from drift checks (mirrors
                                # decision_agent.py's high_cardinality_ratio logic).
HIST_BINS = 20                 # bins used for KL divergence on numeric columns
_EPS = 1e-9                    # smoothing to avoid log(0) / divide-by-zero


# ---------------------------------------------------------------------------
# Divergence helpers
# ---------------------------------------------------------------------------
def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) for two discrete distributions (already summing to ~1)."""
    p = np.asarray(p, dtype=float) + _EPS
    q = np.asarray(q, dtype=float) + _EPS
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric Jensen-Shannon divergence in [0, 1] (log base 2)."""
    p = np.asarray(p, dtype=float) + _EPS
    q = np.asarray(q, dtype=float) + _EPS
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def _psi(ref: np.ndarray, cur: np.ndarray) -> float:
    """Population Stability Index between two frequency vectors."""
    ref = np.asarray(ref, dtype=float) + _EPS
    cur = np.asarray(cur, dtype=float) + _EPS
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


# ---------------------------------------------------------------------------
# Per-column drift checks
# ---------------------------------------------------------------------------
def _numeric_drift(ref: pd.Series, cur: pd.Series) -> Dict:
    ref_v = ref.dropna().to_numpy(dtype=float)
    cur_v = cur.dropna().to_numpy(dtype=float)

    if len(ref_v) == 0 or len(cur_v) == 0:
        return {"type": "numeric", "skipped": "empty_column", "drift": False}

    ks_stat, p_value = ks_2samp(ref_v, cur_v)

    # KL divergence over a shared histogram grid.
    lo = float(min(ref_v.min(), cur_v.min()))
    hi = float(max(ref_v.max(), cur_v.max()))
    if hi <= lo:
        hi = lo + 1.0  # degenerate (constant) column
    bins = np.linspace(lo, hi, HIST_BINS + 1)
    ref_hist, _ = np.histogram(ref_v, bins=bins)
    cur_hist, _ = np.histogram(cur_v, bins=bins)
    kl = _kl_divergence(ref_hist, cur_hist)

    drift = bool(p_value < KS_PVALUE_THRESHOLD)
    return {
        "type": "numeric",
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": round(float(p_value), 4),
        "kl_divergence": round(kl, 4),
        "ref_mean": round(float(np.mean(ref_v)), 4),
        "cur_mean": round(float(np.mean(cur_v)), 4),
        "drift": drift,
    }


def _categorical_drift(ref: pd.Series, cur: pd.Series) -> Dict:
    # Align both series onto the union of observed categories.
    categories = sorted(
        set(ref.dropna().astype(str)) | set(cur.dropna().astype(str))
    )
    if not categories:
        return {"type": "categorical", "skipped": "empty_column", "drift": False}

    ref_counts = ref.astype(str).value_counts()
    cur_counts = cur.astype(str).value_counts()
    ref_vec = np.array([ref_counts.get(c, 0) for c in categories], dtype=float)
    cur_vec = np.array([cur_counts.get(c, 0) for c in categories], dtype=float)

    psi = _psi(ref_vec, cur_vec)
    js = _js_divergence(ref_vec, cur_vec)

    drift = bool(psi >= PSI_THRESHOLD)
    return {
        "type": "categorical",
        "psi": round(psi, 4),
        "js_divergence": round(js, 4),
        "num_categories": len(categories),
        "drift": drift,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    dataset_name: str = "unknown",
    id_like_ratio_threshold: float = ID_LIKE_RATIO_THRESHOLD,
) -> Dict:
    """Compare two batches column-by-column and return a drift report dict.

    Columns whose unique-value ratio in the reference batch exceeds
    `id_like_ratio_threshold` (e.g. Name, Ticket, Cabin, PassengerId) are
    treated as identifier-like/free-text and excluded from drift checks:
    PSI and KL divergence are not statistically meaningful when almost every
    value is unique, and including them produces false-positive drift on
    every run regardless of whether real distributional drift occurred.
    """
    shared_cols = [c for c in reference.columns if c in current.columns]

    columns_report: Dict[str, Dict] = {}
    drifted: List[str] = []
    skipped_id_like: List[str] = []

    for col in shared_cols:
        ref_s, cur_s = reference[col], current[col]

        non_null_count = ref_s.notna().sum()
        unique_ratio = ref_s.nunique(dropna=True) / non_null_count if non_null_count else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(ref_s) and pd.api.types.is_numeric_dtype(cur_s)

        if not is_numeric and unique_ratio > id_like_ratio_threshold:
            columns_report[col] = {
                "type": "categorical",
                "skipped": "id_like_column",
                "unique_ratio": round(float(unique_ratio), 4),
                "drift": False,
            }
            skipped_id_like.append(col)
            continue

        if is_numeric:
            result = _numeric_drift(ref_s, cur_s)
        else:
            result = _categorical_drift(ref_s, cur_s)
        columns_report[col] = result
        if result.get("drift"):
            drifted.append(col)

    n_checked = len(shared_cols) - len(skipped_id_like)
    report = {
        "dataset": dataset_name,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "columns_checked": n_checked,
        "columns_skipped_id_like": skipped_id_like,
        "columns_drifted": len(drifted),
        "drift_share": round(len(drifted) / n_checked, 4) if n_checked else 0.0,
        "drift_detected": len(drifted) > 0,
        "drifted_columns": drifted,
        "columns": columns_report,
    }
    return report


def save_report(report: Dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def log_to_mlflow(
    report: Dict,
    tracking_uri: str = "http://localhost:5000",
    experiment: str = "etl_drift_detection",
    report_path: Optional[str] = None,
) -> bool:
    """Log drift metrics/artifacts to MLflow. Returns False (no raise) if the
    tracking server or the mlflow package is unavailable, so the pipeline keeps
    running offline."""
    try:
        import mlflow
    except ImportError:
        print("[drift] mlflow not installed - skipping MLflow logging.")
        return False

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=f"drift_{report['dataset']}"):
            mlflow.log_param("dataset", report["dataset"])
            mlflow.log_param("reference_rows", report["reference_rows"])
            mlflow.log_param("current_rows", report["current_rows"])
            mlflow.log_metric("columns_checked", report["columns_checked"])
            mlflow.log_metric("columns_drifted", report["columns_drifted"])
            mlflow.log_metric("drift_share", report["drift_share"])
            mlflow.log_metric("drift_detected", int(report["drift_detected"]))
            for col, res in report["columns"].items():
                if res.get("type") == "numeric" and "ks_p_value" in res:
                    mlflow.log_metric(f"ks_pvalue__{col}", res["ks_p_value"])
                elif res.get("type") == "categorical" and "psi" in res:
                    mlflow.log_metric(f"psi__{col}", res["psi"])
            if report_path and os.path.exists(report_path):
                mlflow.log_artifact(report_path)
        return True
    except Exception as exc:  # pragma: no cover - depends on server availability
        print(f"[drift] MLflow logging skipped ({type(exc).__name__}: {exc}).")
        return False


def _simulate_drifted_batch(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic 'current' batch with injected drift, for demos and the
    'Simulation Test' evaluation method named in Section 10 of the proposal.
    Shifts numeric columns and re-weights categorical columns."""
    rng = np.random.default_rng(seed)
    drifted = df.copy()
    for col in drifted.columns:
        s = drifted[col]
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            std = float(s.std(skipna=True)) or 1.0
            drifted[col] = s + rng.normal(1.5 * std, 0.5 * std, size=len(s))
        elif s.notna().any():
            # Bias sampling towards the most frequent category to shift the mix
            # hard enough to move PSI past its 0.2 significance threshold.
            top = s.dropna().astype(str).value_counts().idxmax()
            mask = rng.random(len(s)) < 0.65
            drifted.loc[mask, col] = top
    return drifted


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL-layer data drift detection (Layer 8.3)")
    parser.add_argument("--reference", help="Path to the reference-batch CSV.")
    parser.add_argument("--current", help="Path to the current-batch CSV.")
    parser.add_argument(
        "--demo",
        metavar="CSV",
        help="Run a self-contained demo: load this CSV, simulate a drifted "
             "batch from it, and report the drift.",
    )
    parser.add_argument("--name", default=None, help="Dataset name for the report.")
    parser.add_argument(
        "--out",
        default=None,
        help="Where to save the JSON report (default: reports/<name>_drift_report.json).",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args()

    if args.demo:
        name = args.name or os.path.splitext(os.path.basename(args.demo))[0]
        reference = pd.read_csv(args.demo)
        current = _simulate_drifted_batch(reference)
        print(f"[drift] DEMO mode: simulated a drifted batch from '{args.demo}'.")
    elif args.reference and args.current:
        name = args.name or os.path.splitext(os.path.basename(args.reference))[0]
        reference = pd.read_csv(args.reference)
        current = pd.read_csv(args.current)
    else:
        parser.error("Provide either --demo CSV, or both --reference and --current.")

    report = detect_drift(reference, current, dataset_name=name)
    out_path = args.out or os.path.join("reports", f"{name}_drift_report.json")
    save_report(report, out_path)

    print("\n=== DRIFT DETECTION REPORT ===")
    print(f"Dataset:          {report['dataset']}")
    print(f"Columns checked:  {report['columns_checked']}")
    if report.get('columns_skipped_id_like'):
        print(f"Columns skipped (id-like, e.g. Name/Ticket/Cabin): {report['columns_skipped_id_like']}")
    print(f"Columns drifted:  {report['columns_drifted']}  {report['drifted_columns']}")
    print(f"Drift detected:   {report['drift_detected']}")
    print(f"Report saved to:  {out_path}")

    if not args.no_mlflow:
        log_to_mlflow(report, report_path=out_path)


if __name__ == "__main__":
    main()