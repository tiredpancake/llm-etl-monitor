#!/bin/bash
# Setup script for MLflow tracking server (Linux/Mac)
# Prerequisite: pip install -r requirements.txt

set -e

MLFLOW_DIR="$(dirname "$0")/../mlflow_data"
mkdir -p "$MLFLOW_DIR"

echo ">>> Starting MLflow Tracking Server at http://localhost:5000 ..."
mlflow server \
  --backend-store-uri "sqlite:///${MLFLOW_DIR}/mlflow.db" \
  --default-artifact-root "${MLFLOW_DIR}/artifacts" \
  --host 0.0.0.0 \
  --port 5000
