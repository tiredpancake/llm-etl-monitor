# Setup script for MLflow tracking server (Windows PowerShell)
# Run: .\scripts\setup_mlflow.ps1
# Prerequisite: pip install -r requirements.txt (in active venv)

$MlflowDir = Join-Path $PSScriptRoot "..\mlflow_data"
New-Item -ItemType Directory -Force -Path $MlflowDir | Out-Null

Write-Host ">>> Starting MLflow Tracking Server at http://localhost:5000 ..."

mlflow server `
  --backend-store-uri "sqlite:///$MlflowDir/mlflow.db" `
  --default-artifact-root "$MlflowDir/artifacts" `
  --host 0.0.0.0 `
  --port 5000
