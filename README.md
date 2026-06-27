# LLM-based Intelligent ETL Monitor — Setup Guide (Phase 1)

This repository is the project skeleton for "Design and Implementation of an
LLM-based Intelligent ETL Monitor for Tabular Data Cleaning and Integration".

## Folder structure

```
etl_llm_project/
├── config/                # Settings (config.yaml)
├── data/
│   ├── raw/                # Raw datasets (Titanic, Adult Income, OpenML Dirty, Synthetic)
│   └── processed/          # Cleaned output
├── src/
│   ├── ingestion/           # Layer 6.1 - Data ingestion
│   ├── profiling/            # Layer 6.2 - Data profiling
│   ├── validation/           # Layer 6.3 - Rule-based validation (Great Expectations)
│   ├── decision_engine/      # Layer 6.4 - Hybrid decision engine
│   ├── llm_layer/            # Layer 6.5 - Local large language model (Ollama)
│   ├── transformation/       # Layer 6.7 - Output merging and final transformation
│   ├── monitoring/           # Layer 6.8 - Monitoring and logging (MLflow)
│   ├── evaluation/           # Layer 6.9 - Data quality evaluation
│   └── agents/               # Profiler / Cleaner / Reviewer agents (ReAct)
├── scripts/
│   ├── setup_ollama.sh / .ps1   # Install Ollama + pull model
│   ├── setup_mlflow.sh / .ps1   # Start MLflow tracking server
│   └── test_infrastructure.py
├── notebooks/                # Exploratory analysis
├── reports/                  # Final evaluation reports
├── tests/
└── requirements.txt
```

## Installation steps

### 1. Create a Python virtual environment
```bash
cd etl_llm_project
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Install and start Ollama

**Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_ollama.ps1
```

**Linux/Mac:**
```bash
bash scripts/setup_ollama.sh
```

This installs Ollama, starts the service (port 11434), and pulls a local model.

> Hardware note: at least 8 GB of free RAM, ideally a GPU with 6+ GB VRAM.
> It also works without a GPU, just slower at inference.

### 3. Start MLflow (in a separate terminal)

**Windows (PowerShell):**
```powershell
.\scripts\setup_mlflow.ps1
```

**Linux/Mac:**
```bash
bash scripts/setup_mlflow.sh
```

Once running, the UI is available at http://localhost:5000

### 4. Test the infrastructure
```bash
python scripts/test_infrastructure.py
```
If both checks pass, Phase 1 is complete and you're ready for Phase 2 (dataset preparation).

## Progress so far

### Phase 1 - Infrastructure (done)
- Ollama running locally with `mistral:latest` (`http://localhost:11434`)
- MLflow Tracking Server running (`http://localhost:5000`)
- Verified via `scripts/test_infrastructure.py`

### Phase 2 - Dataset preparation (in progress, Titanic dataset)
- `data/raw/titanic.csv` downloaded (891 rows, 12 columns)
- `src/ingestion/load_data.py` - Layer 6.1, loads + standardizes column names
- `src/profiling/profile_data.py` - Layer 6.2, computes missing values /
  duplicates / per-column stats, saved as baseline to
  `reports/titanic_profile_before.json`
  - Result: 866 missing cells (8.1%) - mostly in `age` (19.87%),
    `cabin` (77.1%), `embarked` (0.22%). No duplicate rows.

### Next: rule-based validation (Layer 6.3, Great Expectations)

## Phase checklist
- [x] Phase 1: Infrastructure setup
- [~] Phase 2: Dataset preparation (Titanic done, 3 more datasets pending)
- [ ] Phase 3: Core ETL layer implementation
- [ ] Phase 4: Multi-agent architecture
- [ ] Phase 5: Monitoring and evaluation
- [ ] Phase 6: Final evaluation and report
