# Reports

Files created during pipeline runs are generated outputs and are not committed to Git. MLflow stores the complete run history and artifacts.

- `examples/` contains a small mock example for checking the output structure.
- `final/` contains the final JSON, Markdown, and HTML evaluation reports after a complete run.

Generate the final reports with:

```bash
python scripts/generate_final_report.py
```

The included mock files are software-flow examples only and must not be reported as real model-performance results.
