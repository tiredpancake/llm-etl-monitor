"""
Phase 1 infrastructure connectivity test:
1. Is the Ollama API reachable?
2. Is the MLflow Tracking Server reachable?

Run: python scripts/test_infrastructure.py
"""
import sys
import requests
import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_ollama(base_url: str, model_name: str) -> bool:
    print(f"\n[1/2] Testing connection to Ollama ({base_url}) ...")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"   OK  Ollama is reachable. Installed models: {models}")
        if model_name not in models:
            print(f"   WARN Model '{model_name}' not found. Run: ollama pull {model_name}")
            return False
        return True
    except Exception as e:
        print(f"   FAIL Could not connect to Ollama: {e}")
        print("   Hint: make sure the Ollama service is running ('ollama serve').")
        return False


def test_mlflow(tracking_uri: str) -> bool:
    print(f"\n[2/2] Testing connection to MLflow ({tracking_uri}) ...")
    try:
        resp = requests.get(tracking_uri, timeout=5)
        if resp.status_code == 200:
            print("   OK  MLflow Tracking Server is reachable.")
            return True
        print(f"   FAIL Unexpected response: {resp.status_code}")
        return False
    except Exception as e:
        print(f"   FAIL Could not connect to MLflow: {e}")
        print("   Hint: run scripts/setup_mlflow.ps1 (Windows) or setup_mlflow.sh (Linux/Mac).")
        return False


if __name__ == "__main__":
    config = load_config()
    ok_ollama = test_ollama(config["llm"]["base_url"], config["llm"]["model_name"])
    ok_mlflow = test_mlflow(config["mlflow"]["tracking_uri"])

    print("\n" + "=" * 50)
    if ok_ollama and ok_mlflow:
        print("SUCCESS: Phase 1 infrastructure is ready. You can move on to Phase 2.")
        sys.exit(0)
    else:
        print("FAILED: Infrastructure is not fully ready. Fix the issues above.")
        sys.exit(1)
