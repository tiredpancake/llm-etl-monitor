"""
Shared LLM client used by Agent Cleaner (Section 6.5 / 7.2 of the proposal).

Two implementations:
- OllamaClient: talks to a real local Ollama server (http://localhost:11434),
  same endpoint already used by src/llm_layer/semantic_cleaner.py.
- MockLLMClient: deterministic, offline stand-in used for unit-testing the
  agent orchestration logic (routing, confidence thresholds, MLflow logging)
  in environments without a GPU / Ollama running, e.g. CI.

Both implement the same `.generate(prompt) -> str` interface so the rest of
the agent code never has to know which one it's talking to.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral:latest",
                 temperature: float = 0.0, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False


class MockLLMClient:
    """Offline stand-in for Ollama. Returns a deterministic JSON-ish answer so
    the orchestrator, confidence-threshold logic, and MLflow logging can be
    exercised without any model running. NOT a substitute for the real model
    at evaluation time -- swap in OllamaClient for that.
    """

    def __init__(self, seed_note: str = "mock"):
        self.seed_note = seed_note

    def generate(self, prompt: str) -> str:
        # Pull "Allowed values: [...]" out of the prompt if present and just
        # pick one deterministically (hash-based) so repeated calls for the
        # same prompt are stable, mimicking temperature=0 behaviour.
        m = re.search(r"Allowed values:\s*\[([^\]]*)\]", prompt)
        digest = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        if m:
            options = [o.strip().strip("'\"") for o in m.group(1).split(",") if o.strip()]
            if options:
                value = options[digest % len(options)]
                confidence = 60 + (digest % 35)  # 60-94, varies per prompt
                return json.dumps({"value": value, "confidence": confidence})
        # Free-text fallback.
        return json.dumps({"value": "UNKNOWN", "confidence": 40})

    def is_available(self) -> bool:
        return True


def get_llm_client(cfg: dict, prefer_mock: bool = False):
    """Builds the appropriate client from config/config.yaml's `llm:` block.
    Falls back to MockLLMClient automatically if Ollama isn't reachable, so
    the pipeline can still be demoed/tested end-to-end offline."""
    llm_cfg = cfg.get("llm", {})
    agents_cfg = cfg.get("agents", {})
    client = OllamaClient(
        base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        model=llm_cfg.get("model_name", "mistral:latest"),
        temperature=llm_cfg.get("temperature", 0.0),
        timeout=agents_cfg.get("llm_timeout_seconds", 30),
    )
    if prefer_mock or not client.is_available():
        return MockLLMClient(), False
    return client, True
