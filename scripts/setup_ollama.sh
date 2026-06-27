#!/bin/bash
# Setup script for Ollama + local LLM model (Linux/Mac)
# Windows users: use scripts/setup_ollama.ps1 instead

set -e

echo ">>> Checking if Ollama is installed..."
if ! command -v ollama &> /dev/null; then
    echo ">>> Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo ">>> Ollama is already installed."
fi

echo ">>> Starting Ollama service (background)..."
ollama serve &
sleep 3

echo ">>> Pulling model: mistral:latest ..."
ollama pull mistral:latest

echo ">>> (Optional) Pulling alternative model llama3:8b ..."
# ollama pull llama3:8b

echo ">>> Quick model test..."
ollama run mistral:latest "Reply with just: OK"

echo ">>> Ollama API is available at http://localhost:11434"
