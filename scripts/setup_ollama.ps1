# Setup script for Ollama + local LLM model (Windows PowerShell)
# Run: .\scripts\setup_ollama.ps1

Write-Host ">>> Checking if Ollama is installed..."

$ollamaInstalled = Get-Command ollama -ErrorAction SilentlyContinue

if (-not $ollamaInstalled) {
    Write-Host ">>> Ollama not found. Please download and install it from:"
    Write-Host "    https://ollama.com/download/windows"
    Write-Host ">>> After installing, re-run this script."
    exit 1
} else {
    Write-Host ">>> Ollama is already installed."
}

Write-Host ">>> Checking if Ollama service is running..."
try {
    Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null
    Write-Host ">>> Ollama service is already running."
} catch {
    Write-Host ">>> Starting Ollama service in background..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

Write-Host ">>> Pulling model: mistral:latest ..."
ollama pull mistral:latest

Write-Host ">>> (Optional) Pulling alternative model llama3:8b ..."
# ollama pull llama3:8b

Write-Host ">>> Quick test..."
ollama run mistral:latest "Reply with just: OK"

Write-Host ">>> Ollama API is available at http://localhost:11434"
