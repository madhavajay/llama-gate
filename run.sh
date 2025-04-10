#!/bin/sh
# ollama pull llama3.1:8b
uv venv
uv pip install -r requirements.txt
uv run app.py
