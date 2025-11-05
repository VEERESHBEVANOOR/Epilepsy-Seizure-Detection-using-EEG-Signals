#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

echo "🟢 Training model..."
python3 -m src.train
if [ $? -ne 0 ]; then
    echo "❌ Training failed! Exiting."
    exit 1
fi

echo "🟢 Launching Streamlit app..."
streamlit run app.py
