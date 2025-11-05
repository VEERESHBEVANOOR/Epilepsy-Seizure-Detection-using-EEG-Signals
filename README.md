# Epilepsy Seizure Detection — End‑to‑End (Train + Inference + UI)

This project lets you **train** a seizure detector on your EEG CSV files and **predict** on a new uploaded report (CSV). A simple Streamlit UI is included.

> ⚠️ **Input format expected**: CSV with numeric EEG samples.
> - Rows = time samples
> - Columns = EEG channels (e.g., `Fp1`, `Fp2`, `F3`, ...). Non‑numeric columns will be ignored.
> - Optional: a `label` column for training (`0` = non‑seizure, `1` = seizure).

## Quick Start

### 1) Create & activate env (example)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train with your data
Put your **CSV** files into `data/raw`. You have 3 options to provide labels:
- **Option A (per-row labels):** Each CSV contains a `label` column (0/1).
- **Option B (whole-file label):** Add a JSON file `data/labels.json` mapping filename to label, e.g.
  ```json
  {"patient1.csv": 0, "patient2.csv": 1}
  ```
- **Option C (filename hint):** If filename contains `seizure` -> 1 else 0.

Then run:
```bash
python src/train.py --raw_dir data/raw --out_model models/seizure_clf.joblib
```

### 3) Predict on a new report (CSV)
```bash
python src/predictor.py --model models/seizure_clf.joblib --input your_report.csv
```
This prints class and probability and writes a JSON to `data/processed/prediction.json`.

### 4) Streamlit UI
```bash
streamlit run app/app.py
```
Upload a CSV and see the prediction with confidence.

## Notes
- The included model is **not pre‑trained**. Train on your data for best accuracy.
- Feature engineering: per‑channel stats + Welch bandpowers (δ, θ, α, β, γ), Hjorth params, entropy.
- Classifier: GradientBoostingClassifier (scikit‑learn). You can switch to XGBoost/LightGBM if available.
- Sampling rate: Features are sampling-rate agnostic for stats; spectral features assume default `fs=256` Hz (set via `--fs`).
