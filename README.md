# ConvoGuard

Conversation Guardian: early forecasting of personal attacks in Wikipedia Talk-page discussions.

This repository is being initialized for a fresh, evidence-driven implementation. The implementation plan, evaluation protocol, trained-model artifacts, NoSQL benchmark outputs, and application will be added in controlled milestones. No prior implementation or reported metric is assumed valid until reproduced.

## Project goals

- Forecast future discussion derailment from only the observed conversation prefix.
- Compare published baselines and a proposed interaction-aware model.
- Evaluate MongoDB data layouts and Redis cache policies on measured workloads.
- Deliver one integrated Streamlit application and separate Sequential Modeling and NoSQL reports.

## Status

Phase 1 & Phase 2 completed:
- **Phase 1 (Clean Foundation & Verification):** Reusable dataset loader, strict schema verification, SHA-256 cryptographic provenance, and automated testing operational.
- **Phase 2 (Forecasting Dataset & Baselines):**
  - Contract validation locking pre-target observation histories (strictly excluding terminal target comments).
  - Static (full history) and dynamic (prefix sequences) dataset generators.
  - CPU Baselines evaluated on validation split: Class-Prior, Static Word+Char TF-IDF + Logistic Regression, Dynamic TF-IDF + Logistic Regression.
  - Published BERT-SC and BERT-SC+ models implemented with turn-preserving recursive truncation (Kementchedjhieva & Søgaard, EMNLP 2021).
  - Forecast-horizon evaluation framework quantifying early warnings and false alert rates.
  - Kaggle GPU training notebook and execution guide prepared under `experiments/kaggle/`.
  - The official test split was strictly untouched and excluded from all evaluations.

## Project Structure

```
ConvoGuard/
├── src/
│   └── convoguard/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── audit.py
│       │   └── forecasting.py       # Pre-target static & dynamic example builders
│       └── models/
│           ├── __init__.py
│           ├── baselines.py          # Class-prior, Static & Dynamic TF-IDF + LR
│           ├── bert.py               # Published BERT-SC & BERT-SC+ architectures
│           └── evaluation.py         # Metrics, threshold tuning, and forecast-horizon
├── scripts/
│   ├── audit_dataset.py              # Provenance and integrity audit CLI
│   └── run_cpu_baselines.py          # CPU baselines execution on train/val
├── tests/
│   ├── test_data_pipeline.py         # 14 data integrity tests
│   └── test_forecasting_pipeline.py  # 11 forecasting & horizon tests
├── experiments/
│   ├── kaggle/
│   │   ├── 02_bert_forecasting_baselines.ipynb  # Self-contained Kaggle GPU notebook
│   │   └── README.md                            # Beginner-friendly Kaggle guide
│   └── colab/
├── artifacts/
│   ├── data_audit/
│   └── benchmarks/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Setup and Installation

### 1. Environment Installation
Install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

For GPU/transformer training dependencies:

```bash
pip install -e ".[dev,bert]"
```

### 2. Dataset Configuration
Configure your local dataset root in `.env` or point directly via CLI arguments:

```bash
cp .env.example .env
```

Ensure `DATA_ROOT` points to the directory containing `train_conversations.parquet`, `val_conversations.parquet`, and `test_conversations.parquet`.

## Running the CPU Baselines

Run the class-prior, static TF-IDF, and dynamic TF-IDF baselines on train and validation data:

```bash
python scripts/run_cpu_baselines.py --data-root "data/raw" --output "artifacts/benchmarks/cpu_baselines_metrics.json"
```

Evaluated models:
1. **Class-Prior Baseline:** Majority / empirical training class rate.
2. **Static TF-IDF + Logistic Regression:** Word (1-2) and Char (3-5) n-grams on complete pre-target history.
3. **Dynamic TF-IDF + Logistic Regression:** Word and Char n-grams on successive prefix sequences.
4. **Forecast-Horizon Analysis:** Measures turns ahead ($H = N - k$) at which positive conversations are alerted.

Outputs are saved locally to `artifacts/benchmarks/` and excluded from git tracking.

## Kaggle GPU Training for BERT-SC Baselines

Follow the step-by-step instructions in [`experiments/kaggle/README.md`](experiments/kaggle/README.md):
- Open `experiments/kaggle/02_bert_forecasting_baselines.ipynb` in Kaggle.
- Attach the CGA-Wiki dataset and enable a GPU accelerator (P100 or T4 x 2) with internet access.
- Trains both **BERT-SC** (static) and **BERT-SC+** (dynamic) with validation checkpointing and packages results into a downloadable ZIP archive.

## Testing and Linting

Run the complete test suite (25 automated tests):

```bash
python -m pytest -v
```

Run code quality and style linting:

```bash
python -m ruff check .
```
