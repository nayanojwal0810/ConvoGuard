# ConvoGuard

Conversation Guardian: early forecasting of personal attacks in Wikipedia Talk-page discussions.

This repository contains a clean, evidence-driven implementation. Previous code and unverified metrics are not treated as valid baselines unless they are reproduced.

## Project goals

- Forecast future discussion derailment from only the observed conversation prefix.
- Compare published baselines and an interaction-aware model.
- Evaluate MongoDB data layouts and Redis cache policies on measured workloads.
- Deliver one integrated Streamlit application and separate Sequential Modeling and NoSQL reports.

## Current status

- **Phase 1 — Data foundation: complete.** Dataset loading, schema/label contract checks, audit tooling, and synthetic-fixture tests are in place.
- **Phase 2 — Forecasting pipeline and CPU baselines: implemented.** Static and dynamic pre-target examples, word/character TF-IDF baselines, threshold selection, and forecast-horizon calculations have been run on training and validation data.
- **BERT-SC and BERT-SC+ experiment: prepared, GPU execution pending.** The published methods have been implemented and a Kaggle notebook is available, but no successful GPU training run or BERT validation result has been demonstrated yet. Do not interpret this repository as having validated BERT performance.
- **Later phases: not started.** Graph modeling, NoSQL benchmarks, the integrated Streamlit application, public deployment, and subject reports remain pending.

### Evaluation integrity note

The current clean baseline script and Kaggle notebook are designed to load training and validation data only; they must not load the test split during model development. Test metrics were examined in an earlier project iteration, however, so the test split cannot honestly be described as never viewed across the full project history. Do not use it for further model selection, threshold tuning, or experimentation; disclose this historical exposure in any final evaluation.

## Project structure

```
ConvoGuard/
├── src/
│   └── convoguard/
│       ├── config.py
│       ├── data/
│       │   ├── loader.py
│       │   ├── audit.py
│       │   └── forecasting.py
│       └── models/
│           ├── __init__.py
│           ├── artifacts.py           # Checkpoint persistence and ZIP packaging
│           ├── baselines.py           # Class-prior and TF-IDF baselines
│           ├── bert.py                # Published BERT-SC and BERT-SC+ architectures
│           └── evaluation.py           # Classification and forecast-horizon metrics
├── scripts/
│   ├── audit_dataset.py
│   └── run_cpu_baselines.py
├── tests/
│   ├── test_data_pipeline.py
│   └── test_forecasting_pipeline.py
├── experiments/
│   ├── kaggle/
│   │   ├── 02_bert_forecasting_baselines.ipynb
│   │   └── README.md
│   └── colab/
├── artifacts/
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Setup and installation

Create and activate a Python environment, then install the project and development tools:

```bash
pip install -e ".[dev]"
```

The CPU baselines and evaluation utilities do not require PyTorch or Transformers. BERT model functionality uses the optional `bert` dependency group:

```bash
pip install -e ".[dev,bert]"
```

## Dataset configuration

Point `DATA_ROOT` to a directory containing the train, validation, and test Parquet files. Keep dataset files local; they are excluded from Git. For model development, the baseline runner and BERT notebook should load only the train and validation files.

## Run CPU baselines

```bash
python scripts/run_cpu_baselines.py --data-root "data/raw" --output "artifacts/benchmarks/cpu_baselines_metrics.json"
```

This fits the class-prior, static word-plus-character TF-IDF, and dynamic-prefix TF-IDF baselines on training data and reports validation classification and forecast-horizon diagnostics. Outputs are written locally under `artifacts/benchmarks/` and excluded from Git.

## Run the BERT experiment

Follow `experiments/kaggle/README.md` and open `experiments/kaggle/02_bert_forecasting_baselines.ipynb` in Kaggle. Attach the CGA-Wiki dataset, enable a GPU accelerator and internet access, and run the notebook. The experiment is not complete until BERT-SC and BERT-SC+ both finish, validation results are saved, and the generated ZIP is inspected. Do not load the test split.

## Tests and linting

Run the tests:

```bash
python -m pytest -v
```

Run Ruff:

```bash
python -m ruff check .
```
