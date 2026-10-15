# ConvoGuard

Conversation Guardian: early forecasting of personal attacks in Wikipedia Talk-page discussions.

This repository is being initialized for a fresh, evidence-driven implementation. The implementation plan, evaluation protocol, trained-model artifacts, NoSQL benchmark outputs, and application will be added in controlled milestones. No prior implementation or reported metric is assumed valid until reproduced.

## Project goals

- Forecast future discussion derailment from only the observed conversation prefix.
- Compare published baselines and a proposed interaction-aware model.
- Evaluate MongoDB data layouts and Redis cache policies on measured workloads.
- Deliver one integrated Streamlit application and separate Sequential Modeling and NoSQL reports.

## Status

Phase 1 completed: Clean foundation and dataset verification.
- Reusable dataset loading and schema validation pipeline established.
- Automated tests and linting operational.
- Cryptographic provenance and statistical audit executed on the verified Wikipedia Talk-page dataset.
- Models, persistence layers (MongoDB/Redis), and application UI are scheduled for subsequent phases.

## Project Structure

```
ConvoGuard/
├── src/
│   └── convoguard/
│       ├── __init__.py
│       ├── config.py
│       └── data/
│           ├── __init__.py
│           ├── loader.py
│           └── audit.py
├── scripts/
│   └── audit_dataset.py
├── tests/
│   └── test_data_pipeline.py
├── experiments/
│   ├── kaggle/
│   └── colab/
├── artifacts/
│   └── data_audit/
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

Or standard editable install:

```bash
pip install -e .
```

### 2. Dataset Configuration
Configure your local dataset root in `.env` or point directly via CLI arguments:

```bash
cp .env.example .env
```

Ensure `DATA_ROOT` points to the directory containing `train_conversations.parquet`, `val_conversations.parquet`, and `test_conversations.parquet`.

## Dataset Verification and Audit

Run the audit script to verify dataset provenance, schema integrity, and label distributions:

```bash
python scripts/audit_dataset.py --data-root "data/raw" --output "artifacts/data_audit/summary.json"
```

The audit calculates:
- File inventory, cryptographic SHA-256 provenance hashes, shapes, and schemas.
- Distinct conversation-level (`has_attack`) and comment-level (`is_comment_attack`) distributions.
- Monotonic sequence ordering and chronological integrity.
- Cross-split conversation and utterance leakage verification.
- Observation prefix lengths prior to annotated attack turns.

Audit outputs are saved locally under `artifacts/data_audit/` and excluded from git tracking.

## Testing and Linting

Run automated unit and integration tests using synthetic test fixtures:

```bash
python -m pytest -v
```

Run code quality and style linting:

```bash
python -m ruff check .
```
