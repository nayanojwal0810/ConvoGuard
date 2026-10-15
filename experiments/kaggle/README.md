# ConvoGuard Kaggle GPU Training Guide

This directory contains the self-contained GPU training notebook for Phase 2:
`02_bert_forecasting_baselines.ipynb`

## Overview
The notebook reproduces the published conversational derailment forecasting baselines from:
> **Dynamic Forecasting of Conversation Derailment**  
> Yova Kementchedjhieva and Anders Søgaard (EMNLP 2021)  
> Paper: https://aclanthology.org/2021.emnlp-main.624/  
> Code: https://github.com/YovaKem/forecasting_derailment

It implements:
1. **BERT-SC (Static):** Standard sequence classification on the complete pre-target conversation history.
2. **BERT-SC+ (Dynamic):** Prefix-trained sequence classification on successive conversational prefixes with forecast-horizon evaluation.

---

## Step-by-Step Kaggle Execution Guide

### 1. Create a Kaggle Notebook
1. Log in to [Kaggle](https://www.kaggle.com).
2. Click **Create** -> **New Notebook**.
3. Click **File** -> **Import Notebook** and upload `02_bert_forecasting_baselines.ipynb` (or copy/paste the cells).

### 2. Configure Hardware and Network Settings
In the right-hand panel (**Notebook Settings**):
1. **Accelerator:** Select **GPU P100** or **GPU T4 x 2**.
2. **Internet:** Turn **ON** (required to download pretrained `bert-base-uncased` weights from Hugging Face).
3. **Language:** Python 3.

### 3. Attach the CGA-Wiki Dataset
1. In the right-hand panel under **Input**, click **Add Input** (or **Upload Dataset**).
2. Upload or select the dataset containing:
   - `train_conversations.parquet`
   - `val_conversations.parquet`
3. The notebook will automatically search `/kaggle/input/` and locate the files.

### 4. Protocol Verification
- The notebook **only loads `train_conversations.parquet` and `val_conversations.parquet`**.
- The official test split is **never accessed or evaluated** during baseline model development and hyperparameter selection.
- All target attack comments are strictly excluded from input features.

### 5. Running the Notebook
- **Smoke-Test Mode:** In Cell 5, set `SMOKE_TEST = True` to run a 1-epoch test on a sub-sampled batch in under 2 minutes to confirm pipeline execution.
- **Full Baseline Training:** Keep `SMOKE_TEST = False` to train 3 epochs with linear warmup and validation checkpointing.
- Click **Run All** (or `Shift + Enter` cell by cell).

### 6. Downloading Outputs
Cell 11 packages all trained artifacts into `convoguard_phase2_bert_outputs.zip`:
- `checkpoints/bert_sc_static_best.pt`
- `checkpoints/bert_sc_dynamic_best.pt`
- `bert_baselines_validation_metrics.json`

Download the ZIP archive from the `/kaggle/working/` output directory in Kaggle's right-hand file viewer.
