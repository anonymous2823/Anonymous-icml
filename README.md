# Scientific Claim Unlearning

Pipeline for building scientific question-answer datasets and running unlearning experiments.

This repository contains the code used to construct and filter two related dataset families:

- Forget set: claim-level questions extracted from source papers.
- Retain set: questions from reference papers that should remain available.

The code also includes export utilities, synchronization steps, evaluation helpers, and experiment scripts for multiple unlearning methods.

## Repository Layout

- `SciUnlearn/` - main Python package with the data pipeline and utilities.
- `SciUnlearn/export_main.py` - builds the forget set export.
- `SciUnlearn/retain_external_main.py` - builds the external retain pipeline.
- `SciUnlearn/retain_internal_main.py` - builds the internal retain pipeline.
- `SciUnlearn/dataset_export/` - exports JSON outputs to Parquet and JSONL.
- `SciUnlearn/dataset_sync/` - synchronization between forget and retain sets.
- `SciUnlearn/evaluation/` - filtering, coverage, and survival checks.
- `SciUnlearn/semantic_scholar/` - Semantic Scholar API helpers.
- `SciUnlearn/llm_client/` - Azure GPT client utilities.
- `Experiments/` - unlearning and evaluation experiment scripts.
- `Experiments/data/` - example exported datasets in JSONL and Parquet format.

## Requirements

- Python 3.10+ recommended.
- Install dependencies with:

```bash
pip install -r requirements.txt
```

Some workflows also depend on model weights and API access:

- Azure OpenAI credentials for GPT-based generation and filtering.
- An optional Semantic Scholar API key for metadata fetching.
- OLMo model access for validation steps.

## Environment Variables

The main pipelines expect these variables to be set:

- `AZURE_API_KEY`
- `AZURE_API_BASE`
- `AZURE_API_VERSION`

Optional configuration used by some helpers:

- `S2_API_KEY`

Create a `.env` file or set them in your shell before running the scripts.

## Workflow Overview

The typical pipeline is:

1. Collect and process source papers.
2. Generate claim-level QA for the forget set.
3. Build retain sets from related papers.
4. Synchronize the forget and retain outputs.
5. Export the final datasets to JSONL and Parquet.

## Running the Pipelines

Run these scripts from the repository root:

```bash
python SciUnlearn/export_main.py
python SciUnlearn/retain_external_main.py
python SciUnlearn/retain_internal_main.py
```

Each script reads the corresponding JSON inputs, applies validation and filtering, and writes updated outputs in the repository root or in `SciUnlearn/export_data/` depending on the stage.

## Main Outputs

Common output files include:

- `claims_output_forget.json`
- `retain_paper_claims.json`
- `retain_set_internal.json`
- `forget_common.json`
- `retain_external_common.json`
- `retain_internal_common.json`
- `export_data/`

## Experiments

The `Experiments/` folder contains scripts for running unlearning methods such as GA, GD, NPO, RMU, and SIMNPO, plus evaluation helpers for LoRA-based checkpoints and downstream scoring.

## Configuration

Most runtime behavior is controlled from `SciUnlearn/config.py` through the `AppConfig` dataclass. That file defines:

- input and output paths,
- model names,
- validation flags,
- pruning behavior,
- export names,
- and threshold settings for similarity, coverage, and OLMo agreement.

If you want to change the pipeline without editing code, start there.

## Notes

- The repository contains generated data artifacts as well as source code.
- Some scripts are designed to be run only after earlier stages have produced their inputs.
- Large model validation can be expensive, so make sure the required API and model access is available before running full runs.
