# Repository Guidelines

## Project Structure & Module Organization

This repository contains a reproducible pipeline for the DM2026 Assignment 3 human activity recognition Kaggle task. Core reusable code lives in `src/dm2026_asg3/`: `data.py` loads and validates window CSVs, `features.py` builds feature sets, `modeling.py` handles model utilities, and `reporting.py` supports report outputs. Runnable experiment and submission entry points are in `scripts/`. Reports and figures are in `report/`, while local Kaggle data, generated features, model artifacts, and CSV submissions belong in `data/`, `artifacts/`, and `submissions/`.

## Build, Test, and Development Commands

Set up the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use `make smoke` to verify CLI imports and argument parsing. Use `make experiment` to run the default group-aware CV experiment against `data/raw`. Use `make submission` to train the final pipeline and write `submissions/submission_ensemble.csv`. For targeted runs, call scripts directly, for example `python scripts/run_experiment.py --help`.

## Coding Style & Naming Conventions

Use Python with 4-space indentation, `from __future__ import annotations`, and type hints for public helpers. Keep reusable logic in `src/dm2026_asg3/`; keep orchestration, sweeps, and one-off training jobs in `scripts/`. Name modules and scripts with `snake_case.py`, functions and variables with `snake_case`, constants with `UPPER_SNAKE_CASE`, and dataclasses/classes with `PascalCase`. Prefer `pathlib.Path` for filesystem paths and explicit seeds such as `--seed 2026` for reproducibility.

## Testing Guidelines

There is no dedicated test suite currently. Treat `make smoke` as the minimum preflight check after edits. For modeling or feature changes, run a small CV command such as `python scripts/run_experiment.py --data-dir data/raw --output-dir artifacts/fast --n-splits 5 --seed 2026 --fast` and record meaningful metric changes in `experiments.md` or `SUBMISSION_LOG.md`. If adding tests, place them under `tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines

Recent commits use concise, sentence-case summaries such as `Complete DM2026 Assignment 3 pipeline and report`. Keep commit messages imperative and scoped to one logical change. Pull requests should describe the changed pipeline behavior, list commands run, note metric or Kaggle score impact, and link any relevant assignment or competition context. Include updated report figures or screenshots when visual outputs change.

## Data, Artifacts, and Configuration

Do not commit raw Kaggle data, generated artifacts, virtual environments, compiled Python files, or generated submission CSVs; these paths are already covered by `.gitignore`. Keep required input data under `data/raw/` with the layout documented in `README.md`, and write reproducible outputs to a clearly named subdirectory under `artifacts/`.
