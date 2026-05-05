.PHONY: help smoke experiment submission

help:
	@printf "Targets:\n"
	@printf "  smoke       Check CLI imports\n"
	@printf "  experiment  Run group-aware CV\n"
	@printf "  submission  Train final ensemble and write Kaggle CSV\n"

smoke:
	python scripts/run_experiment.py --help
	python scripts/make_submission.py --help

experiment:
	python scripts/run_experiment.py --data-dir data/raw --output-dir artifacts --n-splits 5 --seed 2026

submission:
	python scripts/make_submission.py --data-dir data/raw --experiment-dir artifacts --output submissions/submission_ensemble.csv --seed 2026

