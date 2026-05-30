.PHONY: help smoke experiment submission final repro

help:
	@printf "Targets:\n"
	@printf "  smoke       Check CLI imports\n"
	@printf "  experiment  Run group-aware CV\n"
	@printf "  submission  Train final ensemble and write Kaggle CSV\n"
	@printf "  final       Reproduce the rank-1 0.827 submissions from committed artifacts\n"
	@printf "  repro       Alias for 'final' (graded byte-identical path)\n"

smoke:
	python scripts/run_experiment.py --help
	python scripts/make_submission.py --help

experiment:
	python scripts/run_experiment.py --data-dir data/raw --output-dir artifacts --n-splits 5 --seed 2026

submission:
	python scripts/make_submission.py --data-dir data/raw --experiment-dir artifacts --output submissions/submission_ensemble.csv --seed 2026

# Reproduce the graded rank-1 submissions byte-for-byte from committed artifacts.
# synth_agg_consensus.csv -> public F1 0.8270 (rank 1); synth_safe_flip37.csv -> 0.8262.
final:
	python scripts/build_synth_candidates.py
	@echo "---- SHA-256 (expect 54075fcb... and 9921d49d...) ----"
	@sha256sum submissions/synth_agg_consensus.csv submissions/synth_safe_flip37.csv

repro: final
