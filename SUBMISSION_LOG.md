# Kaggle Submission Log

Competition: `nycu-data-mining-assignment-3`

Daily submission limit: `3`

Rule for future work: every time a Kaggle submission is made, update this file immediately after `kaggle competitions submissions` reports the status/score. Record the exact CSV, description, public score, private score if available, status, SHA256, and any modeling notes needed to reproduce the file.

Last updated: `2026-05-05 17:49:38 CST`

| Kaggle timestamp | CSV | Description | Status | Public score | Private score | SHA256 | Notes |
|---|---|---|---|---:|---:|---|---|
| 2026-05-05 09:40:12.760000 | `submissions/submission_lgbm_leaves63_calibrated.csv` | `lgbm_leaves63 position calibrated` | COMPLETE | 0.8106 |  | `f78034a7e72d25ab4d52baebed585e130ad810e1f5cfae7469205008d5d5035e` | Current best. Full-train `lgbm_leaves63`, position context features, OOF global calibration weights from `artifacts/model_search_position_proba/results_position.csv`. |
| 2026-05-05 09:36:36.137000 | `submissions/submission_xgb_base_calibrated.csv` | `xgb_base position calibrated` | COMPLETE | 0.7933 |  | `fb300d038fb74d1c38d0658ec738afa74c0ddbc113a58b6547c963ad20cd4e79` | Full-train `xgb_base`, position context features, OOF global calibration weights from `artifacts/model_search_position_xgb/results_position.csv`. |
| 2026-05-05 09:28:50.077000 | `submissions/submission_selected.csv` | `selected calibrated HGB baseline` | COMPLETE | 0.7922 |  | `6da50ee25ce4311d4acc0a20eefb7ffb50a999c8ddda64213edda73626230632` | Same predictions as `submission_hgb_fast_calibrated.csv`; stable calibrated HGB baseline. |
