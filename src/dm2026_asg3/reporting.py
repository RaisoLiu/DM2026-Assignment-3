from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_metrics_json(output_dir: Path, metrics: pd.DataFrame, metadata: dict) -> None:
    payload = {
        "metadata": metadata,
        "metrics": metrics.to_dict(orient="records"),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_report_tables(output_dir: Path, metrics: pd.DataFrame, ablation: pd.DataFrame | None = None) -> None:
    lines = ["# Report-ready Tables", ""]
    lines.extend(["## Cross-validation macro-F1", "", metrics.to_markdown(index=False, floatfmt=".5f"), ""])
    if ablation is not None and not ablation.empty:
        lines.extend(["## Ablation study", "", ablation.to_markdown(index=False, floatfmt=".5f"), ""])
    lines.append(
        "Use these tables in the report only after verifying that the same code and seed produced the Kaggle submission."
    )
    (output_dir / "report_tables.md").write_text("\n".join(lines), encoding="utf-8")

