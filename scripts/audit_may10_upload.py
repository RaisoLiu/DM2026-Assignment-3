#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the scheduled May 10 Kaggle upload results.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/blend_search/upload_may10_0810_submitted.tsv"),
    )
    parser.add_argument(
        "--status-csv",
        type=Path,
        default=Path("artifacts/blend_search/upload_may10_0810_status.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/blend_search/upload_may10_0810_audit.json"),
    )
    parser.add_argument("--target", type=float, default=0.82)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submitted = read_manifest(args.manifest)
    status_rows = read_status(args.status_csv)
    by_key = {(row.get("fileName", ""), row.get("description", "")): row for row in status_rows}

    rows: list[dict[str, object]] = []
    complete_scores: list[float] = []
    for csv_path, message in submitted:
        filename = Path(csv_path).name
        row = by_key.get((filename, message), {})
        public_score = parse_score(row.get("publicScore", ""))
        status = normalize_status(row.get("status", ""))
        record = {
            "csv": csv_path,
            "message": message,
            "filename": filename,
            "date": row.get("date", ""),
            "status": status,
            "public_score": public_score,
            "meets_target": public_score is not None and public_score > args.target,
        }
        rows.append(record)
        if status == "COMPLETE" and public_score is not None:
            complete_scores.append(public_score)

    best_public = max(complete_scores) if complete_scores else None
    payload = {
        "target": args.target,
        "manifest": str(args.manifest),
        "status_csv": str(args.status_csv),
        "submitted_count": len(submitted),
        "complete_count": sum(1 for row in rows if row["status"] == "COMPLETE"),
        "best_public_score": best_public,
        "target_met": best_public is not None and best_public > args.target,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)

    if len(submitted) != 3:
        raise SystemExit(f"Expected 3 submitted rows, found {len(submitted)}")
    if payload["complete_count"] != len(submitted):
        raise SystemExit("Not all scheduled submissions are complete")
    if not payload["target_met"]:
        raise SystemExit(f"Best public score did not exceed {args.target}")


def read_manifest(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing manifest: {path}")
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        left, right = line.split("\t", 1)
        rows.append((left, right))
    return rows


def read_status(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing status CSV: {path}")
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def parse_score(raw: str) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def normalize_status(raw: str) -> str:
    return (raw or "").split(".")[-1].upper()


if __name__ == "__main__":
    main()
