from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

IMMUTABLE_PREDICTION_FIELDS = {
    "created_at",
    "model_version",
    "data_as_of",
    "home_win",
    "draw",
    "away_win",
    "expected_home_goals",
    "expected_away_goals",
    "predicted_outcome",
    "predicted_score",
}


def _is_prediction_field(name: str) -> bool:
    return name in IMMUTABLE_PREDICTION_FIELDS or name.startswith("predicted_")


def _rows(payload: Any, source: str) -> list[dict[str, Any]]:
    rows = payload.get("predictions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError(f"{source} does not contain a predictions list")
    if any(not isinstance(row, dict) or not row.get("fixture_key") for row in rows):
        raise ValueError(f"{source} contains an invalid prediction row")
    return rows


def merge_prediction_snapshots(
    durable_payload: Any,
    live_payload: Any,
) -> dict[str, Any]:
    """Merge a live ledger without deleting or rewriting durable official picks."""
    durable = {
        str(row["fixture_key"]): dict(row) for row in _rows(durable_payload, "durable snapshot")
    }
    live = _rows(live_payload, "live snapshot")

    for live_row in live:
        key = str(live_row["fixture_key"])
        existing = durable.get(key)
        if existing is None:
            durable[key] = dict(live_row)
            continue

        merged = dict(existing)
        for field, value in live_row.items():
            if _is_prediction_field(field) and existing.get(field) is not None:
                continue
            if value is None and existing.get(field) is not None:
                continue
            if field == "updated_at" and existing.get(field) and value:
                merged[field] = max(str(existing[field]), str(value))
            else:
                merged[field] = value
        durable[key] = merged

    predictions = sorted(
        durable.values(),
        key=lambda row: (str(row.get("kickoff_utc", "")), str(row["fixture_key"])),
        reverse=True,
    )
    return {
        "snapshot_version": 2,
        "generated_at": datetime.now(UTC).isoformat(),
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge immutable prediction-ledger snapshots")
    parser.add_argument("durable", type=Path)
    parser.add_argument("live", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    durable = json.loads(args.durable.read_text())
    live = json.loads(args.live.read_text())
    merged = merge_prediction_snapshots(durable, live)
    args.output.write_text(json.dumps(merged, separators=(",", ":")))


if __name__ == "__main__":
    main()
