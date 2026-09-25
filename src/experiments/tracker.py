from datetime import datetime, timezone
from pathlib import Path
import json


class ExperimentTracker:
    """Small dependency-free experiment tracker.

    Each experiment is written as a JSON file plus a simple summary CSV.
    It is intentionally lightweight so the project does not depend on a
    full tracking server.
    """

    def __init__(self, output_dir="results/experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name, config, metrics, notes=None):
        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        record = {
            "experiment": name,
            "timestamp_utc": timestamp,
            "config": config,
            "metrics": metrics,
            "notes": notes or "",
        }

        json_path = self.output_dir / f"{name}_{timestamp}.json"
        json_path.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

        summary_path = self.output_dir / "experiments.jsonl"
        with summary_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(record, separators=(",", ":"))
                + "\n"
            )

        return json_path
