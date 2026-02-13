"""Stage 0: Preprocess raw input data.

Reads raw Google Takeout MyActivity JSON from data/raw/<user_id>/,
parses and sorts entries, and returns text lines for downstream chunking.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, is_zipfile

from prefect import task
from prefect.cache_policies import INPUTS

from causal_ssm_agent.utils.data import RAW_DIR

TAKEOUT_ZIP_PATH = "Takeout/My Activity/Search/MyActivity.json"


def _extract_location(entry: dict) -> str | None:
    """Extract location coordinates from entry."""
    location_infos = entry.get("locationInfos", [])
    if not location_infos:
        return None

    url = location_infos[0].get("url", "")
    match = re.search(r"center=([0-9.-]+),([0-9.-]+)", url)
    if match:
        return f"{match.group(1)},{match.group(2)}"
    return None


def _process_activity(entries: list[dict]) -> list[dict]:
    """Extract all activity with timestamps into flat records."""
    processed = []

    for entry in entries:
        title = entry.get("title", "")
        time_str = entry.get("time", "")
        if not time_str or not title:
            continue

        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

        if title.startswith("Searched for "):
            activity_type = "search"
            content = title.removeprefix("Searched for ")
        elif title.startswith("Visited "):
            activity_type = "visit"
            content = title.removeprefix("Visited ")
        elif title.startswith("Viewed "):
            activity_type = "view"
            content = title.removeprefix("Viewed ")
        else:
            activity_type = "other"
            content = title

        location = _extract_location(entry)

        processed.append(
            {
                "datetime": dt,
                "activity_type": activity_type,
                "content": content,
                "location": location,
            }
        )

    processed.sort(key=lambda r: r["datetime"])
    return processed


def _records_to_lines(records: list[dict]) -> list[str]:
    """Convert processed records to text lines."""
    lines = []
    for r in records:
        ts = r["datetime"].isoformat()
        loc_str = f" @ {r['location']}" if r["location"] else ""
        lines.append(f"[{ts}]{loc_str} [{r['activity_type']}] {r['content']}")
    return lines


def _parse_json(json_path: Path) -> list[dict]:
    """Parse a standalone MyActivity.json file."""
    raw_data = json.loads(json_path.read_text())
    if not raw_data:
        raise ValueError(f"Empty JSON data in {json_path}")
    return _process_activity(raw_data)


def _parse_takeout_zip(archive_path: Path) -> list[dict]:
    """Parse Google Takeout MyActivity from zip archive."""
    with archive_path.open("rb") as f:
        if not is_zipfile(f):
            raise ValueError(f"{archive_path} is not a valid zip archive")

        with ZipFile(f, "r") as zip_ref:
            if TAKEOUT_ZIP_PATH not in zip_ref.namelist():
                raise ValueError(f"{TAKEOUT_ZIP_PATH} not found in archive")
            with zip_ref.open(TAKEOUT_ZIP_PATH) as zip_f:
                raw_data = json.load(zip_f)

    if not raw_data:
        raise ValueError("Empty JSON data in archive")

    return _process_activity(raw_data)


def _find_raw_input(user_id: str) -> Path:
    """Find the raw input file for a user.

    Searches data/raw/<user_id>/ for .json or .zip files.
    """
    user_dir = RAW_DIR / user_id
    if not user_dir.is_dir():
        raise FileNotFoundError(f"No raw data directory: {user_dir}")

    for pattern in ("*.json", "*.zip"):
        files = sorted(user_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return files[0]

    raise FileNotFoundError(f"No .json or .zip files in {user_dir}")


@task(cache_policy=INPUTS, result_serializer="json")
def preprocess_raw_input(user_id: str = "test_user") -> list[str]:
    """Preprocess raw input data into text lines.

    Finds the most recent .json or .zip in data/raw/<user_id>/,
    parses it, and returns sorted text lines ready for chunking.

    Args:
        user_id: User subdirectory under data/raw/

    Returns:
        List of preprocessed text lines, one per activity record
    """
    raw_path = _find_raw_input(user_id)
    print(f"Preprocessing {raw_path.name} from {raw_path.parent.name}/")

    if raw_path.suffix == ".json":
        records = _parse_json(raw_path)
    else:
        records = _parse_takeout_zip(raw_path)

    lines = _records_to_lines(records)
    print(f"Preprocessed {len(lines)} activity records")
    return lines
