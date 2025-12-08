#!/usr/bin/env python
"""
Preprocess Google Takeout / ChatGPT export data into text chunks.

Reads zip archives from data/google-takeout/ and outputs sorted text files
to data/preprocessed/.

Usage:
    uv run python scripts/preprocess_google_takeout.py
    uv run python scripts/preprocess_google_takeout.py --input data/google-takeout/export.zip
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, is_zipfile

import polars as pl

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_DIR = DATA_DIR / "google-takeout"
OUTPUT_DIR = DATA_DIR / "preprocessed"


def parse_conversations_zip(
    archive_path: Path,
    expected_file: str = "conversations.json",
) -> pl.DataFrame:
    """Parse ChatGPT/OpenAI conversation export from zip."""
    with archive_path.open("rb") as f:
        if not is_zipfile(f):
            raise ValueError(f"{archive_path} is not a valid zip archive")

        with ZipFile(f, "r") as zip_ref:
            if expected_file not in zip_ref.namelist():
                raise ValueError(f"{expected_file} not found in archive")
            with zip_ref.open(expected_file) as zip_f:
                raw_data = json.load(zip_f)

    if not raw_data:
        raise ValueError("Empty JSON data in archive")

    return _process_conversations(raw_data)


def _process_conversations(conversations: list[dict]) -> pl.DataFrame:
    """Extract Q&A pairs with timestamps from conversation data."""
    processed = []

    for conv in conversations:
        conversation_id = conv["id"]
        title = conv.get("title", "")
        mapping = conv.get("mapping", {})
        messages = list(mapping.values())

        for i, message in enumerate(messages):
            msg = message.get("message")
            if not msg:
                continue

            author = msg.get("author", {})
            if author.get("role") != "user":
                continue

            create_time = msg.get("create_time")
            content = msg.get("content", {})
            if not create_time or not content:
                continue

            dt = datetime.fromtimestamp(create_time)

            # Extract question
            parts = content.get("parts", [""])
            question = "\n".join(
                str(p) if isinstance(p, dict) else p for p in parts
            )

            # Extract assistant answer from next message
            answer = ""
            if i + 1 < len(messages):
                next_msg = messages[i + 1].get("message")
                if next_msg and next_msg.get("author", {}).get("role") == "assistant":
                    next_content = next_msg.get("content", {})
                    answer_parts = next_content.get("parts", [""])
                    answer = "\n".join(
                        str(p) if isinstance(p, dict) else p for p in answer_parts
                    )

            processed.append({
                "conversation_id": conversation_id,
                "title": title,
                "datetime": dt,
                "question": question,
                "answer": answer,
            })

    df = pl.DataFrame(processed)
    return df.sort("datetime")


def export_as_text_chunks(df: pl.DataFrame, output_path: Path) -> None:
    """Export dataframe as newline-delimited text chunks."""
    chunks = []
    for row in df.iter_rows(named=True):
        # Format: timestamp, title, Q&A
        chunk = f"[{row['datetime']}] {row['title']}\nQ: {row['question']}\nA: {row['answer']}"
        chunks.append(chunk)

    output_path.write_text("\n\n---\n\n".join(chunks))
    print(f"Wrote {len(chunks)} chunks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Google Takeout data")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Specific zip file to process (default: all zips in data/google-takeout/)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for preprocessed files",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    if args.input:
        input_files = [args.input]
    else:
        input_files = list(INPUT_DIR.glob("*.zip"))

    if not input_files:
        print(f"No zip files found in {INPUT_DIR}")
        return

    for zip_path in input_files:
        print(f"Processing {zip_path.name}...")
        try:
            df = parse_conversations_zip(zip_path)
            output_path = args.output_dir / f"{zip_path.stem}.txt"
            export_as_text_chunks(df, output_path)
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
