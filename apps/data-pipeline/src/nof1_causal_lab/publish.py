"""Publish a local workspace to the hosted (R2) store.

Publishing is a deliberate act: the hosted read facade serves whatever
lives under the R2 store, so copying a workspace there makes it — and
every artifact payload in it — publicly viewable. Raw N-of-1 data is
personal data: exclude it unless the workspace is synthetic/demo.

The store is append-only with immutable versions and journal entries, so
publishing is an idempotent file copy: keys that already exist are
skipped, except the mutable read models (``episode/state.json`` and the
``run/`` stage projections), which are always overwritten. Re-running
publish while a local episode executes gives the hosted viewer a live
tail through its normal polling.

Usage (needs the ``cloud`` dependency group and the production R2 env:
``R2_ENDPOINT_URL``, ``R2_ACCESS_KEY_ID``, ``R2_SECRET_ACCESS_KEY``,
``R2_BUCKET``, ``R2_PREFIX``)::

    uv run nof1-publish WORKSPACE_ID [--exclude raw_data --exclude input]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _dest_fs():
    import fsspec

    return fsspec.filesystem(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        key=os.environ["R2_ACCESS_KEY_ID"],
        secret=os.environ["R2_SECRET_ACCESS_KEY"],
    )


def _is_mutable(rel: str) -> bool:
    """The only non-append-only files in a workspace: latest-state read models."""
    return rel == "episode/state.json" or rel.startswith("run/")


def _is_excluded(rel: str, excludes: list[str]) -> bool:
    for name in excludes:
        prefix = "input/" if name == "input" else f"store/{name}/"
        if rel.startswith(prefix):
            return True
    return False


def publish_workspace(workspace_id: str, excludes: list[str]) -> dict[str, int]:
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils import storage

    if storage.is_remote():
        raise RuntimeError("publish copies FROM the local store; unset DEPLOYMENT_ENV=production")
    src_root = Path(data_module.DATA_URI) / workspace_id
    if not src_root.is_dir():
        raise FileNotFoundError(f"No local workspace at {src_root}")

    bucket = os.environ["R2_BUCKET"]
    prefix = os.environ.get("R2_PREFIX", "data")
    dest_root = f"{bucket}/{prefix}/{workspace_id}"
    fs = _dest_fs()
    existing = {found.lstrip("/") for found in fs.find(dest_root)}

    counts = {"uploaded": 0, "skipped": 0, "excluded": 0}
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src_root).as_posix()
        if _is_excluded(rel, excludes):
            counts["excluded"] += 1
            continue
        dest = f"{dest_root}/{rel}"
        if dest in existing and not _is_mutable(rel):
            counts["skipped"] += 1
            continue
        fs.put_file(str(path), dest)
        counts["uploaded"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a local workspace to the hosted (R2) store."
    )
    parser.add_argument("workspace_id")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="ARTIFACT_ID|input",
        help=(
            "Withhold store/<artifact_id>/ payloads (or the raw 'input' uploads) "
            "from publication; repeatable."
        ),
    )
    args = parser.parse_args()
    counts = publish_workspace(args.workspace_id, args.exclude)
    print(
        f"{args.workspace_id}: uploaded {counts['uploaded']}, "
        f"skipped {counts['skipped']} existing, excluded {counts['excluded']}"
    )


if __name__ == "__main__":
    main()
