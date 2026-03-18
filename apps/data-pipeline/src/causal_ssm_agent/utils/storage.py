"""Pluggable storage backend — local filesystem or Cloudflare R2.

Production (``DEPLOYMENT_ENV=production``) uses Cloudflare R2.
All other environments default to local filesystem.

Environment variables for R2::

    DEPLOYMENT_ENV=production
    R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
    R2_ACCESS_KEY_ID=...
    R2_SECRET_ACCESS_KEY=...
    R2_BUCKET=...
    R2_PREFIX=data              # key prefix inside bucket (default: "data")
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    import fsspec

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def is_remote() -> bool:
    """True when using Cloudflare R2 (remote) storage."""
    return os.getenv("DEPLOYMENT_ENV") == "production"


# ---------------------------------------------------------------------------
# Base URI & filesystem
# ---------------------------------------------------------------------------


def _find_local_data_dir() -> Path:
    """Find the repository ``data/`` directory by walking up from this file."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "data"
        if candidate.exists():
            return candidate
    return Path.cwd() / "data"


def get_base_uri() -> str:
    """Return the root URI for data storage.

    Local: absolute path like ``/Users/.../data``
    R2:    ``s3://<bucket>/<prefix>``
    """
    if is_remote():
        bucket = os.environ["R2_BUCKET"]
        prefix = os.getenv("R2_PREFIX", "data")
        return f"s3://{bucket}/{prefix}"
    return str(_find_local_data_dir())


@lru_cache(maxsize=1)
def get_fs() -> fsspec.AbstractFileSystem:
    """Return an fsspec filesystem for the active backend (cached)."""
    import fsspec as _fsspec

    if is_remote():
        return _fsspec.filesystem(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            key=os.environ["R2_ACCESS_KEY_ID"],
            secret=os.environ["R2_SECRET_ACCESS_KEY"],
        )
    return _fsspec.filesystem("file")


def polars_storage_options() -> dict[str, str] | None:
    """Storage options dict for Polars cloud I/O, or *None* for local."""
    if not is_remote():
        return None
    return {
        "aws_endpoint_url": os.environ["R2_ENDPOINT_URL"],
        "aws_access_key_id": os.environ["R2_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"],
        "aws_region": "auto",
    }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def join(*parts: str) -> str:
    """Join path segments — works for both local paths and ``s3://`` URIs."""
    if not parts:
        return ""
    base = parts[0]
    for part in parts[1:]:
        base = f"{base.rstrip('/')}/{part.strip('/')}"
    return base


# ---------------------------------------------------------------------------
# I/O primitives
# ---------------------------------------------------------------------------


def exists(path: str) -> bool:
    if is_remote():
        return get_fs().exists(path)
    return Path(path).exists()


def makedirs(path: str) -> None:
    """Create directories. No-op for S3 (directories are implicit)."""
    if is_remote():
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def listdir(path: str) -> list[str]:
    """List entries in *path*. Returns full paths/URIs."""
    if is_remote():
        try:
            entries = get_fs().ls(path, detail=False)
        except FileNotFoundError:
            return []
        return [f"s3://{e}" if not e.startswith("s3://") else e for e in entries]
    p = Path(path)
    if not p.is_dir():
        return []
    return [str(child) for child in p.iterdir()]


def file_info(path: str) -> dict[str, Any]:
    """Return file metadata (size, type, last_modified / mtime)."""
    if is_remote():
        return get_fs().info(path)
    p = Path(path)
    stat = p.stat()
    return {
        "name": str(p),
        "size": stat.st_size,
        "type": "file" if p.is_file() else "directory",
        "mtime": stat.st_mtime,
    }


@contextmanager
def open_file(path: str, mode: str = "rb") -> Iterator[Any]:
    """Open a file for reading or writing. Works for both local and remote."""
    if is_remote():
        with get_fs().open(path, mode) as f:
            yield f
    else:
        if "w" in mode:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open(mode) as f:
            yield f


def read_text(path: str) -> str:
    if is_remote():
        with get_fs().open(path, "r") as f:
            return f.read()
    return Path(path).read_text()


def write_text(path: str, content: str) -> None:
    if is_remote():
        with get_fs().open(path, "w") as f:
            f.write(content)
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def read_bytes(path: str) -> bytes:
    if is_remote():
        with get_fs().open(path, "rb") as f:
            return f.read()
    return Path(path).read_bytes()


def write_bytes(path: str, content: bytes) -> None:
    if is_remote():
        with get_fs().open(path, "wb") as f:
            f.write(content)
    else:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def read_json(path: str) -> Any:
    return json.loads(read_text(path))


def write_json(path: str, data: Any) -> None:
    write_text(path, json.dumps(data))
