"""Consume encrypted single-use BYOK secret refs minted by the web server.

The store uses libSQL with a local file URL by default so development and CI do
not need extra services. Deployed environments can point the same code at a
remote Turso/libSQL database by setting one URL plus an optional auth token.
"""

from __future__ import annotations

import os
import time
from base64 import urlsafe_b64decode, urlsafe_b64encode
from contextlib import closing
from hashlib import sha256
from pathlib import Path

import libsql
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

BYOK_SECRET_TABLE = "byok_secret_refs"
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_BYOK_SECRET_STORE_PATH = REPO_ROOT / ".local" / "byok-secret-store.db"
DEFAULT_BYOK_SECRET_STORE_URL = f"file:{DEFAULT_BYOK_SECRET_STORE_PATH}"


def _get_store_url() -> str:
    raw_value = os.getenv("BYOK_SECRET_STORE_URL", "").strip()
    return raw_value or DEFAULT_BYOK_SECRET_STORE_URL


def _get_auth_token() -> str | None:
    raw_value = os.getenv("BYOK_SECRET_STORE_AUTH_TOKEN", "").strip()
    return raw_value or None


def _resolve_database() -> str:
    raw_url = _get_store_url()
    if not raw_url.startswith("file:"):
        return raw_url

    raw_path = raw_url[len("file:") :]
    path = Path(raw_path).expanduser()
    resolved = path if path.is_absolute() else REPO_ROOT / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def _get_cipher_key() -> bytes:
    secret = os.getenv("BYOK_SECRET_STORE_ENCRYPTION_KEY")
    if not secret or len(secret) < 32:
        raise RuntimeError(
            "BYOK_SECRET_STORE_ENCRYPTION_KEY must be set and at least 32 characters"
        )
    return sha256(secret.encode("utf-8")).digest()


def _encode_base64url(raw: bytes) -> str:
    return urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_base64url(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return urlsafe_b64decode(f"{value}{padding}")


def _decrypt_api_key(payload: str) -> str:
    try:
        version, nonce_part, ciphertext_part, auth_tag_part = payload.split(".", maxsplit=3)
    except ValueError as exc:
        raise RuntimeError("Invalid BYOK secret payload version") from exc
    if version != "v1" or not nonce_part or not ciphertext_part or not auth_tag_part:
        raise RuntimeError("Invalid BYOK secret payload version")
    ciphertext_with_tag = _decode_base64url(ciphertext_part) + _decode_base64url(auth_tag_part)
    plaintext = AESGCM(_get_cipher_key()).decrypt(
        _decode_base64url(nonce_part),
        ciphertext_with_tag,
        None,
    )
    return plaintext.decode("utf-8")


def _connect() -> libsql.Connection:
    auth_token = _get_auth_token()
    connection = libsql.connect(
        _resolve_database(),
        **({"auth_token": auth_token} if auth_token else {}),
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {BYOK_SECRET_TABLE} (
            ref TEXT PRIMARY KEY,
            ciphertext TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            expires_at_ms INTEGER NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def consume_byok_secret_ref(ref: str) -> str | None:
    now_ms = int(time.time() * 1000)

    with closing(_connect()) as connection:
        row = connection.execute(
            f"""
            DELETE FROM {BYOK_SECRET_TABLE}
            WHERE ref = ? AND expires_at_ms > ?
            RETURNING ciphertext
            """,
            (ref, now_ms),
        ).fetchone()
        connection.execute(
            f"DELETE FROM {BYOK_SECRET_TABLE} WHERE expires_at_ms <= ?",
            (now_ms,),
        )
        connection.commit()

    if row is None:
        return None
    return _decrypt_api_key(str(row[0]))
