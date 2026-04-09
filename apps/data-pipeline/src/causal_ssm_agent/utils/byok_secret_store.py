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
from hmac import new as hmac_new
from pathlib import Path

import libsql
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

BYOK_SECRET_TABLE = "byok_secret_refs"
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_BYOK_SECRET_STORE_PATH = REPO_ROOT / ".local" / "byok-secret-store.db"
DEFAULT_BYOK_SECRET_STORE_URL = f"file:{DEFAULT_BYOK_SECRET_STORE_PATH}"
DELETE_ACTIVE_BYOK_SECRET_SQL = """
DELETE FROM byok_secret_refs
WHERE ref = ? AND expires_at_ms > ?
RETURNING ciphertext
"""
PURGE_EXPIRED_BYOK_SECRET_SQL = "DELETE FROM byok_secret_refs WHERE expires_at_ms <= ?"


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
    return sha256(_derive_app_secret("byok-secret-store").encode("utf-8")).digest()


def _get_app_secret() -> str:
    secret = os.getenv("APP_SECRET", "").strip()
    if secret:
        if len(secret) < 32:
            raise RuntimeError("APP_SECRET must be set and at least 32 characters")
        return secret
    raise RuntimeError("APP_SECRET is not configured")


def _derive_app_secret(scope: str) -> str:
    return hmac_new(
        _get_app_secret().encode("utf-8"),
        scope.encode("utf-8"),
        sha256,
    ).hexdigest()


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


def _connect():
    auth_token = _get_auth_token()
    connect_fn = getattr(libsql, "connect", None)
    if connect_fn is None:
        raise RuntimeError("libsql.connect is unavailable in the current environment")
    connection = connect_fn(
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
            DELETE_ACTIVE_BYOK_SECRET_SQL,
            (ref, now_ms),
        ).fetchone()
        connection.execute(PURGE_EXPIRED_BYOK_SECRET_SQL, (now_ms,))
        connection.commit()

    if row is None:
        return None
    return _decrypt_api_key(str(row[0]))
