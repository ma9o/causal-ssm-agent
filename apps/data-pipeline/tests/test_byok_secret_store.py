import time
from base64 import urlsafe_b64encode
from hashlib import sha256
from hmac import new as hmac_new

import libsql
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from causal_ssm_agent.utils.byok_secret_store import consume_byok_secret_ref


def _encode_base64url(raw: bytes) -> str:
    return urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _build_web_payload(api_key: str, secret: str) -> str:
    nonce = b"0" * 12
    key = sha256(secret.encode("utf-8")).digest()
    ciphertext_with_tag = AESGCM(key).encrypt(nonce, api_key.encode("utf-8"), None)
    ciphertext = ciphertext_with_tag[:-16]
    auth_tag = ciphertext_with_tag[-16:]
    return (
        f"v1.{_encode_base64url(nonce)}.{_encode_base64url(ciphertext)}."
        f"{_encode_base64url(auth_tag)}"
    )


def _derive_local_secret(root_secret: str, scope: str) -> str:
    return hmac_new(root_secret.encode("utf-8"), scope.encode("utf-8"), sha256).hexdigest()


def _seed_byok_ref(tmp_path, ref: str, payload: str, *, expires_at_ms: int) -> None:
    db_path = tmp_path / "byok-secret-store.db"
    connect = getattr(libsql, "connect", None)
    if connect is None:
        raise RuntimeError("libsql.connect is unavailable in the test environment")
    connection = connect(str(db_path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS byok_secret_refs (
            ref TEXT PRIMARY KEY,
            ciphertext TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            expires_at_ms INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO byok_secret_refs (ref, ciphertext, created_at_ms, expires_at_ms)
        VALUES (?, ?, ?, ?)
        """,
        (ref, payload, expires_at_ms - 1000, expires_at_ms),
    )
    connection.commit()


def test_consume_byok_secret_ref_reads_web_payload_and_is_single_use(monkeypatch, tmp_path):
    app_secret = "0123456789abcdef0123456789abcdef"
    now_ms = int(time.time() * 1000)
    ref = "ref-123"
    monkeypatch.setenv("BYOK_SECRET_STORE_URL", f"file:{tmp_path / 'byok-secret-store.db'}")
    monkeypatch.setenv("APP_SECRET", app_secret)
    monkeypatch.delenv("BYOK_SECRET_STORE_AUTH_TOKEN", raising=False)

    _seed_byok_ref(
        tmp_path,
        ref,
        _build_web_payload("user-key", _derive_local_secret(app_secret, "byok-secret-store")),
        expires_at_ms=now_ms + 60_000,
    )

    assert consume_byok_secret_ref(ref) == "user-key"
    assert consume_byok_secret_ref(ref) is None


def test_consume_byok_secret_ref_returns_none_for_expired_rows(monkeypatch, tmp_path):
    app_secret = "0123456789abcdef0123456789abcdef"
    now_ms = int(time.time() * 1000)
    ref = "ref-expired"
    monkeypatch.setenv("BYOK_SECRET_STORE_URL", f"file:{tmp_path / 'byok-secret-store.db'}")
    monkeypatch.setenv("APP_SECRET", app_secret)
    monkeypatch.delenv("BYOK_SECRET_STORE_AUTH_TOKEN", raising=False)

    _seed_byok_ref(
        tmp_path,
        ref,
        _build_web_payload("user-key", _derive_local_secret(app_secret, "byok-secret-store")),
        expires_at_ms=now_ms - 1,
    )

    assert consume_byok_secret_ref(ref) is None
