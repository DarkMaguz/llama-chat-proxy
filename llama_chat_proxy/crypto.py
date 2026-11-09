"""Authenticated encryption (AES-256-GCM) used to keep chat history at rest.

The operator supplies a 128 character hexadecimal key (64 raw bytes) through
``LLAMA_PROXY_KEY``.  That 64 byte value is treated as a secret *input* and
expanded with HKDF-SHA256 into a 32 byte AES-256 master key, so the same
key material always yields the same deterministic key while we still get a
proper AES key size.  Every encrypted record carries a random 12 byte GCM
nonce, so the same plaintext encrypts to a different ciphertext each time.

On-disk layout of an encrypted record::

    b"LCP1" || nonce(12) || ciphertext || tag(16)

The ``b"LCP1"`` magic lets us detect a key mismatch (decrypt -> ``InvalidTag``)
versus a corrupt/truncated row (``ValueError``) and fail with a clear message.
"""

from __future__ import annotations

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_MAGIC = b"LCP1"
_NONCE_SIZE = 12  # standard GCM nonce length
_MASTER_INFO = b"llama-chat-proxy/v1"


class VaultError(RuntimeError):
    """Raised when a record cannot be decrypted (wrong key or corruption)."""


class Vault:
    """Symmetric encrypt/decrypt helper backed by a single derived key."""

    def __init__(self, key_hex: str) -> None:
        key_bytes = bytes.fromhex(key_hex)  # 64 bytes for a 128-hex-char key
        master = self._derive_master(key_bytes)
        self._aead = AESGCM(master)

    @staticmethod
    def _derive_master(secret: bytes) -> bytes:
        kdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=_MASTER_INFO,
        )
        return kdf.derive(secret)

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt ``plaintext`` into a self-contained, authenticated blob."""
        if not isinstance(plaintext, (bytes, bytearray)):
            raise TypeError("plaintext must be bytes")
        nonce = os.urandom(_NONCE_SIZE)
        # AESGCM.encrypt returns ciphertext || 16-byte tag.
        ciphertext = self._aead.encrypt(nonce, bytes(plaintext), None)
        return _MAGIC + nonce + ciphertext

    def decrypt(self, blob: bytes) -> bytes:
        """Decrypt a blob produced by :meth:`encrypt`."""
        if not isinstance(blob, (bytes, bytearray)) or not bytes(blob).startswith(_MAGIC):
            raise VaultError("record is not a llama-chat-proxy encrypted blob")
        body = bytes(blob)[len(_MAGIC):]
        if len(body) < _NONCE_SIZE + 16:
            raise VaultError("encrypted record is too short to be valid")
        nonce = body[:_NONCE_SIZE]
        ciphertext = body[_NONCE_SIZE:]
        try:
            return self._aead.decrypt(nonce, ciphertext, None)
        except InvalidTag as exc:
            raise VaultError(
                "decryption failed (wrong LLAMA_PROXY_KEY or corrupt record)"
            ) from exc

    # Convenience wrappers --------------------------------------------------
    def encrypt_obj(self, obj) -> bytes:
        import json

        return self.encrypt(json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    def decrypt_obj(self, blob: bytes) -> dict:
        import json

        return json.loads(self.decrypt(blob).decode("utf-8"))
