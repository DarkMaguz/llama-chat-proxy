import pytest

from llama_chat_proxy.crypto import Vault, VaultError

KEY = "ab" * 64  # 128 hex chars -> 64 bytes


def test_roundtrip_bytes():
    v = Vault(KEY)
    blob = v.encrypt(b"hello \xe2\x9c\x93 world")
    assert v.decrypt(blob) == b"hello \xe2\x9c\x93 world"


def test_roundtrip_obj():
    v = Vault(KEY)
    obj = {"a": 1, "b": "héllo", "c": ["x", "y"]}
    assert v.decrypt_obj(v.encrypt_obj(obj)) == obj


def test_nonce_makes_ciphertexts_differ():
    v = Vault(KEY)
    assert v.encrypt(b"same") != v.encrypt(b"same")


def test_wrong_key_is_detected():
    v1 = Vault(KEY)
    v2 = Vault("cd" * 64)
    blob = v1.encrypt_obj({"secret": 1})
    with pytest.raises(VaultError):
        v2.decrypt_obj(blob)


def test_garbage_magic_rejected():
    v = Vault(KEY)
    with pytest.raises(VaultError):
        v.decrypt(b"this is not a valid blob at all")


def test_derivation_is_deterministic():
    assert Vault(KEY)._derive_master(bytes.fromhex(KEY)) == Vault(KEY)._derive_master(
        bytes.fromhex(KEY)
    )
