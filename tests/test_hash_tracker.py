import pytest

from ingestion import hash_tracker


@pytest.fixture(autouse=True)
def manifest_path(tmp_path, monkeypatch):
    path = tmp_path / "store" / "manifest.json"
    monkeypatch.setattr(hash_tracker, "MANIFEST_PATH", path)
    return path


def test_sha256():
    assert hash_tracker.sha256(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_missing_manifest_is_empty():
    assert hash_tracker.load_manifest() == {}


def test_manifest_round_trip(manifest_path):
    manifest = {"a.pdf": {"sha256": "x", "embedding_model": "m"}}
    hash_tracker.save_manifest(manifest)
    assert manifest_path.exists()
    assert hash_tracker.load_manifest() == manifest
