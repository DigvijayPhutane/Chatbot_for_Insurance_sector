from pathlib import Path

import pytest

from ingestion import hash_tracker, index
from tests.helpers import FakeS3, ephemeral_collection, fake_embed, word_count

FILES = Path(__file__).parent.parent / "files"
HEALTH = "Health_Insurance_Policy_Suraksha_Family_Health_Plus.pdf"
HOME = "Home_Insurance_Policy_Griha_Kavach.pdf"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """index.main() wired to a fake S3, an in-memory collection and a counting fake embedder."""
    s3 = FakeS3({
        HEALTH: (FILES / HEALTH).read_bytes(),
        HOME: (FILES / HOME).read_bytes(),
        "database/insurance.db": b"not a pdf",
    })
    collection = ephemeral_collection()
    embedded = []

    def embed(texts):
        embedded.append(len(texts))
        return fake_embed(texts)

    monkeypatch.setattr(index, "s3", s3)
    monkeypatch.setattr(index, "get_collection", lambda: collection)
    monkeypatch.setattr(index, "embed", embed)
    monkeypatch.setattr(index, "count_tokens", word_count)
    monkeypatch.setattr(hash_tracker, "MANIFEST_PATH", tmp_path / "manifest.json")
    return s3, collection, embedded


def sources(collection):
    return {m["source"] for m in collection.get(include=["metadatas"])["metadatas"]}


def test_first_run_indexes_every_pdf_and_ignores_other_keys(env, capsys):
    s3, collection, embedded = env
    index.main()
    assert sources(collection) == {HEALTH, HOME}
    assert len(embedded) == 2
    assert capsys.readouterr().out.count("New, embedded") == 2
    assert set(hash_tracker.load_manifest()) == {HEALTH, HOME}


def test_unchanged_documents_are_not_embedded_again(env, capsys):
    s3, collection, embedded = env
    index.main()
    count = collection.count()
    index.main()
    assert len(embedded) == 2
    assert collection.count() == count
    assert capsys.readouterr().out.count("Unchanged, skipped") == 2


def test_changed_document_is_re_embedded_and_old_chunks_replaced(env, capsys):
    s3, collection, embedded = env
    index.main()
    s3.objects[HOME] = s3.objects[HEALTH]  # new content under the same key
    index.main()
    assert len(embedded) == 3
    assert "Updated, embedded" in capsys.readouterr().out
    hashes = {m["doc_hash"] for m in collection.get(where={"source": HOME}, include=["metadatas"])["metadatas"]}
    assert hashes == {hash_tracker.sha256(s3.objects[HEALTH])}


def test_document_removed_from_s3_is_removed_from_the_index(env):
    s3, collection, embedded = env
    index.main()
    del s3.objects[HOME]
    index.main()
    assert sources(collection) == {HEALTH}
    assert set(hash_tracker.load_manifest()) == {HEALTH}


def test_changing_the_embedding_model_re_embeds_everything(env, monkeypatch):
    s3, collection, embedded = env
    index.main()
    monkeypatch.setattr(index, "EMBEDDING_MODEL", "another-model")
    index.main()
    assert len(embedded) == 4


def test_wiped_vector_store_is_rebuilt_even_if_manifest_says_indexed(env):
    s3, collection, embedded = env
    index.main()
    collection.delete(where={"source": HEALTH})
    index.main()
    assert sources(collection) == {HEALTH, HOME}
    assert len(embedded) == 3
