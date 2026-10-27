"""Runs against the Floci S3 emulator (AWS_ENDPOINT_URL), each test in its own bucket."""

import sqlite3
import uuid

import pytest

from customer_api import db
from ingestion import dms, index

pytestmark = pytest.mark.integration


@pytest.fixture
def bucket(monkeypatch):
    name = f"test-{uuid.uuid4().hex[:12]}"
    for module in (dms, index, db):
        monkeypatch.setattr(module, "BUCKET", name)
    yield name
    s3 = dms.s3
    for obj in s3.list_objects_v2(Bucket=name).get("Contents", []):
        s3.delete_object(Bucket=name, Key=obj["Key"])
    s3.delete_bucket(Bucket=name)


def test_upload_pdfs_and_list_them(bucket, tmp_path, monkeypatch):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (tmp_path / "notes.txt").write_text("not uploaded")
    monkeypatch.setattr(dms, "FILES_DIR", tmp_path)

    dms.ensure_bucket()
    dms.ensure_bucket()  # second call must not fail
    dms.upload_pdfs()
    dms.s3.put_object(Bucket=bucket, Key="database/insurance.db", Body=b"x")

    assert index.list_pdf_keys() == ["a.pdf"]
    head = dms.s3.head_object(Bucket=bucket, Key="a.pdf")
    assert head["ContentType"] == "application/pdf"


def test_database_is_seeded_to_s3_then_downloaded(bucket, tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "first.db")
    db.load_db()  # nothing in S3 yet: seeds the sample data and uploads it
    assert dms.s3.head_object(Bucket=bucket, Key=db.DB_KEY)["ContentLength"] > 0

    # Change the copy in S3; a fresh start must download it rather than reseed.
    with sqlite3.connect(tmp_path / "first.db") as conn:
        conn.execute("UPDATE claims SET status = 'Settled' WHERE claim_id = 'CLM-HM-2026-00087'")
    conn.close()
    db.upload_db()

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "second.db")
    db.load_db()
    with db.connect() as conn:
        status = conn.execute("SELECT status FROM claims WHERE claim_id = 'CLM-HM-2026-00087'").fetchone()[0]
    conn.close()
    assert status == "Settled"
