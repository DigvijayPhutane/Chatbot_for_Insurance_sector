"""Offline stand-ins for the models, S3 and the vector store."""

import io
import math
import re
import uuid
import zlib
from types import SimpleNamespace

import chromadb
import pymupdf
from chromadb.config import Settings

DIM = 64


def word_count(text: str) -> int:
    """Token counter that needs no tokenizer download."""
    return len(text.split())


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic bag-of-words vectors, so texts sharing words are close in cosine distance."""
    vectors = []
    for text in texts:
        v = [1e-3] * DIM
        for word in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(word.encode()) % DIM] += 1
        norm = math.sqrt(sum(x * x for x in v))
        vectors.append([x / norm for x in v])
    return vectors


class FakeReranker:
    """Scores a (query, doc) pair by the number of shared words."""

    def predict(self, pairs):
        return [len(set(re.findall(r"\w+", q.lower())) & set(re.findall(r"\w+", d.lower()))) for q, d in pairs]


def ephemeral_collection():
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    return client.create_collection(f"test-{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})


class FakeS3:
    """The subset of the boto3 S3 client used by ingestion.index."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})

    def get_paginator(self, name):
        return self

    def paginate(self, Bucket):
        yield {"Contents": [{"Key": k} for k in self.objects]}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}


def chat_response(content: str):
    """Shape of a non-streaming ollama.chat() response."""
    return SimpleNamespace(message=SimpleNamespace(content=content))


def build_pdf(pages: list[list[tuple[str, float, bool]]]) -> bytes:
    """Build a PDF from pages of (text, font size, bold) lines."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 50
        for text, size, bold in lines:
            page.insert_text((50, y), text, fontsize=size, fontname="hebo" if bold else "helv")
            y += size * 1.8
    data = doc.tobytes()
    doc.close()
    return data
