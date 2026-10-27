import os
from functools import lru_cache
from pathlib import Path

import chromadb
import ollama
from chromadb.config import Settings
from tokenizers import Tokenizer

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-minilm:l6-v2")
# Ollama does not expose its tokenizer, so token counts for chunk sizing use the same model's HF tokenizer.
EMBEDDING_TOKENIZER = os.getenv("EMBEDDING_TOKENIZER", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_MAX_TOKENS = int(os.getenv("EMBEDDING_MAX_TOKENS", "256"))
CHROMA_DIR = Path(__file__).parent.parent / "vectorstore" / "chroma"
COLLECTION = "insurance_docs"


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer.from_pretrained(EMBEDDING_TOKENIZER)
    tokenizer.no_truncation()  # the hub config truncates at 128, which would hide oversized chunks
    return tokenizer


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text).ids)


def max_tokens() -> int:
    return EMBEDDING_MAX_TOKENS


def embed(texts: list[str]) -> list[list[float]]:
    return ollama.embed(model=EMBEDDING_MODEL, input=texts).embeddings


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
