import os
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingestion.vector_store import embed, get_collection

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CANDIDATES = 20   # taken from each of dense and BM25 search
RRF_K = 60        # standard Reciprocal Rank Fusion constant


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


class _BM25Index:
    """BM25 over every chunk in the vector store; rebuilt when the chunk count changes."""

    def __init__(self):
        self.count = -1

    def get(self, collection):
        count = collection.count()
        if count != self.count:
            data = collection.get(include=["documents", "metadatas"])
            self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
            self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs]) if self.docs else None
            self.count = count
        return self


_bm25_index = _BM25Index()


@lru_cache(maxsize=1)
def _reranker():
    return CrossEncoder(RERANKER_MODEL)


def _dense_search(collection, query, sources):
    res = collection.query(
        query_embeddings=embed([query]),
        n_results=CANDIDATES,
        where={"source": {"$in": sources}},
        include=["documents", "metadatas"],
    )
    return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0], strict=True))


def _bm25_search(collection, query, sources):
    index = _bm25_index.get(collection)
    if index.bm25 is None:
        return []
    scores = index.bm25.get_scores(_tokenize(query))
    ranked = sorted(
        (i for i in range(len(index.ids)) if index.metas[i]["source"] in sources),
        key=lambda i: scores[i], reverse=True,
    )
    return [(index.ids[i], index.docs[i], index.metas[i]) for i in ranked[:CANDIDATES] if scores[i] > 0]


def _rrf(*result_lists):
    """Reciprocal Rank Fusion of several ranked (id, doc, meta) lists."""
    fused, items = {}, {}
    for results in result_lists:
        for rank, (id_, doc, meta) in enumerate(results):
            fused[id_] = fused.get(id_, 0) + 1 / (RRF_K + rank + 1)
            items[id_] = (doc, meta)
    return [(id_, *items[id_]) for id_ in sorted(fused, key=fused.get, reverse=True)]


def retrieve(query: str, sources: list[str], top_k: int = 5) -> list[dict]:
    """Hybrid search (dense + BM25, fused with RRF) followed by cross-encoder reranking.

    `sources` limits the search to these S3 keys, so a customer only sees their own documents.
    """
    if not sources:
        return []
    collection = get_collection()
    candidates = _rrf(_dense_search(collection, query, sources), _bm25_search(collection, query, sources))
    if not candidates:
        return []
    scores = _reranker().predict([(query, doc) for _, doc, _ in candidates])
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"text": doc, "metadata": meta, "score": float(score)} for (_, doc, meta), score in ranked]
