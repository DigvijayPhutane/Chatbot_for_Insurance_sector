import pytest

from rag import retriever
from tests.helpers import FakeReranker, ephemeral_collection, fake_embed

HEALTH, HOME = "health.pdf", "home.pdf"
CHUNKS = [
    (HEALTH, "6. Waiting Periods", "Pre-existing diseases such as diabetes have a waiting period of 36 months."),
    (HEALTH, "7. Sub-limits", "Cataract surgery is limited to Rs. 40,000 per eye per policy year."),
    (HEALTH, "9. Permanent Exclusions", "Infertility treatments including IVF are excluded."),
    (HOME, "6. Insured Perils", "Flood, storm and inundation damage to the building is covered."),
    (HOME, "7. Deductibles", "Earthquake deductible is 5% of the claim, minimum Rs. 25,000."),
    (HOME, "10. General Exclusions", "Mechanical breakdown of appliances is excluded. Diabetes is not relevant here."),
]


@pytest.fixture(autouse=True)
def store(monkeypatch):
    collection = ephemeral_collection()
    collection.add(
        ids=[f"{src}::{i}" for i, (src, _, _) in enumerate(CHUNKS)],
        documents=[text for _, _, text in CHUNKS],
        embeddings=fake_embed([text for _, _, text in CHUNKS]),
        metadatas=[{"source": src, "section": sec} for src, sec, _ in CHUNKS],
    )
    monkeypatch.setattr(retriever, "get_collection", lambda: collection)
    monkeypatch.setattr(retriever, "embed", fake_embed)
    monkeypatch.setattr(retriever, "_reranker", FakeReranker)
    monkeypatch.setattr(retriever, "_bm25_index", retriever._BM25Index())  # no BM25 cache shared between tests
    return collection


def test_finds_the_relevant_section():
    results = retriever.retrieve("waiting period for diabetes", [HEALTH])
    assert results[0]["metadata"]["section"] == "6. Waiting Periods"
    assert isinstance(results[0]["score"], float)


def test_only_searches_the_customers_own_documents():
    # "Diabetes" also appears in the home policy chunk, which must never be returned for a health-only customer.
    results = retriever.retrieve("diabetes", [HEALTH])
    assert results
    assert {r["metadata"]["source"] for r in results} == {HEALTH}
    assert {r["metadata"]["source"] for r in retriever.retrieve("diabetes", [HOME])} == {HOME}


def test_no_documents_means_no_results(store, monkeypatch):
    monkeypatch.setattr(retriever, "get_collection", lambda: pytest.fail("vector store should not be queried"))
    assert retriever.retrieve("anything", []) == []


def test_results_are_ordered_by_reranker_score_and_limited_to_top_k():
    results = retriever.retrieve("earthquake deductible claim", [HEALTH, HOME], top_k=2)
    assert len(results) == 2
    assert results[0]["metadata"]["section"] == "7. Deductibles"
    assert results[0]["score"] >= results[1]["score"]


def test_bm25_ignores_chunks_without_matching_words(store):
    hits = retriever._bm25_search(store, "cataract", [HEALTH, HOME])
    assert [meta["section"] for _, _, meta in hits] == ["7. Sub-limits"]


def test_bm25_index_is_rebuilt_when_chunks_are_added(store):
    assert retriever._bm25_search(store, "burglary", [HOME]) == []
    store.add(ids=["home.pdf::99"], documents=["Burglary needs an FIR within 24 hours."],
              embeddings=fake_embed(["burglary"]), metadatas=[{"source": HOME, "section": "13. Claims"}])
    assert len(retriever._bm25_search(store, "burglary", [HOME])) == 1


def test_rrf_rewards_documents_found_by_both_searches():
    a, b, c = ("a", "doc a", {}), ("b", "doc b", {}), ("c", "doc c", {})
    fused = retriever._rrf([a, b], [c, b])
    assert [item[0] for item in fused] == ["b", "a", "c"]


def test_rrf_with_empty_lists():
    assert retriever._rrf([], []) == []


def test_tokenize():
    assert retriever._tokenize("Rs. 40,000 per EYE!") == ["rs", "40", "000", "per", "eye"]
