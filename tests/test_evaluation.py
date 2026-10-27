"""The evaluation's scoring rules, and the dataset's consistency with the documents and sample data."""

import json
from pathlib import Path

import pytest

from customer_api.db import CUSTOMERS
from evaluation.run_eval import DATASET, missing_facts, normalize, routing_ok, summarize
from ingestion.chunker import chunk_pdf
from rag.intent import ROUTES
from tests.helpers import word_count

FILES = Path(__file__).parent.parent / "files"
CASES = json.loads(DATASET.read_text(encoding="utf-8"))


def test_normalize():
    assert normalize("Rs. **8,50,000**  per\nyear, don’t") == "rs. 850000 per year don't"


@pytest.mark.parametrize(("answer", "missing"), [
    ("The sub-limit is Rs. 40,000 per eye.", []),
    ("The sub-limit is Rs 40000 per eye.", []),
    ("There is a sub-limit per eye.", [["40,000"]]),
])
def test_missing_facts(answer, missing):
    assert missing_facts(answer, [["40,000"]]) == missing


def test_every_fact_group_must_match():
    groups = [["active"], ["2027-04-01", "1 april 2027"]]
    assert missing_facts("The policy is Active; next premium due on 1 April, 2027.", groups) == []
    assert missing_facts("The policy is active.", groups) == [groups[1]]


@pytest.mark.parametrize(("expected", "predicted", "ok"), [
    (["rag"], ["rag"], True),
    (["policy_status"], ["policy_status", "rag"], True),
    (["policy_status"], ["rag"], False),
    (["general"], ["general"], True),
    (["general"], ["rag"], False),
])
def test_routing_ok(expected, predicted, ok):
    assert routing_ok(expected, predicted) is ok


def test_summarize():
    results = [
        {"routing": {"ok": True}, "retrieval": {"hit": True, "rr": 1.0},
         "answer": {"ok": True, "scored": True, "foreign_sources": [], "seconds": 2.0}},
        {"routing": {"ok": False}, "retrieval": {"hit": False, "rr": 0.0},
         "answer": {"ok": False, "scored": True, "foreign_sources": ["Other"], "seconds": 4.0}},
        {"routing": {"ok": True},
         "answer": {"ok": True, "scored": False, "foreign_sources": [], "seconds": 3.0}},
    ]
    assert summarize(results) == {
        "routing_accuracy": 0.667, "retrieval_hit_at_5": 0.5, "retrieval_mrr": 0.5, "answer_accuracy": 0.5,
        "isolation_violations": 1, "answer_seconds_avg": 3.0, "cases": 3}


# --- dataset consistency: catches typos that would make a case impossible to pass ---

def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_case_refers_to_real_customers_routes_and_sections(case):
    assert case["customer_id"] in {c[0] for c in CUSTOMERS}
    assert set(case["routes"]) <= set(ROUTES)
    if case["sections"]:
        sections = {c["metadata"]["section"] for pdf in FILES.glob("*.pdf")
                    for c in chunk_pdf(pdf.read_bytes(), pdf.name, word_count, 200)}
        assert set(case["sections"]) <= sections
