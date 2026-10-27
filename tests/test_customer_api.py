import pytest
from fastapi.testclient import TestClient

from customer_api import db, server


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Seed a temporary SQLite file instead of downloading the database from S3.
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "insurance.db")
    monkeypatch.setattr(server, "load_db", db.init_db)
    with TestClient(server.app) as client:
        yield client


def test_list_customers(client):
    customers = client.get("/customers").json()
    assert [c["name"] for c in customers] == sorted(c["name"] for c in customers)
    assert {c["customer_id"] for c in customers} == {c[0] for c in db.CUSTOMERS}
    assert set(customers[0]) == {"customer_id", "name", "city"}


def test_get_customer(client):
    customer = client.get("/customers/MSG-CU-30419277").json()
    assert customer["name"] == "Sneha Prakash Iyer"
    assert customer["kyc_status"] == "Verified"


@pytest.mark.parametrize("path", ["/customers/NOPE", "/customers/NOPE/policies"])
def test_unknown_customer_is_404(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert response.json()["detail"] == "Customer not found"


def test_policies_include_claims_newest_first(client):
    policies = client.get("/customers/CUST-88274615/policies").json()
    assert [p["policy_number"] for p in policies] == ["NSG/HL/2026/0048213", "NSG/HL/2025/0031877"]
    assert policies[0]["document_key"] == "Health_Insurance_Policy_Suraksha_Family_Health_Plus.pdf"
    assert all(p["claims"] == [] for p in policies)


def test_claim_is_attached_to_its_policy(client):
    [policy] = client.get("/customers/MSG-CU-30419277/policies").json()
    assert [(c["claim_id"], c["status"]) for c in policy["claims"]] == [("CLM-HM-2026-00087", "Under Review")]


def test_init_db_is_idempotent(client):
    db.init_db()
    assert len(client.get("/customers").json()) == len(db.CUSTOMERS)
