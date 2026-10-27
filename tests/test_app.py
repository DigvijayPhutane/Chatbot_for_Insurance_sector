from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main


class FakeChatbot:
    created = []

    def __init__(self, customer_id):
        self.customer_id = customer_id
        self.questions = []
        FakeChatbot.created.append(self)

    def ask(self, message):
        self.questions.append(message)
        self.last_turn = {"routes": ["rag"], "search_query": message, "sources": []}
        yield from ["Answer ", "for ", self.customer_id]


@pytest.fixture
def client(monkeypatch):
    FakeChatbot.created = []
    main.sessions.clear()
    monkeypatch.setattr(main, "Chatbot", FakeChatbot)
    return TestClient(main.app)


def chat(client, customer_id, message, session_id=None):
    response = client.post("/chat", json={"customer_id": customer_id, "message": message, "session_id": session_id})
    assert response.status_code == 200
    return response.json()


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Insurance Support Assistant" in response.text


def test_chat_returns_answer_and_route(client):
    data = chat(client, "C1", "Is flood covered?")
    assert data["answer"] == "Answer for C1"
    assert data["routes"] == ["rag"]
    assert data["session_id"]


def test_session_keeps_the_same_conversation(client):
    first = chat(client, "C1", "q1")
    second = chat(client, "C1", "q2", first["session_id"])
    assert second["session_id"] == first["session_id"]
    assert len(FakeChatbot.created) == 1
    assert FakeChatbot.created[0].questions == ["q1", "q2"]


def test_switching_customer_starts_a_new_conversation(client):
    first = chat(client, "C1", "q1")
    data = chat(client, "C2", "q2", first["session_id"])
    assert data["answer"] == "Answer for C2"
    assert [b.customer_id for b in FakeChatbot.created] == ["C1", "C2"]


def test_message_is_required(client):
    assert client.post("/chat", json={"customer_id": "C1"}).status_code == 422


def test_customer_endpoints_proxy_the_customer_api(client, monkeypatch):
    data = {
        "/customers": [{"customer_id": "C1"}],
        "/customers/C1": {"customer_id": "C1", "name": "Sneha"},
        "/customers/C1/policies": [{"policy_number": "P1"}],
    }
    monkeypatch.setattr(main.httpx, "get", lambda url, timeout: SimpleNamespace(
        json=lambda: data[url.removeprefix(main.CUSTOMER_API_URL)]))
    assert client.get("/api/customers").json() == [{"customer_id": "C1"}]
    assert client.get("/api/customers/C1").json() == {
        "customer": {"customer_id": "C1", "name": "Sneha"}, "policies": [{"policy_number": "P1"}]}
