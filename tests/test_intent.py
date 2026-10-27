import json

import pytest

from rag import intent
from tests.helpers import chat_response


@pytest.fixture
def model(monkeypatch):
    """Replace the intent model; set `model.reply` to what it should return."""
    class Model:
        reply = "{}"
        calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return chat_response(self.reply)

    fake = Model()
    monkeypatch.setattr(intent.ollama, "chat", fake.chat)
    return fake


def reply(routes, search_query="q", policy_number=""):
    return json.dumps({"routes": routes, "search_query": search_query, "policy_number": policy_number})


@pytest.mark.parametrize(("routes", "expected"), [
    (["rag"], ["rag"]),
    (["policy_status"], ["policy_status", "rag"]),            # document search is always added
    (["customer_details", "rag"], ["customer_details", "rag"]),
    (["general"], ["general"]),                               # small talk skips retrieval
    (["general", "policy_status"], ["policy_status", "rag"]),  # general dropped when mixed
    (["policy_status", "policy_status"], ["policy_status", "rag"]),
    (["made_up_route"], ["rag"]),
    ([], ["rag"]),
])
def test_routes(model, routes, expected):
    model.reply = reply(routes)
    assert intent.classify("question", [])["routes"] == expected


def test_invalid_json_falls_back_to_rag_with_the_original_message(model):
    model.reply = "not json"
    assert intent.classify("Is IVF covered?", []) == {
        "routes": ["rag"], "search_query": "Is IVF covered?", "policy_number": ""}


def test_empty_search_query_falls_back_to_the_message(model):
    model.reply = reply(["rag"], search_query="")
    assert intent.classify("Is IVF covered?", [])["search_query"] == "Is IVF covered?"


def test_passes_through_query_and_policy_number(model):
    model.reply = reply(["policy_status"], "status of policy MSG/HM/2026/7731940", "MSG/HM/2026/7731940")
    result = intent.classify("What about MSG/HM/2026/7731940?", [])
    assert result["search_query"] == "status of policy MSG/HM/2026/7731940"
    assert result["policy_number"] == "MSG/HM/2026/7731940"


def test_request_uses_schema_zero_temperature_and_recent_history(model):
    history = [{"role": "user", "content": f"message {i}"} for i in range(10)]
    model.reply = reply(["rag"])
    intent.classify("and for cataract?", history)
    call = model.calls[-1]
    assert call["model"] == intent.INTENT_MODEL
    assert call["format"] == intent.SCHEMA
    assert call["options"]["temperature"] == 0
    prompt = call["messages"][1]["content"]
    assert "message 9" in prompt and "message 4" in prompt and "message 3" not in prompt  # last 6 turns only
    assert "and for cataract?" in prompt
