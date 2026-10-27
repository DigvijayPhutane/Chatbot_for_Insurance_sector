import json
from types import SimpleNamespace

import pytest

from rag import chatbot

CUSTOMER = {"customer_id": "C1", "name": "Sneha", "email": "sneha@example.com"}
POLICIES = [
    {"policy_number": "HOME/1", "status": "Active", "document_key": "home.pdf", "claims": []},
    {"policy_number": "OLD/1", "status": "Expired", "document_key": None, "claims": []},
]
CHUNK = {"text": "Flood is covered.", "score": 3.0,
         "metadata": {"title": "Home Policy", "section": "6. Insured Perils", "subsection": "", "page": 3}}


@pytest.fixture
def bot(monkeypatch):
    calls = {"retrieve": []}

    def retrieve(query, sources):
        calls["retrieve"].append((query, sources))
        return [CHUNK]

    monkeypatch.setattr(chatbot, "retrieve", retrieve)
    bot = chatbot.Chatbot("C1")
    monkeypatch.setattr(bot, "_get", lambda path: {
        "/customers/C1": CUSTOMER, "/customers/C1/policies": POLICIES}[path])
    bot.calls = calls
    return bot


def intent(*routes, policy_number=""):
    return {"routes": list(routes), "search_query": "flood cover", "policy_number": policy_number}


def customer_data(context):
    return json.loads(context.split("<customer_data>\n")[1].split("\n</customer_data>")[0])


def test_rag_searches_only_the_customers_documents(bot):
    context, chunks = bot._build_context(intent("rag"))
    assert bot.calls["retrieve"] == [("flood cover", ["home.pdf"])]
    assert chunks == [CHUNK]
    assert 'section="6. Insured Perils ' in context and "Flood is covered." in context
    assert "<customer_data>" not in context


def test_no_matching_document(bot, monkeypatch):
    monkeypatch.setattr(chatbot, "retrieve", lambda query, sources: [])
    context, chunks = bot._build_context(intent("rag"))
    assert "No matching policy document found." in context


def test_policy_status_adds_policies_but_not_the_profile(bot):
    context, _ = bot._build_context(intent("policy_status"))
    assert bot.calls["retrieve"] == []
    data = customer_data(context)
    assert [p["policy_number"] for p in data["policies"]] == ["HOME/1", "OLD/1"]
    assert "customer" not in data


def test_customer_details_adds_profile_and_policies(bot):
    data = customer_data(bot._build_context(intent("customer_details"))[0])
    assert data["customer"] == CUSTOMER
    assert len(data["policies"]) == 2


def test_policy_number_narrows_the_policies(bot):
    data = customer_data(bot._build_context(intent("policy_status", policy_number="old/1"))[0])
    assert [p["policy_number"] for p in data["policies"]] == ["OLD/1"]


def test_policy_number_of_someone_else_is_flagged(bot):
    data = customer_data(bot._build_context(intent("policy_status", policy_number="LIFE/99"))[0])
    assert "LIFE/99 is not held by this customer" in data["note"]
    assert len(data["policies"]) == 2


def test_general_route_has_no_context(bot):
    assert bot._build_context(intent("general")) == ("", [])


def test_ask_streams_the_answer_and_records_the_turn(bot, monkeypatch):
    sent = {}

    def chat(**kwargs):
        sent.update(kwargs)
        return iter(SimpleNamespace(message=SimpleNamespace(content=t)) for t in ["Yes, ", "", "flood is covered."])

    monkeypatch.setattr(chatbot, "classify", lambda message, history: intent("rag"))
    monkeypatch.setattr(chatbot.ollama, "chat", chat)

    assert "".join(bot.ask("Is flood covered?")) == "Yes, flood is covered."
    assert sent["model"] == chatbot.MAIN_MODEL and sent["stream"] is True
    assert sent["messages"][0] == {"role": "system", "content": chatbot.SYSTEM}
    assert "<policy_documents>" in sent["messages"][-1]["content"]
    assert "<question>\nIs flood covered?\n</question>" in sent["messages"][-1]["content"]
    # History keeps the plain question, not the retrieved context.
    assert bot.history == [{"role": "user", "content": "Is flood covered?"},
                           {"role": "assistant", "content": "Yes, flood is covered."}]
    assert bot.last_turn == {"routes": ["rag"], "search_query": "flood cover", "sources": [
        {"title": "Home Policy", "section": "6. Insured Perils", "subsection": "", "page": 3}]}


def test_small_talk_is_sent_without_context(bot, monkeypatch):
    sent = {}
    monkeypatch.setattr(chatbot, "classify", lambda message, history: intent("general"))
    monkeypatch.setattr(chatbot.ollama, "chat", lambda **kw: sent.update(kw) or iter([]))
    list(bot.ask("Thanks!"))
    assert sent["messages"][-1] == {"role": "user", "content": "Thanks!"}
    assert bot.last_turn["sources"] == []
