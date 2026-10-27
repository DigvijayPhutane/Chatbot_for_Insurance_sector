import json
import os

import httpx
import ollama
from dotenv import load_dotenv

load_dotenv()

from rag.intent import classify  # noqa: E402  (needs .env loaded first)
from rag.retriever import retrieve  # noqa: E402

MAIN_MODEL = os.getenv("MAIN_MODEL", "llama3.2:latest")
CONTEXT_WINDOW = int(os.getenv("OLLAMA_NUM_CTX", "8192"))  # Ollama's small default would truncate the context
CUSTOMER_API_URL = os.getenv("CUSTOMER_API_URL", "http://localhost:8001")

SYSTEM = """You are an internal assistant for an insurance company's customer-support team. A support agent \
is helping one customer (on a call or chat) and asks you about that customer's policies, so they can answer quickly.

Answer using only the information inside <policy_documents> and <customer_data>. If the answer is not \
there, say so plainly so the agent can escalate - never guess amounts, limits, dates or coverage, and never \
say something is or is not covered unless the policy document says so. When you use the policy document, \
mention the section so the agent can quote it. Refer to the customer in the third person ("the customer", \
or by name). Lead with the direct answer, keep it short, and use Rs. for amounts."""

class Chatbot:
    """One conversation with one logged-in customer."""

    def __init__(self, customer_id: str):
        self.customer_id = customer_id
        self.history: list[dict] = []

    def _get(self, path):
        response = httpx.get(f"{CUSTOMER_API_URL}{path}", timeout=10)
        response.raise_for_status()
        return response.json()

    def _customer_sources(self, policies):
        return [p["document_key"] for p in policies if p.get("document_key")]

    def _build_context(self, intent):
        routes = set(intent["routes"])
        policies = self._get(f"/customers/{self.customer_id}/policies")
        parts, chunks = [], []

        if "rag" in routes:
            chunks = retrieve(intent["search_query"], self._customer_sources(policies))
            docs = "\n\n".join(
                f"<chunk source=\"{c['metadata']['title']}\" section=\"{c['metadata']['section']} "
                f"{c['metadata']['subsection']}\" page=\"{c['metadata']['page']}\">\n{c['text']}\n</chunk>"
                for c in chunks
            )
            parts.append(f"<policy_documents>\n{docs or 'No matching policy document found.'}\n</policy_documents>")

        data = {}
        if "customer_details" in routes:
            data["customer"] = self._get(f"/customers/{self.customer_id}")
        if routes & {"customer_details", "policy_status"}:
            wanted = intent.get("policy_number", "").strip().upper()
            data["policies"] = [p for p in policies if p["policy_number"].upper() == wanted] or policies
            if wanted and not any(p["policy_number"].upper() == wanted for p in policies):
                data["note"] = f"Policy {wanted} is not held by this customer; showing their own policies instead."
        if data:
            parts.append(f"<customer_data>\n{json.dumps(data, indent=2, default=str)}\n</customer_data>")

        return "\n\n".join(parts), chunks

    def ask(self, message: str):
        """Yield the answer text as it streams. Afterwards `self.last_turn` holds route and sources."""
        intent = classify(message, self.history)
        context, chunks = self._build_context(intent)

        user_content = f"{context}\n\n<question>\n{message}\n</question>" if context else message
        answer = ""
        for part in ollama.chat(
            model=MAIN_MODEL,
            stream=True,
            options={"temperature": 0.2, "num_ctx": CONTEXT_WINDOW},
            messages=[{"role": "system", "content": SYSTEM}, *self.history,
                      {"role": "user", "content": user_content}],
        ):
            text = part.message.content
            if text:
                answer += text
                yield text

        # Store the plain question (not the retrieved context) to keep history small.
        self.history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        self.last_turn = {
            "routes": intent["routes"],
            "search_query": intent["search_query"],
            "sources": [
                {k: c["metadata"][k] for k in ("title", "section", "subsection", "page")} for c in chunks
            ],
        }
