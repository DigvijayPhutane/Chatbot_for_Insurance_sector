import json
import os

import ollama

INTENT_MODEL = os.getenv("INTENT_MODEL", "llama3.2:1b")

ROUTES = {
    "rag": "Questions about policy terms, coverage, benefits, exclusions, waiting periods, sub-limits, "
           "premium breakup, claim procedure, definitions - anything answered by the policy document.",
    "customer_details": "The customer's own profile: name, contact details, KYC status, list of policies held.",
    "policy_status": "Live status: is the policy active/lapsed/in grace, premium paid or due, next due date, "
                     "renewal date, status of a claim the customer has filed.",
    "general": "Greetings, thanks, small talk, or questions unrelated to insurance.",
}

SYSTEM = (
    "You are the intent classifier of an insurance customer-support chatbot. "
    "Pick every route needed to answer the latest user message (a message can need more than one, "
    "e.g. 'is my policy active and does it cover maternity?' needs policy_status and rag).\n\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in ROUTES.items())
    + "\n\nAlso rewrite the latest message as a standalone search query (resolve pronouns and follow-ups "
    "using the conversation, but do not carry over topics the user has moved away from), and extract a "
    "policy number if the user mentions one. Reply in JSON only.\n\n"
    "Examples:\n"
    'Message: "What is the waiting period for diabetes?"\n'
    '{"routes": ["rag"], "search_query": "waiting period for diabetes", "policy_number": ""}\n'
    'Message: "Is my policy active? When is my premium due?"\n'
    '{"routes": ["policy_status"], "search_query": "policy status and premium due date", "policy_number": ""}\n'
    'Message: "What is my registered email address?"\n'
    '{"routes": ["customer_details"], "search_query": "registered email address", "policy_number": ""}\n'
    'Message: "What is the status of my claim and is flood damage covered?"\n'
    '{"routes": ["policy_status", "rag"], "search_query": "claim status; flood damage coverage", "policy_number": ""}\n'
    'Message: "Hi, thanks a lot!"\n'
    '{"routes": ["general"], "search_query": "thanks", "policy_number": ""}'
)

SCHEMA = {
    "type": "object",
    "properties": {
        "routes": {"type": "array", "items": {"type": "string", "enum": list(ROUTES)}},
        "search_query": {"type": "string"},
        "policy_number": {"type": "string", "description": "Empty string if none mentioned."},
    },
    "required": ["routes", "search_query", "policy_number"],
}


def classify(message: str, history: list[dict]) -> dict:
    """Return {"routes": [...], "search_query": str, "policy_number": str}."""
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    response = ollama.chat(
        model=INTENT_MODEL,
        format=SCHEMA,
        options={"temperature": 0},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<conversation>\n{transcript}\n</conversation>\n\n"
                                        f"<latest_message>\n{message}\n</latest_message>"},
        ],
    )
    try:
        intent = json.loads(response.message.content)
    except json.JSONDecodeError:
        intent = {}
    # Small local models can drift from the schema; fall back to safe defaults.
    routes = list(dict.fromkeys(r for r in intent.get("routes", []) if r in ROUTES)) or ["rag"]
    if routes != ["general"]:
        # Any insurance question also gets document search: retrieval is local and cheap, and a missed
        # "rag" route leaves the answer model without the policy text it needs.
        routes = [r for r in routes if r != "general"]
        routes += [] if "rag" in routes else ["rag"]
    return {
        "routes": routes,
        "search_query": intent.get("search_query") or message,
        "policy_number": intent.get("policy_number") or "",
    }
