import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from rag.chatbot import CUSTOMER_API_URL, Chatbot

app = FastAPI(title="Insurance Support Assistant")
sessions: dict[str, Chatbot] = {}  # in-memory; use Redis or a DB for production
INDEX_HTML = Path(__file__).parent / "static" / "index.html"


class ChatRequest(BaseModel):
    customer_id: str          # the customer the support agent is currently helping
    message: str
    session_id: str | None = None


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.get("/api/customers")
def customers():
    return httpx.get(f"{CUSTOMER_API_URL}/customers", timeout=10).json()


@app.get("/api/customers/{customer_id}")
def customer_profile(customer_id: str):
    customer = httpx.get(f"{CUSTOMER_API_URL}/customers/{customer_id}", timeout=10).json()
    policies = httpx.get(f"{CUSTOMER_API_URL}/customers/{customer_id}/policies", timeout=10).json()
    return {"customer": customer, "policies": policies}


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    bot = sessions.get(session_id)
    if bot is None or bot.customer_id != req.customer_id:
        bot = sessions[session_id] = Chatbot(req.customer_id)
    answer = "".join(bot.ask(req.message))
    return {"session_id": session_id, "answer": answer, **bot.last_turn}
