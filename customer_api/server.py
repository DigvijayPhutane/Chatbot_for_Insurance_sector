from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from customer_api.db import connect, load_db


@asynccontextmanager
async def lifespan(app):
    load_db()
    yield


app = FastAPI(title="Customer & Policy Status API", lifespan=lifespan)


def _customer_or_404(conn, customer_id):
    row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return dict(row)


@app.get("/customers")
def list_customers():
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT customer_id, name, city FROM customers ORDER BY name")]


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    with connect() as conn:
        return _customer_or_404(conn, customer_id)


@app.get("/customers/{customer_id}/policies")
def get_policies(customer_id: str):
    """All policies of a customer, each with its claims."""
    with connect() as conn:
        _customer_or_404(conn, customer_id)
        policies = [dict(r) for r in conn.execute(
            "SELECT * FROM policies WHERE customer_id = ? ORDER BY start_date DESC", (customer_id,))]
        for p in policies:
            p["claims"] = [dict(r) for r in conn.execute(
                "SELECT * FROM claims WHERE policy_number = ? ORDER BY filed_on DESC", (p["policy_number"],))]
        return policies
