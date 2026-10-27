import os
import sqlite3
from pathlib import Path

from botocore.exceptions import ClientError

from ingestion.dms import BUCKET, ensure_bucket, s3

DB_PATH = Path(__file__).parent / "insurance.db"
DB_KEY = os.getenv("DB_S3_KEY", "database/insurance.db")  # S3 holds the master copy; DB_PATH is a local working copy

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT,
    mobile      TEXT,
    city        TEXT,
    kyc_status  TEXT
);
CREATE TABLE IF NOT EXISTS policies (
    policy_number  TEXT PRIMARY KEY,
    customer_id    TEXT NOT NULL REFERENCES customers(customer_id),
    product        TEXT,
    policy_type    TEXT,
    insurer        TEXT,
    status         TEXT,       -- Active / Grace Period / Lapsed / Expired
    start_date     TEXT,
    end_date       TEXT,
    sum_insured    REAL,
    total_premium  REAL,
    premium_status TEXT,       -- Paid / Due / Overdue
    next_due_date  TEXT,
    document_key   TEXT        -- S3 key of the policy PDF, used to scope RAG retrieval
);
CREATE TABLE IF NOT EXISTS claims (
    claim_id        TEXT PRIMARY KEY,
    policy_number   TEXT NOT NULL REFERENCES policies(policy_number),
    claim_type      TEXT,
    filed_on        TEXT,
    amount_claimed  REAL,
    amount_approved REAL,
    status          TEXT,      -- Submitted / Under Review / Approved / Settled / Rejected
    remarks         TEXT
);
"""

# Fictional sample data. The first two customers match the specimen PDFs in files/.
CUSTOMERS = [
    ("CUST-88274615", "Rohan Vikram Mehta", "rohan.mehta.demo@example.com", "+91 98220 41736", "Pune", "Verified"),
    ("MSG-CU-30419277", "Sneha Prakash Iyer", "sneha.iyer.demo@example.com", "+91 99001 58324", "Bengaluru", "Verified"),
    ("CUST-10000001", "Priya Sharma", "priya.sharma.demo@example.com", "+91 90000 11111", "Delhi", "Verified"),
    ("CUST-10000002", "Amit Verma", "amit.verma.demo@example.com", "+91 90000 22222", "Mumbai", "Pending"),
]

POLICIES = [
    ("NSG/HL/2026/0048213", "CUST-88274615", "Suraksha Family Health Plus (Platinum)", "Health - Family Floater",
     "Navshakti General Insurance Co. Ltd.", "Active", "2026-04-01", "2027-03-31", 1000000, 35400, "Paid",
     "2027-04-01", "Health_Insurance_Policy_Suraksha_Family_Health_Plus.pdf"),
    ("NSG/HL/2025/0031877", "CUST-88274615", "Suraksha Family Health Plus (Platinum)", "Health - Family Floater",
     "Navshakti General Insurance Co. Ltd.", "Expired (Renewed)", "2025-04-01", "2026-03-31", 1000000, 33100, "Paid",
     None, None),
    ("MSG/HM/2026/7731940", "MSG-CU-30419277", "Griha Kavach Comprehensive Home Insurance", "Home - Comprehensive",
     "Meridian Shield General Insurance Ltd.", "Active", "2026-06-15", "2027-06-14", 6226000, 10620, "Paid",
     "2027-06-15", "Home_Insurance_Policy_Griha_Kavach.pdf"),
    ("MTR/PC/2025/5512087", "CUST-10000001", "Private Car Package Policy", "Motor - Private Car",
     "Navshakti General Insurance Co. Ltd.", "Grace Period", "2025-10-20", "2026-10-19", 650000, 18450, "Overdue",
     "2026-10-19", None),
    ("LIF/TL/2022/0098123", "CUST-10000002", "Term Life Secure", "Life - Term",
     "Meridian Shield General Insurance Ltd.", "Lapsed", "2022-02-01", "2052-01-31", 10000000, 14200, "Overdue",
     "2026-02-01", None),
]

CLAIMS = [
    ("CLM-MTR-2026-00412", "MTR/PC/2025/5512087", "Own damage - accident repair", "2026-03-14", 42000, 38500,
     "Settled", "Settled after deductible of Rs. 1,000 and depreciation on parts."),
    ("CLM-HM-2026-00087", "MSG/HM/2026/7731940", "Water damage - burst pipe (contents)", "2026-09-02", 65000, None,
     "Under Review", "Surveyor visit completed on 2026-09-06; awaiting repair estimate."),
]


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables and (re)load the sample data."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?,?)", CUSTOMERS)
        conn.executemany("INSERT OR REPLACE INTO policies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", POLICIES)
        conn.executemany("INSERT OR REPLACE INTO claims VALUES (?,?,?,?,?,?,?,?)", CLAIMS)
    print(f"Sample database ready: {DB_PATH}")


def upload_db():
    ensure_bucket()
    s3.upload_file(str(DB_PATH), BUCKET, DB_KEY, ExtraArgs={"ContentType": "application/vnd.sqlite3"})
    print(f"Uploaded: {DB_PATH.name} -> s3://{BUCKET}/{DB_KEY}")


def load_db():
    """Download the database from S3; if it is not there yet, seed it with the sample data and upload it."""
    try:
        s3.download_file(BUCKET, DB_KEY, str(DB_PATH))
        print(f"Downloaded: s3://{BUCKET}/{DB_KEY} -> {DB_PATH}")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NoSuchBucket"):
            raise
        init_db()
        upload_db()


if __name__ == "__main__":
    # Reseed: reload the sample data and overwrite the copy in S3.
    init_db()
    upload_db()
