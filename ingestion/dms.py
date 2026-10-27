import os
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

FILES_DIR = Path(__file__).parent.parent / "files"
BUCKET = os.getenv("S3_BUCKET", "insurance-docs")

# Path-style URLs (http://floci:4566/bucket/key) so the endpoint also resolves as a Compose service name.
s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566"),
    config=Config(s3={"addressing_style": "path"}),
)


def ensure_bucket():
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if BUCKET not in existing:
        s3.create_bucket(Bucket=BUCKET)
        print(f"Created bucket: {BUCKET}")


def upload_pdfs():
    pdfs = sorted(FILES_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {FILES_DIR}")
        return
    for pdf in pdfs:
        s3.upload_file(str(pdf), BUCKET, pdf.name, ExtraArgs={"ContentType": "application/pdf"})
        print(f"Uploaded: {pdf.name} -> s3://{BUCKET}/{pdf.name}")


if __name__ == "__main__":
    ensure_bucket()
    upload_pdfs()
