import os
import socket
from urllib.parse import urlparse

import pytest

# Set before any project module is imported: the S3 client is created at import time.
# Only the emulator's fake credentials are ever used.
for key, value in {
    "AWS_ENDPOINT_URL": "http://localhost:4566",
    "AWS_ACCESS_KEY_ID": "test",
    "AWS_SECRET_ACCESS_KEY": "test",
    "AWS_DEFAULT_REGION": "us-east-1",
}.items():
    os.environ.setdefault(key, value)


def _emulator_reachable() -> bool:
    url = urlparse(os.environ["AWS_ENDPOINT_URL"])
    try:
        with socket.create_connection((url.hostname, url.port or 80), timeout=1):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip integration tests when Floci is not running, unless REQUIRE_INTEGRATION=1 (CI)."""
    if _emulator_reachable() or os.getenv("REQUIRE_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason=f"Floci not reachable at {os.environ['AWS_ENDPOINT_URL']}")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
