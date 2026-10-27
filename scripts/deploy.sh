#!/usr/bin/env bash
# Runs on the server (sent over SSH by .github/workflows/deploy.yml).
# Deploys exactly the commit CI tested, re-indexes changed PDFs, then smoke-tests the app.
set -euo pipefail

cd "$DEPLOY_PATH"
git fetch --quiet origin
git checkout --quiet --detach "$SHA"
echo "Deploying $(git log -1 --format='%h %s')"

docker compose up -d --build --wait
docker compose run --rm indexer        # unchanged documents are skipped by the hash check

# Smoke test: the page loads and the customer API answers through the app.
curl -fsS http://127.0.0.1:8000/ > /dev/null
curl -fsS http://127.0.0.1:8000/api/customers | grep -q customer_id
docker image prune -f > /dev/null
echo "Deployed $SHA"
