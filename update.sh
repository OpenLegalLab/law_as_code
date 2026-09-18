#!/usr/bin/env bash
# Update the repo, reinstall dependencies, and refresh the deployed frontend.
#
# NOTE: oll_project_law_as_code/ and oll_2026_law_as_code_case_classifier/ became
# ordinary tracked directories in commit b0804a1 ("Replace git submodules with
# inline source directories"). The `git pull --rebase` below already updates
# them -- do NOT re-add per-submodule fetch/pull steps, they abort the script.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

UV="${UV:-$HOME/.local/bin/uv}"
WEBROOT="/var/www/oll-chatbox"

echo "=== Pulling repo ==="
git pull --rebase

echo ""
echo "=== Syncing dependencies ==="
"$UV" sync

echo ""
echo "=== Publishing chatbox to $WEBROOT ==="
# Served from /var/www because Caddy runs as user `caddy`, which cannot
# traverse /home/ubuntu (mode 0750). App.js is the unbuilt React variant and
# is deliberately not published.
sudo install -m 0644 -o root -g root \
    OLL_Chatbox/index.html \
    OLL_Chatbox/shared.js \
    OLL_Chatbox/App.css \
    OLL_Chatbox/logo.jpeg \
    OLL_Chatbox/open-legal-lab.png \
    "$WEBROOT/"

echo ""
echo "=== Restarting API ==="
sudo systemctl restart law-as-code-api
sleep 10
if systemctl is-active --quiet law-as-code-api; then
    echo "law-as-code-api is active"
else
    echo "law-as-code-api FAILED to start:"
    journalctl -u law-as-code-api -n 60 --no-pager
    exit 1
fi

echo ""
echo "=== Smoke test ==="
curl -fsS --max-time 30 localhost:8040/api/v1/health && echo
curl -fsS --max-time 30 https://vps-13d3e726.vps.ovh.net/oll/api/v1/health && echo

echo ""
echo "Done. Logs: journalctl -u law-as-code-api -f"
