#!/usr/bin/env bash
# Update all repos and reinstall dependencies
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=== Pulling main repo ==="
git pull --rebase

echo ""
echo "=== Updating submodule: oll_project_law_as_code ==="
cd "$PROJECT_DIR/oll_project_law_as_code"
git fetch origin
git pull origin "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)" || git pull origin main

echo ""
echo "=== Updating submodule: oll_2026_law_as_code_case_classifier ==="
cd "$PROJECT_DIR/oll_2026_law_as_code_case_classifier"
git fetch origin
git pull origin "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)" || git pull origin main

echo ""
echo "=== Reinstalling dependencies ==="
cd "$PROJECT_DIR"
~/.local/bin/uv sync

echo ""
echo "=== Restarting service ==="
sudo systemctl restart law-as-code-api

echo ""
echo "Done! Check status with: systemctl status law-as-code-api"
