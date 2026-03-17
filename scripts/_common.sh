#!/bin/bash
set -euo pipefail

readonly SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)"

cd_repo_root() {
    cd "$REPO_ROOT"
}

ensure_dir() {
    mkdir -p "$1"
}
