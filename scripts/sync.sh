#!/usr/bin/env bash
# Push the working tree to the compute box (code only -- never datasets or results).
#
#   bash scripts/sync.sh              # push code up
#   bash scripts/sync.sh pull         # bring results back down
#
# Uses tar-over-ssh rather than rsync: Git Bash on Windows ships no rsync, and this
# needs nothing beyond ssh on either end.
#
# Deliberately not git-based -- iterating on a server should not require a commit per
# edit. Git stays the record of what was *decided*; this carries work in progress.
#
# Note on excludes: only whole-name patterns that cannot collide with package
# directories are listed. An earlier version excluded 'data', which silently matched
# src/arc/data/ and shipped a package with a hole in it.

set -euo pipefail

HOST="${ARC_HOST:-prashant@100.103.26.120}"
REMOTE="${ARC_REMOTE:-/home/prashant/arc}"
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Only source trees are pushed, so top-level data/ results/ logs/ are excluded by
# simply not being listed.
PAYLOAD=(src experiments scripts tests docs configs pyproject.toml)

EXCLUDES=(--exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache')

case "${1:-push}" in
    push)
        echo "[sync] $LOCAL -> $HOST:$REMOTE"
        cd "$LOCAL"
        present=()
        for p in "${PAYLOAD[@]}"; do [ -e "$p" ] && present+=("$p"); done
        tar czf - "${EXCLUDES[@]}" "${present[@]}" \
            | ssh "$HOST" "mkdir -p '$REMOTE' && cd '$REMOTE' && tar xzf -"
        ;;
    pull)
        echo "[sync] $HOST:$REMOTE/results -> $LOCAL/results"
        mkdir -p "$LOCAL/results"
        ssh "$HOST" "cd '$REMOTE' && tar czf - results" | tar xzf - -C "$LOCAL"
        ;;
    *)
        echo "usage: $0 [push|pull]" >&2
        exit 1
        ;;
esac
echo "[sync] done"
