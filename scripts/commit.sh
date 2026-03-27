#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: ./scripts/commit.sh \"commit message\""
  exit 1
fi

git add .
git commit -m "$1"
git push origin main

echo "Committed and pushed: $1"