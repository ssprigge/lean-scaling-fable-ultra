#!/usr/bin/env bash
# Fetch per-language source corpora as pinned shallow git clones of public
# open-source repositories. Records a manifest with the exact commit SHA of
# every clone for reproducibility. Usage: fetch_corpora.sh <dest-dir>
set -uo pipefail

DEST="${1:?usage: fetch_corpora.sh <dest-dir>}"
mkdir -p "$DEST"
MANIFEST="$DEST/manifest.tsv"
: > "$MANIFEST"

clone() { # clone <lang> <owner/repo>
  local lang="$1" slug="$2" name dir sha
  name="${slug##*/}"
  dir="$DEST/$lang/$name"
  if [ -d "$dir/.git" ]; then
    sha=$(git -C "$dir" rev-parse HEAD)
    echo -e "$lang\t$slug\t$sha\tcached" >> "$MANIFEST"
    return 0
  fi
  mkdir -p "$DEST/$lang"
  for attempt in 1 2 3; do
    if git clone --quiet --depth 1 --single-branch "https://github.com/$slug.git" "$dir"; then
      sha=$(git -C "$dir" rev-parse HEAD)
      echo -e "$lang\t$slug\t$sha\tok" >> "$MANIFEST"
      echo "OK $lang $slug $sha"
      return 0
    fi
    rm -rf "$dir"
    sleep $((attempt * 5))
  done
  echo -e "$lang\t$slug\t-\tFAILED" >> "$MANIFEST"
  echo "FAILED $lang $slug"
  return 1
}

# Lean 4: math library, stdlib extensions, and the compiler/stdlib itself
clone lean leanprover-community/mathlib4
clone lean leanprover-community/batteries
clone lean leanprover/lean4

# Python: math (sympy, topic-matched vs mathlib), web, ORM, microframework
clone python sympy/sympy
clone python django/django
clone python sqlalchemy/sqlalchemy
clone python pallets/flask

# Haskell: theorem prover (topic-matched vs lean4), documents, optics, JSON
clone haskell agda/agda
clone haskell jgm/pandoc
clone haskell ekmett/lens
clone haskell haskell/aeson

# Rust: math (nalgebra), async runtime, serialization, CLI
clone rust dimforge/nalgebra
clone rust tokio-rs/tokio
clone rust serde-rs/serde
clone rust clap-rs/clap

# C: compression (zlib, the essay's example), KV store, transfer, SQL engine
clone c madler/zlib
clone c redis/redis
clone c curl/curl
clone c sqlite/sqlite

# JavaScript: web framework(s), utility library, HTTP client
clone javascript expressjs/express
clone javascript lodash/lodash
clone javascript axios/axios
clone javascript fastify/fastify

echo "=== manifest ==="
cat "$MANIFEST"
