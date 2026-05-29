#!/usr/bin/env bash
# Fetch reference papers from arxiv into docs/papers/ (PDFs are gitignored).
#
# Usage:
#   scripts/fetch_papers.sh                    Download every manifest entry not already present.
#   scripts/fetch_papers.sh <arxiv-id> [slug]  Add a paper to the manifest, then download it.
#
# The arxiv id may be given bare (2301.12345), with a version (2301.12345v2),
# or as a full abs/pdf URL — it is normalized to the bare id.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
papers_dir="$repo_root/docs/papers"
manifest="$papers_dir/papers.txt"

mkdir -p "$papers_dir"
touch "$manifest"

normalize_id() {
  local id="$1"
  id="${id##*arxiv.org/abs/}"
  id="${id##*arxiv.org/pdf/}"
  id="${id%.pdf}"
  printf '%s' "$id"
}

download_one() {
  local id="$1" slug="$2"
  local out="$papers_dir/$slug.pdf"
  if [[ -f "$out" ]]; then
    printf '✓ %s (%s) already present\n' "$slug" "$id"
    return
  fi
  printf '↓ %s (%s)\n' "$slug" "$id"
  curl -fsSL "https://arxiv.org/pdf/$id" -o "$out"
}

# Add-and-fetch mode.
if [[ $# -ge 1 ]]; then
  id="$(normalize_id "$1")"
  slug="${2:-$id}"
  if grep -qE "^[[:space:]]*${id}([[:space:]]|$)" "$manifest"; then
    printf 'manifest already lists %s\n' "$id"
  else
    printf '%s\t%s\n' "$id" "$slug" >>"$manifest"
    printf 'added %s -> %s to manifest\n' "$id" "$slug"
  fi
  download_one "$id" "$slug"
  exit 0
fi

# Manifest mode: download anything listed but not yet present.
while read -r id slug _; do
  [[ -z "${id:-}" || "$id" == \#* ]] && continue
  download_one "$id" "${slug:-$id}"
done <"$manifest"
