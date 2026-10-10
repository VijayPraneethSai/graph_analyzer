#!/usr/bin/env bash
# Downloads the three SNAP graphs used in the benchmark into data/.
# Usage: scripts/download_data.sh [--large]
#   --large  additionally downloads SNAP com-LiveJournal (about 35M edges, ~1.3 GB as text, ~250 MB gzipped)
#            to data/com-LiveJournal.txt. Without the flag nothing changes.
# Each graph is fetched to a temp name, then gunzip -c'd into the final file, so an
# interrupted download never leaves a half-written data/X.txt. Existing non-empty files are skipped.
set -u
LARGE=0
for a in "$@"; do
  case "$a" in
    --large) LARGE=1 ;;
    *) echo "usage: $0 [--large]" >&2; exit 2 ;;
  esac
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data"
cd "$ROOT/data" || exit 1

# "remote name on SNAP (case matters!)|local name the project expects"
GRAPHS="wiki-Vote.txt.gz|Wiki-Vote.txt
email-Enron.txt.gz|email-Enron.txt
as-skitter.txt.gz|as-Skitter.txt"
BASE=https://snap.stanford.edu/data
# A remote that starts with http is used as a full URL instead of being appended to BASE.
if [ "$LARGE" -eq 1 ]; then
  GRAPHS="$GRAPHS
https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz|com-LiveJournal.txt"
fi

fetch() {  # fetch URL OUTFILE  (retries up to 3 times; wget preferred, curl fallback)
  if command -v wget >/dev/null 2>&1; then
    wget -q --tries=3 --waitretry=2 -O "$2" "$1"
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 --retry-delay 2 -o "$2" "$1"
  else
    echo "ERROR: neither wget nor curl is installed" >&2; return 1
  fi
}

edge_lines() { grep -vc '^#' "$1" 2>/dev/null || true; }

FILES=""
while IFS='|' read -r remote local; do
  [ -z "$remote" ] && continue
  FILES="$FILES $local"
  if [ -s "$local" ]; then echo "skip $local (already present)"; continue; fi
  case "$remote" in http*) url="$remote" ;; *) url="$BASE/$remote" ;; esac
  tmp="$local.gz.part"
  echo "downloading $url"
  if ! fetch "$url" "$tmp"; then
    rm -f "$tmp"; echo "ERROR: download failed for $url" >&2; exit 1
  fi
  if ! gunzip -c "$tmp" > "$local.part"; then
    rm -f "$tmp" "$local.part"; echo "ERROR: could not gunzip download from $url" >&2; exit 1
  fi
  rm -f "$tmp"
  mv "$local.part" "$local"
  n=$(edge_lines "$local")
  if [ ! -s "$local" ] || [ "${n:-0}" -lt 100 ]; then
    echo "ERROR: $local from $url has only ${n:-0} edge lines (expected >= 100)" >&2
    rm -f "$local"; exit 1
  fi
done <<EOT
$GRAPHS
EOT

echo "== summary"
for f in $FILES; do
  if [ ! -s "$f" ]; then echo "ERROR: $f missing or empty" >&2; exit 1; fi
  printf '%-20s %12s bytes %12s edge lines\n' "data/$f" "$(wc -c < "$f" | tr -d ' ')" "$(edge_lines "$f")"
done
