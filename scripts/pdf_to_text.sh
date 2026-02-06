#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [SRC_DIR] [DEST_DIR] [-f|--force]

Recursively finds PDF files in SRC_DIR (defaults to current dir) and
converts them to UTF-8 text files under DEST_DIR (defaults to "books").
Existing .txt files are skipped unless -f/--force is given.

Example:
  $(basename "$0") b books
  $(basename "$0") ./pdfs ./books -f
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SRC_DIR="${1:-.}"
DEST_DIR="${2:-books}"
FORCE=0
if [[ "${3:-}" == "-f" || "${3:-}" == "--force" ]]; then
  FORCE=1
fi

if ! command -v pdftotext >/dev/null 2>&1; then
  echo "Error: pdftotext not found. Install poppler-utils (apt) or poppler (brew)."
  exit 2
fi

mkdir -p "$DEST_DIR"

# Find PDFs (case-insensitive) under SRC_DIR and convert them.
# We avoid cd-ing into SRC_DIR to keep paths stable; preserve relative
# directory structure under DEST_DIR.
find "$SRC_DIR" -type f \( -iname '*.pdf' \) -print0 |
while IFS= read -r -d '' pdf_abs; do
  # normalize leading ./ if present
  pdf_abs=${pdf_abs#./}
  out_rel="${pdf_abs%.*}.txt"
  out_abs="$DEST_DIR/$out_rel"
  mkdir -p "$(dirname "$out_abs")"
  if [[ -f "$out_abs" && $FORCE -ne 1 ]]; then
    printf "Skipping (exists): %s\n" "$out_abs"
    continue
  fi
  printf "Converting: %s -> %s\n" "$pdf_abs" "$out_abs"
  pdftotext -layout -nopgbrk -enc UTF-8 "$pdf_abs" "$out_abs"
done

printf "Done. Text files written under: %s\n" "$DEST_DIR"
