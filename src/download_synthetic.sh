#!/usr/bin/env bash
# Szybkie pobieranie synthetic RarePlanes przez rownolegly HTTPS (curl).
# ~20x szybciej niz aws-cli (omija throttling/narzut klienta). Resume: pomija
# pliki ktore juz sa i maja niezerowy rozmiar.
#
# Uzycie: bash src/download_synthetic.sh [split] [parallel]
#   split: train|test (domyslnie train)
#   parallel: liczba rownoleglych pobran (domyslnie 24)
set -uo pipefail

SPLIT="${1:-train}"
P="${2:-24}"
BASE="https://rareplanes-public.s3.amazonaws.com/synthetic/${SPLIT}/images"
DEST="data/synthetic/images/${SPLIT}"
mkdir -p "$DEST"

echo "[$(date)] lista obiektow ${SPLIT}..."
aws s3 ls "s3://rareplanes-public/synthetic/${SPLIT}/images/" 2>/dev/null \
  | awk '{print $NF}' | grep '\.png$' > "/tmp/all_${SPLIT}.txt"
total=$(wc -l < "/tmp/all_${SPLIT}.txt")

# brakujace = w buckecie, a na dysku brak LUB plik pusty
: > "/tmp/missing_${SPLIT}.txt"
while read -r f; do
  if [ ! -s "$DEST/$f" ]; then echo "$f"; fi
done < "/tmp/all_${SPLIT}.txt" > "/tmp/missing_${SPLIT}.txt"
missing=$(wc -l < "/tmp/missing_${SPLIT}.txt")
echo "[$(date)] ${SPLIT}: total=$total, brakuje=$missing, parallel=$P"

# pobieranie rownolegle; -f = blad HTTP -> niezerowy exit (pomijamy 403/404 cicho)
export BASE DEST
cat "/tmp/missing_${SPLIT}.txt" | xargs -P"$P" -I{} sh -c '
  curl -sf -o "$DEST/{}.part" "$BASE/{}" && mv "$DEST/{}.part" "$DEST/{}" || rm -f "$DEST/{}.part"
'

have=$(ls "$DEST"/*.png 2>/dev/null | wc -l)
echo "[$(date)] ${SPLIT} DONE: na dysku $have / $total"
