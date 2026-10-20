#!/usr/bin/env bash
# Rebuild Candy's shimeji images (images/shimeji/*.webp) from images/candy_stare.jpg.
#
# usage: _shimeji/build.sh [work_dir]        (default work dir: /tmp/candy-shimeji-work)
# Set PYTHON to an interpreter that has requirements.txt installed. The cutout (slow, needs the ~1 GB
# BiRefNet model) is cached in the work dir; delete cutout.png there to redo it.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
photo="$repo/images/candy_stare.jpg"
work=${1:-/tmp/candy-shimeji-work}
py=${PYTHON:-python3}

mkdir -p "$work/frames" "$work/tail" "$work/assets"
[ -f "$work/cutout.png" ] || "$py" "$here/cutout.py" "$photo" "$work/cutout.png"
"$py" "$here/build_master.py" "$photo" "$work/cutout.png" "$work"
"$py" "$here/eyes.py" "$photo" "$work/master.png" "$work/master_meta.json" "$work/frames"
"$py" "$here/tail_spine.py" "$work/master_cat.png" "$work/tail_spine.json" "$work/tail_spine.png"
"$py" "$here/tail.py" "$work/master_cat.png" "$work/master_shadow.png" "$work/tail_spine.json" \
  "$work/frames/frame_c.png" "$work/tail"
"$py" "$here/build_assets.py" "$work/frames" "$work/tail" "$work/master_cat.png" "$work/assets"
"$py" "$here/preview.py" "$work"
cp "$work/assets/candy.webp" "$work/assets/candy-eyes.webp" "$work/assets/candy-tail.webp" "$repo/images/shimeji/"

echo
echo "Copied the images to images/shimeji/. The sprite boxes and grab outline for the CSS are in"
echo "$work/assets/candy-shimeji.json; if they changed, update _includes/candy-shimeji.html to match."
